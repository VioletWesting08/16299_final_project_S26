import mujoco
import mujoco.viewer
import numpy as np
import os

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/debug_scene.xml"

# --- Jerk PID Parameters ---
# The PID watches jerk magnitude and outputs a speed scalar in [T_SCALE_MIN, T_SCALE_MAX].
# High jerk  -> scale decreases (slow down)
# Low jerk   -> scale increases (speed up toward 1.0)
JERK_THRESH  = 5.0    # m/s³ — acceptable jerk; PID drives toward this
JERK_KP      = 0.08
JERK_KI      = 0.002
JERK_KD      = 0.005
T_SCALE_MIN  = 0.05
T_SCALE_MAX  = 1.0

# Inner-loop IK gain (fraction of position error closed per step)
IK_GAIN      = 10.0   # higher = stiffer tracking; tune per robot stiffness
IK_LAMBDA_SQ = 1e-4   # damping for DLS IK

# Waypoints: trajectory is FIXED. Only the speed along it changes.
PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 3.0},
    {"name": "2. Descend",   "target_xyz": [0.6, 0.0, 0.12], "gripper": 0.04, "duration": 2.0},
    {"name": "3. Grasp",     "target_xyz": [0.6, 0.0, 0.12], "gripper": 0.00, "duration": 1.5},
    {"name": "4. Lift",      "target_xyz": [0.6, 0.0, 0.40], "gripper": 0.00, "duration": 2.0},
    {"name": "5. Transport", "target_xyz": [0.4, 0.4, 0.40], "gripper": 0.00, "duration": 4.0},
    {"name": "6. Place",     "target_xyz": [0.4, 0.4, 0.12], "gripper": 0.00, "duration": 2.0},
    {"name": "7. Release",   "target_xyz": [0.4, 0.4, 0.12], "gripper": 0.04, "duration": 1.0},
]


# ═══════════════════════════════════════════════════════════════
# JERK PID  (outer loop)
# ═══════════════════════════════════════════════════════════════
class JerkController:
    """
    Measures EE jerk magnitude, compares to threshold, and outputs a
    trajectory speed scalar t_scale ∈ [T_SCALE_MIN, T_SCALE_MAX].

    Convention:
        error > 0  →  jerk below threshold  →  speed up  →  t_scale ↑
        error < 0  →  jerk above threshold  →  slow down →  t_scale ↓
    """
    def __init__(self):
        self.integral   = 0.0
        self.prev_error = 0.0
        self.t_scale    = 1.0          # start at full speed

    def update(self, jerk_mag: float, dt: float) -> float:
        error = JERK_THRESH - jerk_mag  # positive when jerk is low (safe to go faster)

        self.integral   += error * dt
        derivative       = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error  = error

        delta = JERK_KP * error + JERK_KI * self.integral + JERK_KD * derivative
        self.t_scale = float(np.clip(self.t_scale + delta, T_SCALE_MIN, T_SCALE_MAX))
        return self.t_scale


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY SAMPLER
# ═══════════════════════════════════════════════════════════════
class TrajectorySampler:
    """
    Stores the fixed sequence of waypoints and maps a single
    'effective time' value t_eff → (interpolated xyz, gripper width).

    The trajectory shape never changes; only how fast t_eff advances
    (controlled by t_scale from the JerkController) changes the speed.
    """
    def __init__(self, phases):
        # Build cumulative time boundaries from the *nominal* durations
        self.phases     = phases
        self.durations  = [p["duration"] for p in phases]
        self.boundaries = np.cumsum(self.durations)   # shape (N,)
        self.total_time = self.boundaries[-1]

        # Waypoint start positions: phase i starts where phase i-1 ended.
        # Phase 0 implicitly starts wherever the robot currently is (set at runtime).
        self.starts = [None] * len(phases)   # filled in update() on first call

    def set_start(self, xyz_start: np.ndarray):
        """Call once with the robot's actual EE position at t=0."""
        self.starts[0] = xyz_start.copy()
        for i in range(1, len(self.phases)):
            self.starts[i] = np.array(self.phases[i - 1]["target_xyz"])

    def sample(self, t_eff: float):
        """Return (target_xyz, gripper_width) at effective time t_eff."""
        t_eff = float(np.clip(t_eff, 0.0, self.total_time))

        # Which phase are we in?
        idx = int(np.searchsorted(self.boundaries, t_eff, side="right"))
        idx = min(idx, len(self.phases) - 1)

        phase    = self.phases[idx]
        t_start  = self.boundaries[idx - 1] if idx > 0 else 0.0
        duration = self.durations[idx]

        # Normalized progress within this phase [0, 1]
        alpha = (t_eff - t_start) / max(duration, 1e-9)
        alpha = float(np.clip(alpha, 0.0, 1.0))

        # Smooth step (ease-in / ease-out) — keeps trajectory shape smooth
        # so the PID has less work to do at phase transitions
        alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)   # smoothstep

        target_xyz = (1.0 - alpha_s) * self.starts[idx] + alpha_s * np.array(phase["target_xyz"])
        return target_xyz, phase["gripper"]


# ═══════════════════════════════════════════════════════════════
# SCENE BUILDER
# ═══════════════════════════════════════════════════════════════
def build_scene(scene_path, out_path):
    xml = f"""<mujoco>
    <include file="{os.path.abspath(scene_path)}"/>
    <worldbody>
        <body name="centrifuge_tube" pos="0.6 0.0 0.07">
            <freejoint name="tube_joint"/>
            <geom name="tube_geom" type="cylinder" size="0.015 0.05"
                  rgba="0.2 0.7 1.0 0.9" mass="0.05"/>
        </body>
    </worldbody>
</mujoco>"""
    with open(out_path, "w") as f:
        f.write(xml)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    hand_id  = model.body("hand").id
    jerk_pid = JerkController()
    traj     = TrajectorySampler(PHASES)

    # ── warm-up: let the robot settle at its rest pose ──────────
    for _ in range(500):
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    traj.set_start(data.xpos[hand_id].copy())

    # ── Generate and print full trajectory ──────────────────────
    # Sample trajectory at regular intervals to show the full path
    trajectory_samples = 100  # number of points to sample
    full_trajectory = []
    for i in range(trajectory_samples + 1):
        t_sample = i / trajectory_samples * traj.total_time
        xyz, _ = traj.sample(t_sample)
        full_trajectory.append(xyz)
    
    full_trajectory = np.array(full_trajectory)  # shape: (sample_pts, 3)
    
    # Store sample times for tracking during execution
    trajectory_sample_times = np.linspace(0, traj.total_time, trajectory_samples + 1)
    
    print("\n" + "="*70)
    print("FULL TRAJECTORY (linearly interpolated, 100 pts)")
    print("="*70)
    print("Time [s] | X [m]     | Y [m]     | Z [m]")
    print("-"*70)
    for i, xyz in enumerate(full_trajectory[::10]):  # print every 10th point
        t_sample = (i * 10) / trajectory_samples * traj.total_time
        print(f"{t_sample:7.2f} | {xyz[0]:9.4f} | {xyz[1]:9.4f} | {xyz[2]:9.4f}")
    print("-"*70)
    print(f"Full trajectory array shape: {full_trajectory.shape}")
    print("="*70 + "\n")

    # ── history buffers for finite-difference jerk ──────────────
    prev_vel = np.zeros(3)
    prev_acc = np.zeros(3)

    # ── time counters ────────────────────────────────────────────
    # sim_t   : wall-clock simulation time (advances by dt every step)
    # t_eff   : effective trajectory time  (advances by dt * t_scale)
    sim_t = 0.0
    t_eff = 0.0    
    prev_phase_idx = -1  # track phase transitions
    next_sample_idx = 0  # track which trajectory sample point we're at

    print("Starting trajectory …")
    print("\nTracking position error at sampled trajectory points:\n")
    print("Sample# | Time [s] | Desired XYZ                    | Actual XYZ                     | Error [m]")
    print("-"*115)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and t_eff < traj.total_time:

            # ── 1. MEASURE EE VELOCITY → finite-diff acc & jerk ──
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0
            )
            cur_vel = vel6[3:].copy()           # linear velocity (world frame)

            acc  = (cur_vel - prev_vel) / dt
            jerk = (acc - prev_acc) / dt
            jerk_mag = float(np.linalg.norm(jerk))

            # ── 2. OUTER LOOP: jerk PID → speed scalar ────────────
            # Disabled jerk PID for now — run at constant speed
            # t_scale = jerk_pid.update(jerk_mag, dt)
            t_scale = 1.0  # constant speed

            # Advance the effective trajectory clock by the scaled step
            t_eff += dt * t_scale

            # ── 3. SAMPLE FIXED TRAJECTORY at t_eff ──────────────
            target_xyz, gripper_w = traj.sample(t_eff)

            # ── 4. TASK-SPACE IK → Δq  (3-DOF position only) ────
            current_xyz = data.xpos[hand_id].copy()
            dx = target_xyz - current_xyz       # position error in world frame

            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, hand_id)
            J = jacp[:, :7]                     # first 7 DOF

            # DLS: dq = Jᵀ (J Jᵀ + λ²I)⁻¹ dx
            JJT   = J @ J.T
            dq    = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), dx)

            # Target joint position = current + IK_GAIN * dq
            q_des = data.qpos[:7] + IK_GAIN * dq

            # ── 5. SEND TO INNER-LOOP ACTUATORS ──────────────────
            data.ctrl[:7] = q_des

            if model.nu >= 9:
                data.ctrl[7] = gripper_w
                data.ctrl[8] = gripper_w

            # ── 6. STEP PHYSICS ───────────────────────────────────
            mujoco.mj_step(model, data)

            # ── 7. UPDATE HISTORY ─────────────────────────────────
            prev_vel = cur_vel
            prev_acc = acc
            sim_t   += dt

            # ── 8. LOGGING ────────────────────────────────────────
            # Check if we've reached a trajectory sample point
            while next_sample_idx < len(trajectory_sample_times) and t_eff >= trajectory_sample_times[next_sample_idx]:
                sample_time = trajectory_sample_times[next_sample_idx]
                desired_xyz = full_trajectory[next_sample_idx]
                actual_xyz = data.xpos[hand_id].copy()
                error = np.linalg.norm(desired_xyz - actual_xyz)
                
                print(
                    f"{next_sample_idx:7d} | {sample_time:7.2f} | "
                    f"[{desired_xyz[0]:7.4f}, {desired_xyz[1]:7.4f}, {desired_xyz[2]:7.4f}] | "
                    f"[{actual_xyz[0]:7.4f}, {actual_xyz[1]:7.4f}, {actual_xyz[2]:7.4f}] | {error:7.4f}"
                )
                next_sample_idx += 1
            
            phase_idx = int(np.searchsorted(traj.boundaries, t_eff, side="right"))
            phase_idx = min(phase_idx, len(PHASES) - 1)
            
            # Print phase details when phase changes
            if phase_idx != prev_phase_idx:
                phase = PHASES[phase_idx]
                print(
                    f"\n>>> PHASE {phase_idx + 1}: {phase['name']}\n"
                    f"    Target XYZ: {phase['target_xyz']}\n"
                    f"    Gripper:    {phase['gripper']:.4f}\n"
                    f"    Duration:   {phase['duration']:.1f}s\n"
                )
                prev_phase_idx = phase_idx
            
            # Print progress every 200 steps
            if int(sim_t / dt) % 200 == 0:
                print(
                    f"sim={sim_t:6.2f}s | t_eff={t_eff:6.2f}s | "
                    f"jerk={jerk_mag:7.3f} m/s³ | scale={t_scale:.2%} | "
                    f"phase={PHASES[phase_idx]['name']}"
                )

            viewer.sync()

    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)
    print("Trajectory complete.")


if __name__ == "__main__":
    main()