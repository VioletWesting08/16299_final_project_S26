"""
ik_jerk_min.py  –  Pure IK controller with gradient-descent jerk minimisation.

Architecture
============
No PID feedback of any kind. Every timestep we solve:

    min  ||J dq - v_des||²  +  λ||dq||²
         + α_jerk * ||dq - dq_prev||²      ← jerk penalty (limits Δjerk)
         + α_null * ||dq - dq_null||²       ← null-space posture

via one step of gradient descent on the joint-velocity level.  Because the
objective is quadratic in dq the gradient step is exact (one Newton step =
global minimum), giving a closed-form solution that is equivalent to
Damped-Least-Squares IK with jerk regularisation baked in.

Math
----
Objective:  E(dq) = ||J dq - v_des||²
                  + λ²||dq||²
                  + α_j||dq - dq_prev||²
                  + α_n||dq - dq_null||²

∂E/∂(dq) = 0  →

  (JᵀJ + (λ²+α_j+α_n)I) dq = Jᵀ v_des + α_j dq_prev + α_n dq_null

Liquid-inertia evaluation
-------------------------
Identical first-order lag model and metrics as test_picking.py so results
can be compared directly:
  - EE speed (avg / max)
  - Position tracking error (avg / max)
  - Tube tilt error  (avg / max)  — angle between tube axis and a_liquid
  - Liquid mixing angle (avg / max / integrated °·s)

Phases: identical to test_picking.py / kinematics.py.
"""

from __future__ import annotations
import os
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise SystemExit("MuJoCo Python package is not installed.")

# ───────────────────────────────────────────────────────────────
# SCENE / FILE CONFIG
# ───────────────────────────────────────────────────────────────
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/ik_jerk_scene.xml"

# ───────────────────────────────────────────────────────────────
# IK + JERK-MINIMISATION PARAMETERS
# ───────────────────────────────────────────────────────────────
IK_LAMBDA_SQ   = 1e-4    # Tikhonov damping  (standard DLS regularisation)
JERK_ALPHA     = 0.05    # weight on ||dq - dq_prev||²  (jerk penalty)
NULL_ALPHA     = 0.30    # weight on null-space posture term
NULL_GAIN      = 0.5     # scaling of posture error → null velocity
K_FB           = 5.0     # proportional position-feedback gain (closes the loop
                         #   without a PID: adds K_fb*err to the IK target vel)

# Nominal (rest) joint configuration — identical to test_picking.py
Q_NOMINAL = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])

# ───────────────────────────────────────────────────────────────
# LIQUID INERTIA MODEL — identical constants to test_picking.py
# ───────────────────────────────────────────────────────────────
LIQUID_TAU           = 1.0    # liquid reorientation time constant (seconds)
ACCEL_LPFILTER_ALPHA = 0.03   # low-pass filter on raw finite-diff acceleration

# ───────────────────────────────────────────────────────────────
# TRAJECTORY PHASES — identical to test_picking.py / kinematics.py
# ───────────────────────────────────────────────────────────────
PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.04, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6000,  0.0000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4000,  0.4000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "6. Place",     "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.04, "duration": 0.5},
]


# ───────────────────────────────────────────────────────────────
# MINIMUM-JERK TRAJECTORY SAMPLER  (5th-order polynomial)
# ───────────────────────────────────────────────────────────────
class TrajectorySampler:
    """5th-order minimum-jerk spline across all phases."""

    def __init__(self, phases: list[dict]) -> None:
        self.phases     = phases
        self.durations  = [p["duration"] for p in phases]
        self.boundaries = np.cumsum(self.durations)
        self.total_time = float(self.boundaries[-1])
        self._starts: list[np.ndarray] = []
        self._coeffs:  list[tuple]     = []

    def set_start(self, xyz_start: np.ndarray) -> None:
        self._starts = [xyz_start.copy()]
        for i in range(1, len(self.phases)):
            self._starts.append(np.array(self.phases[i - 1]["target_xyz"], dtype=float))

        self._coeffs = []
        for i, phase in enumerate(self.phases):
            s = self._starts[i]
            e = np.array(phase["target_xyz"], dtype=float)
            T = self.durations[i]
            a0 = s
            a3 =  10 * (e - s) / T**3
            a4 = -15 * (e - s) / T**4
            a5 =   6 * (e - s) / T**5
            self._coeffs.append((a0, a3, a4, a5))

    def _phase_idx(self, t: float) -> tuple[int, float]:
        idx = int(np.searchsorted(self.boundaries, t, side="right"))
        idx = min(idx, len(self.phases) - 1)
        t0  = self.boundaries[idx - 1] if idx > 0 else 0.0
        return idx, t - t0

    def sample(self, t: float) -> tuple[np.ndarray, float]:
        t = float(np.clip(t, 0.0, self.total_time))
        idx, tl = self._phase_idx(t)
        a0, a3, a4, a5 = self._coeffs[idx]
        xyz = a0 + a3 * tl**3 + a4 * tl**4 + a5 * tl**5
        return xyz, self.phases[idx]["gripper"]

    def sample_velocity(self, t: float) -> np.ndarray:
        """Analytical first derivative of the spline (m/s)."""
        t = float(np.clip(t, 0.0, self.total_time))
        idx, tl = self._phase_idx(t)
        _, a3, a4, a5 = self._coeffs[idx]
        return 3 * a3 * tl**2 + 4 * a4 * tl**3 + 5 * a5 * tl**4


# ───────────────────────────────────────────────────────────────
# JERK-MINIMISING IK SOLVER
# ───────────────────────────────────────────────────────────────
class JerkMinIKSolver:
    """
    Closed-form (one Newton step) solution to the regularised IK problem.

    At each step solves:
        A dq = b
        A = JᵀJ + (λ² + α_j + α_n) I
        b = Jᵀ v_des  +  α_j·dq_prev  +  α_n·dq_null

    The α_j·dq_prev term pulls the new joint velocity toward the previous
    one, directly penalising joint-space acceleration (jerk proxy).
    """

    def __init__(self,
                 n_joints:   int   = 7,
                 lambda_sq:  float = IK_LAMBDA_SQ,
                 jerk_alpha: float = JERK_ALPHA,
                 null_alpha: float = NULL_ALPHA,
                 null_gain:  float = NULL_GAIN) -> None:
        self.n       = n_joints
        self.lam_sq  = lambda_sq
        self.j_alpha = jerk_alpha
        self.n_alpha = null_alpha
        self.n_gain  = null_gain
        self.dq_prev = np.zeros(n_joints)

    def solve(self,
              J:         np.ndarray,  # (3, n) positional Jacobian
              v_des:     np.ndarray,  # (3,) desired EE velocity
              q_current: np.ndarray  # (n,) current joint angles
              ) -> np.ndarray:
        n = self.n
        dq_null = self.n_gain * (Q_NOMINAL - q_current)
        A   = J.T @ J + (self.lam_sq + self.j_alpha + self.n_alpha) * np.eye(n)
        rhs = J.T @ v_des + self.j_alpha * self.dq_prev + self.n_alpha * dq_null
        dq  = np.linalg.solve(A, rhs)
        self.dq_prev = dq.copy()
        return dq


# ───────────────────────────────────────────────────────────────
# SCENE BUILDER
# ───────────────────────────────────────────────────────────────
def build_scene(scene_path: str, out_path: str) -> None:
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


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────
def main() -> None:
    # ── Build & load scene ────────────────────────────────────
    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    hand_id = model.body("hand").id

    # ── Warm-up (let robot settle at rest pose) ───────────────
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # ── Build trajectory ─────────────────────────────────────
    traj = TrajectorySampler(PHASES)
    traj.set_start(data.xpos[hand_id].copy())

    # ── Controller ───────────────────────────────────────────
    solver = JerkMinIKSolver()

    # ── Print plan ───────────────────────────────────────────
    print("=" * 70)
    print("JERK-MINIMISING IK CONTROLLER  –  phase plan")
    print("=" * 70)
    for p in PHASES:
        print(f"  {p['name']:14s}  target={p['target_xyz']}  "
              f"gripper={p['gripper']:.2f}  T={p['duration']:.1f}s")
    print("=" * 70)
    print(f"\nSolver parameters:")
    print(f"  lambda_sq    (DLS damping)       = {IK_LAMBDA_SQ}")
    print(f"  alpha_jerk   (jerk penalty)      = {JERK_ALPHA}")
    print(f"  alpha_null   (posture)           = {NULL_ALPHA}")
    print(f"  null gain                        = {NULL_GAIN}")
    print(f"  K_fb         (position feedback) = {K_FB}")
    print(f"  liquid tau                       = {LIQUID_TAU} s")
    print(f"  accel LP-filter alpha            = {ACCEL_LPFILTER_ALPHA}")
    print()

    # ── Statistics buffers — identical names/semantics to test_picking.py ──
    speed_values        = []   # EE speed each step
    pos_error_values    = []   # |target_xyz - current_xyz|
    tilt_error_values   = []   # angle(tube_axis, a_liquid)
    mixing_score_values = []   # same angle (separate list mirrors test_picking.py)
    joint_jerk_values   = []   # extra: joint-space jerk proxy

    # ── Liquid inertia state — identical to test_picking.py ──
    a_liquid          = np.array([0.0, 0.0, -9.81])
    prev_ee_vel       = np.zeros(3)
    filtered_ee_accel = np.zeros(3)
    g                 = np.array([0.0, 0.0, -9.81])

    # ── Timing ───────────────────────────────────────────────
    t_eff      = 0.0
    sim_t      = 0.0
    prev_phase = -1

    print("Starting trajectory ...\n")

    # ── Main loop ────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and t_eff < traj.total_time:

            # ── 1. MEASURE EE VELOCITY ────────────────────────
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0
            )
            cur_vel = vel6[3:].copy()
            speed_values.append(float(np.linalg.norm(cur_vel)))

            # ── 2. LIQUID INERTIA MODEL ───────────────────────
            # Identical computation to test_picking.py
            ee_accel_raw = (cur_vel - prev_ee_vel) / max(dt, 1e-9)
            prev_ee_vel  = cur_vel.copy()

            # Low-pass filter — same alpha as test_picking.py
            filtered_ee_accel = (ACCEL_LPFILTER_ALPHA * ee_accel_raw +
                                 (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_ee_accel)

            # Effective gravity: what the liquid "feels" (gravity + inertial)
            a_effective = g + filtered_ee_accel

            # First-order lag — same tau as test_picking.py
            a_liquid += ((a_effective - a_liquid) / LIQUID_TAU) * dt

            # ── 3. ADVANCE TRAJECTORY TIME ────────────────────
            t_eff += dt
            target_xyz, gripper_w = traj.sample(t_eff)
            v_ff = traj.sample_velocity(t_eff)

            # ── 4. POSITION ERROR ─────────────────────────────
            current_xyz = data.xpos[hand_id].copy()
            pos_error   = target_xyz - current_xyz
            pos_error_values.append(float(np.linalg.norm(pos_error)))

            # Feedforward velocity + proportional position feedback
            # (closes the loop without integrators or derivative terms)
            v_des = v_ff + K_FB * pos_error

            # ── 5. COMPUTE JACOBIAN ───────────────────────────
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, hand_id)
            J = jacp[:, :7]

            # ── 6. JERK-MINIMISING IK SOLVE ───────────────────
            dq_prev_snapshot = solver.dq_prev.copy()
            dq    = solver.solve(J, v_des, data.qpos[:7].copy())
            q_des = data.qpos[:7] + dq

            # Joint-space jerk proxy: change in joint velocity / dt
            joint_jerk_values.append(
                float(np.linalg.norm(dq - dq_prev_snapshot) / max(dt, 1e-9))
            )

            # ── 7. APPLY COMMANDS ─────────────────────────────
            data.ctrl[:7] = q_des
            if model.nu >= 8:
                data.ctrl[7] = gripper_w

            # ── 8. STEP PHYSICS ───────────────────────────────
            mujoco.mj_step(model, data)
            sim_t += dt

            # ── 9. MIXING / TILT ANGLE ────────────────────────
            # Identical calculation to test_picking.py
            hand_rot  = data.xmat[hand_id].reshape(3, 3)
            tube_axis = hand_rot[:, 2]
            norm_tube = np.linalg.norm(tube_axis)
            norm_liq  = np.linalg.norm(a_liquid)
            if norm_tube > 1e-9 and norm_liq > 1e-9:
                cos_angle = np.clip(
                    np.dot(tube_axis, a_liquid / norm_liq), -1.0, 1.0
                )
                mix_angle = float(np.degrees(np.arccos(cos_angle)))
            else:
                mix_angle = 0.0

            tilt_error_values.append(mix_angle)
            mixing_score_values.append(mix_angle)

            # ── 10. PHASE TRANSITION LOGGING ──────────────────
            phase_idx = int(np.searchsorted(traj.boundaries, t_eff, side="right"))
            phase_idx = min(phase_idx, len(PHASES) - 1)
            if phase_idx != prev_phase:
                ph = PHASES[phase_idx]
                print(
                    f"\n>>> PHASE {phase_idx + 1}: {ph['name']}\n"
                    f"    Target XYZ: {ph['target_xyz']}\n"
                    f"    Gripper:    {ph['gripper']:.4f}\n"
                    f"    Duration:   {ph['duration']:.1f}s\n"
                )
                prev_phase = phase_idx

            # ── 11. PERIODIC CONSOLE LOG (every 100 steps) ────
            if int(sim_t / dt) % 100 == 0:
                joint_angles = np.degrees(data.qpos[:7])
                joints_str   = " | ".join(
                    f"J{i+1}={ang:7.2f}°" for i, ang in enumerate(joint_angles)
                )
                print(
                    f"sim={sim_t:6.2f}s | t_eff={t_eff:6.2f}s | "
                    f"pos_err={np.linalg.norm(pos_error):.4f}m | "
                    f"mix_angle={mix_angle:6.1f}° | "
                    f"a_eff={np.linalg.norm(a_effective):.2f}m/s²"
                )
                print(f"  Joints: {joints_str}")

            viewer.sync()

    # ── Clean up temp file ────────────────────────────────────
    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)

    # ── PRINT STATISTICS — identical layout to test_picking.py ──────────
    speed_array        = np.array(speed_values)
    pos_error_array    = np.array(pos_error_values)
    tilt_error_array   = np.array(tilt_error_values)
    mixing_score_array = np.array(mixing_score_values)
    joint_jerk_array   = np.array(joint_jerk_values)

    print("\n" + "=" * 70)
    print("TRAJECTORY STATISTICS (IK + Gradient-Descent Jerk Minimisation)")
    print("=" * 70)
    print(f"Total simulation time:     {sim_t:.2f} s")
    print(f"Total trajectory time:     {traj.total_time:.2f} s")

    print(f"\nAverage EE speed:          {np.mean(speed_array):.4f} m/s")
    print(f"Max EE speed:              {np.max(speed_array):.4f} m/s")

    print(f"\nAverage position error:    {np.mean(pos_error_array):.6f} m")
    print(f"Max position error:        {np.max(pos_error_array):.6f} m")

    print(f"\nAverage tube tilt error:   {np.mean(tilt_error_array):.2f}°")
    print(f"Max tube tilt error:       {np.max(tilt_error_array):.2f}°")

    print(f"\nAverage liquid mixing angle: {np.mean(mixing_score_array):.2f}°")
    print(f"Max liquid mixing angle:     {np.max(mixing_score_array):.2f}°")
    print(f"Integrated mixing score:     {np.sum(mixing_score_array) * dt:.2f}°*s")

    # Extra diagnostics not in test_picking.py
    print(f"\n-- Jerk diagnostics (controller-specific) ---------------")
    print(f"Average joint jerk:        {np.mean(joint_jerk_array):.4f} rad/s^2")
    print(f"Max joint jerk:            {np.max(joint_jerk_array):.4f} rad/s^2")
    print(f"Integrated joint jerk:     {np.sum(joint_jerk_array) * dt:.4f} rad/s")
    print("=" * 70)

    print("\nTrajectory complete.")


if __name__ == "__main__":
    main()