"""
trial_runs.py
=============
Sweep LIQUID_TAU from 0.0 → 2.0 seconds, running exact test_picking.py logic each time.
Uses identical settings and controllers as test_picking.py.
"""

import mujoco
import numpy as np
import os
import csv
import time

# ═══════════════════════════════════════════════════════════════
# FIXED CONFIGURATION  (not swept)
# ═══════════════════════════════════════════════════════════════
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/debug_scene.xml"

USE_TASK_SPACE_PID = True
TASK_KP = 0.5
TASK_KI = 0.0
TASK_KD = 0.2

IK_LAMBDA_SQ = 1e-4
IK_GAIN      = 1.0

Q_NOMINAL  = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])
NULL_GAIN  = 0.5

USE_LQR_WEIGHT       = False          # <<< always False for these trials
ACCEL_LPFILTER_ALPHA = 0.03

AGGRESSIVE_TRANSPORT = True
INIT_TUBE_MISALIGNED = False

PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6, 0.0, 0.12],          "gripper": 0.04, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6, 0.0, 0.12],          "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6, 0.0, 0.40],          "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4, 0.4, 0.40],          "gripper": 0.00, "duration": 0.3 if AGGRESSIVE_TRANSPORT else 1.0},
    {"name": "6. Place",     "target_xyz": [0.4, 0.4, 0.12],          "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4, 0.4, 0.12],          "gripper": 0.04, "duration": 0.5},
]

NUM_TRIALS      = 50
TAU_MIN         = 0.0
TAU_MAX         = 2.0
OUTPUT_CSV      = "trial_results.csv"

# ─── PID gain sets to sweep ────────────────────────────────────
WRIST_KP = 0.5
WRIST_KI = 0.01
WRIST_KD = 0.1
WRIST_WEIGHT_FIXED = 0.27   # used when USE_LQR_WEIGHT=False


# ═══════════════════════════════════════════════════════════════
# CONTROLLERS  (self-contained copies — no module dependency)
# ═══════════════════════════════════════════════════════════════
class TaskSpacePID:
    def __init__(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.integral_clamp = 2.0

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def update(self, position_error, dt):
        self.integral += position_error * dt
        self.integral = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)
        derivative = (position_error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = position_error.copy()
        return TASK_KP * position_error + TASK_KI * self.integral + TASK_KD * derivative


class JointPIDController:
    def __init__(self, ndof=7):
        self.Kp = np.array([100.0] * ndof)
        self.Kd = np.array([20.0]  * ndof)
        self.prev_error = np.zeros(ndof)

    def reset(self):
        self.prev_error[:] = 0.0

    def update(self, q_current, q_desired, dq_current, dt):
        error = q_desired - q_current
        derivative = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error.copy()
        return self.Kp * error - self.Kd * dq_current


class MixingPID:
    def __init__(self, kp=0.5, ki=0.01, kd=0.1):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self):
        """Reset integral and derivative state."""
        self.integral   = 0.0
        self.prev_error = 0.0

    def update(self, theta_mix_deg, dt):
        error          = theta_mix_deg
        self.integral += error * dt
        deriv          = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * deriv


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY
# ═══════════════════════════════════════════════════════════════
class TrajectorySampler:
    def __init__(self, phases):
        self.phases     = phases
        self.durations  = [p["duration"] for p in phases]
        self.boundaries = np.cumsum(self.durations)
        self.total_time = self.boundaries[-1]
        self.poly_coeffs = []

    def set_start(self, xyz_start):
        starts = [xyz_start.copy()]
        for i in range(1, len(self.phases)):
            starts.append(np.array(self.phases[i - 1]["target_xyz"]))
        self.poly_coeffs = []
        for i, phase in enumerate(self.phases):
            s = starts[i]
            e = np.array(phase["target_xyz"])
            T = self.durations[i]
            a0 = s
            a3 =  10 * (e - s) / T**3
            a4 = -15 * (e - s) / T**4
            a5 =   6 * (e - s) / T**5
            self.poly_coeffs.append([a0, np.zeros(3), np.zeros(3), a3, a4, a5])

    def sample(self, t_eff):
        t_eff = float(np.clip(t_eff, 0.0, self.total_time))
        idx   = min(int(np.searchsorted(self.boundaries, t_eff, side="right")),
                    len(self.phases) - 1)
        t_start = self.boundaries[idx - 1] if idx > 0 else 0.0
        tl      = t_eff - t_start
        a0, a1, a2, a3, a4, a5 = self.poly_coeffs[idx]
        xyz = a0 + a1*tl + a2*tl**2 + a3*tl**3 + a4*tl**4 + a5*tl**5
        return xyz, self.phases[idx]["gripper"]


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
# SINGLE TRIAL
# ═══════════════════════════════════════════════════════════════
def run_trial(liquid_tau: float, use_wrist_pid: bool) -> dict:
    """
    Run one complete trajectory and return a dict of statistics.

    Parameters
    ----------
    liquid_tau    : lag time constant for liquid reorientation (seconds)
    use_wrist_pid : whether to enable closed-loop wrist orientation control
    """
    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    hand_id   = model.body("hand").id
    task_pid  = TaskSpacePID()
    joint_pid = JointPIDController(ndof=7)
    wrist_pid = MixingPID(kp=WRIST_KP, ki=WRIST_KI, kd=WRIST_KD)
    traj      = TrajectorySampler(PHASES)

    # Warm-up
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    traj.set_start(data.xpos[hand_id].copy())

    # State
    prev_ee_vel       = np.zeros(3)
    filtered_ee_accel = np.zeros(3)
    a_liquid          = np.array([0.0, 0.0, -9.81])
    prev_phase_idx    = -1

    # Accumulators
    speed_values         = []
    pos_error_values     = []
    mixing_score_values  = []

    sim_t = 0.0
    t_eff = 0.0

    while t_eff < traj.total_time:
        # ── EE velocity ────────────────────────────────────────
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0
        )
        cur_vel = vel6[3:].copy()
        speed_values.append(float(np.linalg.norm(cur_vel)))

        # ── Effective gravity + liquid lag ──────────────────────
        g = np.array([0.0, 0.0, -9.81])
        ee_accel_raw      = (cur_vel - prev_ee_vel) / max(dt, 1e-9)
        prev_ee_vel       = cur_vel.copy()
        filtered_ee_accel = (ACCEL_LPFILTER_ALPHA * ee_accel_raw +
                             (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_ee_accel)
        a_effective = g - filtered_ee_accel

        # Liquid inertia model — use trial's tau
        if liquid_tau > 1e-9:
            da_liquid = (a_effective - a_liquid) / liquid_tau
            a_liquid += da_liquid * dt
        else:
            # tau=0 → liquid instantly tracks effective gravity
            a_liquid = a_effective.copy()

        # ── Sample trajectory ───────────────────────────────────
        t_eff += dt
        target_xyz, gripper_w = traj.sample(t_eff)

        # ── IK ──────────────────────────────────────────────────
        current_xyz = data.xpos[hand_id].copy()
        dx = target_xyz - current_xyz
        pos_error_values.append(float(np.linalg.norm(dx)))

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        J  = jacp[:, :7]
        Jr = jacr[:, :7]

        JJT   = J @ J.T
        dq    = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), dx)
        J_pinv = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), np.eye(3))
        null_proj = np.eye(7) - J_pinv @ J

        # ── Tube axis and mixing angle ──────────────────────────
        hand_rot  = data.xmat[hand_id].reshape(3, 3)
        tube_axis = hand_rot[:, 2]

        norm_tube = np.linalg.norm(tube_axis)
        norm_liq  = np.linalg.norm(a_liquid)
        if norm_tube > 1e-9 and norm_liq > 1e-9:
            cos_angle = np.clip(
                np.dot(tube_axis, a_liquid / norm_liq), -1.0, 1.0
            )
            mix_angle = np.degrees(np.arccos(cos_angle))
        else:
            mix_angle = 0.0

        # Determine phase for gating logic (move earlier)
        phase_idx = min(
            int(np.searchsorted(traj.boundaries, t_eff, side="right")),
            len(PHASES) - 1
        )
        
        # Only collect mixing metrics during grasping/transport phases (3-6, indices 2-5)
        if 2 <= phase_idx <= 5:
            mixing_score_values.append(float(mix_angle))

        # ── Null-space goal ─────────────────────────────────────
        q_current       = data.qpos[:7]
        q_posture_error = Q_NOMINAL - q_current

        # Wrist orientation control only during grasping/transport phases (3-6, indices 2-5)
        if use_wrist_pid and 2 <= phase_idx <= 5:
            correction_magnitude = wrist_pid.update(mix_angle, dt)
            z_desired    = a_liquid / norm_liq if norm_liq > 1e-9 else np.array([0., 0., -1.])
            z_actual     = tube_axis / norm_tube if norm_tube > 1e-9 else np.array([0., 0., 1.])
            rotation_axis = np.cross(z_actual, z_desired)
            if np.linalg.norm(rotation_axis) > 1e-9:
                rotation_axis = (rotation_axis / np.linalg.norm(rotation_axis)
                                 * correction_magnitude)
            JrJrT = Jr @ Jr.T
            dq_wrist_goal = Jr.T @ np.linalg.solve(
                JrJrT + IK_LAMBDA_SQ * np.eye(3), rotation_axis
            )
            null_space_goal = (WRIST_WEIGHT_FIXED * dq_wrist_goal +
                               (1.0 - WRIST_WEIGHT_FIXED) * NULL_GAIN * q_posture_error)
        else:
            null_space_goal = NULL_GAIN * q_posture_error

        dq = dq + null_proj @ null_space_goal

        # ── Task-space PID ──────────────────────────────────────
        q_des = data.qpos[:7] + IK_GAIN * dq
        if USE_TASK_SPACE_PID:
            correction = task_pid.update(dx, dt)
            q_des += J.T @ correction * dt
        else:
            task_pid.update(dx, dt)

        # ── Joint PID → torques ─────────────────────────────────
        dq_current = data.qvel[:7]
        tau = joint_pid.update(data.qpos[:7], q_des, dq_current, dt)
        data.ctrl[:7] = tau
        if model.nu >= 8:
            data.ctrl[7] = gripper_w

        # ── Step physics ────────────────────────────────────────
        mujoco.mj_step(model, data)
        sim_t += dt

        # ── Phase transitions: reset PIDs ───────────────────────
        phase_idx = min(
            int(np.searchsorted(traj.boundaries, t_eff, side="right")),
            len(PHASES) - 1
        )
        if phase_idx != prev_phase_idx:
            task_pid.reset()
            joint_pid.reset()
            wrist_pid.reset()
            prev_phase_idx = phase_idx

    # Clean up temp file
    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)

    # ── Compute statistics ──────────────────────────────────────
    sa  = np.array(speed_values)
    pa  = np.array(pos_error_values)
    ma  = np.array(mixing_score_values)

    return {
        "liquid_tau":              liquid_tau,
        "use_wrist_pid":           int(use_wrist_pid),
        "avg_ee_speed":            float(np.mean(sa)),
        "max_ee_speed":            float(np.max(sa)),
        "avg_pos_error":           float(np.mean(pa)),
        "max_pos_error":           float(np.max(pa)),
        "avg_tube_tilt_error_deg": float(np.mean(ma)),
        "max_tube_tilt_error_deg": float(np.max(ma)),
        "avg_mix_angle_deg":       float(np.mean(ma)),
        "max_mix_angle_deg":       float(np.max(ma)),
        "integrated_mix_score":    float(np.sum(ma) * model.opt.timestep),
        "sim_time_s":              float(sim_t),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    # Linearly interpolate tau values
    tau_values = [TAU_MIN + (TAU_MAX - TAU_MIN) * i / (NUM_TRIALS - 1)
                  for i in range(NUM_TRIALS)]

    # Build the full experiment plan
    # Format: (liquid_tau, use_wrist_pid, condition_label)
    experiment_plan = []
    for tau in tau_values:
        experiment_plan.append((tau, False, "no_pid"))
    for tau in tau_values:
        experiment_plan.append((tau, True,  "pid_no_lqr"))

    total = len(experiment_plan)
    results = []

    print("="*70)
    print(f"BATCH TRIAL RUNNER — {total} trials total")
    print(f"  TAU sweep:  {TAU_MIN:.2f}s → {TAU_MAX:.2f}s  ({NUM_TRIALS} steps)")
    print(f"  Conditions: no_pid | pid_no_lqr")
    print(f"  Output:     {OUTPUT_CSV}")
    print("="*70 + "\n")

    wall_start = time.time()

    for trial_idx, (tau, use_wrist, label) in enumerate(experiment_plan):
        wall_elapsed = time.time() - wall_start
        print(f"[{trial_idx+1:3d}/{total}] condition={label:12s}  tau={tau:.4f}s  "
              f"(elapsed {wall_elapsed:.1f}s)", flush=True)

        stats = run_trial(liquid_tau=tau, use_wrist_pid=use_wrist)
        stats["condition"] = label
        stats["trial_index"] = trial_idx
        results.append(stats)

        # Print key metrics inline
        print(f"         avg_pos_err={stats['avg_pos_error']:.4f}m  "
              f"avg_mix={stats['avg_mix_angle_deg']:.2f}°  "
              f"integrated={stats['integrated_mix_score']:.2f}°·s")

    # ── Save CSV ────────────────────────────────────────────────
    fieldnames = [
        "trial_index", "condition", "liquid_tau", "use_wrist_pid",
        "avg_ee_speed", "max_ee_speed",
        "avg_pos_error", "max_pos_error",
        "avg_tube_tilt_error_deg", "max_tube_tilt_error_deg",
        "avg_mix_angle_deg", "max_mix_angle_deg",
        "integrated_mix_score", "sim_time_s",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"\n✓ Results written to {OUTPUT_CSV}")

    # ── Summary table ───────────────────────────────────────────
    no_pid_results  = [r for r in results if r["condition"] == "no_pid"]
    pid_results     = [r for r in results if r["condition"] == "pid_no_lqr"]

    def mean(vals, key):
        return sum(v[key] for v in vals) / len(vals)

    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL TRIALS")
    print("="*70)
    print(f"{'Metric':<35} {'no_pid':>12} {'pid_no_lqr':>12}")
    print("-"*70)
    for key, label in [
        ("avg_ee_speed",            "Mean avg EE speed (m/s)"),
        ("max_ee_speed",            "Mean max EE speed (m/s)"),
        ("avg_pos_error",           "Mean avg pos error (m)"),
        ("max_pos_error",           "Mean max pos error (m)"),
        ("avg_mix_angle_deg",       "Mean avg mix angle (°)"),
        ("max_mix_angle_deg",       "Mean max mix angle (°)"),
        ("integrated_mix_score",    "Mean integrated mix (°·s)"),
    ]:
        print(f"  {label:<33} {mean(no_pid_results, key):>12.4f} {mean(pid_results, key):>12.4f}")
    print("="*70)
    print(f"\nTotal wall-clock time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()