"""
trial_runs_6dik.py
==================
Sweep ORI_WEIGHT from 0.0 → 1.0, comparing:
  - condition "3d_ik"  : 3D position-only IK, no wrist PID, posture null-space only
  - condition "6d_ik"  : 6D IK (position + orientation jointly) at each ORI_WEIGHT

Uses identical settings and controllers as test_picking_6dik.py.
"""

import mujoco
import numpy as np
import os
import csv
import time

# ═══════════════════════════════════════════════════════════════
# FIXED CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/debug_scene.xml"

USE_TASK_SPACE_PID   = True
TASK_KP              = 0.5
TASK_KI              = 0.0
TASK_KD              = 0.2

IK_LAMBDA_SQ         = 1e-4
IK_GAIN              = 1.0

Q_NOMINAL            = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])
NULL_GAIN            = 0.5

ACCEL_LPFILTER_ALPHA = 0.03
LIQUID_TAU           = 1.0

AGGRESSIVE_TRANSPORT = True

PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6, 0.0, 0.12],          "gripper": 0.04, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6, 0.0, 0.12],          "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6, 0.0, 0.40],          "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4, 0.4, 0.40],          "gripper": 0.00, "duration": 0.3 if AGGRESSIVE_TRANSPORT else 1.0},
    {"name": "6. Place",     "target_xyz": [0.4, 0.4, 0.12],          "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4, 0.4, 0.12],          "gripper": 0.04, "duration": 0.5},
]

NUM_TRIALS     = 50
ORI_WEIGHT_MIN = 0.0
ORI_WEIGHT_MAX = 1.0
OUTPUT_CSV     = "trial_results_6dik.csv"


# ═══════════════════════════════════════════════════════════════
# CONTROLLERS
# ═══════════════════════════════════════════════════════════════
class TaskSpacePID:
    def __init__(self):
        self.integral       = np.zeros(3)
        self.prev_error     = np.zeros(3)
        self.integral_clamp = 2.0

    def reset(self):
        self.integral[:]    = 0.0
        self.prev_error[:]  = 0.0

    def update(self, position_error, dt):
        self.integral  += position_error * dt
        self.integral   = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)
        derivative      = (position_error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = position_error.copy()
        return TASK_KP * position_error + TASK_KI * self.integral + TASK_KD * derivative


class JointPIDController:
    def __init__(self, ndof=7):
        self.Kp         = np.array([100.0] * ndof)
        self.Kd         = np.array([20.0]  * ndof)
        self.prev_error = np.zeros(ndof)

    def reset(self):
        self.prev_error[:] = 0.0

    def update(self, q_current, q_desired, dq_current, dt):
        error           = q_desired - q_current
        derivative      = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error.copy()
        return self.Kp * error - self.Kd * dq_current


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY
# ═══════════════════════════════════════════════════════════════
class TrajectorySampler:
    def __init__(self, phases):
        self.phases      = phases
        self.durations   = [p["duration"] for p in phases]
        self.boundaries  = np.cumsum(self.durations)
        self.total_time  = self.boundaries[-1]
        self.poly_coeffs = []

    def set_start(self, xyz_start):
        starts = [xyz_start.copy()]
        for i in range(1, len(self.phases)):
            starts.append(np.array(self.phases[i-1]["target_xyz"]))
        self.poly_coeffs = []
        for i, phase in enumerate(self.phases):
            s  = starts[i]
            e  = np.array(phase["target_xyz"])
            T  = self.durations[i]
            a3 =  10 * (e - s) / T**3
            a4 = -15 * (e - s) / T**4
            a5 =   6 * (e - s) / T**5
            self.poly_coeffs.append([s, np.zeros(3), np.zeros(3), a3, a4, a5])

    def sample(self, t_eff):
        t_eff = float(np.clip(t_eff, 0.0, self.total_time))
        idx   = min(int(np.searchsorted(self.boundaries, t_eff, side="right")),
                    len(self.phases) - 1)
        t0    = self.boundaries[idx-1] if idx > 0 else 0.0
        tl    = t_eff - t0
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
def run_trial(ori_weight: float, use_6d_ik: bool) -> dict:
    """
    Run one complete trajectory and return statistics.

    Parameters
    ----------
    ori_weight : orientation error scale in 6D IK (ignored when use_6d_ik=False)
    use_6d_ik  : True  = 6D IK (position + orientation jointly)
                 False = 3D IK + posture-only null-space (pure baseline)
    """
    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    hand_id   = model.body("hand").id
    task_pid  = TaskSpacePID()
    joint_pid = JointPIDController(ndof=7)
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
    speed_values        = []
    pos_error_values    = []
    mixing_score_values = []

    sim_t = 0.0
    t_eff = 0.0

    while t_eff < traj.total_time:

        # ── EE velocity ─────────────────────────────────────────
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0)
        cur_vel = vel6[3:].copy()
        speed_values.append(float(np.linalg.norm(cur_vel)))

        # ── Effective gravity + liquid lag ───────────────────────
        g             = np.array([0.0, 0.0, -9.81])
        ee_accel_raw  = (cur_vel - prev_ee_vel) / max(dt, 1e-9)
        prev_ee_vel   = cur_vel.copy()
        filtered_ee_accel = (ACCEL_LPFILTER_ALPHA * ee_accel_raw +
                             (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_ee_accel)
        a_effective = g - filtered_ee_accel
        a_liquid   += ((a_effective - a_liquid) / LIQUID_TAU) * dt

        # ── Trajectory sample ────────────────────────────────────
        t_eff += dt
        target_xyz, gripper_w = traj.sample(t_eff)

        # ── Jacobians ────────────────────────────────────────────
        current_xyz = data.xpos[hand_id].copy()
        dx_pos      = target_xyz - current_xyz
        pos_error_values.append(float(np.linalg.norm(dx_pos)))

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
        J  = jacp[:, :7]
        Jr = jacr[:, :7]

        # ── Tube axis & mixing angle ─────────────────────────────
        hand_rot  = data.xmat[hand_id].reshape(3, 3)
        tube_axis = hand_rot[:, 2]
        norm_tube = np.linalg.norm(tube_axis)
        norm_liq  = np.linalg.norm(a_liquid)

        if norm_tube > 1e-9 and norm_liq > 1e-9:
            cos_angle = np.clip(np.dot(tube_axis, a_liquid / norm_liq), -1.0, 1.0)
            mix_angle = np.degrees(np.arccos(cos_angle))
        else:
            mix_angle = 0.0

        # ── Phase ────────────────────────────────────────────────
        phase_idx = min(
            int(np.searchsorted(traj.boundaries, t_eff, side="right")),
            len(PHASES) - 1)
        grasping = (2 <= phase_idx <= 5)

        if grasping:
            mixing_score_values.append(float(mix_angle))

        # ── IK solve ─────────────────────────────────────────────
        q_posture_error = Q_NOMINAL - data.qpos[:7]

        if use_6d_ik and grasping:
            # 6D IK: position + orientation jointly
            z_desired = a_liquid  / norm_liq  if norm_liq  > 1e-9 else np.array([0., 0., -1.])
            z_actual  = tube_axis / norm_tube if norm_tube > 1e-9 else np.array([0., 0.,  1.])
            dx_ori    = np.cross(z_actual, z_desired) * ori_weight   # scaled orientation error

            dx_full  = np.concatenate([dx_pos, dx_ori])  # (6,)
            J_full   = np.vstack([J, Jr])                 # (6×7)

            J_full_JT   = J_full @ J_full.T               # (6×6)
            dq          = J_full.T @ np.linalg.solve(
                J_full_JT + IK_LAMBDA_SQ * np.eye(6), dx_full)

            J_full_pinv = J_full.T @ np.linalg.solve(
                J_full_JT + IK_LAMBDA_SQ * np.eye(6), np.eye(6))
            null_proj   = np.eye(7) - J_full_pinv @ J_full

            # Only posture in null-space (orientation already primary)
            dq = dq + null_proj @ (NULL_GAIN * q_posture_error)

        else:
            # 3D IK + posture-only null-space (baseline)
            JJT       = J @ J.T
            dq        = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), dx_pos)
            J_pinv    = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), np.eye(3))
            null_proj = np.eye(7) - J_pinv @ J
            dq        = dq + null_proj @ (NULL_GAIN * q_posture_error)

        # ── Task-space PID ───────────────────────────────────────
        q_des = data.qpos[:7] + IK_GAIN * dq
        if USE_TASK_SPACE_PID:
            correction = task_pid.update(dx_pos, dt)
            q_des += J.T @ correction * dt
        else:
            task_pid.update(dx_pos, dt)

        # ── Joint PID → torques ──────────────────────────────────
        dq_current    = data.qvel[:7]
        tau           = joint_pid.update(data.qpos[:7], q_des, dq_current, dt)
        data.ctrl[:7] = tau
        if model.nu >= 8:
            data.ctrl[7] = gripper_w

        # ── Step physics ─────────────────────────────────────────
        mujoco.mj_step(model, data)
        sim_t += dt

        # ── Phase transitions: reset PIDs ────────────────────────
        phase_idx = min(
            int(np.searchsorted(traj.boundaries, t_eff, side="right")),
            len(PHASES) - 1)
        if phase_idx != prev_phase_idx:
            task_pid.reset()
            joint_pid.reset()
            prev_phase_idx = phase_idx

    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)

    sa = np.array(speed_values)
    pa = np.array(pos_error_values)
    ma = np.array(mixing_score_values) if mixing_score_values else np.array([0.0])

    return {
        "ori_weight":              ori_weight,
        "use_6d_ik":               int(use_6d_ik),
        "avg_ee_speed":            float(np.mean(sa)),
        "max_ee_speed":            float(np.max(sa)),
        "avg_pos_error":           float(np.mean(pa)),
        "max_pos_error":           float(np.max(pa)),
        "avg_mix_angle_deg":       float(np.mean(ma)),
        "max_mix_angle_deg":       float(np.max(ma)),
        "integrated_mix_score":    float(np.sum(ma) * dt),
        "sim_time_s":              float(sim_t),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    ori_weights = [ORI_WEIGHT_MIN + (ORI_WEIGHT_MAX - ORI_WEIGHT_MIN) * i / (NUM_TRIALS - 1)
                   for i in range(NUM_TRIALS)]

    # Plan: 3D IK baseline (ori_weight=0, repeated) + 6D IK sweep
    experiment_plan = []
    for w in ori_weights:
        experiment_plan.append((0.0, False, "3d_ik"))   # baseline: no orientation control
    for w in ori_weights:
        experiment_plan.append((w, True, "6d_ik"))       # 6D IK at each ori_weight

    total   = len(experiment_plan)
    results = []

    print("=" * 70)
    print(f"BATCH TRIAL RUNNER — {total} trials total")
    print(f"  ORI_WEIGHT sweep: {ORI_WEIGHT_MIN:.2f} → {ORI_WEIGHT_MAX:.2f}  ({NUM_TRIALS} steps)")
    print(f"  Conditions:       3d_ik (baseline) | 6d_ik (swept)")
    print(f"  Output:           {OUTPUT_CSV}")
    print("=" * 70 + "\n")

    wall_start = time.time()

    for trial_idx, (ori_w, use_6d, label) in enumerate(experiment_plan):
        wall_elapsed = time.time() - wall_start
        print(f"[{trial_idx+1:3d}/{total}] condition={label:8s}  ori_weight={ori_w:.4f}  "
              f"(elapsed {wall_elapsed:.1f}s)", flush=True)

        stats = run_trial(ori_weight=ori_w, use_6d_ik=use_6d)
        stats["condition"]   = label
        stats["trial_index"] = trial_idx
        results.append(stats)

        print(f"         avg_pos_err={stats['avg_pos_error']:.4f}m  "
              f"avg_mix={stats['avg_mix_angle_deg']:.2f}°  "
              f"integrated={stats['integrated_mix_score']:.2f}°·s")

    # ── Save CSV ─────────────────────────────────────────────────
    fieldnames = [
        "trial_index", "condition", "ori_weight", "use_6d_ik",
        "avg_ee_speed", "max_ee_speed",
        "avg_pos_error", "max_pos_error",
        "avg_mix_angle_deg", "max_mix_angle_deg",
        "integrated_mix_score", "sim_time_s",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"\n✓ Results written to {OUTPUT_CSV}")

    # ── Summary table ─────────────────────────────────────────────
    baseline_results = [r for r in results if r["condition"] == "3d_ik"]
    ik6d_results     = [r for r in results if r["condition"] == "6d_ik"]

    def mean(vals, key):
        return sum(v[key] for v in vals) / len(vals)

    # Find best 6D IK trial by integrated mixing score (lower = better)
    best_6d = min(ik6d_results, key=lambda r: r["integrated_mix_score"])

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL TRIALS")
    print("=" * 70)
    print(f"{'Metric':<35} {'3d_ik (mean)':>14} {'6d_ik (mean)':>14}")
    print("-" * 70)
    for key, label in [
        ("avg_ee_speed",         "Mean avg EE speed (m/s)"),
        ("max_ee_speed",         "Mean max EE speed (m/s)"),
        ("avg_pos_error",        "Mean avg pos error (m)"),
        ("max_pos_error",        "Mean max pos error (m)"),
        ("avg_mix_angle_deg",    "Mean avg mix angle (°)"),
        ("max_mix_angle_deg",    "Mean max mix angle (°)"),
        ("integrated_mix_score", "Mean integrated mix (°·s)"),
    ]:
        print(f"  {label:<33} {mean(baseline_results, key):>14.4f} {mean(ik6d_results, key):>14.4f}")

    print("-" * 70)
    print(f"\nBest 6D IK trial:")
    print(f"  ori_weight={best_6d['ori_weight']:.4f}  "
          f"avg_mix={best_6d['avg_mix_angle_deg']:.2f}°  "
          f"integrated={best_6d['integrated_mix_score']:.2f}°·s  "
          f"avg_pos_err={best_6d['avg_pos_error']:.4f}m")
    print("=" * 70)
    print(f"\nTotal wall-clock time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()