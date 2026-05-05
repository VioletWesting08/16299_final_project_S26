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
from only_kinematics import TaskSpacePID, build_scene, JointPIDController, MixingPID, TrajectorySampler

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/debug_scene.xml"

# Task-space PID for position feedback
USE_TASK_SPACE_PID = True
TASK_KP = 0.5
TASK_KI = 0.0
TASK_KD = 0.2

# IK solver parameters
IK_LAMBDA_SQ = 1e-4
IK_GAIN = 1.0

# Null-space posture control
Q_NOMINAL = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])
NULL_GAIN = 0.5

# ── IK MODE ──────────────────────────────────────────────────
# True  = 6D IK (position + orientation jointly, no wrist PID, no null-space wrist)
# False = 3D IK (position only) + wrist PID in null-space (original approach)
USE_6D_IK = True

# Orientation weight relative to position in 6D IK
# Lower = orientation softer, position dominates
ORI_WEIGHT = 0.3   # scales dx_ori before stacking with dx_pos

# Wrist orientation control — only used when USE_6D_IK = False
USE_WRIST_PID = True
USE_LQR_WEIGHT = False
WRIST_WEIGHT = 0.27

# Liquid inertia model
ACCEL_LPFILTER_ALPHA = 0.03

# Testing
AGGRESSIVE_TRANSPORT = True
INIT_TUBE_MISALIGNED = False

PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 200, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6, 0.0, 0.11], "gripper": 200, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6, 0.0, 0.11], "gripper": 0, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6, 0.0, 0.40], "gripper": 0, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4, 0.4, 0.40], "gripper": 0, "duration": 0.3 if AGGRESSIVE_TRANSPORT else 1.0},
    {"name": "6. Place",     "target_xyz": [0.4, 0.4, 0.12], "gripper": 0, "duration": 1.5},
    {"name": "7. Release",   "target_xyz": [0.4, 0.4, 0.12], "gripper": 200, "duration": 1},
]

# ═══════════════════════════════════════════════════════════════
# LQR WEIGHT (only used when USE_6D_IK = False)
# ═══════════════════════════════════════════════════════════════
def compute_lqr_orientation_weight(q_orientation_error, q_position_error):
    relative_priority = q_orientation_error / max(q_position_error, 1e-6)
    return np.tanh(relative_priority) * 0.75 + 0.15

Q_ORIENTATION_ERROR = 300.0
Q_POSITION_ERROR    = 1000.0
WRIST_WEIGHT_LQR    = compute_lqr_orientation_weight(Q_ORIENTATION_ERROR, Q_POSITION_ERROR)

NUM_TRIALS      = 50
TAU_MIN         = 0.15
TAU_MAX         = 2.0
OUTPUT_CSV      = "outputs/trial_results_ik.csv"

# ─── PID gain sets to sweep ────────────────────────────────────
WRIST_KP = 0.5
WRIST_KI = 0.01
WRIST_KD = 0.1
WRIST_WEIGHT_FIXED = 0.27   # used when USE_LQR_WEIGHT=False


# ═══════════════════════════════════════════════════════════════
# SINGLE TRIAL
# ═══════════════════════════════════════════════════════════════
def run_trial(liquid_tau: float, use_6D: bool) -> dict:
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
    left_finger_id = model.body("left_finger").id
    right_finger_id = model.body("right_finger").id
    task_pid  = TaskSpacePID()
    joint_pid = JointPIDController(ndof=7)
    wrist_pid = MixingPID(kp=0.5, ki=0.01, kd=0.1)
    traj      = TrajectorySampler(PHASES)

    prev_ee_vel       = np.zeros(3)
    filtered_ee_accel = np.zeros(3)

    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    
    # Calculate starting TCP as the exact midpoint between the fingers
    tcp_start = (data.xpos[left_finger_id] + data.xpos[right_finger_id]) / 2.0
    
    # Pass virtual TCP start to the trajectory sampler
    traj.set_start(tcp_start)
        
    sim_t          = 0.0
    t_eff          = 0.0
    prev_phase_idx = -1
    speed_values        = []
    pos_error_values    = []
    tilt_error_values   = []
    mixing_score_values = []
    a_liquid = np.array([0.0, 0.0, -9.81])


    while t_eff < traj.total_time:
        # ── 1. EE VELOCITY ───────────────────────────────────
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0)
        cur_vel = vel6[3:].copy()
        speed_values.append(float(np.linalg.norm(cur_vel)))

        # ── 2. EFFECTIVE GRAVITY & LIQUID LAG ────────────────
        g = np.array([0.0, 0.0, -9.81])
        ee_accel_raw = (cur_vel - prev_ee_vel) / max(dt, 1e-9)
        prev_ee_vel  = cur_vel.copy()
        filtered_ee_accel = (ACCEL_LPFILTER_ALPHA * ee_accel_raw +
                                (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_ee_accel)
        a_effective = g - filtered_ee_accel
        a_liquid   += ((a_effective - a_liquid) / liquid_tau) * dt

        # ── 3. TRAJECTORY SAMPLE ─────────────────────────────
        t_eff += dt
        target_xyz, gripper_w = traj.sample(t_eff)

        # ── 4. JACOBIANS ─────────────────────────────────────
        tcp_global_pos = (data.xpos[left_finger_id] + data.xpos[right_finger_id]) / 2.0
        
        # Error is calculated from this floating midpoint
        dx_pos = target_xyz - tcp_global_pos 
        pos_error_values.append(float(np.linalg.norm(dx_pos)))

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, tcp_global_pos, hand_id)
        J  = jacp[:, :7]   # (3×7) positional
        Jr = jacr[:, :7]   # (3×7) rotational

        # ── 5. TUBE AXIS & MIXING ANGLE ──────────────────────
        hand_rot  = data.xmat[hand_id].reshape(3, 3)
        tube_axis = hand_rot[:, 2]
        norm_tube = np.linalg.norm(tube_axis)
        norm_liq  = np.linalg.norm(a_liquid)

        if norm_tube > 1e-9 and norm_liq > 1e-9:
            cos_angle = np.clip(np.dot(tube_axis, a_liquid/norm_liq), -1.0, 1.0)
            mix_angle = np.degrees(np.arccos(cos_angle))
        else:
            mix_angle = 0.0

        # ── 6. PHASE ─────────────────────────────────────────
        phase_idx = int(np.searchsorted(traj.boundaries, t_eff, side="right"))
        phase_idx = min(phase_idx, len(PHASES)-1)
        grasping  = (2 <= phase_idx <= 5)

        # ── 7. IK SOLVE ──────────────────────────────────────
        if use_6D and grasping:
            # ── 6D IK: position + orientation jointly ────────
            # Orientation error: rotate tube_axis toward a_liquid direction
            z_desired = a_liquid / norm_liq if norm_liq > 1e-9 else np.array([0.,0.,-1.])
            z_actual  = tube_axis / norm_tube if norm_tube > 1e-9 else np.array([0.,0., 1.])
            dx_ori    = np.cross(z_actual, z_desired) * ORI_WEIGHT  # (3,) scaled

            # Stack into 6D error and 6D Jacobian
            dx_full  = np.concatenate([dx_pos, dx_ori])   # (6,)
            J_full   = np.vstack([J, Jr])                  # (6×7)

            # Damped least squares on 6D system
            J_full_JT = J_full @ J_full.T                  # (6×6)
            dq = J_full.T @ np.linalg.solve(
                J_full_JT + IK_LAMBDA_SQ * np.eye(6), dx_full)

            # Null-space projector from 6D Jacobian (1D null space remains)
            J_full_pinv = J_full.T @ np.linalg.solve(
                J_full_JT + IK_LAMBDA_SQ * np.eye(6), np.eye(6))
            null_proj = np.eye(7) - J_full_pinv @ J_full

            # Only posture control in null space (wrist already in primary task)
            q_posture_error = Q_NOMINAL - data.qpos[:7]
            dq = dq + null_proj @ (NULL_GAIN * q_posture_error)

        else:
            # ── 3D IK + null-space wrist PID (original) ──────
            JJT    = J @ J.T
            dq     = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), dx_pos)
            J_pinv = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), np.eye(3))
            null_proj = np.eye(7) - J_pinv @ J

            q_posture_error = Q_NOMINAL - data.qpos[:7]

            if USE_WRIST_PID and grasping:
                correction_magnitude = wrist_pid.update(mix_angle, dt)
                z_desired     = a_liquid  / norm_liq  if norm_liq  > 1e-9 else np.array([0.,0.,-1.])
                z_actual      = tube_axis / norm_tube if norm_tube > 1e-9 else np.array([0.,0., 1.])
                rotation_axis = np.cross(z_actual, z_desired)
                if np.linalg.norm(rotation_axis) > 1e-9:
                    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) * correction_magnitude
                JrJrT         = Jr @ Jr.T
                dq_wrist_goal = Jr.T @ np.linalg.solve(
                    JrJrT + IK_LAMBDA_SQ * np.eye(3), rotation_axis)
                blend_weight  = WRIST_WEIGHT_LQR if USE_LQR_WEIGHT else WRIST_WEIGHT
                null_space_goal = (blend_weight * dq_wrist_goal +
                                    (1.0 - blend_weight) * NULL_GAIN * q_posture_error)
            else:
                null_space_goal = NULL_GAIN * q_posture_error

            dq = dq + null_proj @ null_space_goal

        # ── 8. TASK-SPACE PID ────────────────────────────────
        q_des = data.qpos[:7] + IK_GAIN * dq
        if USE_TASK_SPACE_PID:
            correction = task_pid.update(dx_pos, dt)
            q_des += J.T @ correction * dt
        else:
            _ = task_pid.update(dx_pos, dt)

        # ── 9. JOINT PID → TORQUES ───────────────────────────
        dq_current    = data.qvel[:7]
        tau           = joint_pid.update(data.qpos[:7], q_des, dq_current, dt)
        data.ctrl[:7] = tau
        if model.nu >= 8:
            data.ctrl[7] = gripper_w

        # ── 10. STEP ─────────────────────────────────────────
        mujoco.mj_step(model, data)
        sim_t += dt

        if grasping:
            mixing_score_values.append(float(mix_angle))
            tilt_error_values.append(float(mix_angle))

    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)


    # ── Compute statistics ──────────────────────────────────────
    sa  = np.array(speed_values)
    pa  = np.array(pos_error_values)
    ma  = np.array(mixing_score_values)

    return {
        "liquid_tau":              liquid_tau,
        "use_6D":                  int(use_6D),
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
        experiment_plan.append((tau, True, "ik"))

    total = len(experiment_plan)
    results = []

    print("="*70)
    print(f"BATCH TRIAL RUNNER — {total} trials total")
    print(f"  TAU sweep:  {TAU_MIN:.2f}s → {TAU_MAX:.2f}s  ({NUM_TRIALS} steps)")
    print(f"  Conditions: no_pid | pid_no_lqr")
    print(f"  Output:     {OUTPUT_CSV}")
    print("="*70 + "\n")

    wall_start = time.time()

    for trial_idx, (tau, use_6D, label) in enumerate(experiment_plan):
        wall_elapsed = time.time() - wall_start
        print(f"[{trial_idx+1:3d}/{total}] condition={label:12s}  tau={tau:.4f}s  "
              f"(elapsed {wall_elapsed:.1f}s)", flush=True)

        stats = run_trial(liquid_tau=tau, use_6D=use_6D)
        stats["condition"] = label
        stats["trial_index"] = trial_idx
        results.append(stats)

        # Print key metrics inline
        print(f"         avg_pos_err={stats['avg_pos_error']:.4f}m  "
              f"avg_mix={stats['avg_mix_angle_deg']:.2f}°  "
              f"integrated={stats['integrated_mix_score']:.2f}°·s")

    # ── Save CSV ────────────────────────────────────────────────
    fieldnames = [
        "trial_index", "condition", "liquid_tau", "use_6D",
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
    ik_results  = results

    def mean(vals, key):
        return sum(v[key] for v in vals) / len(vals)

    print("\n" + "="*70)
    print("SUMMARY ACROSS ALL TRIALS")
    print("="*70)
    print(f"{'Metric':<35} {'6D_IK':>12}")
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
        print(f"  {label:<33} {mean(ik_results, key):>12.4f}")
    print("="*70)
    print(f"\nTotal wall-clock time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()