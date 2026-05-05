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
from test_picking import TaskSpacePID, build_scene, JointPIDController, MixingPID, TrajectorySampler

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/debug_scene.xml"

# Task-space PID for position feedback
USE_TASK_SPACE_PID = True     # toggle position PID feedback on/off
TASK_KP = 0.5     # position gain (higher = stiffer tracking)
TASK_KI = 0.0     # integral gain
TASK_KD = 0.2     # derivative gain (damping)

# IK solver parameters
IK_LAMBDA_SQ = 1e-4   # damping for damped least squares
IK_GAIN = 1.0         # scaling factor for IK joint velocity commands (1.0 = full application)

# Null-space posture control (keeps arm in natural configuration)
Q_NOMINAL = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])  # comfortable Franka pose (radians)
NULL_GAIN = 0.5  # how strongly to pull toward nominal pose

# Wrist orientation control (tube alignment with gravity)
USE_WRIST_PID = False          # toggle wrist orientation control on/off
USE_LQR_WEIGHT = False         # use LQR-computed weight for null-space blending
ACCEL_LPFILTER_ALPHA = 0.03   # low-pass filter on acceleration measurement (lower=more smoothing)


# Testing/validation options
AGGRESSIVE_TRANSPORT = True   # fast transport phase (0.3s vs 1.0s) for stress-testing
INIT_TUBE_MISALIGNED = False  # start with tube deliberately tilted
DEBUG_WRIST_PID = True        # print wrist control diagnostics
PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 200, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.62, 0.0, 0.075], "gripper": 200, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.62, 0.0, 0.075], "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.62, 0.0, 0.40], "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4, 0.4, 0.40], "gripper": 0.00, "duration": 0.3 if AGGRESSIVE_TRANSPORT else 1.0},
    {"name": "6. Place",     "target_xyz": [0.4, 0.4, 0.08], "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4, 0.4, 0.08], "gripper": 200, "duration": 0.5},
]

# ═══════════════════════════════════════════════════════════════
# LQR-TUNED ORIENTATION CONTROL
# ═══════════════════════════════════════════════════════════════
def compute_lqr_orientation_weight(q_orientation_error: float, q_position_error: float) -> float:
    """Compute optimal null-space blending weight from LQR cost ratio."""
    relative_priority = q_orientation_error / max(q_position_error, 1e-6)
    wrist_weight = np.tanh(relative_priority) * 0.75 + 0.15  # max weight ~0.85 for strong wrist control
    return wrist_weight

# LQR cost weights
Q_ORIENTATION_ERROR = 300.0  # heavily prioritize wrist alignment
Q_POSITION_ERROR = 1000.0
WRIST_WEIGHT_LQR = compute_lqr_orientation_weight(Q_ORIENTATION_ERROR, Q_POSITION_ERROR)
WRIST_WEIGHT = 0.27  # fallback weight if USE_LQR_WEIGHT is False
print(f"[LQR] WRIST_WEIGHT={WRIST_WEIGHT_LQR:.3f}")

NUM_TRIALS      = 50
TAU_MIN         = 0.15
TAU_MAX         = 2.0
OUTPUT_CSV      = "outputs/trial_results.csv"

# ─── PID gain sets to sweep ────────────────────────────────────
WRIST_KP = 0.5
WRIST_KI = 0.01
WRIST_KD = 0.1
WRIST_WEIGHT_FIXED = 0.27   # used when USE_LQR_WEIGHT=False


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

    left_finger_id = model.body("left_finger").id
    right_finger_id = model.body("right_finger").id
    hand_id  = model.body("hand").id
    task_pid = TaskSpacePID()
    joint_pid = JointPIDController(ndof=7)
    wrist_pid = MixingPID(kp=0.5, ki=0.01, kd=0.1)
    traj     = TrajectorySampler(PHASES)
    
    # For finite-difference acceleration estimation + filtering
    prev_ee_vel = np.zeros(3)
    filtered_ee_accel = np.zeros(3)  # low-pass filtered acceleration

    # ── warm-up: let the robot settle at its rest pose ──────────
    for _ in range(500):
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    
    # Calculate starting TCP as the exact midpoint between the fingers
    tcp_start = (data.xpos[left_finger_id] + data.xpos[right_finger_id]) / 2.0
    
    # Pass virtual TCP start to the trajectory sampler
    traj.set_start(tcp_start)
    
    sim_t = 0.0
    t_eff = 0.0    
    prev_phase_idx = -1  # track phase transitions
    
    # Statistics collection
    speed_values = []
    pos_error_values = []
    tilt_error_values = []
    mixing_score_values = []
    
    # Liquid inertia state
    a_liquid = np.array([0.0, 0.0, -9.81])  # starts aligned with gravity

    while t_eff < traj.total_time:

        # ── 1. MEASURE EE VELOCITY ──────────────────────────────
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0
        )
        cur_vel = vel6[3:].copy()           # linear velocity (world frame)
        
        # Collect statistics
        speed_values.append(float(np.linalg.norm(cur_vel)))
        
        # ── COMPUTE EFFECTIVE GRAVITY ────────────────────────────────
        # a_effective = true gravity + inertial acceleration of EE
        # Inertial accel estimated by finite differencing EE velocity
        g = np.array([0.0, 0.0, -9.81])
        
        # Raw finite-difference acceleration (very noisy!)
        ee_accel_raw = (cur_vel - prev_ee_vel) / max(dt, 1e-9)
        prev_ee_vel = cur_vel.copy()
        
        # Low-pass filter to smooth out noise from finite differencing
        # filtered_accel = alpha * raw + (1 - alpha) * prev_filtered
        filtered_ee_accel = (ACCEL_LPFILTER_ALPHA * ee_accel_raw + 
                            (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_ee_accel)
        
        a_effective = g + filtered_ee_accel   # what the liquid "feels" (smoothed)
        
        # ── LIQUID INERTIA MODEL ─────────────────────────────────────
        # Liquid doesn't instantly reorient; it lags behind a_effective
        # First-order lag: liquid "catches up" to effective gravity slowly
        if liquid_tau > 1e-9:
            da_liquid = (a_effective - a_liquid) / liquid_tau
            a_liquid += da_liquid * dt
        else:
            # tau=0 → liquid instantly tracks effective gravity
            a_liquid = a_effective.copy()

        # ── 2. SAMPLE FIXED TRAJECTORY ─────────────────────────
        # Trajectory runs at constant speed
        t_eff += dt
        target_xyz, gripper_w = traj.sample(t_eff)

        # ── 3. COMPUTE IK ──────────────────────────────────────
        # tcp_global_pos = (data.xpos[left_finger_id] + data.xpos[right_finger_id]) / 2.0
        TCP_OFFSET = np.array([0.0, 0.0, 0.1034]) 
            
        hand_pos = data.xpos[hand_id].copy()
        hand_rot = data.xmat[hand_id].reshape(3, 3)
        
        # Compute the global position of the TCP (between the fingers)
        tcp_global_pos = hand_pos + hand_rot @ TCP_OFFSET 

        # Error is calculated from this floating midpoint
        dx = target_xyz - tcp_global_pos       
        
        # Collect position tracking error
        pos_error_values.append(float(np.linalg.norm(dx)))

        # Compute Jacobians for the virtual point, anchored to the hand body kinematics
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, jacr, tcp_global_pos, hand_id)
        J  = jacp[:, :7]                    # first 7 DOF positional Jacobian
        Jr = jacr[:, :7]                    # first 7 DOF rotational Jacobian

        # Compute position-task IK (damped least squares)
        JJT   = J @ J.T
        dq    = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), dx)
        
        # Null-space projector
        J_pinv = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), np.eye(3))
        null_proj = np.eye(7) - J_pinv @ J
        
        # Compute tube axis and mixing score
        hand_rot  = data.xmat[hand_id].reshape(3, 3)
        tube_axis = hand_rot[:, 2]
        
        # Compute mixing angle (feedback signal for closed-loop control)
        norm_tube = np.linalg.norm(tube_axis)
        norm_liq  = np.linalg.norm(a_liquid)
        if norm_tube > 1e-9 and norm_liq > 1e-9:
            cos_angle = np.clip(
                np.dot(tube_axis, a_liquid / norm_liq),
                -1.0, 1.0
            )
            mix_angle = np.degrees(np.arccos(cos_angle))
        else:
            mix_angle = 0.0
        mixing_score_values.append(float(mix_angle))
        
        # Null-space blending: posture + wrist orientation
        q_current = data.qpos[:7]
        q_posture_error = Q_NOMINAL - q_current
        
        # Wrist orientation control (via null-space)
        if use_wrist_pid:
            # Closed-loop control: use mixing angle as feedback
            correction_magnitude = wrist_pid.update(mix_angle, dt)
            
            # Compute rotation axis toward alignment (tube → liquid direction)
            z_desired = a_liquid / norm_liq if norm_liq > 1e-9 else np.array([0., 0., -1.])
            z_actual = tube_axis / norm_tube if norm_tube > 1e-9 else np.array([0., 0., 1.])
            rotation_axis = np.cross(z_actual, z_desired)
            
            # Scale rotation axis by PID correction magnitude
            if np.linalg.norm(rotation_axis) > 1e-9:
                rotation_axis = rotation_axis / np.linalg.norm(rotation_axis) * correction_magnitude
            
            # Map rotation axis to joint velocity via rotational Jacobian
            JrJrT = Jr @ Jr.T
            dq_wrist_goal = Jr.T @ np.linalg.solve(
                JrJrT + IK_LAMBDA_SQ * np.eye(3),
                rotation_axis
            )
            
            # Blend posture and wrist goals
            blend_weight = WRIST_WEIGHT_LQR if USE_LQR_WEIGHT else WRIST_WEIGHT
            null_space_goal = (blend_weight * dq_wrist_goal + 
                                (1.0 - blend_weight) * NULL_GAIN * q_posture_error)
        else:
            # Pure posture control
            null_space_goal = NULL_GAIN * q_posture_error
        
        # Project blended goal into null-space
        dq = dq + null_proj @ null_space_goal

        # Task-space PID feedback (outer-loop position correction)
        q_des = data.qpos[:7] + IK_GAIN * dq
        
        if USE_TASK_SPACE_PID:
            correction = task_pid.update(dx, dt)
            q_des += J.T @ correction * dt
        else:
            _ = task_pid.update(dx, dt)

        # Joint-level PID controller to track desired positions
        dq_current = data.qvel[:7]
        tau = joint_pid.update(data.qpos[:7], q_des, dq_current, dt)
        
        # Send torque commands to robot actuators
        data.ctrl[:7] = tau
        
        # Gripper is controlled via tendon (actuator7)
        if model.nu >= 8:
            data.ctrl[7] = gripper_w

        # ── 6. STEP PHYSICS ───────────────────────────────────
        mujoco.mj_step(model, data)
        sim_t   += dt
        
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
        
        # ── Collect mixing angle statistics ───────────────────────
        # mix_angle already computed above; track it as tilt/mixing error
        tilt_error_values.append(float(mix_angle))

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