import mujoco
import mujoco.viewer
import numpy as np
import os
import imageio.v2 as imageio

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/debug_scene.xml"

# Recording
RECORD_VIDEO    = True
VIDEO_PATH      = "test_pid_simulation.mp4"
VIDEO_FPS       = 30
VIDEO_WIDTH     = 1280
VIDEO_HEIGHT    = 720
CAMERA_NAME     = None   # None = default free camera, or e.g. "track"

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
USE_WRIST_PID = True          # toggle wrist orientation control on/off
USE_LQR_WEIGHT = False         # use LQR-computed weight for null-space blending
ACCEL_LPFILTER_ALPHA = 0.03   # low-pass filter on acceleration measurement (lower=more smoothing)

# Liquid inertia model (first-order lag in reorientation)
LIQUID_TAU = 1.0             # liquid reorientation time constant (seconds)

# Testing/validation options
AGGRESSIVE_TRANSPORT = True   # fast transport phase (0.3s vs 1.0s) for stress-testing
INIT_TUBE_MISALIGNED = False  # start with tube deliberately tilted
DEBUG_WRIST_PID = True        # print wrist control diagnostics
# PHASES = [
#     {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 200, "duration": 1.0},
#     {"name": "2. Descend",   "target_xyz": [0.6, 0.0, 0.09], "gripper": 200, "duration": 1.0},
#     {"name": "3. Grasp",     "target_xyz": [0.6, 0.0, 0.09], "gripper": 0, "duration": 0.5},
#     {"name": "4. Lift",      "target_xyz": [0.6, 0.0, 0.40], "gripper": 0, "duration": 1.0},
#     {"name": "5. Transport", "target_xyz": [0.4, 0.4, 0.40], "gripper": 0, "duration": 0.3 if AGGRESSIVE_TRANSPORT else 1.0},
#     {"name": "6. Place",     "target_xyz": [0.4, 0.4, 0.12], "gripper": 0, "duration": 1.5},
#     {"name": "7. Release",   "target_xyz": [0.4, 0.4, 0.12], "gripper": 200, "duration": 1},
# ]
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

# ═══════════════════════════════════════════════════════════════
# TASK-SPACE PID  (position tracking)
# ═══════════════════════════════════════════════════════════════
class TaskSpacePID:
    """PID for end-effector position feedback."""
    def __init__(self):
        self.integral = np.zeros(3)
        self.prev_error = np.zeros(3)
        self.integral_clamp = 2.0

    def reset(self):
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def update(self, position_error: np.ndarray, dt: float) -> np.ndarray:
        self.integral += position_error * dt
        self.integral = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)
        derivative = (position_error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = position_error.copy()
        return TASK_KP * position_error + TASK_KI * self.integral + TASK_KD * derivative


# JOINT-LEVEL PID CONTROLLER
class JointPIDController:
    """Low-level PID controller for tracking desired joint positions."""
    def __init__(self, ndof=7):
        self.Kp = np.array([100.0] * ndof)   # position gains
        self.Kd = np.array([20.0] * ndof)    # velocity damping
        self.prev_error = np.zeros(ndof)

    def reset(self):
        self.prev_error[:] = 0.0

    def update(self, q_current: np.ndarray, q_desired: np.ndarray, dq_current: np.ndarray, dt: float) -> np.ndarray:
        """Compute torque commands to track desired positions."""
        error = q_desired - q_current
        derivative = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error.copy()
        # tau = Kp * error - Kd * velocity
        return self.Kp * error - self.Kd * dq_current


# ═══════════════════════════════════════════════════════════════
# MIXING PID (closed-loop wrist control on mixing angle)
# ═══════════════════════════════════════════════════════════════
class MixingPID:
    """
    Sensor:     theta_mix = angle between tube axis and a_liquid
    Setpoint:   0 degrees (tube aligned with liquid settlement)
    Output:     angular correction magnitude fed into null-space controller
    """
    def __init__(self, kp=0.5, ki=0.01, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self):
        """Reset integral and derivative state."""
        self.integral   = 0.0
        self.prev_error = 0.0

    def update(self, theta_mix_deg: float, dt: float) -> float:
        """Compute PID output for wrist correction magnitude."""
        error           = theta_mix_deg   # setpoint is 0
        self.integral  += error * dt
        deriv           = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error
        return self.kp*error + self.ki*self.integral + self.kd*deriv


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY SAMPLER
# ═══════════════════════════════════════════════════════════════
class TrajectorySampler:
    """5th-order minimum-jerk trajectory between waypoints."""
    def __init__(self, phases):
        self.phases = phases
        self.durations = [p["duration"] for p in phases]
        self.boundaries = np.cumsum(self.durations)
        self.total_time = self.boundaries[-1]
        self.starts = [None] * len(phases)
        
        # Precompute polynomial coefficients for each phase
        self.poly_coeffs = []  # will be filled in set_start()

    def set_start(self, xyz_start: np.ndarray):
        """Call once with the robot's actual EE position at t=0."""
        self.starts[0] = xyz_start.copy()
        for i in range(1, len(self.phases)):
            self.starts[i] = np.array(self.phases[i - 1]["target_xyz"])
        
        # Precompute 5th-order polynomial coefficients for each segment
        self.poly_coeffs = []
        for i in range(len(self.phases)):
            start = self.starts[i]
            end = np.array(self.phases[i]["target_xyz"])
            T = self.durations[i]  # segment duration
            
            # 5th-order minimum-jerk polynomial: x(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
            # Boundary conditions: x(0)=start, x(T)=end, dx/dt(0)=0, dx/dt(T)=0, d2x/dt2(0)=0, d2x/dt2(T)=0
            # Solve for coefficients a0..a5
            
            a0 = start
            a1 = np.zeros(3)  # initial velocity = 0
            a2 = np.zeros(3)  # initial acceleration = 0
            
            # From boundary conditions: solve for a3, a4, a5
            # a3 = 10*(end - start) / T^3
            # a4 = -15*(end - start) / T^4
            # a5 = 6*(end - start) / T^5
            a3 = 10 * (end - start) / (T ** 3)
            a4 = -15 * (end - start) / (T ** 4)
            a5 = 6 * (end - start) / (T ** 5)
            
            self.poly_coeffs.append([a0, a1, a2, a3, a4, a5])

    def sample(self, t_eff: float):
        """Return (target_xyz, gripper_width) at effective time t_eff."""
        t_eff = float(np.clip(t_eff, 0.0, self.total_time))

        # Which phase are we in?
        idx = int(np.searchsorted(self.boundaries, t_eff, side="right"))
        idx = min(idx, len(self.phases) - 1)

        phase = self.phases[idx]
        t_start = self.boundaries[idx - 1] if idx > 0 else 0.0
        t_local = t_eff - t_start  # time within this segment
        
        # Evaluate 5th-order polynomial
        a0, a1, a2, a3, a4, a5 = self.poly_coeffs[idx]
        target_xyz = (a0 + 
                      a1 * t_local + 
                      a2 * (t_local ** 2) + 
                      a3 * (t_local ** 3) + 
                      a4 * (t_local ** 4) + 
                      a5 * (t_local ** 5))
        
        return target_xyz, phase["gripper"]


# ═══════════════════════════════════════════════════════════════
# SCENE BUILDER
# ═══════════════════════════════════════════════════════════════
def build_scene(scene_path, out_path):
    xml = f"""<mujoco>
    <include file="{os.path.abspath(scene_path)}"/>
    <worldbody>
        <body name="centrifuge_tube" pos="0.6 -0.025 0.05">
            <freejoint name="tube_joint"/>
            <geom name="tube_geom" type="cylinder" size="0.015 0.05"
                  rgba="0.2 0.7 1.0 0.9" mass="0.01"
                  condim="4" friction="2.0 0.5 0.0001" 
                  solimp="0.95 0.99 0.001" solref="0.01 1"/>
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
    print("\n=== MODEL DOF LAYOUT ===")
    for i in range(model.njnt):
        name = model.joint(i).name
        dof_addr = model.jnt_dofadr[i]
        jtype = model.jnt_type[i]
        print(f"Joint {i}: '{name}' | type={jtype} | dof_addr={dof_addr}")

    print("\n=== ACTUATORS ===")
    for i in range(model.nu):
        print(f"Actuator {i}: '{model.actuator(i).name}' | trnid={model.actuator(i).trnid[0]}")
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

    # Initialize tube deliberately misaligned if testing
    if INIT_TUBE_MISALIGNED:
        data.qpos[5] += 0.5   # tilt joint 6
        mujoco.mj_forward(model, data)

    # ── set up offscreen renderer ─────────────────────────────────
    if RECORD_VIDEO:
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer   = imageio.get_writer(VIDEO_PATH, fps=VIDEO_FPS, macro_block_size=None)
        steps_per_frame = max(1, int(1.0 / (VIDEO_FPS * dt)))
        
        # Set up camera for bottom-right view
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat = np.array([0.5, 0.2, 0.2])
        camera.distance = 1.2
        camera.elevation = 0
        camera.azimuth = 220
        
        print(f"[REC] Recording to {VIDEO_PATH} @ {VIDEO_FPS}fps "
              f"(1 frame every {steps_per_frame} steps) | Camera: bottom-right")

    # Generate and log full trajectory
    trajectory_samples = 100
    full_trajectory = []
    for i in range(trajectory_samples + 1):
        t_sample = i / trajectory_samples * traj.total_time
        xyz, _ = traj.sample(t_sample)
        full_trajectory.append(xyz)
    
    full_trajectory = np.array(full_trajectory)
    
    print("\n" + "="*70)
    print("FULL TRAJECTORY (5th-order minimum-jerk, 100 points)")
    print("="*70)
    print("Time [s] | X [m]     | Y [m]     | Z [m]")
    print("-"*70)
    for i, xyz in enumerate(full_trajectory[::10]):
        t_sample = (i * 10) / trajectory_samples * traj.total_time
        print(f"{t_sample:7.2f} | {xyz[0]:9.4f} | {xyz[1]:9.4f} | {xyz[2]:9.4f}")
    print("-"*70)
    print("="*70 + "\n")

    # ── time counters ────────────────────────────────────────────
    # sim_t   : wall-clock simulation time (advances by dt every step)
    # t_eff   : effective trajectory time  (advances by dt every step)
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

    print("Starting trajectory …\n")
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and t_eff < traj.total_time:

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
            da_liquid = (a_effective - a_liquid) / LIQUID_TAU
            a_liquid += da_liquid * dt

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

            # # Error is calculated from this floating midpoint
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
            if USE_WRIST_PID:
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

            if RECORD_VIDEO and int(sim_t / dt) % steps_per_frame == 0:
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                writer.append_data(frame)

            # ── 7. UPDATE TRACKING ────────────────────────────────
            # (no longer tracking acceleration for jerk)
            sim_t   += dt

            # ── 8. LOGGING ────────────────────────────────────────
            phase_idx = int(np.searchsorted(traj.boundaries, t_eff, side="right"))
            phase_idx = min(phase_idx, len(PHASES) - 1)
            
            # Print phase details when phase changes
            if phase_idx != prev_phase_idx:
                task_pid.reset()
                joint_pid.reset()
                # Note: MixingPID doesn't require explicit reset (stateless gains)
                phase = PHASES[phase_idx]
                print(
                    f"\n>>> PHASE {phase_idx + 1}: {phase['name']}\n"
                    f"    Target XYZ: {phase['target_xyz']}\n"
                    f"    Gripper:    {phase['gripper']:.4f}\n"
                    f"    Duration:   {phase['duration']:.1f}s\n"
                )
                prev_phase_idx = phase_idx
            
            # Print progress every 100 steps (more frequent)
            if int(sim_t / dt) % 100 == 0:
                error_mag = np.linalg.norm(dx) if 'dx' in locals() else 0.0
                # mix_angle already computed above; reuse it
                mix_angle_ref = mix_angle
                
                # Print all joint angles
                joint_angles = np.degrees(data.qpos[:7])
                joints_str = " | ".join([f"J{i+1}={ang:7.2f}°" for i, ang in enumerate(joint_angles)])
                
                print(
                    f"sim={sim_t:6.2f}s | t_eff={t_eff:6.2f}s | "
                    f"pos_err={error_mag:.4f}m | "
                    f"mix_angle={mix_angle_ref:6.1f}° | "
                    f"a_eff={np.linalg.norm(a_effective):.2f}m/s²"
                )
                print(f"  Joints: {joints_str}")
            
            # ── Collect mixing angle statistics ───────────────────────
            # mix_angle already computed above; track it as tilt/mixing error
            tilt_error_values.append(float(mix_angle))

            viewer.sync()

    if RECORD_VIDEO:
        writer.close()
        try:
            renderer.close()
        except (AttributeError, RuntimeError):
            pass  # renderer cleanup may fail; that's ok
        print(f"\n[REC] Saved → {VIDEO_PATH}")

    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)
    
    # ── PRINT STATISTICS ──────────────────────────────────────────
    speed_array = np.array(speed_values)
    pos_error_array = np.array(pos_error_values)
    tilt_error_array = np.array(tilt_error_values)
    mixing_score_array = np.array(mixing_score_values)
    
    print("\n" + "="*70)
    pid_status = []
    if USE_TASK_SPACE_PID:
        pid_status.append("TaskSpace-PID")
    if USE_WRIST_PID:
        pid_status.append("Wrist-PID")
    status_str = " + ".join(pid_status) if pid_status else "IK-feedforward only"
    print(f"TRAJECTORY STATISTICS ({status_str})")
    print("="*70)
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
    print(f"Integrated mixing score:     {np.sum(mixing_score_array) * dt:.2f}°·s")
    print("="*70 + "\n")
    
    print("Trajectory complete.")


if __name__ == "__main__":
    main()