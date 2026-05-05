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
VIDEO_PATH      = "no_pid_simulation.mp4"
VIDEO_FPS       = 30
VIDEO_WIDTH     = 1280
VIDEO_HEIGHT    = 720
CAMERA_NAME     = None   # None = default free camera, or e.g. "track"

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

# Wrist orientation control
USE_WRIST_PID = False
USE_LQR_WEIGHT = False
ACCEL_LPFILTER_ALPHA = 0.03

# Liquid inertia model
LIQUID_TAU = 1.0

# Testing/validation options
AGGRESSIVE_TRANSPORT = True
INIT_TUBE_MISALIGNED = False
DEBUG_WRIST_PID = True

PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6, 0.0, 0.12],          "gripper": 0.04, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6, 0.0, 0.12],          "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6, 0.0, 0.40],          "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4, 0.4, 0.40],          "gripper": 0.00, "duration": 0.3 if AGGRESSIVE_TRANSPORT else 1.0},
    {"name": "6. Place",     "target_xyz": [0.4, 0.4, 0.12],          "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4, 0.4, 0.12],          "gripper": 0.04, "duration": 0.5},
]


# ═══════════════════════════════════════════════════════════════
# LQR-TUNED ORIENTATION CONTROL
# ═══════════════════════════════════════════════════════════════
def compute_lqr_orientation_weight(q_orientation_error: float, q_position_error: float) -> float:
    relative_priority = q_orientation_error / max(q_position_error, 1e-6)
    return np.tanh(relative_priority) * 0.75 + 0.15

Q_ORIENTATION_ERROR = 300.0
Q_POSITION_ERROR    = 1000.0
WRIST_WEIGHT_LQR    = compute_lqr_orientation_weight(Q_ORIENTATION_ERROR, Q_POSITION_ERROR)
WRIST_WEIGHT        = 0.27
print(f"[LQR] WRIST_WEIGHT={WRIST_WEIGHT_LQR:.3f}")


# ═══════════════════════════════════════════════════════════════
# CONTROLLERS
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
        self.Kd = np.array([20.0] * ndof)
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
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral   = 0.0
        self.prev_error = 0.0

    def reset(self):
        """Reset integral and derivative state."""
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, theta_mix_deg, dt):
        error          = theta_mix_deg
        self.integral += error * dt
        deriv          = (error - self.prev_error) / max(dt, 1e-9)
        self.prev_error = error
        return self.kp * error + self.ki * self.integral + self.kd * deriv


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY SAMPLER
# ═══════════════════════════════════════════════════════════════
class TrajectorySampler:
    def __init__(self, phases):
        self.phases     = phases
        self.durations  = [p["duration"] for p in phases]
        self.boundaries = np.cumsum(self.durations)
        self.total_time = self.boundaries[-1]
        self.starts     = [None] * len(phases)
        self.poly_coeffs = []

    def set_start(self, xyz_start):
        self.starts[0] = xyz_start.copy()
        for i in range(1, len(self.phases)):
            self.starts[i] = np.array(self.phases[i - 1]["target_xyz"])
        self.poly_coeffs = []
        for i in range(len(self.phases)):
            start = self.starts[i]
            end   = np.array(self.phases[i]["target_xyz"])
            T     = self.durations[i]
            a0 = start
            a1 = np.zeros(3)
            a2 = np.zeros(3)
            a3 = 10  * (end - start) / T**3
            a4 = -15 * (end - start) / T**4
            a5 = 6   * (end - start) / T**5
            self.poly_coeffs.append([a0, a1, a2, a3, a4, a5])

    def sample(self, t_eff):
        t_eff = float(np.clip(t_eff, 0.0, self.total_time))
        idx   = int(np.searchsorted(self.boundaries, t_eff, side="right"))
        idx   = min(idx, len(self.phases) - 1)
        phase   = self.phases[idx]
        t_start = self.boundaries[idx - 1] if idx > 0 else 0.0
        t_local = t_eff - t_start
        a0, a1, a2, a3, a4, a5 = self.poly_coeffs[idx]
        target_xyz = (a0 + a1*t_local + a2*t_local**2 +
                      a3*t_local**3 + a4*t_local**4 + a5*t_local**5)
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

    hand_id   = model.body("hand").id
    task_pid  = TaskSpacePID()
    joint_pid = JointPIDController(ndof=7)
    wrist_pid = MixingPID(kp=0.5, ki=0.01, kd=0.1)
    traj      = TrajectorySampler(PHASES)

    prev_ee_vel      = np.zeros(3)
    filtered_ee_accel = np.zeros(3)

    # ── warm-up ──────────────────────────────────────────────────
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    traj.set_start(data.xpos[hand_id].copy())

    if INIT_TUBE_MISALIGNED:
        data.qpos[5] += 0.5
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

    # ── counters & stats ─────────────────────────────────────────
    sim_t          = 0.0
    t_eff          = 0.0
    prev_phase_idx = -1
    speed_values        = []
    pos_error_values    = []
    tilt_error_values   = []
    mixing_score_values = []
    a_liquid = np.array([0.0, 0.0, -9.81])

    print("Starting trajectory …\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and t_eff < traj.total_time:

            # ── 1. EE VELOCITY ───────────────────────────────────
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0)
            cur_vel = vel6[3:].copy()
            speed_values.append(float(np.linalg.norm(cur_vel)))

            # ── 2. EFFECTIVE GRAVITY (sign fixed) ────────────────
            g = np.array([0.0, 0.0, -9.81])
            ee_accel_raw = (cur_vel - prev_ee_vel) / max(dt, 1e-9)
            prev_ee_vel  = cur_vel.copy()
            filtered_ee_accel = (ACCEL_LPFILTER_ALPHA * ee_accel_raw +
                                 (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_ee_accel)
            a_effective = g - filtered_ee_accel          # ← fixed sign

            # ── 3. LIQUID LAG ────────────────────────────────────
            da_liquid = (a_effective - a_liquid) / LIQUID_TAU
            a_liquid += da_liquid * dt

            # ── 4. TRAJECTORY SAMPLE ─────────────────────────────
            t_eff += dt
            target_xyz, gripper_w = traj.sample(t_eff)

            # ── 5. IK ────────────────────────────────────────────
            current_xyz = data.xpos[hand_id].copy()
            dx = target_xyz - current_xyz
            pos_error_values.append(float(np.linalg.norm(dx)))

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)
            J  = jacp[:, :7]
            Jr = jacr[:, :7]

            JJT    = J @ J.T
            dq     = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), dx)
            J_pinv = J.T @ np.linalg.solve(JJT + IK_LAMBDA_SQ * np.eye(3), np.eye(3))
            null_proj = np.eye(7) - J_pinv @ J

            # ── 6. MIXING ANGLE ──────────────────────────────────
            hand_rot  = data.xmat[hand_id].reshape(3, 3)
            tube_axis = hand_rot[:, 2]
            norm_tube = np.linalg.norm(tube_axis)
            norm_liq  = np.linalg.norm(a_liquid)
            if norm_tube > 1e-9 and norm_liq > 1e-9:
                cos_angle = np.clip(
                    np.dot(tube_axis, a_liquid / norm_liq), -1.0, 1.0)
                mix_angle = np.degrees(np.arccos(cos_angle))
            else:
                mix_angle = 0.0

            # ── 7. NULL-SPACE GOAL ───────────────────────────────
            q_current       = data.qpos[:7]
            q_posture_error = Q_NOMINAL - q_current
            
            # Determine phase for gating logic
            phase_idx = int(np.searchsorted(traj.boundaries, t_eff, side="right"))
            phase_idx = min(phase_idx, len(PHASES) - 1)

            # Wrist orientation control only during grasping/transport phases (3-6, indices 2-5)
            if USE_WRIST_PID and 2 <= phase_idx <= 5:
                correction_magnitude = wrist_pid.update(mix_angle, dt)
                z_desired     = a_liquid  / norm_liq  if norm_liq  > 1e-9 else np.array([0., 0., -1.])
                z_actual      = tube_axis / norm_tube if norm_tube > 1e-9 else np.array([0., 0.,  1.])
                rotation_axis = np.cross(z_actual, z_desired)
                if np.linalg.norm(rotation_axis) > 1e-9:
                    rotation_axis = (rotation_axis / np.linalg.norm(rotation_axis)
                                     * correction_magnitude)
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
                correction = task_pid.update(dx, dt)
                q_des += J.T @ correction * dt
            else:
                _ = task_pid.update(dx, dt)

            # ── 9. JOINT PID → TORQUES ───────────────────────────
            dq_current    = data.qvel[:7]
            tau           = joint_pid.update(data.qpos[:7], q_des, dq_current, dt)
            data.ctrl[:7] = tau
            if model.nu >= 8:
                data.ctrl[7] = gripper_w

            # ── 10. STEP PHYSICS ─────────────────────────────────
            mujoco.mj_step(model, data)
            sim_t += dt

            # ── 11. RECORD FRAME ─────────────────────────────────
            if RECORD_VIDEO and int(sim_t / dt) % steps_per_frame == 0:
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                writer.append_data(frame)

            # ── 12. LOGGING ──────────────────────────────────────
            # phase_idx already calculated above for phase-aware control
            if phase_idx != prev_phase_idx:
                task_pid.reset()
                joint_pid.reset()
                wrist_pid.reset()  # clear integral state between phases
                phase = PHASES[phase_idx]
                print(f"\n>>> PHASE {phase_idx+1}: {phase['name']}\n"
                      f"    Target XYZ: {phase['target_xyz']}\n"
                      f"    Gripper:    {phase['gripper']:.4f}\n"
                      f"    Duration:   {phase['duration']:.1f}s\n")
                prev_phase_idx = phase_idx

            if int(sim_t / dt) % 100 == 0:
                joint_angles = np.degrees(data.qpos[:7])
                joints_str   = " | ".join(
                    [f"J{i+1}={a:7.2f}°" for i, a in enumerate(joint_angles)])
                print(f"sim={sim_t:6.2f}s | t_eff={t_eff:6.2f}s | "
                      f"pos_err={np.linalg.norm(dx):.4f}m | "
                      f"mix_angle={mix_angle:6.1f}° | "
                      f"a_eff={np.linalg.norm(a_effective):.2f}m/s²")
                print(f"  Joints: {joints_str}")

            # Collect mixing metrics only during grasping/transport phases (3-6, indices 2-5)
            if 2 <= phase_idx <= 5:
                mixing_score_values.append(float(mix_angle))
                tilt_error_values.append(float(mix_angle))
            viewer.sync()

    # ── CLEANUP ───────────────────────────────────────────────────
    if RECORD_VIDEO:
        writer.close()
        try:
            renderer.close()
        except (AttributeError, RuntimeError):
            pass  # renderer cleanup may fail; that's ok
        print(f"\n[REC] Saved → {VIDEO_PATH}")

    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)

    # ── STATISTICS ───────────────────────────────────────────────
    speed_array         = np.array(speed_values)
    pos_error_array     = np.array(pos_error_values)
    tilt_error_array    = np.array(tilt_error_values)
    mixing_score_array  = np.array(mixing_score_values)

    print("\n" + "="*70)
    pid_status = []
    if USE_TASK_SPACE_PID: pid_status.append("TaskSpace-PID")
    if USE_WRIST_PID:      pid_status.append("Wrist-PID")
    status_str = " + ".join(pid_status) if pid_status else "IK-feedforward only"
    print(f"TRAJECTORY STATISTICS ({status_str})")
    print("="*70)
    print(f"Total simulation time:       {sim_t:.2f} s")
    print(f"Total trajectory time:       {traj.total_time:.2f} s")
    print(f"\nAverage EE speed:            {np.mean(speed_array):.4f} m/s")
    print(f"Max EE speed:                {np.max(speed_array):.4f} m/s")
    print(f"\nAverage position error:      {np.mean(pos_error_array):.6f} m")
    print(f"Max position error:          {np.max(pos_error_array):.6f} m")
    print(f"\nAverage tube tilt error:     {np.mean(tilt_error_array):.2f}°")
    print(f"Max tube tilt error:         {np.max(tilt_error_array):.2f}°")
    print(f"\nAverage liquid mixing angle: {np.mean(mixing_score_array):.2f}°")
    print(f"Max liquid mixing angle:     {np.max(mixing_score_array):.2f}°")
    print(f"Integrated mixing score:     {np.sum(mixing_score_array) * dt:.2f}°·s")
    print("="*70 + "\n")
    print("Trajectory complete.")


if __name__ == "__main__":
    main()