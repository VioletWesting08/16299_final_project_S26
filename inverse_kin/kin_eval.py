"""hybrid_eval.py — kinematics.py control tests evaluated with test_picking.py metrics.

Control architecture (from kinematics.py):
  - TrajectorySampler with min-jerk interpolation and time_scale speed control
  - solve_cartesian_ik (damped least-squares IK + null-space posture control)
  - TaskSpacePID (optional outer-loop position correction)
  - JointPIDController (joint-level PD torques)
  - Optional wrist orientation PID aligning tube to effective liquid gravity

Evaluation metrics (from test_picking.py):
  - Average / max EE speed                        [m/s]
  - Average / max position tracking error         [m]
  - Average / max tube tilt error                 [°]   (same as mixing angle)
  - Average / max liquid mixing angle             [°]
  - Integrated mixing score                       [°·s]

Run examples
  python hybrid_eval.py                          # interactive single run (viewer)
  python hybrid_eval.py --speed 1.5             # faster interactive run
  python hybrid_eval.py --sweep                 # headless grid sweep -> CSV

Requires: mujoco, numpy. Scene XMLs from franka_emika_panda/ must be present.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Sequence

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except Exception:
    mujoco = None  # type: ignore

# ═══════════════════════════════════════════════════════════════
# PATHS & CONSTANTS
# ═══════════════════════════════════════════════════════════════
SCENE_XML   = "franka_emika_panda/scene.xml"
ENV_XML     = "franka_emika_panda/hybrid_eval_scene.xml"
OUTPUT_DIR  = "outputs"
LOG_PREFIX  = "hybrid_eval"

Q_HOME = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])

# Joint PID defaults (from kinematics.py)
JOINT_KP_DEFAULT = np.array([100.0, 100.0, 100.0, 100.0,  50.0,  50.0, 20.0], dtype=float)
JOINT_KD_DEFAULT = np.array([ 20.0,  20.0,  20.0,  20.0,  10.0,  10.0,  5.0], dtype=float)

# Task-space PID defaults (from kinematics.py / test_picking.py)
TASK_KP_DEFAULT = 0.5
TASK_KI_DEFAULT = 0.0
TASK_KD_DEFAULT = 0.2

# IK / wrist defaults
IK_LAMBDA_SQ        = 1e-4
WRIST_ORIENT_KP_DEFAULT = 5.0
LIQUID_TAU          = 1.0          # liquid reorientation time constant (s)
TUBE_AXIS_LOCAL     = np.array([0.0, 0.0, 1.0])
ACCEL_LPFILTER_ALPHA = 0.04        # smoothing for finite-difference accel

# Null-space posture gain (from test_picking.py)
NULL_GAIN           = 0.5
Q_NOMINAL           = Q_HOME.copy()  # comfortable target pose

VISUAL_SLEEP_SEC    = 0.02
POST_RUN_HOLD_SEC   = 4.0

# Trajectory phases (identical in both source files; kinematics.py version used)
PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.04, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6000,  0.0000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4000,  0.4000, 0.4000], "gripper": 0.00, "duration": 0.6},
    {"name": "6. Place",     "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.04, "duration": 0.5},
]


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════
def min_jerk(t: float, duration: float, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    if t <= 0.0:
        return start.copy()
    if t >= duration:
        return goal.copy()
    tau = t / duration
    s   = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return start + s * (goal - start)


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    return vec / n if n > 1e-9 else fallback.copy()


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Return angle in degrees between vectors a and b."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    cos = float(np.clip(np.dot(a / na, b / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


# ═══════════════════════════════════════════════════════════════
# CONTROLLERS  (from kinematics.py)
# ═══════════════════════════════════════════════════════════════
class TaskSpacePID:
    """Outer-loop Cartesian position correction (from kinematics.py / test_picking.py)."""
    def __init__(self, kp: float = TASK_KP_DEFAULT,
                 ki: float = TASK_KI_DEFAULT,
                 kd: float = TASK_KD_DEFAULT) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = np.zeros(3)
        self.prev      = np.zeros(3)
        self.integral_clamp = 2.0

    def reset(self) -> None:
        self.integral.fill(0.0)
        self.prev.fill(0.0)

    def update(self, err: np.ndarray, dt: float) -> np.ndarray:
        self.integral += err * dt
        self.integral  = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)
        deriv = (err - self.prev) / max(dt, 1e-9)
        self.prev = err.copy()
        return self.kp * err + self.ki * self.integral + self.kd * deriv


class JointPIDController:
    """Joint-level PD torque controller (from kinematics.py)."""
    def __init__(self, kp: np.ndarray | None = None,
                 kd: np.ndarray | None = None) -> None:
        self.kp = kp.copy() if kp is not None else JOINT_KP_DEFAULT.copy()
        self.kd = kd.copy() if kd is not None else JOINT_KD_DEFAULT.copy()
        self.prev_err = np.zeros(7)

    def reset(self) -> None:
        self.prev_err.fill(0.0)

    def update(self, q: np.ndarray, qd: np.ndarray, dq: np.ndarray, dt: float) -> np.ndarray:
        err = qd - q
        self.prev_err = err.copy()
        return self.kp * err - self.kd * dq


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY SAMPLER  (from kinematics.py)
# ═══════════════════════════════════════════════════════════════
class TrajectorySampler:
    """Min-jerk waypoint sampler with time_scale acceleration."""
    def __init__(self, phases: Sequence[dict], time_scale: float = 1.0) -> None:
        self.phases = list(phases)
        base            = np.array([p["duration"] for p in self.phases], dtype=float)
        self.durations  = base / max(time_scale, 1e-9)
        self.boundaries = np.cumsum(self.durations)
        self.total_time = float(self.boundaries[-1])
        self.starts: List[np.ndarray] = []

    def set_start(self, start_xyz: np.ndarray) -> None:
        self.starts = [start_xyz.copy()]
        for i in range(1, len(self.phases)):
            self.starts.append(np.array(self.phases[i - 1]["target_xyz"], dtype=float))

    def sample(self, t: float) -> tuple[np.ndarray, str, float]:
        idx        = int(np.searchsorted(self.boundaries, t, side="right"))
        idx        = min(idx, len(self.phases) - 1)
        phase_start = self.boundaries[idx - 1] if idx > 0 else 0.0
        t_local    = t - phase_start
        start      = self.starts[idx]
        goal       = np.array(self.phases[idx]["target_xyz"], dtype=float)
        return (min_jerk(t_local, self.durations[idx], start, goal),
                self.phases[idx]["name"],
                self.phases[idx]["gripper"])


# ═══════════════════════════════════════════════════════════════
# IK  (from kinematics.py)
# ═══════════════════════════════════════════════════════════════
def body_id(model, names: Sequence[str]) -> int:
    for n in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if i != -1:
            return i
    raise RuntimeError(f"Could not find body among {names}")


def site_id(model, names: Sequence[str]) -> int:
    for n in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        if i != -1:
            return i
    raise RuntimeError(f"Could not find site among {names}")


def solve_cartesian_ik(model, data, site_index: int,
                       target_xyz: np.ndarray,
                       q_home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Damped least-squares IK with null-space posture control (kinematics.py)."""
    current = data.site_xpos[site_index].copy()
    err     = target_xyz - current

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_index)
    J  = jacp[:, :7]
    Jr = jacr[:, :7]

    JJT   = J @ J.T
    damp  = JJT + IK_LAMBDA_SQ * np.eye(3)
    dq    = J.T @ np.linalg.solve(damp, err)

    J_pinv   = J.T @ np.linalg.solve(damp, np.eye(3))
    null     = np.eye(7) - J_pinv @ J
    posture  = q_home - data.qpos[:7]

    q_des = data.qpos[:7] + dq + null @ (NULL_GAIN * posture)
    return q_des, J, Jr, err


# ═══════════════════════════════════════════════════════════════
# SCENE BUILDER  (from kinematics.py)
# ═══════════════════════════════════════════════════════════════
def build_scene(scene_path: str, out_path: str) -> None:
    scene_dir  = os.path.dirname(os.path.abspath(scene_path))
    panda_path = os.path.join(scene_dir, "panda.xml")

    with open(scene_path, "r", encoding="utf-8") as f:
        base = f.read()

    if os.path.exists(panda_path):
        with open(panda_path, "r", encoding="utf-8") as pf:
            panda_xml = pf.read()
        panda_xml = re.sub(r"\n\s*<keyframe>.*?</keyframe>\s*", "\n", panda_xml, flags=re.S)
        if 'name="gripper"' not in panda_xml:
            panda_xml = panda_xml.replace(
                '<geom mesh="hand_c" class="collision"/>',
                '<geom mesh="hand_c" class="collision"/>\n'
                '                      <site name="gripper" pos="0 0 0.1" size="0.004" rgba="1 0 0 0.35"/>',
            )
        runtime_panda = os.path.join(scene_dir, "panda_runtime.xml")
        with open(runtime_panda, "w", encoding="utf-8") as rf:
            rf.write(panda_xml)
        base = base.replace('<include file="panda.xml"/>',
                            f'<include file="{os.path.basename(runtime_panda)}"/>')

    tube_block = (
        '    <body name="centrifuge_tube" pos="0.6 0.0 0.07">\n'
        '        <freejoint name="tube_joint"/>\n'
        '        <geom name="tube_geom" type="cylinder" size="0.015 0.05" '
        'rgba="0.2 0.7 1.0 0.9" mass="0.05"/>\n'
        '    </body>'
    )
    out_xml = base.replace("</worldbody>", f"{tube_block}\n  </worldbody>")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out_xml)


# ═══════════════════════════════════════════════════════════════
# TRIAL CONFIG  (from kinematics.py)
# ═══════════════════════════════════════════════════════════════
@dataclass
class TrialConfig:
    kp_scale:     float = 1.0
    kd_scale:     float = 1.0
    task_kp:      float = TASK_KP_DEFAULT
    time_scale:   float = 1.0
    use_task_pid: bool  = True
    use_wrist_pid: bool = False
    wrist_kp:     float = WRIST_ORIENT_KP_DEFAULT
    name:         str   = "trial"


# ═══════════════════════════════════════════════════════════════
# METRICS COLLECTOR  (test_picking.py evaluation)
# ═══════════════════════════════════════════════════════════════
@dataclass
class PickingMetrics:
    """Mirrors the statistics block printed at the end of test_picking.py."""
    speed_values:         list = field(default_factory=list)
    pos_error_values:     list = field(default_factory=list)
    tilt_error_values:    list = field(default_factory=list)
    mixing_score_values:  list = field(default_factory=list)

    def record(self, speed: float, pos_error: float, mix_angle: float) -> None:
        self.speed_values.append(speed)
        self.pos_error_values.append(pos_error)
        self.tilt_error_values.append(mix_angle)   # tilt == mix angle (same signal)
        self.mixing_score_values.append(mix_angle)

    def summary(self, dt: float, sim_t: float, total_time: float,
                pid_status: str) -> dict:
        s  = np.array(self.speed_values)
        pe = np.array(self.pos_error_values)
        te = np.array(self.tilt_error_values)
        ms = np.array(self.mixing_score_values)
        return {
            "pid_status":             pid_status,
            "sim_time":               sim_t,
            "total_traj_time":        total_time,
            "avg_ee_speed":           float(np.mean(s)),
            "max_ee_speed":           float(np.max(s)),
            "avg_pos_error":          float(np.mean(pe)),
            "max_pos_error":          float(np.max(pe)),
            "avg_tilt_error_deg":     float(np.mean(te)),
            "max_tilt_error_deg":     float(np.max(te)),
            "avg_mixing_angle_deg":   float(np.mean(ms)),
            "max_mixing_angle_deg":   float(np.max(ms)),
            "integrated_mixing_score": float(np.sum(ms) * dt),
        }

    def print_summary(self, dt: float, sim_t: float, total_time: float,
                      pid_status: str) -> None:
        """Reproduce the test_picking.py statistics block verbatim."""
        d = self.summary(dt, sim_t, total_time, pid_status)
        print("\n" + "=" * 70)
        print(f"TRAJECTORY STATISTICS ({pid_status})")
        print("=" * 70)
        print(f"Total simulation time:       {d['sim_time']:.2f} s")
        print(f"Total trajectory time:       {d['total_traj_time']:.2f} s")
        print(f"\nAverage EE speed:            {d['avg_ee_speed']:.4f} m/s")
        print(f"Max EE speed:                {d['max_ee_speed']:.4f} m/s")
        print(f"\nAverage position error:      {d['avg_pos_error']:.6f} m")
        print(f"Max position error:          {d['max_pos_error']:.6f} m")
        print(f"\nAverage tube tilt error:     {d['avg_tilt_error_deg']:.2f}°")
        print(f"Max tube tilt error:         {d['max_tilt_error_deg']:.2f}°")
        print(f"\nAverage liquid mixing angle: {d['avg_mixing_angle_deg']:.2f}°")
        print(f"Max liquid mixing angle:     {d['max_mixing_angle_deg']:.2f}°")
        print(f"Integrated mixing score:     {d['integrated_mixing_score']:.2f}°·s")
        print("=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════
# MAIN TRIAL RUNNER
# ═══════════════════════════════════════════════════════════════
def run_trial(config: TrialConfig,
              use_viewer: bool = False) -> dict:
    """
    Execute one pick-and-place trial using kinematics.py's control stack,
    evaluated with test_picking.py's metrics.

    Returns a dict containing the PickingMetrics summary plus config fields.
    """
    if mujoco is None:
        raise RuntimeError("MuJoCo is required.")

    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    hand         = body_id(model, ["hand", "panda_hand"])
    gripper_site = site_id(model, ["gripper"])

    # Warm-up (let robot settle)
    data.qpos[:7] = Q_HOME
    mujoco.mj_forward(model, data)
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # Controllers
    traj      = TrajectorySampler(PHASES, time_scale=config.time_scale)
    traj.set_start(data.site_xpos[gripper_site].copy())

    task_pid  = TaskSpacePID(kp=config.task_kp)
    joint_pid = JointPIDController(kp=(JOINT_KP_DEFAULT * config.kp_scale),
                                   kd=(JOINT_KD_DEFAULT * config.kd_scale))
    metrics   = PickingMetrics()

    # Liquid inertia state (from test_picking.py)
    a_liquid        = np.array([0.0, 0.0, -9.81], dtype=float)
    gravity         = np.array([0.0, 0.0, -9.81], dtype=float)
    filtered_acc    = np.zeros(3)
    prev_vel        = np.zeros(3)

    t_sim         = 0.0
    prev_phase_idx = -1

    # Print header
    print(f"\n{'='*70}")
    print(f"TRIAL: {config.name}  |  time_scale={config.time_scale}  "
          f"kp_scale={config.kp_scale}  kd_scale={config.kd_scale}  "
          f"task_kp={config.task_kp}")
    print(f"  use_task_pid={config.use_task_pid}  use_wrist_pid={config.use_wrist_pid}")
    print(f"{'='*70}")

    def step_once() -> None:
        nonlocal t_sim, filtered_acc, a_liquid, prev_vel, prev_phase_idx

        # ── 1. EE velocity (world frame) ──────────────────────────────
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_SITE, gripper_site, vel6, 0
        )
        ee_vel = vel6[3:6].copy()
        speed  = float(np.linalg.norm(ee_vel))

        # ── 2. Effective gravity + liquid inertia (test_picking.py) ───
        acc_raw      = (ee_vel - prev_vel) / max(dt, 1e-9)
        prev_vel[:]  = ee_vel
        filtered_acc = (ACCEL_LPFILTER_ALPHA * acc_raw
                        + (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_acc)
        a_effective  = gravity + filtered_acc          # test_picking.py sign convention
        a_liquid    += ((a_effective - a_liquid) / max(LIQUID_TAU, 1e-9)) * dt

        # ── 3. Trajectory sample ─────────────────────────────────────
        target_xyz, phase_name, grip = traj.sample(t_sim)

        # ── 4. IK (kinematics.py) ────────────────────────────────────
        q_des, J, Jr, err = solve_cartesian_ik(
            model, data, gripper_site, target_xyz, Q_HOME
        )
        pos_error = float(np.linalg.norm(err))

        # Task-space PID correction (optional, kinematics.py style)
        if config.use_task_pid:
            corr  = task_pid.update(err, dt)
            q_des += J.T @ corr
        else:
            _ = task_pid.update(err, dt)     # keep integral ticking

        # Wrist orientation PID (kinematics.py)
        if config.use_wrist_pid:
            hand_rot  = data.xmat[hand].reshape(3, 3)
            z_actual  = normalize(hand_rot @ TUBE_AXIS_LOCAL, np.array([0., 0., 1.]))
            z_desired = normalize(a_liquid, np.array([0., 0., -1.]))
            rot_axis  = np.cross(z_actual, z_desired)
            if np.linalg.norm(rot_axis) > 1e-9:
                rot_axis  = rot_axis * float(config.wrist_kp)
                JrJrT    = Jr @ Jr.T
                dq_wrist = Jr.T @ np.linalg.solve(JrJrT + IK_LAMBDA_SQ * np.eye(3),
                                                    rot_axis)
                q_des   += dq_wrist

        # ── 5. Joint PID torques ─────────────────────────────────────
        dq_current        = data.qvel[:7].copy()
        data.ctrl[:7]     = joint_pid.update(data.qpos[:7], q_des, dq_current, dt)
        if model.nu >= 8:
            data.ctrl[7]  = grip

        # ── 6. Physics step ──────────────────────────────────────────
        mujoco.mj_step(model, data)
        t_sim += dt

        # ── 7. Compute mixing / tilt angle (test_picking.py) ─────────
        hand_rot  = data.xmat[hand].reshape(3, 3)
        tube_axis = hand_rot[:, 2]               # local Z in world frame
        mix_angle = angle_between_deg(tube_axis, a_liquid)

        # ── 8. Record test_picking.py metrics ────────────────────────
        metrics.record(speed, pos_error, mix_angle)

        # ── 9. Per-phase logging (test_picking.py style) ──────────────
        nonlocal prev_phase_idx
        phase_idx = int(np.searchsorted(traj.boundaries, t_sim, side="right"))
        phase_idx = min(phase_idx, len(PHASES) - 1)
        if phase_idx != prev_phase_idx:
            task_pid.reset()
            joint_pid.reset()
            p = PHASES[phase_idx]
            print(
                f"\n>>> PHASE {phase_idx + 1}: {p['name']}\n"
                f"    Target XYZ: {p['target_xyz']}\n"
                f"    Gripper:    {p['gripper']:.4f}\n"
                f"    Duration:   {p['duration'] / config.time_scale:.2f}s "
                f"(base {p['duration']:.1f}s, time_scale={config.time_scale})\n"
            )
            prev_phase_idx = phase_idx

        step_no = int(round(t_sim / dt))
        if step_no % 100 == 0:
            joint_angles = np.degrees(data.qpos[:7])
            joints_str   = " | ".join(
                [f"J{i+1}={a:7.2f}°" for i, a in enumerate(joint_angles)]
            )
            print(
                f"sim={t_sim:6.2f}s | pos_err={pos_error:.4f}m | "
                f"mix_angle={mix_angle:6.1f}° | "
                f"a_eff={float(np.linalg.norm(a_effective)):.2f}m/s²"
            )
            print(f"  Joints: {joints_str}")

    # ── Run loop ─────────────────────────────────────────────────────
    if use_viewer:
        vw = mujoco.viewer.launch_passive(model, data)
        is_running = getattr(vw, "is_running", lambda: True)
        try:
            while is_running() and t_sim < traj.total_time:
                step_once()
                vw.sync()
                time.sleep(max(dt, VISUAL_SLEEP_SEC))
            t0 = time.time()
            while is_running() and (time.time() - t0) < POST_RUN_HOLD_SEC:
                vw.sync()
                time.sleep(VISUAL_SLEEP_SEC)
        finally:
            if hasattr(vw, "close"):
                vw.close()
    else:
        while t_sim < traj.total_time:
            step_once()

    # Cleanup
    for path in [ENV_XML]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # Build PID status string (mirrors test_picking.py)
    pid_parts = []
    if config.use_task_pid:
        pid_parts.append("TaskSpace-PID")
    if config.use_wrist_pid:
        pid_parts.append("Wrist-PID")
    pid_status = " + ".join(pid_parts) if pid_parts else "IK-feedforward only"

    metrics.print_summary(dt, t_sim, traj.total_time, pid_status)

    summary = metrics.summary(dt, t_sim, traj.total_time, pid_status)
    return {
        "name":           config.name,
        "kp_scale":       config.kp_scale,
        "kd_scale":       config.kd_scale,
        "task_kp":        config.task_kp,
        "time_scale":     config.time_scale,
        "use_task_pid":   config.use_task_pid,
        "use_wrist_pid":  config.use_wrist_pid,
        **summary,
    }


# ═══════════════════════════════════════════════════════════════
# SWEEP  (kinematics.py grid + test_picking.py metrics output)
# ═══════════════════════════════════════════════════════════════
def run_sweep(
    kp_scales    = (0.8, 1.0, 1.2),
    kd_scales    = (0.8, 1.0, 1.2),
    task_kps     = (0.4, 0.6),
    time_scales  = (1.0, 1.5),
    use_task_pid : bool = True,
    use_wrist_pid: bool = False,
    out_csv: str = os.path.join(OUTPUT_DIR, "hybrid_eval_sweep.csv"),
) -> tuple[dict | None, list[dict]]:
    """
    Grid sweep identical to kinematics.py's run_sweep(), but every result
    is evaluated on test_picking.py's metrics (speed, pos error, tilt, mixing).

    Ranking priority (lower is better):
      1. integrated_mixing_score  (primary — liquid disturbance)
      2. avg_pos_error            (trajectory tracking)
      3. max_ee_speed             (peak forces)
      4. sim_time                 (efficiency)
    """
    configs = [
        TrialConfig(
            kp_scale=kp, kd_scale=kd, task_kp=tk, time_scale=ts,
            use_task_pid=use_task_pid, use_wrist_pid=use_wrist_pid,
            name=str(i),
        )
        for i, (kp, kd, tk, ts) in enumerate(
            itertools.product(kp_scales, kd_scales, task_kps, time_scales)
        )
    ]

    results = []
    for cfg in configs:
        print(
            f"\n[SWEEP] Config {cfg.name}: kp={cfg.kp_scale} kd={cfg.kd_scale} "
            f"task_kp={cfg.task_kp} time_scale={cfg.time_scale} "
            f"task_pid={cfg.use_task_pid} wrist_pid={cfg.use_wrist_pid}"
        )
        try:
            res = run_trial(cfg, use_viewer=False)
        except Exception as exc:
            print(f"  Trial {cfg.name} FAILED: {exc}")
            continue
        results.append(res)

    if not results:
        print("[SWEEP] No successful trials.")
        return None, []

    # Save CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "name", "kp_scale", "kd_scale", "task_kp", "time_scale",
        "use_task_pid", "use_wrist_pid",
        "sim_time", "total_traj_time",
        "avg_ee_speed", "max_ee_speed",
        "avg_pos_error", "max_pos_error",
        "avg_tilt_error_deg", "max_tilt_error_deg",
        "avg_mixing_angle_deg", "max_mixing_angle_deg",
        "integrated_mixing_score",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow({k: (f"{r[k]:.6f}" if isinstance(r[k], float) else r[k])
                             for k in fieldnames if k in r})

    print(f"\n[SWEEP] Results saved -> {out_csv}")

    # Rank: minimise (integrated_mixing_score, avg_pos_error, max_ee_speed, sim_time)
    best = min(
        results,
        key=lambda r: (
            r["integrated_mixing_score"],
            r["avg_pos_error"],
            r["max_ee_speed"],
            r["sim_time"],
        ),
    )
    print("\n[SWEEP] Best config (lowest integrated mixing score):")
    print(f"  name={best['name']}  kp_scale={best['kp_scale']}  "
          f"kd_scale={best['kd_scale']}  task_kp={best['task_kp']}  "
          f"time_scale={best['time_scale']}")
    print(f"  integrated_mixing_score = {best['integrated_mixing_score']:.4f}°·s")
    print(f"  avg_pos_error           = {best['avg_pos_error']:.6f} m")
    print(f"  max_ee_speed            = {best['max_ee_speed']:.4f} m/s")
    return best, results


# ═══════════════════════════════════════════════════════════════
# ENTRYPOINT
# ═══════════════════════════════════════════════════════════════
def main_interactive(speed: float = 1.0) -> None:
    if mujoco is None:
        raise RuntimeError("MuJoCo not available.")
    cfg = TrialConfig(
        kp_scale=1.0, kd_scale=1.0, task_kp=TASK_KP_DEFAULT,
        time_scale=speed, use_task_pid=True, use_wrist_pid=False,
        name="interactive",
    )
    run_trial(cfg, use_viewer=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="kinematics.py control tests with test_picking.py metrics"
    )
    p.add_argument("--sweep",        action="store_true",
                   help="Run grid sweep headless and save CSV")
    p.add_argument("--speed",        type=float, default=1.0,
                   help="time_scale for interactive run (>1.0 = faster)")
    p.add_argument("--use-wrist-pid", action="store_true",
                   help="Enable wrist orientation PID during sweep")
    p.add_argument("--no-task-pid",  action="store_true",
                   help="Disable task-space PID (IK feedforward only)")
    args = p.parse_args()

    if args.sweep:
        run_sweep(
            use_task_pid  = not args.no_task_pid,
            use_wrist_pid = args.use_wrist_pid,
            out_csv       = os.path.join(OUTPUT_DIR, "hybrid_eval_sweep.csv"),
        )
    else:
        main_interactive(speed=args.speed)