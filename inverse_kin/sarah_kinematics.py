"""Kinematics sweep: maximize speed while minimizing sudden-motion (jerk/accel).

This file provides:
- PID-only control (task-space PID -> joint PID) using full 7-DOF body
- Controllable speed via `time_scale` (higher => faster execution)
- TrialConfig with tunable `kp_scale`, `kd_scale`, `task_kp`, `time_scale`
- `run_trial()` returns sim_time and risk_integral
- `run_sweep()` runs grid and selects best config using a weighted objective

Run examples:
  python kinematics.py            # interactive single run (viewer)
  python kinematics.py --speed 1.5  # run a single interactive faster trial
  python kinematics.py --sweep     # run grid sweep headless and save results

Note: this script requires MuJoCo and the scene files present in the repo.
"""
from __future__ import annotations

import csv
import itertools
import os
import time
import re
from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover - optional dependency for GIF recording
    imageio = None

# MuJoCo import deferred: user environment must have mujoco installed
try:
    import mujoco
    import mujoco.viewer
except Exception:  # pragma: no cover - runtime environment dependent
    mujoco = None

# --- CONFIGURABLE DEFAULTS ---
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML = "franka_emika_panda/kinematics_scene.xml"
OUTPUT_DIR = "outputs"
LOG_PREFIX = "kinematics"

# Robot & control defaults
Q_HOME = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])
JOINT_KP = np.array([100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 20.0], dtype=float)
JOINT_KD = np.array([20.0, 20.0, 20.0, 20.0, 10.0, 10.0, 5.0], dtype=float)
TASK_KP = 0.5
TASK_KI = 0.0
TASK_KD = 0.2
WRIST_ORIENT_KP = 5.0
LIQUID_TAU = 1.0
TUBE_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

ACCEL_LPFILTER_ALPHA = 0.04
VISUAL_SLEEP_SEC = 0.02
POST_RUN_HOLD_SEC = 4.0

# Phases (start positions are set at runtime)
PHASES = [
    {"name": "hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "descend",   "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.04, "duration": 1.0},
    {"name": "grasp",     "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.00, "duration": 0.5},
    {"name": "lift",      "target_xyz": [0.6000,  0.0000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "transport", "target_xyz": [0.4000,  0.4000, 0.4000], "gripper": 0.00, "duration": 0.6},
    {"name": "place",     "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.00, "duration": 1.0},
    {"name": "release",   "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.04, "duration": 0.5},
]

# --- UTILITIES ---

def min_jerk(t: float, duration: float, start: np.ndarray, goal: np.ndarray) -> np.ndarray:
    if t <= 0.0:
        return start.copy()
    if t >= duration:
        return goal.copy()
    tau = t / duration
    s = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
    return start + s * (goal - start)


def normalize(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec))
    if n < 1e-9:
        return fallback.copy()
    return vec / n

# --- Controllers & helpers ---

class TaskSpacePID:
    def __init__(self, kp: float = TASK_KP, ki: float = TASK_KI, kd: float = TASK_KD) -> None:
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = np.zeros(3)
        self.prev = np.zeros(3)

    def reset(self) -> None:
        self.integral.fill(0.0)
        self.prev.fill(0.0)

    def update(self, err: np.ndarray, dt: float) -> np.ndarray:
        self.integral += err * dt
        derivative = (err - self.prev) / max(dt, 1e-9)
        self.prev = err.copy()
        return self.kp * err + self.ki * self.integral + self.kd * derivative


class JointPIDController:
    def __init__(self, kp: np.ndarray | None = None, kd: np.ndarray | None = None) -> None:
        self.kp = kp.copy() if kp is not None else JOINT_KP.copy()
        self.kd = kd.copy() if kd is not None else JOINT_KD.copy()
        self.prev_err = np.zeros(7)

    def reset(self) -> None:
        self.prev_err.fill(0.0)

    def update(self, q: np.ndarray, qd: np.ndarray, dq: np.ndarray, dt: float) -> np.ndarray:
        err = qd - q
        _ = (err - self.prev_err) / max(dt, 1e-9)
        self.prev_err = err.copy()
        return self.kp * err - self.kd * dq


class TrajectorySampler:
    def __init__(self, phases: Sequence[dict], time_scale: float = 1.0) -> None:
        # time_scale > 1.0 => faster (durations reduced by time_scale)
        self.phases = list(phases)
        base = np.array([p["duration"] for p in self.phases], dtype=float)
        self.durations = base / max(time_scale, 1e-9)
        self.boundaries = np.cumsum(self.durations)
        self.total_time = float(self.boundaries[-1])
        self.starts: List[np.ndarray] = []

    def set_start(self, start_xyz: np.ndarray) -> None:
        self.starts = [start_xyz.copy()]
        for i in range(1, len(self.phases)):
            self.starts.append(np.array(self.phases[i - 1]["target_xyz"], dtype=float))

    def sample(self, t: float) -> tuple[np.ndarray, str, float]:
        idx = int(np.searchsorted(self.boundaries, t, side="right"))
        idx = min(idx, len(self.phases) - 1)
        phase_start = self.boundaries[idx - 1] if idx > 0 else 0.0
        t_local = t - phase_start
        start = self.starts[idx]
        goal = np.array(self.phases[idx]["target_xyz"], dtype=float)
        return min_jerk(t_local, self.durations[idx], start, goal), self.phases[idx]["name"], self.phases[idx]["gripper"]


class EndEffectorKinematics:
    def __init__(self, dt: float) -> None:
        self.dt = dt
        self.prev_vel = np.zeros(3)
        self.prev_acc = np.zeros(3)

    def update(self, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        acc = (vel - self.prev_vel) / max(self.dt, 1e-9)
        jerk = (acc - self.prev_acc) / max(self.dt, 1e-9)
        self.prev_vel = vel.copy()
        self.prev_acc = acc.copy()
        return acc, jerk


class MotionMonitor:
    def __init__(self) -> None:
        self.log = []
        self.risk_integral = 0.0
        self.tilt_integral = 0.0
        self.max_lateral_acc = 0.0
        self.max_jerk = 0.0
        self.max_lat_jerk = 0.0
        self.max_tilt_deg = 0.0

    def update(self, t: float, acc_world: np.ndarray, jerk_world: np.ndarray, xmat: np.ndarray, tilt_deg: float, dt: float) -> None:
        R = xmat.reshape(3, 3)
        tube_axis = normalize(R @ TUBE_AXIS_LOCAL, np.array([0.0, 0.0, 1.0]))
        acc_axial = np.dot(acc_world, tube_axis) * tube_axis
        acc_lat = acc_world - acc_axial
        jerk_axial = np.dot(jerk_world, tube_axis) * tube_axis
        jerk_lat = jerk_world - jerk_axial

        lat_acc_mag = float(np.linalg.norm(acc_lat))
        jerk_mag = float(np.linalg.norm(jerk_world))
        lat_jerk_mag = float(np.linalg.norm(jerk_lat))

        # original risk used a linear combination; we reuse that form
        risk_rate = lat_acc_mag + 1.8 * lat_jerk_mag
        self.risk_integral += risk_rate * dt
        self.tilt_integral += float(tilt_deg) * dt

        self.max_lateral_acc = max(self.max_lateral_acc, lat_acc_mag)
        self.max_jerk = max(self.max_jerk, jerk_mag)
        self.max_lat_jerk = max(self.max_lat_jerk, lat_jerk_mag)
        self.max_tilt_deg = max(self.max_tilt_deg, float(tilt_deg))

        self.log.append((t, lat_acc_mag, jerk_mag, lat_jerk_mag, risk_rate, float(tilt_deg)))

    def summary(self) -> dict:
        return {
            "risk_integral": float(self.risk_integral),
            "tilt_integral": float(self.tilt_integral),
            "peak_lateral_acc": float(self.max_lateral_acc),
            "peak_jerk": float(self.max_jerk),
            "peak_lateral_jerk": float(self.max_lat_jerk),
            "peak_tilt_deg": float(self.max_tilt_deg),
        }

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "lat_acc", "jerk", "lat_jerk", "risk_rate", "tilt_deg"])
            for row in self.log:
                writer.writerow([f"{row[0]:.4f}", f"{row[1]:.6f}", f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}", f"{row[5]:.4f}"])


# --- Scene / IK helpers (assume standard panda scene layout) ---

def body_id(model: "mujoco.MjModel", names: Sequence[str]) -> int:
    for n in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
        if i != -1:
            return i
    raise RuntimeError(f"Could not find body among {names}")


def site_id(model: "mujoco.MjModel", names: Sequence[str]) -> int:
    for n in names:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, n)
        if i != -1:
            return i
    raise RuntimeError(f"Could not find site among {names}")


def solve_cartesian_ik(model: "mujoco.MjModel", data: "mujoco.MjData", site_index: int, target_xyz: np.ndarray, q_home: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    current = data.site_xpos[site_index].copy()
    err = target_xyz - current

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_index)
    J = jacp[:, :7]
    Jr = jacr[:, :7]

    JJT = J @ J.T
    damp = JJT + 1e-4 * np.eye(3)
    dq = J.T @ np.linalg.solve(damp, err)

    J_pinv = J.T @ np.linalg.solve(damp, np.eye(3))
    null = np.eye(7) - J_pinv @ J
    posture_err = q_home - data.qpos[:7]

    q_des = data.qpos[:7] + dq + null @ (0.5 * posture_err)
    return q_des, J, Jr, err


@dataclass
class TrialConfig:
    kp_scale: float = 1.0
    kd_scale: float = 1.0
    task_kp: float = TASK_KP
    time_scale: float = 1.0
    use_task_pid: bool = True
    use_wrist_pid: bool = False
    wrist_kp: float = WRIST_ORIENT_KP
    name: str = "trial"


def run_trial(config: TrialConfig, use_viewer: bool = False, save_prefix: str | None = None) -> dict:
    """Run a single trial and return summary metrics.

    The trajectory is defined for the gripper site, not the hand body center.
    """
    if mujoco is None:
        raise RuntimeError("MuJoCo Python package is required to run trials in this environment.")

    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data = mujoco.MjData(model)

    dt = model.opt.timestep
    hand = body_id(model, ["hand", "panda_hand"])
    gripper_site = site_id(model, ["gripper"])

    data.qpos[:7] = Q_HOME
    mujoco.mj_forward(model, data)

    traj = TrajectorySampler(PHASES, time_scale=config.time_scale)
    traj.set_start(data.site_xpos[gripper_site].copy())

    task_pid = TaskSpacePID(kp=config.task_kp)
    joint_pid = JointPIDController(kp=(JOINT_KP * config.kp_scale), kd=(JOINT_KD * config.kd_scale))
    ee = EndEffectorKinematics(dt)
    monitor = MotionMonitor()

    filtered_acc = np.zeros(3)
    a_liquid = np.array([0.0, 0.0, -9.81], dtype=float)
    gravity = np.array([0.0, 0.0, -9.81], dtype=float)
    t_sim = 0.0
    reach_integral = 0.0

    def step_once() -> None:
        nonlocal t_sim, filtered_acc, a_liquid, reach_integral

        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, gripper_site, vel6, 0)
        vel_world = vel6[3:6].copy()
        acc_raw, jerk = ee.update(vel_world)
        filtered_acc = ACCEL_LPFILTER_ALPHA * acc_raw + (1 - ACCEL_LPFILTER_ALPHA) * filtered_acc
        a_effective = gravity - filtered_acc
        a_liquid += ((a_effective - a_liquid) / max(LIQUID_TAU, 1e-9)) * dt

        t_sim += dt
        target_xyz, _, grip = traj.sample(t_sim)

        q_current = data.qpos[:7].copy()
        dq_current = data.qvel[:7].copy()

        q_des, J, Jr, err = solve_cartesian_ik(model, data, gripper_site, target_xyz, Q_HOME)
        corr = task_pid.update(err, dt) if config.use_task_pid else np.zeros(3)
        q_des += J.T @ corr

        if config.use_wrist_pid:
            hand_rot = data.xmat[hand].reshape(3, 3)
            z_actual = normalize(hand_rot @ TUBE_AXIS_LOCAL, np.array([0.0, 0.0, 1.0]))
            z_desired = normalize(a_liquid, np.array([0.0, 0.0, -1.0]))
            rot_axis = np.cross(z_actual, z_desired)
            if np.linalg.norm(rot_axis) > 1e-9:
                rot_axis = rot_axis * float(config.wrist_kp)
                JrJrT = Jr @ Jr.T
                dq_wrist = Jr.T @ np.linalg.solve(JrJrT + 1e-3 * np.eye(3), rot_axis)
                q_des += dq_wrist

        data.ctrl[:7] = joint_pid.update(q_current, q_des, dq_current, dt)
        if model.nu >= 8:
            data.ctrl[7] = grip

        mujoco.mj_step(model, data)

        gripper_pos = data.site_xpos[gripper_site].copy()
        reach_integral += float(np.linalg.norm(target_xyz - gripper_pos)) * dt

        hand_rot = data.xmat[hand].reshape(3, 3)
        z_axis = normalize(hand_rot @ TUBE_AXIS_LOCAL, np.array([0.0, 0.0, 1.0]))
        a_norm = float(np.linalg.norm(a_liquid))
        if a_norm > 1e-9:
            cosang = float(np.clip(np.dot(z_axis, a_liquid / a_norm), -1.0, 1.0))
            tilt_deg = float(np.degrees(np.arccos(cosang)))
        else:
            tilt_deg = 0.0
        monitor.update(t_sim, filtered_acc, jerk, data.xmat[hand].copy(), tilt_deg, dt)

    if use_viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        # Fallback if is_running doesn't exist on older versions
        is_running = getattr(viewer, "is_running", lambda: True)
        try:
            while is_running() and t_sim < traj.total_time:
                step_once()
                viewer.sync()
                time.sleep(max(dt, VISUAL_SLEEP_SEC))

            t0 = time.time()
            while is_running() and (time.time() - t0) < POST_RUN_HOLD_SEC:
                viewer.sync()
                time.sleep(VISUAL_SLEEP_SEC)
        finally:
            if hasattr(viewer, "close"):
                viewer.close()
    else:
        while t_sim < traj.total_time:
            step_once()

    summary = monitor.summary()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{LOG_PREFIX}_{config.name}.csv")
    monitor.save_csv(csv_path)

    return {
        "name": config.name,
        "kp_scale": config.kp_scale,
        "kd_scale": config.kd_scale,
        "task_kp": config.task_kp,
        "time_scale": config.time_scale,
        "use_task_pid": config.use_task_pid,
        "use_wrist_pid": config.use_wrist_pid,
        "reach_integral": float(reach_integral),
        "sim_time": float(t_sim),
        **summary,
    }


def compute_score(result: dict, w_time: float, w_risk: float, w_tilt: float) -> float:
    # Lower is better; ranking is priority-based with tilt first, then reach, then jerk, then time.
    tilt_term = w_tilt * 1_000_000.0 * result["tilt_integral"]
    reach_term = 100_000.0 * result.get("reach_integral", 0.0)
    risk_term = w_risk * 1_000.0 * result["risk_integral"]
    time_term = w_time * 1.0 * result["sim_time"]
    return tilt_term + reach_term + risk_term + time_term


def run_sweep(
    kp_scales=(0.8, 1.0, 1.2),
    kd_scales=(0.8, 1.0, 1.2),
    task_kps=(0.4, 0.6),
    time_scales=(1.0, 1.5),
    w_time: float = 0.4,
    w_risk: float = 0.3,
    w_tilt: float = 0.3,
    use_task_pid: bool = True,
    use_wrist_pid: bool = False,
    out_csv: str = os.path.join(OUTPUT_DIR, "kinematics_sweep.csv"),
) -> tuple[dict | None, list[dict]]:
    # Run grid sweep headless.
    configs = []
    for i, (kp, kd, tk, ts) in enumerate(itertools.product(kp_scales, kd_scales, task_kps, time_scales)):
        configs.append(
            TrialConfig(
                kp_scale=kp,
                kd_scale=kd,
                task_kp=tk,
                time_scale=ts,
                use_task_pid=use_task_pid,
                use_wrist_pid=use_wrist_pid,
                name=str(i),
            )
        )

    results = []
    for cfg in configs:
        print(
            f"[SWEEP] Running {cfg.name} kp={cfg.kp_scale} kd={cfg.kd_scale} "
            f"task_kp={cfg.task_kp} time_scale={cfg.time_scale} task_pid={cfg.use_task_pid} wrist_pid={cfg.use_wrist_pid}"
        )
        try:
            res = run_trial(cfg, use_viewer=False, save_prefix=os.path.join(OUTPUT_DIR, LOG_PREFIX))
        except Exception as e:
            print(f"  Trial {cfg.name} failed: {e}")
            continue
        res["score"] = float(compute_score(res, w_time=w_time, w_risk=w_risk, w_tilt=w_tilt))
        results.append(res)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "name", "kp_scale", "kd_scale", "task_kp", "time_scale", "use_task_pid", "use_wrist_pid",
                "sim_time", "risk_integral", "tilt_integral", "reach_integral", "score",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r["name"], r["kp_scale"], r["kd_scale"], r["task_kp"], r["time_scale"], r["use_task_pid"], r["use_wrist_pid"],
                    f"{r['sim_time']:.4f}", f"{r['risk_integral']:.6f}", f"{r['tilt_integral']:.6f}", f"{r.get('reach_integral', 0.0):.6f}", f"{r['score']:.6f}",
                ]
            )

    best = min(results, key=lambda x: (x.get("tilt_integral", float("inf")), x.get("reach_integral", float("inf")), x.get("risk_integral", float("inf")), x.get("sim_time", float("inf")))) if results else None
    print("[SWEEP] Done. Best:", best)
    return best, results


def sweep_and_record(
    kp_scales=(0.8, 1.0, 1.2),
    kd_scales=(0.8, 1.0, 1.2),
    task_kps=(0.4, 0.6),
    time_scales=(0.5, 1.0, 1.5, 2.0),
    w_time: float = 0.4,
    w_risk: float = 0.3,
    w_tilt: float = 0.3,
) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Sweep A: full body only
    best_full_body_only, results_fbo = run_sweep(
        kp_scales=kp_scales,
        kd_scales=kd_scales,
        task_kps=task_kps,
        time_scales=time_scales,
        w_time=w_time,
        w_risk=w_risk,
        w_tilt=w_tilt,
        use_task_pid=True,
        use_wrist_pid=False,
        out_csv=os.path.join(OUTPUT_DIR, "kinematics_sweep_full_body_only.csv"),
    )
    # Sweep B: full body + wrist orientation
    best_full_body, results_fb = run_sweep(
        kp_scales=kp_scales,
        kd_scales=kd_scales,
        task_kps=task_kps,
        time_scales=time_scales,
        w_time=w_time,
        w_risk=w_risk,
        w_tilt=w_tilt,
        use_task_pid=True,
        use_wrist_pid=True,
        out_csv=os.path.join(OUTPUT_DIR, "kinematics_sweep_full_body.csv"),
    )

    if best_full_body_only is None or best_full_body is None:
        print("No successful results to record.")
        return

    fbo_cfg = TrialConfig(
        kp_scale=best_full_body_only["kp_scale"],
        kd_scale=best_full_body_only["kd_scale"],
        task_kp=best_full_body_only["task_kp"],
        time_scale=best_full_body_only["time_scale"],
        use_task_pid=True,
        use_wrist_pid=False,
        name="full_body_only",
    )
    fb_cfg = TrialConfig(
        kp_scale=best_full_body["kp_scale"],
        kd_scale=best_full_body["kd_scale"],
        task_kp=best_full_body["task_kp"],
        time_scale=best_full_body["time_scale"],
        use_task_pid=True,
        use_wrist_pid=True,
        name="full_body",
    )
    no_pid_cfg = TrialConfig(
        kp_scale=fb_cfg.kp_scale,
        kd_scale=fb_cfg.kd_scale,
        task_kp=fb_cfg.task_kp,
        time_scale=fb_cfg.time_scale,
        use_task_pid=False,
        use_wrist_pid=False,
        name="no_pid",
    )
    joint_pd_only_cfg = TrialConfig(
        kp_scale=fb_cfg.kp_scale,
        kd_scale=fb_cfg.kd_scale,
        task_kp=fbo_cfg.task_kp,
        time_scale=fb_cfg.time_scale,
        use_task_pid=False,
        use_wrist_pid=False,
        name="joint_pd_only",
    )

    print("Recording no-pid config...")
    record_trial_video(no_pid_cfg, os.path.join(OUTPUT_DIR, f"{LOG_PREFIX}_no_pid.gif"), fps=20, playback_speed=1.0)
    print("Recording joint-pd-only config...")
    record_trial_video(joint_pd_only_cfg, os.path.join(OUTPUT_DIR, f"{LOG_PREFIX}_joint_pd_only.gif"), fps=20, playback_speed=1.0)
    print("Recording full-body-only config...")
    record_trial_video(fbo_cfg, os.path.join(OUTPUT_DIR, f"{LOG_PREFIX}_full_body_only.gif"), fps=20, playback_speed=1.0)
    print("Recording full-body (+wrist) config...")
    record_trial_video(fb_cfg, os.path.join(OUTPUT_DIR, f"{LOG_PREFIX}_full_body.gif"), fps=20, playback_speed=1.0)

    # save aggregate
    out_csv = os.path.join(OUTPUT_DIR, "kinematics_sweep_results.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "name", "kp_scale", "kd_scale", "task_kp", "time_scale", "use_task_pid", "use_wrist_pid", "sim_time", "risk_integral", "tilt_integral", "reach_integral", "score"])
        for r in results_fbo:
            writer.writerow(["full_body_only", r["name"], r["kp_scale"], r["kd_scale"], r["task_kp"], r["time_scale"], r["use_task_pid"], r["use_wrist_pid"], f"{r['sim_time']:.4f}", f"{r['risk_integral']:.6f}", f"{r['tilt_integral']:.6f}", f"{r.get('reach_integral', 0.0):.6f}", f"{r['score']:.6f}"])
        for r in results_fb:
            writer.writerow(["full_body", r["name"], r["kp_scale"], r["kd_scale"], r["task_kp"], r["time_scale"], r["use_task_pid"], r["use_wrist_pid"], f"{r['sim_time']:.4f}", f"{r['risk_integral']:.6f}", f"{r['tilt_integral']:.6f}", f"{r.get('reach_integral', 0.0):.6f}", f"{r['score']:.6f}"])

    print("Saved sweep results and recordings in", OUTPUT_DIR)


def build_scene(scene_path: str, out_path: str) -> None:
    """Create a minimal scene file that includes the main scene and adds the tube body."""
    scene_dir = os.path.dirname(os.path.abspath(scene_path))
    panda_path = os.path.join(scene_dir, "panda.xml")
    # read base scene xml and inject include path
    with open(scene_path, "r", encoding="utf-8") as f:
        base = f.read()

    # create runtime panda include that strips keyframe blocks if present
    if os.path.exists(panda_path):
        with open(panda_path, "r", encoding="utf-8") as pf:
            panda_xml = pf.read()
        panda_xml = re.sub(r"\n\s*<keyframe>.*?</keyframe>\s*", "\n", panda_xml, flags=re.S)
        if 'name="gripper"' not in panda_xml:
            panda_xml = panda_xml.replace(
                '<geom mesh="hand_c" class="collision"/>',
                '<geom mesh="hand_c" class="collision"/>\n                      <site name="gripper" pos="0 0 0.1" size="0.004" rgba="1 0 0 0.35"/>'
            )
        runtime_panda = os.path.join(scene_dir, "panda_runtime.xml")
        with open(runtime_panda, "w", encoding="utf-8") as rf:
            rf.write(panda_xml)
            
        # Resolved absolute/relative path mixing compatibility issue
        include_str = f'<include file="{os.path.basename(runtime_panda)}"/>'
        base = base.replace('<include file="panda.xml"/>', include_str)

    tube_block = (
        "    <body name=\"centrifuge_tube\" pos=\"0.6 0.0 0.07\">\n"
        "        <freejoint name=\"tube_joint\"/>\n"
        "        <geom name=\"tube_geom\" type=\"cylinder\" size=\"0.015 0.05\" rgba=\"0.2 0.7 1.0 0.9\" mass=\"0.05\"/>\n"
        "    </body>"
    )
    out_xml = base.replace("</worldbody>", f"{tube_block}\n  </worldbody>")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(out_xml)


def record_trial_video(config: TrialConfig, out_path: str, fps: int = 20, width: int = 640, height: int = 480, playback_speed: float = 1.0) -> None:
    # Run a trial and record frames to a GIF using MuJoCo offscreen rendering.
    if mujoco is None:
        raise RuntimeError("MuJoCo is required to record videos")
    if imageio is None:
        raise RuntimeError("imageio is required for GIF export. Install with: pip install imageio")

    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    hand = body_id(model, ["hand", "panda_hand"])
    gripper_site = site_id(model, ["gripper"])
    data.qpos[:7] = Q_HOME
    mujoco.mj_forward(model, data)

    traj = TrajectorySampler(PHASES, time_scale=config.time_scale)
    traj.set_start(data.site_xpos[gripper_site].copy())

    task_pid = TaskSpacePID(kp=config.task_kp)
    joint_pid = JointPIDController(kp=(JOINT_KP * config.kp_scale), kd=(JOINT_KD * config.kd_scale))
    ee = EndEffectorKinematics(dt)
    monitor = MotionMonitor()

    filtered_acc = np.zeros(3)
    a_liquid = np.array([0.0, 0.0, -9.81], dtype=float)
    gravity = np.array([0.0, 0.0, -9.81], dtype=float)
    t_sim = 0.0

    renderer = mujoco.Renderer(model, height=height, width=width)
    frame_duration = max(1e-4, 1.0 / (fps * max(playback_speed, 1e-9)))
    writer = imageio.get_writer(out_path, mode="I", duration=frame_duration)
    steps_per_frame = max(1, int(1.0 / (fps * dt)))

    step = 0
    while t_sim < traj.total_time:
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, gripper_site, vel6, 0)
        vel_world = vel6[3:6].copy()
        acc_raw, jerk = ee.update(vel_world)
        filtered_acc = ACCEL_LPFILTER_ALPHA * acc_raw + (1 - ACCEL_LPFILTER_ALPHA) * filtered_acc
        a_effective = gravity - filtered_acc
        a_liquid += ((a_effective - a_liquid) / max(LIQUID_TAU, 1e-9)) * dt

        t_sim += dt
        target_xyz, phase, grip = traj.sample(t_sim)

        q_current = data.qpos[:7].copy()
        dq_current = data.qvel[:7].copy()

        q_des, J, Jr, err = solve_cartesian_ik(model, data, gripper_site, target_xyz, Q_HOME)
        corr = task_pid.update(err, dt) if config.use_task_pid else np.zeros(3)
        q_des += J.T @ corr

        if config.use_wrist_pid:
            hand_rot = data.xmat[hand].reshape(3, 3)
            z_actual = normalize(hand_rot @ TUBE_AXIS_LOCAL, np.array([0.0, 0.0, 1.0]))
            z_desired = normalize(a_liquid, np.array([0.0, 0.0, -1.0]))
            rot_axis = np.cross(z_actual, z_desired)
            if np.linalg.norm(rot_axis) > 1e-9:
                rot_axis = rot_axis * float(config.wrist_kp)
                JrJrT = Jr @ Jr.T
                dq_wrist = Jr.T @ np.linalg.solve(JrJrT + 1e-3 * np.eye(3), rot_axis)
                q_des += 1.0 * dq_wrist

        data.ctrl[:7] = joint_pid.update(q_current, q_des, dq_current, dt)
        if model.nu >= 8:
            data.ctrl[7] = grip

        mujoco.mj_step(model, data)

        hand_rot = data.xmat[hand].reshape(3, 3)
        z_axis = normalize(hand_rot @ TUBE_AXIS_LOCAL, np.array([0.0, 0.0, 1.0]))
        a_norm = float(np.linalg.norm(a_liquid))
        if a_norm > 1e-9:
            cosang = float(np.clip(np.dot(z_axis, a_liquid / a_norm), -1.0, 1.0))
            tilt_deg = float(np.degrees(np.arccos(cosang)))
        else:
            tilt_deg = 0.0
        monitor.update(t_sim, filtered_acc, jerk, data.xmat[hand].copy(), tilt_deg, dt)

        if step % steps_per_frame == 0:
            renderer.update_scene(data)
            frame = renderer.render()
            writer.append_data(frame)

        step += 1

    writer.close()
    try:
        renderer.close()
    except Exception:
        pass

    print(f"Saved recording -> {out_path}")


def main_interactive(speed: float = 1.0) -> None:
    if mujoco is None:
        raise RuntimeError("MuJoCo not available in this environment")

    # single interactive run using viewer
    cfg = TrialConfig(kp_scale=1.0, kd_scale=1.0, task_kp=TASK_KP, time_scale=speed, use_task_pid=True, use_wrist_pid=False, name="interactive")
    run_trial(cfg, use_viewer=True, save_prefix=None)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="store_true", help="Run grid sweep headless")
    p.add_argument("--speed", type=float, default=1.0, help=">1.0 = faster maneuvers")
    p.add_argument("--wtime", type=float, default=0.4, help="secondary weight for completion time")
    p.add_argument("--wrisk", type=float, default=0.3, help="secondary weight for disturbance/jerk risk")
    p.add_argument("--wtilt", type=float, default=0.3, help="secondary weight for liquid tilt integral")
    p.add_argument("--record-gifs", action="store_true", help="After sweeps, record 4 GIFs: no_pid, full_body_only, full_body, wrist_only")
    args = p.parse_args()

    if args.sweep and args.record_gifs:
        sweep_and_record(w_time=args.wtime, w_risk=args.wrisk, w_tilt=args.wtilt)
    elif args.sweep:
        run_sweep(w_time=args.wtime, w_risk=args.wrisk, w_tilt=args.wtilt)
    else:
        main_interactive(speed=args.speed)