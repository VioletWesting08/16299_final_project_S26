"""
Franka Panda - Centrifuge Tube Transport (MJX backend)
16299 Project

Phases:
  0  GRASP   (0.5 s) — hold at start pose, close gripper around tube
  1  MOVE    (3.0 s) — min-jerk transport START → GOAL
  2  SETTLE  (0.5 s) — hold at goal

Physics stepped on JAX (mujoco.mjx). Viewer synced from CPU MjData.
Mixing risk = ∫ (W_lat·|acc_lat| + W_jerk·|jerk|) dt, logged to CSV.

Usage:  mjpython franka_tube_transport.py
"""

import mujoco
import mujoco.viewer
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp
import numpy as np
import csv
import time
from dataclasses import dataclass
from typing import List


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

XML_PATH = "franka_emika_panda/mjx_scene.xml"
LOG_PATH  = "mixing_log.csv"

# Timing
GRASP_TIME    = 0.5     # s — gripper closes over this window
MANEUVER_TIME = 3.0     # s — arm travels START → GOAL
SETTLE_TIME   = 0.5     # s — hold at goal

# Joint configs (radians) — 7 arm joints
START_JOINTS = np.array([ 0.0, -0.5,  0.0, -2.0,  0.0,        1.5,  0.8])
GOAL_JOINTS  = np.array([ 0.5,  0.2, -0.3, -1.5,  np.pi/2,   1.8,  1.2])

# Finger ctrl range: 0 = fully closed, 0.04 = fully open
FINGER_OPEN   = 0.035
FINGER_CLOSED = 0.0

# Mixing risk weights
W_LATERAL_ACC = 1.0
W_JERK        = 2.0

TUBE_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])   # tube axis in gripper Z

# EMA smoothing for velocity before finite-diff jerk (0 = no filter)
VEL_EMA_ALPHA = 0.85

PRINT_EVERY = 100


# ─────────────────────────────────────────────
# TRAJECTORY
# ─────────────────────────────────────────────

def min_jerk(t: float, T: float, q0: np.ndarray, qf: np.ndarray) -> np.ndarray:
    if t <= 0: return q0.copy()
    if t >= T: return qf.copy()
    tau = t / T
    s = 10*tau**3 - 15*tau**4 + 6*tau**5
    return q0 + s*(qf - q0)


# ─────────────────────────────────────────────
# EE KINEMATICS
# ─────────────────────────────────────────────

class EEKinematics:
    def __init__(self, dt: float, alpha: float = 0.85):
        self.dt, self.alpha = dt, alpha
        self.vel_ema  = np.zeros(3)
        self.prev_vel = np.zeros(3)
        self.prev_acc = np.zeros(3)

    def update(self, vel_raw: np.ndarray):
        self.vel_ema = self.alpha*self.vel_ema + (1-self.alpha)*vel_raw
        acc  = (self.vel_ema - self.prev_vel) / self.dt
        jerk = (acc - self.prev_acc) / self.dt
        self.prev_vel = self.vel_ema.copy()
        self.prev_acc = acc.copy()
        return acc, jerk


def decompose_acceleration(acc_local: np.ndarray):
    axial   = np.dot(acc_local, TUBE_AXIS_LOCAL) * TUBE_AXIS_LOCAL
    lateral = acc_local - axial
    return axial, lateral


# ─────────────────────────────────────────────
# MIXING MONITOR
# ─────────────────────────────────────────────

@dataclass
class StepLog:
    t: float; lateral_acc_mag: float; jerk_mag: float; risk: float

class MixingMonitor:
    def __init__(self):
        self.risk_integral = 0.0
        self.log: List[StepLog] = []

    def update(self, t, acc_local, jerk_world, dt) -> float:
        _, lat = decompose_acceleration(acc_local)
        lat_mag  = float(np.linalg.norm(lat))
        jerk_mag = float(np.linalg.norm(jerk_world))
        rate = W_LATERAL_ACC*lat_mag + W_JERK*jerk_mag
        self.risk_integral += rate * dt
        self.log.append(StepLog(t=t, lateral_acc_mag=lat_mag, jerk_mag=jerk_mag, risk=rate))
        return rate

    def save_csv(self, path):
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t","lateral_acc_mag","jerk_mag","risk_rate"])
            for e in self.log:
                w.writerow([f"{e.t:.4f}",f"{e.lateral_acc_mag:.6f}",
                             f"{e.jerk_mag:.6f}",f"{e.risk:.6f}"])
        print(f"[LOG] {len(self.log)} rows → {path}")

    def summary(self):
        if not self.log: return
        lats  = [e.lateral_acc_mag for e in self.log]
        jerks = [e.jerk_mag        for e in self.log]
        print("\n" + "="*52)
        print("  MIXING RISK SUMMARY")
        print("="*52)
        print(f"  Total risk integral : {self.risk_integral:.4f}")
        print(f"  Peak lateral acc    : {max(lats):.4f} m/s²")
        print(f"  Mean lateral acc    : {np.mean(lats):.4f} m/s²")
        print(f"  Peak jerk           : {max(jerks):.4f} m/s³")
        print(f"  Mean jerk           : {np.mean(jerks):.4f} m/s³")
        print("="*52 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def place_tube_at_gripper(model: mujoco.MjModel, data: mujoco.MjData):
    """Set tube free-joint qpos so it sits exactly at the weld target (hand + 0.1 m in hand Z)."""
    hand_id    = model.body("hand").id
    hand_pos   = data.xpos[hand_id].copy()
    hand_rot   = data.xmat[hand_id].reshape(3, 3)
    tube_pos   = hand_pos + hand_rot @ np.array([0.0, 0.0, 0.10])

    tube_quat  = np.zeros(4)
    mujoco.mju_mat2Quat(tube_quat, data.xmat[hand_id])   # match hand orientation

    jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tube_free")
    adr    = model.jnt_qposadr[jnt_id]
    data.qpos[adr:adr+3] = tube_pos
    data.qpos[adr+3:adr+7] = tube_quat


def main():
    print(f"[INIT] Loading: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    print(f"[INIT] dt={dt:.4f}s | JAX backend: {jax.default_backend()}")

    hand_id = model.body("hand").id
    print(f"[INIT] hand body id={hand_id}  actuators={model.nu}")

    # ── Place robot + tube at start ──────────────
    data.qpos[:7] = START_JOINTS
    data.ctrl[:7] = START_JOINTS          # pre-load position targets
    data.ctrl[7]  = FINGER_OPEN           # gripper open before grasp
    mujoco.mj_forward(model, data)
    place_tube_at_gripper(model, data)
    mujoco.mj_forward(model, data)        # recompute with tube in place

    # ── MJX setup ───────────────────────────────
    mjx_model = mjx.put_model(model)
    mjx_data  = mjx.put_data(model, data)
    jit_step  = jax.jit(mjx.step)

    # Warm up JIT
    print("[INIT] Warming up JIT...")
    _ = jit_step(mjx_model, mjx_data)
    print("[INIT] JIT ready.")

    ee_kin  = EEKinematics(dt, alpha=VEL_EMA_ALPHA)
    monitor = MixingMonitor()

    TOTAL_TIME = GRASP_TIME + MANEUVER_TIME + SETTLE_TIME
    t, step, done = 0.0, 0, False

    print(f"[RUN] grasp={GRASP_TIME}s  move={MANEUVER_TIME}s  settle={SETTLE_TIME}s")
    wall_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not done:

            # ── Phase-based ctrl ─────────────────
            ctrl = np.zeros(model.nu)

            if t < GRASP_TIME:
                # Phase 0: hold pose, close gripper linearly
                ctrl[:7]  = START_JOINTS
                alpha      = t / GRASP_TIME
                ctrl[7]    = FINGER_OPEN * (1.0 - alpha) + FINGER_CLOSED * alpha

            else:
                # Phase 1+2: transport (min-jerk) then settle
                t_move    = t - GRASP_TIME
                ctrl[:7]  = min_jerk(t_move, MANEUVER_TIME, START_JOINTS, GOAL_JOINTS)
                ctrl[7]   = FINGER_CLOSED

            # ── MJX step ─────────────────────────
            mjx_data = mjx_data.replace(ctrl=jnp.array(ctrl))
            mjx_data = jit_step(mjx_model, mjx_data)

            # Copy state back into viewer-bound data in-place
            mjx.get_data_into(data, model, mjx_data)
            mujoco.mj_forward(model, data)

            # ── EE velocity → acc → jerk ─────────
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY,
                                     hand_id, vel6, 0)
            acc_world, jerk_world = ee_kin.update(vel6[3:])
            acc_local = data.xmat[hand_id].reshape(3,3).T @ acc_world

            # ── Mixing risk (transport phase only) ─
            if t >= GRASP_TIME:
                risk_rate = monitor.update(t - GRASP_TIME, acc_local, jerk_world, dt)
            else:
                risk_rate = 0.0

            # ── Console ──────────────────────────
            if step % PRINT_EVERY == 0:
                q_actual  = np.array(data.qpos[:7])
                joint_err = float(np.linalg.norm(ctrl[:7] - q_actual))
                phase     = "GRASP" if t < GRASP_TIME else "MOVE"
                print(f"[{phase}] t={t:5.2f}s | err={joint_err:.4f}rad | "
                      f"risk={risk_rate:.2f} | ∫risk={monitor.risk_integral:.2f}")

            viewer.sync()
            t    += dt
            step += 1
            if t >= TOTAL_TIME:
                done = True

    print(f"[DONE] wall={time.time()-wall_start:.1f}s  sim={t:.2f}s")
    monitor.summary()
    monitor.save_csv(LOG_PATH)


if __name__ == "__main__":
    main()
