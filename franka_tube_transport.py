"""
Franka Panda - Centrifuge Tube Transport
16299 Project

Simulates a Franka Panda arm transporting a centrifuge tube while
monitoring end-effector jerk and lateral acceleration as a proxy
for liquid mixing risk.

Dependencies:
    pip install mujoco numpy

Usage:
    python franka_tube_transport.py

    The viewer will open. The arm will execute a min-jerk trajectory
    from start to goal joints. Metrics are logged to:
        - console (live)
        - mixing_log.csv (post-run)
"""

import mujoco
import mujoco.viewer
import numpy as np
import csv
import time
from dataclasses import dataclass, field
from typing import List


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

XML_PATH = "franka_panda.xml"   # path to your Franka MuJoCo XML
LOG_PATH = "mixing_log.csv"

# Trajectory
MANEUVER_TIME   = 4.0           # seconds for the full move
SETTLE_TIME     = 1.0           # seconds to hold at goal before logging ends

# Start and goal joint configurations (radians)
# These are example values — adjust to your scene
START_JOINTS = np.array([ 0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8])
GOAL_JOINTS  = np.array([ 0.5,  0.2, -0.3, -1.5,  0.2,  1.8,  1.2])

# PID gains (per joint, 7 joints)
# Roughly based on real Franka defaults — tune KD first for this project
KP = np.array([4500, 4500, 3500, 3500, 2000, 2000,  500], dtype=float)
KI = np.array([   1,    1,    1,    1,    1,    1,    1], dtype=float)
KD = np.array([ 450,  450,  350,  350,  200,  200,   50], dtype=float)

# Franka torque limits (Nm) per joint
TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

# Mixing risk weights
W_LATERAL_ACC = 1.0
W_JERK        = 2.0            # jerk weighted higher — sudden changes mix more

# Tube axis in gripper local frame (assumes tube is along gripper Z)
TUBE_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

# Print metrics every N steps
PRINT_EVERY = 200


# ─────────────────────────────────────────────
# TRAJECTORY: MINIMUM JERK
# ─────────────────────────────────────────────

def min_jerk(t: float, T: float, q0: np.ndarray, qf: np.ndarray) -> np.ndarray:
    """
    Minimum-jerk trajectory between q0 and qf over duration T.
    Returns desired joint positions at time t.
    Clamps to endpoints outside [0, T].
    """
    if t <= 0:
        return q0.copy()
    if t >= T:
        return qf.copy()
    tau = t / T
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    return q0 + s * (qf - q0)


# ─────────────────────────────────────────────
# PID CONTROLLER
# ─────────────────────────────────────────────

class FrankaPID:
    """
    Manual torque-level PID controller for Franka's 7 joints.
    D term uses measured joint velocity (data.qvel) for stability.
    """

    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral   = np.zeros(7)
        self.integral_clamp = 5.0       # anti-windup clamp (Nm·s)

    def reset(self):
        self.integral[:] = 0.0

    def compute(
        self,
        q_desired:  np.ndarray,
        q_actual:   np.ndarray,
        qd_actual:  np.ndarray,
        dt:         float
    ) -> np.ndarray:
        """
        Returns torque commands for all 7 joints.
        qd_actual: measured joint velocities (data.qvel[:7])
        """
        error = q_desired - q_actual
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)

        torque = (self.kp * error
                + self.ki * self.integral
                - self.kd * qd_actual)      # minus: velocity damping

        return np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)


# ─────────────────────────────────────────────
# END-EFFECTOR KINEMATICS
# ─────────────────────────────────────────────

class EEKinematics:
    """
    Computes velocity, acceleration, and jerk of the end-effector
    via finite differencing of body_xvelp (linear velocity in world frame).
    """

    def __init__(self, dt: float):
        self.dt       = dt
        self.prev_vel = np.zeros(3)
        self.prev_acc = np.zeros(3)

    def update(self, vel_world: np.ndarray):
        acc  = (vel_world - self.prev_vel) / self.dt
        jerk = (acc - self.prev_acc)       / self.dt
        self.prev_vel = vel_world.copy()
        self.prev_acc = acc.copy()
        return acc, jerk


def world_to_tube_frame(vec_world: np.ndarray, xmat: np.ndarray) -> np.ndarray:
    """
    Project a world-frame vector into the tube's local frame.
    xmat: 3x3 rotation matrix of the gripper body (data.body_xmat[id].reshape(3,3))
    """
    R = xmat.reshape(3, 3)
    return R.T @ vec_world


def decompose_acceleration(acc_local: np.ndarray):
    """
    Decompose local-frame acceleration into axial (along tube) and lateral components.
    Returns (acc_axial, acc_lateral) both as 3-vectors.
    """
    axial_proj = np.dot(acc_local, TUBE_AXIS_LOCAL)
    acc_axial   = axial_proj * TUBE_AXIS_LOCAL
    acc_lateral = acc_local - acc_axial
    return acc_axial, acc_lateral


# ─────────────────────────────────────────────
# MIXING RISK MONITOR
# ─────────────────────────────────────────────

@dataclass
class StepLog:
    t:               float
    lateral_acc_mag: float
    jerk_mag:        float
    risk:            float
    torque_norm:     float

class MixingMonitor:
    """
    Accumulates a scalar mixing risk integral over the trajectory.

    risk_rate = W_LATERAL_ACC * |acc_lateral| + W_JERK * |jerk|
    risk_integral = ∫ risk_rate dt
    """

    def __init__(self):
        self.risk_integral = 0.0
        self.log: List[StepLog] = []

    def update(
        self,
        t:           float,
        acc_local:   np.ndarray,
        jerk_world:  np.ndarray,
        torques:     np.ndarray,
        dt:          float
    ) -> float:
        _, acc_lateral = decompose_acceleration(acc_local)
        lat_mag  = float(np.linalg.norm(acc_lateral))
        jerk_mag = float(np.linalg.norm(jerk_world))

        risk_rate = W_LATERAL_ACC * lat_mag + W_JERK * jerk_mag
        self.risk_integral += risk_rate * dt

        entry = StepLog(
            t=t,
            lateral_acc_mag=lat_mag,
            jerk_mag=jerk_mag,
            risk=risk_rate,
            torque_norm=float(np.linalg.norm(torques))
        )
        self.log.append(entry)
        return risk_rate

    def save_csv(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t", "lateral_acc_mag", "jerk_mag", "risk_rate", "torque_norm"])
            for e in self.log:
                writer.writerow([
                    f"{e.t:.4f}",
                    f"{e.lateral_acc_mag:.6f}",
                    f"{e.jerk_mag:.6f}",
                    f"{e.risk:.6f}",
                    f"{e.torque_norm:.4f}"
                ])
        print(f"[LOG] Saved {len(self.log)} rows → {path}")

    def summary(self):
        if not self.log:
            return
        lat_vals  = [e.lateral_acc_mag for e in self.log]
        jerk_vals = [e.jerk_mag        for e in self.log]
        print("\n" + "="*52)
        print("  MIXING RISK SUMMARY")
        print("="*52)
        print(f"  Total risk integral : {self.risk_integral:.4f}")
        print(f"  Peak lateral acc    : {max(lat_vals):.4f} m/s²")
        print(f"  Mean lateral acc    : {np.mean(lat_vals):.4f} m/s²")
        print(f"  Peak jerk           : {max(jerk_vals):.4f} m/s³")
        print(f"  Mean jerk           : {np.mean(jerk_vals):.4f} m/s³")
        print(f"  Steps logged        : {len(self.log)}")
        print("="*52 + "\n")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Load model ──────────────────────────────
    print(f"[INIT] Loading model: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    print(f"[INIT] dt={dt:.4f}s | Maneuver time={MANEUVER_TIME}s")

    # ── Get body IDs ────────────────────────────
    try:
        hand_id = model.body("hand").id
    except KeyError:
        # fallback — some Franka XMLs use different names
        hand_id = model.body("panda_hand").id
    print(f"[INIT] End-effector body id: {hand_id}")

    # ── Reset to start joints ────────────────────
    data.qpos[:7] = START_JOINTS
    mujoco.mj_forward(model, data)

    # ── Init subsystems ──────────────────────────
    pid     = FrankaPID(KP, KI, KD)
    ee_kin  = EEKinematics(dt)
    monitor = MixingMonitor()

    t       = 0.0
    step    = 0
    done    = False

    print("[RUN] Starting simulation...")
    wall_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and not done:

            # ── Trajectory ───────────────────────
            q_des = min_jerk(t, MANEUVER_TIME, START_JOINTS, GOAL_JOINTS)

            # ── PID torque ───────────────────────
            q_actual  = data.qpos[:7].copy()
            qd_actual = data.qvel[:7].copy()
            torques   = pid.compute(q_des, q_actual, qd_actual, dt)
            data.ctrl[:7] = torques

            # ── Step sim ─────────────────────────
            mujoco.mj_step(model, data)

            # ── EE kinematics ────────────────────
            vel_world = data.body_xvelp[hand_id].copy()   # linear vel, world frame
            acc_world, jerk_world = ee_kin.update(vel_world)

            xmat      = data.body_xmat[hand_id].copy()
            acc_local = world_to_tube_frame(acc_world, xmat)

            # ── Mixing monitor ───────────────────
            risk_rate = monitor.update(t, acc_local, jerk_world, torques, dt)

            # ── Console logging ──────────────────
            if step % PRINT_EVERY == 0:
                joint_err = float(np.linalg.norm(q_des - q_actual))
                print(
                    f"t={t:6.3f}s | "
                    f"joint_err={joint_err:.4f}rad | "
                    f"risk_rate={risk_rate:.4f} | "
                    f"risk_int={monitor.risk_integral:.4f}"
                )

            # ── Advance time ─────────────────────
            t    += dt
            step += 1

            # ── Done condition ───────────────────
            if t >= MANEUVER_TIME + SETTLE_TIME:
                done = True

            viewer.sync()

    wall_elapsed = time.time() - wall_start
    print(f"[DONE] Wall time: {wall_elapsed:.1f}s | Sim time: {t:.2f}s")

    # ── Post-run ─────────────────────────────────
    monitor.summary()
    monitor.save_csv(LOG_PATH)


if __name__ == "__main__":
    main()
