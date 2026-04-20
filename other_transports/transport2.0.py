import mujoco
import mujoco.viewer
import numpy as np
import csv
import time
from dataclasses import dataclass
from typing import List


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Updated to look for the file in the current working directory
XML_PATH = "franka_emika_panda/scene.xml"   
LOG_PATH = "mixing_log.csv"

# Trajectory
MANEUVER_TIME   = 1.5046161298138303           
SETTLE_TIME     = 1.0           

START_JOINTS = np.array([ 0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8])
GOAL_JOINTS  = np.array([ 0.5,  0.2, -0.3, -1.5,  0.2,  1.8,  1.2])

KP = np.array([6773.495133002521,
    3457.6958054342663,
    374.38209696982545,
    2385.932929400732,
    125.49682972371897,
    2735.795601515945,
    505.14023683716425], dtype=float)
KI = np.array([9.805291335218723,
    0.3304400492716991,
    3.543002525533466,
    9.999548895605365,
    4.872979520359307,
    9.797508981296728,
    5.37048785882323], dtype=float)
KD = np.array([711.5548748579225,
    653.797154334228,
    417.99773923471173,
    71.71656053528494,
    345.15702565185484,
    119.21777304118488,
    41.163197333547465], dtype=float)


TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

W_LATERAL_ACC = 0.1
W_JERK        = 0.2            

TUBE_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

PRINT_EVERY = 200


# ─────────────────────────────────────────────
# TRAJECTORY: MINIMUM JERK
# ─────────────────────────────────────────────

def min_jerk(t: float, T: float, q0: np.ndarray, qf: np.ndarray) -> np.ndarray:
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
    def __init__(self, kp: np.ndarray, ki: np.ndarray, kd: np.ndarray):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral   = np.zeros(7)
        self.integral_clamp = 5.0       

    def reset(self):
        self.integral[:] = 0.0

    def compute(self, q_desired: np.ndarray, q_actual: np.ndarray, qd_actual: np.ndarray, dt: float) -> np.ndarray:
        error = q_desired - q_actual
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)

        torque = (self.kp * error
                + self.ki * self.integral
                - self.kd * qd_actual)      

        return np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)


# ─────────────────────────────────────────────
# END-EFFECTOR KINEMATICS
# ─────────────────────────────────────────────

class EEKinematics:
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
    R = xmat.reshape(3, 3)
    return R.T @ vec_world


def decompose_acceleration(acc_local: np.ndarray):
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
    def __init__(self):
        self.risk_integral = 0.0
        self.log: List[StepLog] = []

    def update(self, t: float, acc_local: np.ndarray, jerk_world: np.ndarray, torques: np.ndarray, dt: float) -> float:
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
    print(f"[INIT] Loading model: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    print(f"[INIT] dt={dt:.4f}s | Maneuver time={MANEUVER_TIME}s")

    # ── Get body IDs via official API ────────────
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id == -1:
        # Fallback if "hand" doesn't exist
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")
    print(f"[INIT] End-effector body id: {hand_id}")

    data.qpos[:7] = START_JOINTS
    mujoco.mj_forward(model, data)

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

            q_des = min_jerk(t, MANEUVER_TIME, START_JOINTS, GOAL_JOINTS)

            q_actual  = data.qpos[:7].copy()
            qd_actual = data.qvel[:7].copy()
            torques   = pid.compute(q_des, q_actual, qd_actual, dt)
            data.ctrl[:7] = torques

            mujoco.mj_step(model, data)

            # ── Modern API Kinematics ────────────
            # cvel holds spatial velocity [angular, linear] of the body's COM in the world frame
            vel_world = data.cvel[hand_id][3:6].copy()   
            acc_world, jerk_world = ee_kin.update(vel_world)

            # Modern API xmat property
            xmat      = data.xmat[hand_id].copy()
            acc_local = world_to_tube_frame(acc_world, xmat)

            risk_rate = monitor.update(t, acc_local, jerk_world, torques, dt)

            if step % PRINT_EVERY == 0:
                joint_err = float(np.linalg.norm(q_des - q_actual))
                print(
                    f"t={t:6.3f}s | "
                    f"joint_err={joint_err:.4f}rad | "
                    f"risk_rate={risk_rate:.4f} | "
                    f"risk_int={monitor.risk_integral:.4f}"
                )

            t    += dt
            step += 1

            if t >= MANEUVER_TIME + SETTLE_TIME:
                done = True

            viewer.sync()
            
            # Sync to real-time so you can watch the robot
            time.sleep(dt) 

    wall_elapsed = time.time() - wall_start
    print(f"[DONE] Wall time: {wall_elapsed:.1f}s | Sim time: {t:.2f}s")

    monitor.summary()
    monitor.save_csv(LOG_PATH)


if __name__ == "__main__":
    main()