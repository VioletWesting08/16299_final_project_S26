"""
Franka Panda - Centrifuge Tube Transport
16299 Project

Full pick-and-place with:
  - Task-space control via Jacobian pseudoinverse (IK)
  - Inner loop: manual PID → joint torques (smooth tracking)
  - Outer loop: jerk PID → trajectory speed scale (mixing protection)
  - Live tube orientation feedback (xmat) for axial/lateral decomposition
  - Mixing risk monitor → mixing_log.csv

Dependencies:
    pip install mujoco numpy

Usage:
    mjpython franka_tube_transport.py
"""

import mujoco
import mujoco.viewer
import numpy as np
import csv
import time
import os
from dataclasses import dataclass
from typing import List, Optional


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

SCENE_XML = "franka_emika_panda/scene.xml"   # original Franka scene
ENV_XML   = "franka_emika_panda/pick_and_place_scene.xml"
LOG_PATH  = "mixing_log.csv"

# ── Tube axis in gripper LOCAL frame ────────────────────────────
# [0,0,1] means tube points along gripper Z. Adjust if needed.
TUBE_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

# ── Finger tip offset from hand body origin (LOCAL frame) ──────────
# Adjust Z value by printing data.xpos for finger bodies to measure exactly.
# Typically for Franka Panda: 0.1 to 0.12m along local Z axis
FINGER_OFFSET_LOCAL = np.array([0.0, 0.0, 0.105])  # ~10.5cm along gripper Z

# ── Inner PID (joint torque control) ────────────────────────────
# D term is most important — higher KD = smoother = less jerk
KP = np.array([100, 100, 100, 100, 50,  50,  20 ], dtype=float)
KI = np.array([  0,   0,   0,   0,  0,   0,   0 ], dtype=float)  # kill I for now
KD = np.array([ 20,  20,  20,  20, 10,  10,   5 ], dtype=float)
TORQUE_LIMITS = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

# ── Outer PID (jerk → trajectory speed) ─────────────────────────
JERK_THRESHOLD = 50.0     # m/s³ — tune from open-loop CSV first
JERK_KP        = 0.002
JERK_KI        = 0.0001
JERK_KD        = 0.005
T_SCALE_MIN    = 0.1     # never slower than 10% speed
T_SCALE_MAX    = 1.0     # never faster than planned

# ── Mixing risk weights ──────────────────────────────────────────
W_LATERAL_ACC = 1.0
W_JERK        = 2.0      # lateral jerk weighted higher

# ── Console print interval ───────────────────────────────────────
PRINT_EVERY = 200

# ── State machine phases ─────────────────────────────────────────
# target_xyz: world-frame EE position to move toward
# gripper:    finger position in meters (0.04=open, 0.00=closed)
# duration:   seconds to spend in this phase (can stretch under jerk PID)
PHASES = [
    {"name": "1. Hover above tube",  "target_xyz": [0.6, 0.0, 0.30], "gripper": 0.04, "duration": 3.0},
    {"name": "2. Descend to tube",   "target_xyz": [0.6, 0.0, 0.10], "gripper": 0.04, "duration": 2.0},
    {"name": "3. Grasp tube",        "target_xyz": [0.6, 0.0, 0.10], "gripper": 0.00, "duration": 1.5},
    {"name": "4. Lift tube",         "target_xyz": [0.6, 0.0, 0.40], "gripper": 0.00, "duration": 2.0},
    {"name": "5. Transport to goal", "target_xyz": [0.6, 0.4, 0.40], "gripper": 0.00, "duration": 4.0},
    {"name": "6. Place tube",        "target_xyz": [0.6, 0.4, 0.10], "gripper": 0.00, "duration": 2.0},
    {"name": "7. Release tube",      "target_xyz": [0.6, 0.4, 0.10], "gripper": 0.04, "duration": 1.0},
]


# ═══════════════════════════════════════════════════════════════
# SCENE SETUP — inject tube into existing scene XML
# ═══════════════════════════════════════════════════════════════

def build_scene_xml(scene_path: str, out_path: str):
    """
    Wraps the existing Franka scene XML and adds a centrifuge tube
    as a free body. Writes to out_path so MuJoCo can resolve
    the <include> relative path correctly.
    """
    xml = f"""<mujoco>
    <include file="{os.path.abspath(scene_path)}"/>
    <worldbody>
        <body name="centrifuge_tube" pos="0.6 0.0 0.07">
            <freejoint name="tube_joint"/>
            <geom name="tube_geom"
                  type="cylinder"
                  size="0.015 0.05"
                  rgba="0.2 0.7 1.0 0.9"
                  mass="0.05"
                  condim="4"
                  friction="1.0 0.05 0.001"/>
        </body>
    </worldbody>
</mujoco>"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"[SCENE] Written → {out_path}")


# ═══════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════

class StateMachine:
    """
    Advances through PHASES based on effective simulation time t_eff.
    t_eff can slow down under high jerk (controlled by JerkPID),
    so phases automatically stretch when the arm needs to move carefully.
    """
    def __init__(self, phases: list):
        self.phases     = phases
        self.boundaries = []
        cumulative      = 0.0
        for p in phases:
            cumulative += p["duration"]
            self.boundaries.append(cumulative)
        self.total_time = cumulative
        self._last_name = ""

    def get_phase(self, t_eff: float) -> dict:
        for phase, boundary in zip(self.phases, self.boundaries):
            if t_eff < boundary:
                return phase
        return self.phases[-1]

    def log_transition(self, phase: dict):
        if phase["name"] != self._last_name:
            self._last_name = phase["name"]
            print(f"[PHASE] {phase['name']}")


# ═══════════════════════════════════════════════════════════════
# INNER LOOP: JOINT PID → TORQUES
# ═══════════════════════════════════════════════════════════════

class FrankaPID:
    """
    Torque-level PID for all 7 Franka joints.

    This is the INNER feedback loop:
      error     = desired_joint_pos - actual_joint_pos
      output    = torque to apply to each joint

    D term uses measured joint velocity (data.qvel) not finite diff.
    This is more stable and is the standard approach for robot control.
    High KD is critical here — it damps joint overshoot which is the
    primary source of end-effector jerk in this system.
    """
    def __init__(self, kp, ki, kd):
        self.kp             = kp
        self.ki             = ki
        self.kd             = kd
        self.integral       = np.zeros(7)
        self.integral_clamp = 5.0           # anti-windup

    def reset(self):
        self.integral[:] = 0.0

    def compute(self, q_des, q_actual, qd_actual, dt) -> np.ndarray:
        error          = q_des - q_actual
        self.integral += error * dt
        self.integral  = np.clip(self.integral, -self.integral_clamp, self.integral_clamp)

        torque = (self.kp * error
                + self.ki * self.integral
                - self.kd * qd_actual)      # D opposes velocity → damps overshoot

        return np.clip(torque, -TORQUE_LIMITS, TORQUE_LIMITS)


# ═══════════════════════════════════════════════════════════════
# OUTER LOOP: JERK PID → TRAJECTORY SPEED SCALE
# ═══════════════════════════════════════════════════════════════

class JerkPID:
    """
    This is the OUTER feedback loop:
      setpoint  = JERK_THRESHOLD (desired max lateral jerk)
      measured  = lateral_jerk_mag from MixingMonitor
      error     = max(0, measured - threshold)
      output    = t_scale: how fast to advance through phases

    When lateral jerk exceeds threshold:
      → error grows → slowdown increases → t_eff advances slower
      → arm moves more slowly → jerk drops → t_scale recovers

    This is cascade control: outer loop shapes WHEN to move,
    inner loop (FrankaPID) controls HOW to move there.

    Analogous to a balancing robot:
      balancing robot:  angle error    → PID → motor torque
      this outer loop:  jerk error     → PID → trajectory speed
    """
    def __init__(self, kp=JERK_KP, ki=JERK_KI, kd=JERK_KD):
        self.kp         = kp
        self.ki         = ki
        self.kd         = kd
        self.integral   = 0.0
        self.prev_error = 0.0
        self.t_scale    = 1.0

    def update(self, lateral_jerk_mag: float, threshold: float, dt: float) -> float:
        error           = max(0.0, lateral_jerk_mag - threshold)
        self.integral  += error * dt
        derivative      = (error - self.prev_error) / dt
        self.prev_error = error

        slowdown     = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.t_scale = np.clip(self.t_scale - slowdown, T_SCALE_MIN, T_SCALE_MAX)

        # gradually recover speed when jerk is within threshold
        if error == 0.0:
            self.t_scale = min(self.t_scale + 0.001, T_SCALE_MAX)

        return self.t_scale


# ═══════════════════════════════════════════════════════════════
# END-EFFECTOR KINEMATICS
# ═══════════════════════════════════════════════════════════════

class EEKinematics:
    """
    Finite-differences mj_objectVelocity output each timestep
    to get world-frame linear acceleration and jerk at the EE.

    vel  → finite diff → acc
    acc  → finite diff → jerk
    """
    def __init__(self, dt: float):
        self.dt       = dt
        self.prev_vel = np.zeros(3)
        self.prev_acc = np.zeros(3)

    def update(self, vel_linear: np.ndarray):
        acc  = (vel_linear - self.prev_vel) / self.dt
        jerk = (acc - self.prev_acc)        / self.dt
        self.prev_vel = vel_linear.copy()
        self.prev_acc = acc.copy()
        return acc, jerk


# ═══════════════════════════════════════════════════════════════
# MIXING RISK MONITOR
# ═══════════════════════════════════════════════════════════════

@dataclass
class StepLog:
    t:                float
    phase:            str
    lateral_acc_mag:  float
    axial_acc_mag:    float
    jerk_mag:         float
    lateral_jerk_mag: float
    risk:             float
    t_scale:          float
    torque_norm:      float


class MixingMonitor:
    """
    Each timestep:

    1. Gets live tube orientation from xmat (gripper rotation matrix
       from data.xmat[hand_id]). This is the key feedback — we know
       EXACTLY which direction the tube is pointing right now.

    2. Decomposes world-frame acc and jerk into:
         axial:   along the tube   → compresses layers, relatively safe
         lateral: perpendicular    → shears layers, causes mixing

    3. Computes risk_rate = W_LATERAL_ACC * |acc_lat| + W_JERK * |jerk_lat|
       Lateral jerk is weighted higher because sudden shear forces are
       more disruptive than sustained ones.

    4. Integrates risk_rate over time → single scalar for comparison
       across different runs, speeds, or controller tunings.

    The lateral_jerk_mag returned each step feeds directly into
    JerkPID to close the outer control loop.
    """
    def __init__(self):
        self.risk_integral = 0.0
        self.log: List[StepLog] = []

    def update(
        self,
        t:           float,
        phase_name:  str,
        acc_world:   np.ndarray,    # (3,) world-frame linear acceleration
        jerk_world:  np.ndarray,    # (3,) world-frame linear jerk
        xmat:        np.ndarray,    # (9,) flat rotation matrix from data.xmat[hand_id]
        t_scale:     float,
        torques:     np.ndarray,    # (7,) joint torques
        dt:          float,
    ):
        # ── 1. Live tube axis in world frame ────────────────────
        # data.xmat[hand_id] is a 3x3 rotation matrix (flattened to 9).
        # R @ TUBE_AXIS_LOCAL rotates the tube's local-frame axis
        # into world frame. Updates every step as arm moves.
        R               = xmat.reshape(3, 3)
        tube_axis_world = R @ TUBE_AXIS_LOCAL
        tube_axis_world = tube_axis_world / np.linalg.norm(tube_axis_world)

        # ── 2. Decompose acceleration ────────────────────────────
        axial_proj  = np.dot(acc_world, tube_axis_world)
        acc_axial   = axial_proj * tube_axis_world
        acc_lateral = acc_world - acc_axial

        # ── 3. Decompose jerk ────────────────────────────────────
        jerk_axial_proj = np.dot(jerk_world, tube_axis_world)
        jerk_axial      = jerk_axial_proj * tube_axis_world
        jerk_lateral    = jerk_world - jerk_axial

        # ── 4. Magnitudes ────────────────────────────────────────
        lat_mag          = float(np.linalg.norm(acc_lateral))
        axial_mag        = float(np.linalg.norm(acc_axial))
        jerk_mag         = float(np.linalg.norm(jerk_world))
        lateral_jerk_mag = float(np.linalg.norm(jerk_lateral))

        # ── 5. Risk ──────────────────────────────────────────────
        risk_rate          = W_LATERAL_ACC * lat_mag + W_JERK * lateral_jerk_mag
        self.risk_integral += risk_rate * dt

        self.log.append(StepLog(
            t=t,
            phase=phase_name,
            lateral_acc_mag=lat_mag,
            axial_acc_mag=axial_mag,
            jerk_mag=jerk_mag,
            lateral_jerk_mag=lateral_jerk_mag,
            risk=risk_rate,
            t_scale=t_scale,
            torque_norm=float(np.linalg.norm(torques)),
        ))

        return risk_rate, lateral_jerk_mag

    def save_csv(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "t", "phase",
                "lateral_acc_mag", "axial_acc_mag",
                "jerk_mag", "lateral_jerk_mag",
                "risk_rate", "t_scale", "torque_norm"
            ])
            for e in self.log:
                writer.writerow([
                    f"{e.t:.4f}", e.phase,
                    f"{e.lateral_acc_mag:.6f}", f"{e.axial_acc_mag:.6f}",
                    f"{e.jerk_mag:.6f}",        f"{e.lateral_jerk_mag:.6f}",
                    f"{e.risk:.6f}",            f"{e.t_scale:.4f}",
                    f"{e.torque_norm:.4f}",
                ])
        print(f"[LOG] Saved {len(self.log)} rows → {path}")

    def summary(self):
        if not self.log:
            return
        lat   = [e.lateral_acc_mag  for e in self.log]
        axial = [e.axial_acc_mag    for e in self.log]
        jerk  = [e.jerk_mag         for e in self.log]
        ljerk = [e.lateral_jerk_mag for e in self.log]
        print("\n" + "="*54)
        print("  MIXING RISK SUMMARY")
        print("="*54)
        print(f"  Total risk integral   : {self.risk_integral:.4f}")
        print(f"  Peak lateral acc      : {max(lat):.4f}  m/s²")
        print(f"  Mean lateral acc      : {np.mean(lat):.4f}  m/s²")
        print(f"  Peak axial acc        : {max(axial):.4f}  m/s²")
        print(f"  Peak jerk (total)     : {max(jerk):.4f}  m/s³")
        print(f"  Peak lateral jerk     : {max(ljerk):.4f}  m/s³")
        print(f"  Mean lateral jerk     : {np.mean(ljerk):.4f}  m/s³")
        print(f"  Steps logged          : {len(self.log)}")
        print("="*54 + "\n")


# ═══════════════════════════════════════════════════════════════
# END-EFFECTOR POSITION WITH FINGER TIP OFFSET
# ═══════════════════════════════════════════════════════════════

def get_ee_position(data, model, hand_id) -> np.ndarray:
    """
    Compute true end-effector position at fingertip center in world frame.
    
    Instead of using the hand body origin (data.xpos[hand_id]), this accounts
    for the offset from hand body to actual fingertips, giving better IK targets.
    
    Args:
        data: MuJoCo data object
        model: MuJoCo model object
        hand_id: body ID of the hand/gripper
        
    Returns:
        (3,) np.ndarray with EE position in world frame
    """
    R = data.xmat[hand_id].copy().reshape(3, 3)
    return data.xpos[hand_id] + R @ FINGER_OFFSET_LOCAL


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():

    # ── Build scene ─────────────────────────────────────────────
    build_scene_xml(SCENE_XML, ENV_XML)
    print(f"[INIT] Loading model: {ENV_XML}")
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep
    print(f"[INIT] dt={dt:.4f}s | nq={model.nq} | nu={model.nu}")

    # ── Body IDs ─────────────────────────────────────────────────
    hand_id: Optional[int] = None
    for name in ["hand", "panda_hand"]:
        try:
            hand_id = model.body(name).id
            print(f"[INIT] EE body: '{name}' id={hand_id}")
            break
        except KeyError:
            continue
    if hand_id is None:
        raise RuntimeError("Could not find hand/panda_hand body. "
                           "Check printed body list below and update the name.")

    print("[INIT] All bodies in model:")
    for i in range(model.nbody):
        print(f"       {i}: {model.body(i).name}")

    # ── Init subsystems ──────────────────────────────────────────
    sm        = StateMachine(PHASES)
    pid       = FrankaPID(KP, KI, KD)
    jerk_pid  = JerkPID()
    ee_kin    = EEKinematics(dt)
    monitor   = MixingMonitor()

    # ── Sim state ────────────────────────────────────────────────
    sim_t            = 0.0      # real sim time — always advances by dt
    t_eff            = 0.0      # effective phase time — slows under high jerk
    step             = 0
    t_scale          = 1.0
    lateral_jerk_mag = 0.0      # will be updated by monitor, read by jerk_pid

    mujoco.mj_forward(model, data)
    
    # ── DEBUG: Print body positions to measure finger offset ──────
    # Uncomment to see positions at startup. Hand body position + offset
    # should equal fingertip position for accurate IK targeting.
    print("[DEBUG] Body positions at startup:")
    for i in range(model.nbody):
        name = model.body(i).name
        if "finger" in name.lower() or "hand" in name.lower() or "link8" in name.lower():
            print(f"  {name}: {data.xpos[i]}")
    print("[DEBUG] Hand body (for offset reference):")
    print(f"  panda_hand: {data.xpos[hand_id]}")
    
    print("[RUN] Starting — run with: mjpython franka_tube_transport.py")
    wall_start = time.time()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            # ════════════════════════════════════════════════
            # A. OUTER LOOP — jerk PID → trajectory speed
            #
            # lateral_jerk_mag from LAST step feeds in here.
            # Output: t_scale in [T_SCALE_MIN, T_SCALE_MAX]
            # Skip first 3 steps — finite diff is noisy at startup.
            # ════════════════════════════════════════════════
            if step > 3:
                t_scale = jerk_pid.update(lateral_jerk_mag, JERK_THRESHOLD, dt)
            else:
                t_scale = 1.0

            t_eff += dt * t_scale       # effective time advances at scaled rate

            # ════════════════════════════════════════════════
            # B. STATE MACHINE — which phase are we in
            #
            # Uses t_eff not sim_t — so phases stretch automatically
            # when jerk PID slows things down.
            # ════════════════════════════════════════════════
            phase = sm.get_phase(t_eff)
            sm.log_transition(phase)

            target_xyz   = np.array(phase["target_xyz"])
            gripper_ctrl = phase["gripper"]

            # ════════════════════════════════════════════════
            # C. TASK-SPACE IK — Jacobian pseudoinverse
            #
            # Maps Cartesian position error → desired joint delta.
            # J (3x7): how much EE moves per unit joint velocity
            # J_pinv (7x3): inverse — given desired EE move,
            #               what joint moves achieve it?
            # ════════════════════════════════════════════════
            current_xyz = get_ee_position(data, model, hand_id)
            dx          = target_xyz - current_xyz          # (3,) Cartesian error

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, hand_id)

            J      = jacp[:, :7]                            # (3, 7)
            J_pinv = np.linalg.pinv(J)                      # (7, 3)
            dq     = J_pinv @ dx                            # (7,) joint delta from IK

            # IK output becomes the desired joint position for inner PID

            q_des = data.qpos[:7] + dq * 0.05   # drop from 0.5 to 0.05

            # ════════════════════════════════════════════════
            # D. INNER LOOP — joint PID → torques
            #
            # Tracks q_des smoothly using torque commands.
            # D term (KD * qd_actual) damps joint overshoot —
            # overshoot is the primary source of EE jerk,
            # so tuning KD here directly reduces mixing risk.
            # ════════════════════════════════════════════════
            q_actual  = data.qpos[:7].copy()
            qd_actual = data.qvel[:7].copy()
            torques   = pid.compute(q_des, q_actual, qd_actual, dt)
            data.ctrl[:7] = torques

            # gripper fingers
            if model.nu >= 9:
                data.ctrl[7] = gripper_ctrl
                data.ctrl[8] = gripper_ctrl

            # ════════════════════════════════════════════════
            # E. STEP PHYSICS
            # ════════════════════════════════════════════════
            mujoco.mj_step(model, data)

            # ════════════════════════════════════════════════
            # F. SENSING — EE velocity → acc → jerk
            #
            # mj_objectVelocity returns [wx,wy,wz, vx,vy,vz]
            # We take [3:] for linear velocity in world frame,
            # then finite difference twice for acc and jerk.
            # ════════════════════════════════════════════════
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0
            )
            vel_linear            = vel6[3:].copy()
            acc_world, jerk_world = ee_kin.update(vel_linear)

            # live rotation matrix of gripper — tells us tube orientation
            xmat = data.xmat[hand_id].copy()               # (9,) flat

            # ════════════════════════════════════════════════
            # G. MIXING MONITOR — decompose + log
            #
            # Uses live xmat each step to get actual tube axis.
            # Returns lateral_jerk_mag which feeds back into
            # jerk_pid at the TOP of next iteration — closing
            # the outer feedback loop.
            # ════════════════════════════════════════════════
            risk_rate, lateral_jerk_mag = monitor.update(
                t=sim_t,
                phase_name=phase["name"],
                acc_world=acc_world,
                jerk_world=jerk_world,
                xmat=xmat,
                t_scale=t_scale,
                torques=torques,
                dt=dt,
            )

            # ════════════════════════════════════════════════
            # H. CONSOLE + RENDER
            # ════════════════════════════════════════════════
            if step % PRINT_EVERY == 0:
                ee_err = float(np.linalg.norm(dx))
                print(
                    f"t={sim_t:6.3f}s | "
                    f"t_eff={t_eff:6.3f}s | "
                    f"scale={t_scale:.2f} | "
                    f"ee_err={ee_err:.4f}m | "
                    f"lat_jerk={lateral_jerk_mag:.3f}m/s³ | "
                    f"risk={monitor.risk_integral:.3f}"
                )

            sim_t += dt
            step  += 1

            if t_eff > sm.total_time + 1.0:
                print("[RUN] All phases complete.")
                break

            viewer.sync()

    wall_elapsed = time.time() - wall_start
    print(f"[DONE] Wall={wall_elapsed:.1f}s | Sim={sim_t:.2f}s")
    monitor.summary()
    monitor.save_csv(LOG_PATH)

    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)
        print(f"[CLEAN] Removed {ENV_XML}")


if __name__ == "__main__":
    main()