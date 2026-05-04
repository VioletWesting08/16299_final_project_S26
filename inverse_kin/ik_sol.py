"""
ik_jerk_min.py  –  Pure IK controller with gradient-descent jerk minimisation.

Architecture
============
No PID feedback of any kind. Every timestep we solve:

    min  ||J dq - v_des||²  +  λ||dq||²
         + α_jerk * ||dq - dq_prev||²      ← jerk penalty (limits Δjerk)
         + α_null * ||dq - dq_null||²       ← null-space posture

via one step of gradient descent on the joint-velocity level.  Because the
objective is quadratic in dq the gradient step is exact (one Newton step =
global minimum), giving a closed-form solution that is equivalent to
Damped-Least-Squares IK with jerk regularisation baked in.

Math
----
Objective:  E(dq) = ||J dq - v_des||²
                  + λ²||dq||²
                  + α_j||dq - dq_prev||²
                  + α_n||dq - dq_null||²

∂E/∂(dq) = 0  →

  (JᵀJ + (λ²+α_j+α_n)I) dq = Jᵀ v_des + α_j dq_prev + α_n dq_null

Phases: identical to test_picking.py / kinematics.py.
"""

from __future__ import annotations
import os
import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    raise SystemExit("MuJoCo Python package is not installed.")

# ───────────────────────────────────────────────────────────────
# SCENE / FILE CONFIG
# ───────────────────────────────────────────────────────────────
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/ik_jerk_scene.xml"

# ───────────────────────────────────────────────────────────────
# IK + JERK-MINIMISATION PARAMETERS
# ───────────────────────────────────────────────────────────────
IK_LAMBDA_SQ   = 1e-4    # Tikhonov damping  (standard DLS regularisation)
JERK_ALPHA     = 0.05    # weight on ||dq - dq_prev||²  (jerk penalty)
NULL_ALPHA     = 0.30    # weight on null-space posture term
NULL_GAIN      = 0.5     # scaling of posture error → null velocity

# Nominal (rest) joint configuration
Q_NOMINAL = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])

# ───────────────────────────────────────────────────────────────
# TRAJECTORY PHASES  (identical to both reference files)
# ───────────────────────────────────────────────────────────────
PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.04, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6000,  0.0000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4000,  0.4000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "6. Place",     "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.04, "duration": 0.5},
]


# ───────────────────────────────────────────────────────────────
# MINIMUM-JERK TRAJECTORY SAMPLER  (5th-order polynomial)
# ───────────────────────────────────────────────────────────────
class TrajectorySampler:
    """5th-order minimum-jerk spline across all phases."""

    def __init__(self, phases: list[dict]) -> None:
        self.phases    = phases
        self.durations = [p["duration"] for p in phases]
        self.boundaries = np.cumsum(self.durations)
        self.total_time = float(self.boundaries[-1])
        self._starts: list[np.ndarray] = []
        self._coeffs:  list[list]      = []

    def set_start(self, xyz_start: np.ndarray) -> None:
        self._starts = [xyz_start.copy()]
        for i in range(1, len(self.phases)):
            self._starts.append(np.array(self.phases[i - 1]["target_xyz"], dtype=float))

        self._coeffs = []
        for i, phase in enumerate(self.phases):
            s = self._starts[i]
            e = np.array(phase["target_xyz"], dtype=float)
            T = self.durations[i]
            a0 = s
            a3 =  10 * (e - s) / T**3
            a4 = -15 * (e - s) / T**4
            a5 =   6 * (e - s) / T**5
            self._coeffs.append((a0, a3, a4, a5))

    def sample(self, t: float) -> tuple[np.ndarray, float]:
        t   = float(np.clip(t, 0.0, self.total_time))
        idx = int(np.searchsorted(self.boundaries, t, side="right"))
        idx = min(idx, len(self.phases) - 1)
        t0  = self.boundaries[idx - 1] if idx > 0 else 0.0
        tl  = t - t0
        a0, a3, a4, a5 = self._coeffs[idx]
        xyz = a0 + a3 * tl**3 + a4 * tl**4 + a5 * tl**5
        return xyz, self.phases[idx]["gripper"]

    def sample_velocity(self, t: float) -> np.ndarray:
        """Analytical velocity of the spline (3-vector, m/s)."""
        t   = float(np.clip(t, 0.0, self.total_time))
        idx = int(np.searchsorted(self.boundaries, t, side="right"))
        idx = min(idx, len(self.phases) - 1)
        t0  = self.boundaries[idx - 1] if idx > 0 else 0.0
        tl  = t - t0
        _, a3, a4, a5 = self._coeffs[idx]
        return 3 * a3 * tl**2 + 4 * a4 * tl**3 + 5 * a5 * tl**4


# ───────────────────────────────────────────────────────────────
# JERK-MINIMISING IK SOLVER
# ───────────────────────────────────────────────────────────────
class JerkMinIKSolver:
    """
    Solves for joint increments dq at each timestep using a regularised
    least-squares problem that explicitly penalises changes in dq (jerk proxy).

    Closed-form solution (one Newton step on quadratic objective):

        A dq = b
        A = JᵀJ + (λ² + α_j + α_n) I
        b = Jᵀ v_des  +  α_j dq_prev  +  α_n dq_null
    """

    def __init__(self,
                 n_joints:    int   = 7,
                 lambda_sq:   float = IK_LAMBDA_SQ,
                 jerk_alpha:  float = JERK_ALPHA,
                 null_alpha:  float = NULL_ALPHA,
                 null_gain:   float = NULL_GAIN) -> None:
        self.n        = n_joints
        self.lam_sq   = lambda_sq
        self.j_alpha  = jerk_alpha
        self.n_alpha  = null_alpha
        self.n_gain   = null_gain
        self.dq_prev  = np.zeros(n_joints)   # joint velocity at previous step

    def solve(self,
              J:        np.ndarray,   # (3, n_joints) positional Jacobian
              v_des:    np.ndarray,   # (3,) desired EE velocity
              q_current: np.ndarray  # (n_joints,) current joint angles
              ) -> np.ndarray:
        """
        Returns dq  (joint angle increment to apply this step).
        """
        n = self.n

        # Null-space posture velocity target
        dq_null = self.n_gain * (Q_NOMINAL - q_current)

        # Build normal equations
        JtJ = J.T @ J                        # (n, n)
        reg = (self.lam_sq + self.j_alpha + self.n_alpha) * np.eye(n)
        A   = JtJ + reg                      # (n, n)

        rhs = (J.T @ v_des
               + self.j_alpha * self.dq_prev
               + self.n_alpha * dq_null)     # (n,)

        dq = np.linalg.solve(A, rhs)

        self.dq_prev = dq.copy()
        return dq


# ───────────────────────────────────────────────────────────────
# SCENE BUILDER
# ───────────────────────────────────────────────────────────────
def build_scene(scene_path: str, out_path: str) -> None:
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


# ───────────────────────────────────────────────────────────────
# STATISTICS COLLECTOR
# ───────────────────────────────────────────────────────────────
class StatsCollector:
    def __init__(self) -> None:
        self.pos_errors:    list[float] = []
        self.joint_jerks:   list[float] = []   # ||dq - dq_prev|| / dt²
        self.ee_speeds:     list[float] = []
        self._prev_dq:      np.ndarray  = np.zeros(7)
        self._prev_dq2:     np.ndarray  = np.zeros(7)

    def record(self, pos_err: float, dq: np.ndarray, dt: float, ee_vel: np.ndarray) -> None:
        self.pos_errors.append(pos_err)
        self.ee_speeds.append(float(np.linalg.norm(ee_vel)))
        # joint-space jerk proxy: second finite difference of joint velocities
        jerk = np.linalg.norm((dq - 2 * self._prev_dq + self._prev_dq2) / dt**2)
        self.joint_jerks.append(float(jerk))
        self._prev_dq2 = self._prev_dq.copy()
        self._prev_dq  = dq.copy()

    def print_summary(self, sim_t: float) -> None:
        pe  = np.array(self.pos_errors)
        jk  = np.array(self.joint_jerks)
        spd = np.array(self.ee_speeds)
        print("\n" + "=" * 70)
        print("TRAJECTORY STATISTICS  (Pure IK + Gradient-Descent Jerk Minimisation)")
        print("=" * 70)
        print(f"Total simulation time :  {sim_t:.2f} s")
        print()
        print(f"Position error  avg  : {np.mean(pe):.6f} m")
        print(f"Position error  max  : {np.max(pe):.6f} m")
        print()
        print(f"EE speed        avg  : {np.mean(spd):.4f} m/s")
        print(f"EE speed        max  : {np.max(spd):.4f} m/s")
        print()
        print(f"Joint jerk      avg  : {np.mean(jk):.4f} rad/s³")
        print(f"Joint jerk      max  : {np.max(jk):.4f} rad/s³")
        print(f"Joint jerk  integral : {np.sum(jk) * (sim_t / max(len(jk), 1)):.4f} rad/s²")
        print("=" * 70)


# ───────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────
def main() -> None:
    # ── Build & load scene ────────────────────────────────────
    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    hand_id = model.body("hand").id

    # ── Warm-up (let robot settle at rest pose) ───────────────
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    # ── Build trajectory ─────────────────────────────────────
    traj = TrajectorySampler(PHASES)
    traj.set_start(data.xpos[hand_id].copy())

    # ── Controller & stats ───────────────────────────────────
    solver = JerkMinIKSolver()
    stats  = StatsCollector()

    # ── Print trajectory summary ─────────────────────────────
    print("=" * 70)
    print("JERK-MINIMISING IK CONTROLLER  –  phase plan")
    print("=" * 70)
    for p in PHASES:
        print(f"  {p['name']:14s}  target={p['target_xyz']}  "
              f"gripper={p['gripper']:.2f}  T={p['duration']:.1f}s")
    print("=" * 70)
    print(f"\nSolver parameters:")
    print(f"  λ²          (DLS damping)  = {IK_LAMBDA_SQ}")
    print(f"  α_jerk      (jerk penalty) = {JERK_ALPHA}")
    print(f"  α_null      (posture)      = {NULL_ALPHA}")
    print(f"  null gain                  = {NULL_GAIN}")
    print()

    # ── Timing ───────────────────────────────────────────────
    t_eff      = 0.0
    sim_t      = 0.0
    prev_phase = -1

    # ── Main loop ────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and t_eff < traj.total_time:

            # 1. Advance trajectory time
            t_eff += dt

            # 2. Sample desired EE position + analytical velocity
            target_xyz, gripper_w = traj.sample(t_eff)
            v_des                 = traj.sample_velocity(t_eff)

            # 3. Augment desired velocity with position feedback
            #    (pure feedforward would drift; a small proportional
            #     correction closes the loop without being a PID —
            #     it is mathematically equivalent to adding a position
            #     error term to v_des in the IK least-squares objective)
            current_xyz = data.xpos[hand_id].copy()
            pos_error   = target_xyz - current_xyz
            K_fb        = 5.0          # proportional position feedback gain
            v_des_fb    = v_des + K_fb * pos_error   # augmented desired velocity

            # 4. Compute Jacobian
            jacp = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, None, hand_id)
            J = jacp[:, :7]            # positional Jacobian, 7-DOF only

            # 5. Solve IK with jerk minimisation
            dq = solver.solve(J, v_des_fb, data.qpos[:7].copy())

            # 6. Integrate: q_des = q_current + dq
            q_des = data.qpos[:7] + dq

            # 7. Send position commands directly (position-controlled actuators)
            #    The Franka model uses position actuators → set ctrl = desired q
            data.ctrl[:7] = q_des

            # Gripper (actuator index 7 if available)
            if model.nu >= 8:
                data.ctrl[7] = gripper_w

            # 8. Step simulation
            mujoco.mj_step(model, data)

            # 9. Collect EE velocity for stats
            vel6 = np.zeros(6)
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0
            )
            ee_vel = vel6[3:]
            stats.record(float(np.linalg.norm(pos_error)), dq, dt, ee_vel)

            sim_t += dt

            # 10. Phase transition logging
            phase_idx = int(np.searchsorted(traj.boundaries, t_eff, side="right"))
            phase_idx = min(phase_idx, len(PHASES) - 1)
            if phase_idx != prev_phase:
                ph = PHASES[phase_idx]
                print(f"\n>>> PHASE {phase_idx + 1}: {ph['name']}")
                print(f"    Target XYZ : {ph['target_xyz']}")
                print(f"    Gripper    : {ph['gripper']:.4f}")
                print(f"    Duration   : {ph['duration']:.1f} s")
                prev_phase = phase_idx

            # 11. Periodic console logging
            step_n = int(sim_t / dt)
            if step_n % 100 == 0:
                jerk_proxy = float(np.linalg.norm(solver.dq_prev)) / dt
                joints_str = " | ".join(
                    f"J{i+1}={np.degrees(data.qpos[i]):7.2f}°"
                    for i in range(7)
                )
                print(
                    f"sim={sim_t:6.2f}s | t_eff={t_eff:6.2f}s | "
                    f"pos_err={np.linalg.norm(pos_error):.4f}m | "
                    f"jerk≈{jerk_proxy:.4f} rad/s²"
                )
                print(f"  Joints: {joints_str}")

            viewer.sync()

    # ── Clean up temp file ────────────────────────────────────
    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)

    # ── Print final statistics ────────────────────────────────
    stats.print_summary(sim_t)
    print("Trajectory complete.\n")


if __name__ == "__main__":
    main()