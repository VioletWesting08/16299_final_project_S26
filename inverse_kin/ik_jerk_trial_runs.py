"""
ik_jerk_trial_runs.py
=====================
Evaluation sweep for the jerk-minimising IK controller (ik_jerk_min.py).

Mirrors trial_runs.py exactly in structure and CSV schema so results from
both files can be loaded into the same spreadsheet / plotting script.

Two sweep axes are run:
  1. LIQUID_TAU sweep  (0.0 → 2.0 s, NUM_TAU_TRIALS steps)
     — identical to trial_runs.py, allowing apples-to-apples comparison
     — condition label: "jerk_ik"

  2. JERK_ALPHA sweep  (JERK_ALPHA_MIN → JERK_ALPHA_MAX, NUM_ALPHA_TRIALS steps)
     — controller-specific: shows the trade-off between jerk smoothing
       and position tracking as the penalty weight changes
     — condition label: "jerk_alpha_sweep"

Both sweeps use the same fixed tau (LIQUID_TAU_FIXED) and jerk alpha
(JERK_ALPHA_FIXED) respectively when they are not the swept variable.

Output CSV columns match trial_runs.py exactly, with three extra columns:
  jerk_alpha, k_fb, avg_joint_jerk, integrated_joint_jerk
These are appended at the end so existing parsers that read only the shared
columns are unaffected.
"""

import mujoco
import numpy as np
import os
import csv
import time

# ═══════════════════════════════════════════════════════════════
# SCENE CONFIG
# ═══════════════════════════════════════════════════════════════
SCENE_XML = "franka_emika_panda/scene.xml"
ENV_XML   = "franka_emika_panda/ik_jerk_eval_scene.xml"

# ═══════════════════════════════════════════════════════════════
# FIXED CONTROLLER DEFAULTS  (match ik_jerk_min.py)
# ═══════════════════════════════════════════════════════════════
IK_LAMBDA_SQ = 1e-4
NULL_ALPHA   = 0.30
NULL_GAIN    = 0.5
K_FB         = 5.0
Q_NOMINAL    = np.array([0.0, -0.3, 0.0, -1.8, 0.0, 1.5, 0.75])

# ═══════════════════════════════════════════════════════════════
# FIXED LIQUID-MODEL DEFAULTS  (match ik_jerk_min.py / trial_runs.py)
# ═══════════════════════════════════════════════════════════════
ACCEL_LPFILTER_ALPHA = 0.03

# ═══════════════════════════════════════════════════════════════
# SWEEP CONFIGURATION
# ═══════════════════════════════════════════════════════════════
# --- Tau sweep (mirrors trial_runs.py) ---
NUM_TAU_TRIALS  = 10
TAU_MIN         = 0.0
TAU_MAX         = 2.0
JERK_ALPHA_FIXED = 0.05   # held constant while sweeping tau

# --- Jerk-alpha sweep (controller-specific) ---
NUM_ALPHA_TRIALS  = 10
JERK_ALPHA_MIN    = 0.0    # 0 = pure DLS, no jerk penalty
JERK_ALPHA_MAX    = 0.5    # heavy smoothing
LIQUID_TAU_FIXED  = 1.0    # held constant while sweeping alpha

OUTPUT_CSV = "outputs/ik_jerk_trial_results.csv"

# ═══════════════════════════════════════════════════════════════
# TRAJECTORY PHASES  (identical to test_picking.py / trial_runs.py)
# ═══════════════════════════════════════════════════════════════
PHASES = [
    {"name": "1. Hover",     "target_xyz": [0.6182, -0.0470, 0.2958], "gripper": 0.04, "duration": 1.0},
    {"name": "2. Descend",   "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.04, "duration": 1.0},
    {"name": "3. Grasp",     "target_xyz": [0.6000,  0.0000, 0.1200], "gripper": 0.00, "duration": 0.5},
    {"name": "4. Lift",      "target_xyz": [0.6000,  0.0000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "5. Transport", "target_xyz": [0.4000,  0.4000, 0.4000], "gripper": 0.00, "duration": 1.0},
    {"name": "6. Place",     "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.00, "duration": 1.0},
    {"name": "7. Release",   "target_xyz": [0.4000,  0.4000, 0.1200], "gripper": 0.04, "duration": 0.5},
]


# ═══════════════════════════════════════════════════════════════
# TRAJECTORY SAMPLER
# ═══════════════════════════════════════════════════════════════
class TrajectorySampler:
    def __init__(self, phases):
        self.phases     = phases
        self.durations  = [p["duration"] for p in phases]
        self.boundaries = np.cumsum(self.durations)
        self.total_time = float(self.boundaries[-1])
        self._coeffs    = []

    def set_start(self, xyz_start):
        starts = [xyz_start.copy()]
        for i in range(1, len(self.phases)):
            starts.append(np.array(self.phases[i - 1]["target_xyz"], dtype=float))
        self._coeffs = []
        for i, phase in enumerate(self.phases):
            s = starts[i]
            e = np.array(phase["target_xyz"], dtype=float)
            T = self.durations[i]
            self._coeffs.append((s, 10*(e-s)/T**3, -15*(e-s)/T**4, 6*(e-s)/T**5))

    def _local(self, t):
        t   = float(np.clip(t, 0.0, self.total_time))
        idx = min(int(np.searchsorted(self.boundaries, t, side="right")),
                  len(self.phases) - 1)
        t0  = self.boundaries[idx - 1] if idx > 0 else 0.0
        return idx, t - t0

    def sample(self, t):
        idx, tl = self._local(t)
        a0, a3, a4, a5 = self._coeffs[idx]
        return a0 + a3*tl**3 + a4*tl**4 + a5*tl**5, self.phases[idx]["gripper"]

    def sample_velocity(self, t):
        idx, tl = self._local(t)
        _, a3, a4, a5 = self._coeffs[idx]
        return 3*a3*tl**2 + 4*a4*tl**3 + 5*a5*tl**4


# ═══════════════════════════════════════════════════════════════
# JERK-MINIMISING IK SOLVER
# ═══════════════════════════════════════════════════════════════
class JerkMinIKSolver:
    """
    Solves at each step:
        (JᵀJ + (λ² + α_j + α_n)·I) dq = Jᵀ v_des + α_j·dq_prev + α_n·dq_null
    """
    def __init__(self, jerk_alpha):
        self.lam_sq  = IK_LAMBDA_SQ
        self.j_alpha = jerk_alpha
        self.n_alpha = NULL_ALPHA
        self.n_gain  = NULL_GAIN
        self.dq_prev = np.zeros(7)

    def solve(self, J, v_des, q_current):
        dq_null = self.n_gain * (Q_NOMINAL - q_current)
        A   = J.T @ J + (self.lam_sq + self.j_alpha + self.n_alpha) * np.eye(7)
        rhs = J.T @ v_des + self.j_alpha * self.dq_prev + self.n_alpha * dq_null
        dq  = np.linalg.solve(A, rhs)
        self.dq_prev = dq.copy()
        return dq


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
# SINGLE TRIAL
# ═══════════════════════════════════════════════════════════════
def run_trial(liquid_tau: float, jerk_alpha: float) -> dict:
    """
    Run one complete trajectory and return a statistics dict.

    Parameters
    ----------
    liquid_tau  : first-order lag time constant for the liquid model (seconds)
    jerk_alpha  : weight on the ||dq - dq_prev||² jerk penalty term

    Returns a dict whose shared keys match trial_runs.py exactly.
    """
    build_scene(SCENE_XML, ENV_XML)
    model = mujoco.MjModel.from_xml_path(ENV_XML)
    data  = mujoco.MjData(model)
    dt    = model.opt.timestep

    hand_id = model.body("hand").id
    solver  = JerkMinIKSolver(jerk_alpha=jerk_alpha)
    traj    = TrajectorySampler(PHASES)

    # Warm-up
    for _ in range(500):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    traj.set_start(data.xpos[hand_id].copy())

    # State
    prev_ee_vel       = np.zeros(3)
    filtered_ee_accel = np.zeros(3)
    a_liquid          = np.array([0.0, 0.0, -9.81])
    g                 = np.array([0.0, 0.0, -9.81])

    # Accumulators
    speed_values        = []
    pos_error_values    = []
    mixing_score_values = []   # gated to grasp/transport phases (indices 2-5)
    joint_jerk_values   = []

    sim_t = 0.0
    t_eff = 0.0

    while t_eff < traj.total_time:

        # ── 1. EE VELOCITY ──────────────────────────────────────
        vel6 = np.zeros(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_BODY, hand_id, vel6, 0
        )
        cur_vel = vel6[3:].copy()
        speed_values.append(float(np.linalg.norm(cur_vel)))

        # ── 2. LIQUID INERTIA MODEL ─────────────────────────────
        ee_accel_raw      = (cur_vel - prev_ee_vel) / max(dt, 1e-9)
        prev_ee_vel       = cur_vel.copy()
        filtered_ee_accel = (ACCEL_LPFILTER_ALPHA * ee_accel_raw +
                             (1.0 - ACCEL_LPFILTER_ALPHA) * filtered_ee_accel)
        a_effective = g + filtered_ee_accel

        if liquid_tau > 1e-9:
            a_liquid += ((a_effective - a_liquid) / liquid_tau) * dt
        else:
            a_liquid = a_effective.copy()   # tau=0: liquid is instantaneous

        # ── 3. SAMPLE TRAJECTORY ────────────────────────────────
        t_eff += dt
        target_xyz, gripper_w = traj.sample(t_eff)
        v_ff                  = traj.sample_velocity(t_eff)

        # ── 4. POSITION ERROR + AUGMENTED TARGET VELOCITY ───────
        current_xyz = data.xpos[hand_id].copy()
        pos_error   = target_xyz - current_xyz
        pos_error_values.append(float(np.linalg.norm(pos_error)))

        v_des = v_ff + K_FB * pos_error

        # ── 5. JACOBIAN ─────────────────────────────────────────
        jacp = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, None, hand_id)
        J = jacp[:, :7]

        # ── 6. JERK-MINIMISING IK SOLVE ─────────────────────────
        dq_prev_snap = solver.dq_prev.copy()
        dq    = solver.solve(J, v_des, data.qpos[:7].copy())
        q_des = data.qpos[:7] + dq

        joint_jerk_values.append(
            float(np.linalg.norm(dq - dq_prev_snap) / max(dt, 1e-9))
        )

        # ── 7. APPLY COMMANDS ────────────────────────────────────
        data.ctrl[:7] = q_des
        if model.nu >= 8:
            data.ctrl[7] = gripper_w

        # ── 8. STEP PHYSICS ─────────────────────────────────────
        mujoco.mj_step(model, data)
        sim_t += dt

        # ── 9. MIXING ANGLE  (gated to grasp/transport, same as trial_runs.py) ──
        phase_idx = min(
            int(np.searchsorted(traj.boundaries, t_eff, side="right")),
            len(PHASES) - 1
        )
        hand_rot  = data.xmat[hand_id].reshape(3, 3)
        tube_axis = hand_rot[:, 2]
        norm_tube = float(np.linalg.norm(tube_axis))
        norm_liq  = float(np.linalg.norm(a_liquid))
        if norm_tube > 1e-9 and norm_liq > 1e-9:
            cos_angle = float(np.clip(
                np.dot(tube_axis, a_liquid / norm_liq), -1.0, 1.0
            ))
            mix_angle = float(np.degrees(np.arccos(cos_angle)))
        else:
            mix_angle = 0.0

        if 2 <= phase_idx <= 5:
            mixing_score_values.append(mix_angle)

    # Clean up
    if os.path.exists(ENV_XML):
        os.remove(ENV_XML)

    sa = np.array(speed_values)
    pa = np.array(pos_error_values)
    ma = np.array(mixing_score_values) if mixing_score_values else np.zeros(1)
    ja = np.array(joint_jerk_values)

    return {
        # ── shared keys (identical to trial_runs.py) ──────────
        "liquid_tau":              liquid_tau,
        "use_wrist_pid":           0,           # controller has no wrist PID
        "avg_ee_speed":            float(np.mean(sa)),
        "max_ee_speed":            float(np.max(sa)),
        "avg_pos_error":           float(np.mean(pa)),
        "max_pos_error":           float(np.max(pa)),
        "avg_tube_tilt_error_deg": float(np.mean(ma)),
        "max_tube_tilt_error_deg": float(np.max(ma)),
        "avg_mix_angle_deg":       float(np.mean(ma)),
        "max_mix_angle_deg":       float(np.max(ma)),
        "integrated_mix_score":    float(np.sum(ma) * dt),
        "sim_time_s":              float(sim_t),
        # ── extra keys (controller-specific) ──────────────────
        "jerk_alpha":              jerk_alpha,
        "k_fb":                    K_FB,
        "avg_joint_jerk":          float(np.mean(ja)),
        "integrated_joint_jerk":   float(np.sum(ja) * dt),
    }


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    # ── Build experiment plan ───────────────────────────────────
    # Sweep 1: vary tau (jerk_alpha fixed) — mirrors trial_runs.py
    tau_values = [
        TAU_MIN + (TAU_MAX - TAU_MIN) * i / max(NUM_TAU_TRIALS - 1, 1)
        for i in range(NUM_TAU_TRIALS)
    ]

    # Sweep 2: vary jerk_alpha (tau fixed) — controller-specific
    alpha_values = [
        JERK_ALPHA_MIN + (JERK_ALPHA_MAX - JERK_ALPHA_MIN) * i / max(NUM_ALPHA_TRIALS - 1, 1)
        for i in range(NUM_ALPHA_TRIALS)
    ]

    # (liquid_tau, jerk_alpha, condition_label)
    experiment_plan = []
    for tau in tau_values:
        experiment_plan.append((tau, JERK_ALPHA_FIXED, "jerk_ik"))
    for alpha in alpha_values:
        experiment_plan.append((LIQUID_TAU_FIXED, alpha, "jerk_alpha_sweep"))

    total = len(experiment_plan)

    print("=" * 70)
    print(f"IK JERK-MIN EVAL — {total} trials total")
    print(f"  Sweep 1 (tau):   {TAU_MIN:.2f}s → {TAU_MAX:.2f}s  "
          f"({NUM_TAU_TRIALS} steps, jerk_alpha={JERK_ALPHA_FIXED})")
    print(f"  Sweep 2 (alpha): {JERK_ALPHA_MIN:.3f} → {JERK_ALPHA_MAX:.3f}  "
          f"({NUM_ALPHA_TRIALS} steps, tau={LIQUID_TAU_FIXED}s)")
    print(f"  Output:          {OUTPUT_CSV}")
    print("=" * 70 + "\n")

    results    = []
    wall_start = time.time()

    for trial_idx, (tau, alpha, label) in enumerate(experiment_plan):
        elapsed = time.time() - wall_start
        print(
            f"[{trial_idx+1:3d}/{total}]  condition={label:20s}  "
            f"tau={tau:.4f}s  jerk_alpha={alpha:.4f}  "
            f"(elapsed {elapsed:.1f}s)",
            flush=True,
        )

        stats = run_trial(liquid_tau=tau, jerk_alpha=alpha)
        stats["condition"]   = label
        stats["trial_index"] = trial_idx
        results.append(stats)

        print(
            f"           avg_pos_err={stats['avg_pos_error']:.4f}m  "
            f"avg_mix={stats['avg_mix_angle_deg']:.2f}°  "
            f"integrated={stats['integrated_mix_score']:.2f}°*s  "
            f"avg_jerk={stats['avg_joint_jerk']:.4f} rad/s²"
        )

    # ── Save CSV ────────────────────────────────────────────────
    # Shared columns first (same order as trial_runs.py), then extras
    fieldnames = [
        "trial_index", "condition", "liquid_tau", "use_wrist_pid",
        "avg_ee_speed", "max_ee_speed",
        "avg_pos_error", "max_pos_error",
        "avg_tube_tilt_error_deg", "max_tube_tilt_error_deg",
        "avg_mix_angle_deg", "max_mix_angle_deg",
        "integrated_mix_score", "sim_time_s",
        # extras
        "jerk_alpha", "k_fb", "avg_joint_jerk", "integrated_joint_jerk",
    ]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"\n✓ Results written to {OUTPUT_CSV}")

    # ── Summary table ───────────────────────────────────────────
    tau_results   = [r for r in results if r["condition"] == "jerk_ik"]
    alpha_results = [r for r in results if r["condition"] == "jerk_alpha_sweep"]

    def col_mean(rows, key):
        return sum(r[key] for r in rows) / len(rows) if rows else float("nan")

    def col_min(rows, key):
        return min(r[key] for r in rows) if rows else float("nan")

    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL TRIALS")
    print("=" * 70)
    print(f"{'Metric':<35} {'jerk_ik (tau sweep)':>22} {'alpha_sweep':>12}")
    print("-" * 70)
    for key, label in [
        ("avg_ee_speed",          "Mean avg EE speed (m/s)"),
        ("max_ee_speed",          "Mean max EE speed (m/s)"),
        ("avg_pos_error",         "Mean avg pos error (m)"),
        ("max_pos_error",         "Mean max pos error (m)"),
        ("avg_mix_angle_deg",     "Mean avg mix angle (deg)"),
        ("max_mix_angle_deg",     "Mean max mix angle (deg)"),
        ("integrated_mix_score",  "Mean integrated mix (deg*s)"),
        ("avg_joint_jerk",        "Mean avg joint jerk (rad/s^2)"),
        ("integrated_joint_jerk", "Mean integr. joint jerk (rad/s)"),
    ]:
        print(f"  {label:<33} {col_mean(tau_results, key):>22.4f} "
              f"{col_mean(alpha_results, key):>12.4f}")

    print()
    print("  Best (lowest) integrated mix score:")
    if tau_results:
        best_tau = min(tau_results, key=lambda r: r["integrated_mix_score"])
        print(f"    jerk_ik:      tau={best_tau['liquid_tau']:.4f}s  "
              f"integrated={best_tau['integrated_mix_score']:.2f}°*s")
    if alpha_results:
        best_alpha = min(alpha_results, key=lambda r: r["integrated_mix_score"])
        print(f"    alpha_sweep:  jerk_alpha={best_alpha['jerk_alpha']:.4f}  "
              f"integrated={best_alpha['integrated_mix_score']:.2f}°*s")
    print("=" * 70)
    print(f"\nTotal wall-clock time: {time.time() - wall_start:.1f}s")


if __name__ == "__main__":
    main()
