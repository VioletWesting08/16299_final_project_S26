"""
optimize_transport.py
─────────────────────────────────────────────────────────────────────────────
Autonomous, headless optimizer for the Franka transport controller.

Optimizes KP, KD, KI gains and MANEUVER_TIME to minimize a combined cost:

    cost = risk_integral  +  W_JERR * mean_joint_error

Uses CMA-ES (Covariance Matrix Adaptation Evolution Strategy) via the `cma`
package, falling back to scipy's Nelder-Mead if cma is not installed.

Usage
─────
    python optimize_transport.py                    # run forever
    python optimize_transport.py --budget 200       # stop after 200 evals
    python optimize_transport.py --cost-weight 2.0  # tune cost blend

Outputs
───────
    optimizer_log.csv   — one row per evaluation
    best_params.json    — best params found so far (updated every trial)

Requirements
────────────
    pip install mujoco cma          (cma is optional but strongly recommended)
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Tuple

import numpy as np

# ── MuJoCo ───────────────────────────────────────────────────────────────────
try:
    import mujoco
except ImportError:
    sys.exit("ERROR: mujoco not installed.  pip install mujoco")

# ── Optional CMA-ES ───────────────────────────────────────────────────────────
try:
    import cma
    HAS_CMA = True
except ImportError:
    HAS_CMA = False
    from scipy.optimize import minimize as scipy_minimize


# ═════════════════════════════════════════════════════════════════════════════
# FIXED CONFIGURATION  (things we don't optimise)
# ═════════════════════════════════════════════════════════════════════════════

XML_PATH        = "franka_emika_panda/scene.xml"
LOG_PATH        = "optimizer_log.csv"
BEST_PATH       = "best_params.json"

START_JOINTS    = np.array([ 0.0, -0.5,  0.0, -2.0,  0.0,  1.5,  0.8])
GOAL_JOINTS     = np.array([ 0.5,  0.2, -0.3, -1.5,  0.2,  1.8,  1.2])

TORQUE_LIMITS   = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)

W_LATERAL_ACC   = 0.1
W_JERK          = 0.2
TUBE_AXIS_LOCAL = np.array([0.0, 0.0, 1.0])

SETTLE_TIME     = 1.0   # seconds of settle after maneuver (fixed)

# ── Search bounds ─────────────────────────────────────────────────────────────
#   param vector layout:
#   [kp0..kp6, kd0..kd6, ki0..ki6, maneuver_time]   (22 values)

KP_LOW  = np.array([ 500,  500,  300,  300,  100,  100,   50], dtype=float)
KP_HIGH = np.array([8000, 8000, 6000, 6000, 3000, 3000,  800], dtype=float)

KD_LOW  = np.array([ 50,  50,  30,  30,  10,  10,   5], dtype=float)
KD_HIGH = np.array([900, 900, 700, 700, 400, 400,  100], dtype=float)

KI_LOW  = np.zeros(7)
KI_HIGH = np.full(7, 10.0)

T_LOW   = 1.5
T_HIGH  = 8.0

# ── Default (starting) params ─────────────────────────────────────────────────
DEFAULT_KP = np.array([4500, 4500, 3500, 3500, 2000, 2000,  500], dtype=float)
DEFAULT_KD = np.array([ 450,  450,  350,  350,  200,  200,   50], dtype=float)
DEFAULT_KI = np.ones(7)
DEFAULT_T  = 4.0


# ═════════════════════════════════════════════════════════════════════════════
# TRAJECTORY & CONTROLLER  (copied/trimmed from transport2.0.py)
# ═════════════════════════════════════════════════════════════════════════════

def min_jerk(t, T, q0, qf):
    if t <= 0:  return q0.copy()
    if t >= T:  return qf.copy()
    tau = t / T
    s   = 10*tau**3 - 15*tau**4 + 6*tau**5
    return q0 + s*(qf - q0)


class FrankaPID:
    def __init__(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral = np.zeros(7)
        self.clamp    = 5.0

    def reset(self):
        self.integral[:] = 0.0

    def compute(self, q_des, q_act, qd_act, dt):
        e = q_des - q_act
        self.integral = np.clip(self.integral + e*dt, -self.clamp, self.clamp)
        tau = self.kp*e + self.ki*self.integral - self.kd*qd_act
        return np.clip(tau, -TORQUE_LIMITS, TORQUE_LIMITS)


def world_to_tube(vec_world, xmat):
    return xmat.reshape(3, 3).T @ vec_world


def lateral_acc(acc_local):
    proj = np.dot(acc_local, TUBE_AXIS_LOCAL) * TUBE_AXIS_LOCAL
    return acc_local - proj


# ═════════════════════════════════════════════════════════════════════════════
# HEADLESS SIMULATION  → returns scalar cost
# ═════════════════════════════════════════════════════════════════════════════

def run_sim(
    model,
    kp: np.ndarray,
    kd: np.ndarray,
    ki: np.ndarray,
    maneuver_time: float,
    w_jerr: float = 1.0,
) -> Tuple[float, dict]:
    """Run one headless episode and return (cost, info_dict)."""

    data = mujoco.MjData(model)
    dt   = model.opt.timestep

    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    if hand_id == -1:
        hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "panda_hand")

    data.qpos[:7] = START_JOINTS
    mujoco.mj_forward(model, data)

    pid  = FrankaPID(kp, ki, kd)
    pid.reset()

    t_sim        = 0.0
    risk_int     = 0.0
    joint_errs   = []
    lat_accs     = []
    jerks        = []

    prev_vel = np.zeros(3)
    prev_acc = np.zeros(3)

    total_time = maneuver_time + SETTLE_TIME

    while t_sim < total_time:
        q_des     = min_jerk(t_sim, maneuver_time, START_JOINTS, GOAL_JOINTS)
        q_act     = data.qpos[:7].copy()
        qd_act    = data.qvel[:7].copy()
        torques   = pid.compute(q_des, q_act, qd_act, dt)
        data.ctrl[:7] = torques
        mujoco.mj_step(model, data)

        vel_world = data.cvel[hand_id][3:6].copy()
        acc_world = (vel_world - prev_vel) / dt
        jerk      = (acc_world - prev_acc) / dt
        prev_vel  = vel_world.copy()
        prev_acc  = acc_world.copy()

        xmat      = data.xmat[hand_id].copy()
        acc_local = world_to_tube(acc_world, xmat)
        lat       = float(np.linalg.norm(lateral_acc(acc_local)))
        jrk       = float(np.linalg.norm(jerk))

        risk_int += (W_LATERAL_ACC * lat + W_JERK * jrk) * dt
        lat_accs.append(lat)
        jerks.append(jrk)
        joint_errs.append(float(np.linalg.norm(q_des - q_act)))

        t_sim += dt

    # Final joint error (settling quality)
    final_err = float(np.linalg.norm(GOAL_JOINTS - data.qpos[:7]))

    mean_jerr = float(np.mean(joint_errs))
    cost      = risk_int + w_jerr * mean_jerr + 5.0 * final_err

    info = {
        "risk_integral":  round(risk_int, 6),
        "mean_joint_err": round(mean_jerr, 6),
        "final_joint_err":round(final_err, 6),
        "peak_lat_acc":   round(max(lat_accs), 6),
        "peak_jerk":      round(max(jerks), 6),
        "cost":           round(cost, 6),
    }
    return cost, info


# ═════════════════════════════════════════════════════════════════════════════
# PARAMETER ENCODING / DECODING
# ═════════════════════════════════════════════════════════════════════════════

def encode(kp, kd, ki, t) -> np.ndarray:
    return np.concatenate([kp, kd, ki, [t]])

def decode(x: np.ndarray):
    kp = np.clip(x[0:7],  KP_LOW, KP_HIGH)
    kd = np.clip(x[7:14], KD_LOW, KD_HIGH)
    ki = np.clip(x[14:21],KI_LOW, KI_HIGH)
    t  = float(np.clip(x[21], T_LOW, T_HIGH))
    return kp, kd, ki, t

def param_to_dict(kp, kd, ki, t) -> dict:
    return {
        "KP": kp.tolist(), "KD": kd.tolist(),
        "KI": ki.tolist(), "MANEUVER_TIME": t,
    }


# ═════════════════════════════════════════════════════════════════════════════
# LOGGER
# ═════════════════════════════════════════════════════════════════════════════

class TrialLogger:
    def __init__(self, path: str):
        self.path     = path
        self.trial    = 0
        self._wrote_header = False

    def log(self, kp, kd, ki, t, info: dict, is_best: bool):
        self.trial += 1
        row = {
            "trial":         self.trial,
            "timestamp":     time.strftime("%H:%M:%S"),
            "maneuver_time": round(t, 3),
            "is_best":       int(is_best),
            **info,
            **{f"kp{i}": round(kp[i], 1) for i in range(7)},
            **{f"kd{i}": round(kd[i], 1) for i in range(7)},
            **{f"ki{i}": round(ki[i], 3) for i in range(7)},
        }
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerow(row)


# ═════════════════════════════════════════════════════════════════════════════
# MAIN OPTIMIZATION LOOP
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Autonomous Franka transport optimizer")
    parser.add_argument("--budget",      type=int,   default=0,   help="Max evaluations (0 = run forever)")
    parser.add_argument("--cost-weight", type=float, default=1.0, help="Weight on mean joint error in cost")
    parser.add_argument("--sigma0",      type=float, default=0.25, help="CMA-ES initial step size (normalised)")
    parser.add_argument("--seed",        type=int,   default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    print(f"[INIT] Loading model: {XML_PATH}")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    print(f"[INIT] dt={model.opt.timestep:.4f}s")
    print(f"[INIT] Optimizer: {'CMA-ES' if HAS_CMA else 'Nelder-Mead (install cma for better results)'}")
    print(f"[INIT] Budget: {'∞' if args.budget == 0 else args.budget} evaluations")
    print()

    logger      = TrialLogger(LOG_PATH)
    best_cost   = np.inf
    best_params = None
    eval_count  = 0

    x0 = encode(DEFAULT_KP, DEFAULT_KD, DEFAULT_KI, DEFAULT_T)

    # ── Normalise to [0,1] for CMA ────────────────────────────────────────────
    lo = np.concatenate([KP_LOW,  KD_LOW,  KI_LOW,  [T_LOW]])
    hi = np.concatenate([KP_HIGH, KD_HIGH, KI_HIGH, [T_HIGH]])

    def normalise(x):   return (x - lo) / (hi - lo)
    def unnormalise(z): return lo + z * (hi - lo)

    z0 = normalise(x0)

    def objective(z):
        nonlocal best_cost, best_params, eval_count
        eval_count += 1

        x      = unnormalise(np.asarray(z))
        kp, kd, ki, t = decode(x)

        try:
            cost, info = run_sim(model, kp, kd, ki, t, w_jerr=args.cost_weight)
        except Exception as exc:
            # Penalise crashed evaluations
            cost = 1e6
            info = {"risk_integral": 9999, "mean_joint_err": 9999,
                    "final_joint_err": 9999, "peak_lat_acc": 9999,
                    "peak_jerk": 9999, "cost": cost}
            print(f"  [WARN] Simulation crashed: {exc}")

        is_best = cost < best_cost
        if is_best:
            best_cost   = cost
            best_params = param_to_dict(kp, kd, ki, t)
            # Persist immediately so you can ctrl-C anytime
            with open(BEST_PATH, "w") as f:
                json.dump({**best_params, "cost": round(best_cost, 6)}, f, indent=2)

        logger.log(kp, kd, ki, t, info, is_best)

        marker = "  ★ NEW BEST" if is_best else ""
        print(
            f"  [{eval_count:>5}] cost={cost:8.4f} | "
            f"risk={info['risk_integral']:7.4f} | "
            f"jerr={info['mean_joint_err']:.4f} | "
            f"T={t:.2f}s{marker}"
        )

        if args.budget and eval_count >= args.budget:
            raise StopIteration("budget reached")

        return float(cost)

    # ── Run ───────────────────────────────────────────────────────────────────
    print("─" * 62)
    print("  Starting optimization.  Ctrl-C or --budget to stop.")
    print("─" * 62)

    try:
        if HAS_CMA:
            _run_cma(z0, args.sigma0, objective, rng)
        else:
            _run_nelder(z0, objective)
    except (KeyboardInterrupt, StopIteration) as e:
        print(f"\n[STOP] {e}")

    # ── Final report ──────────────────────────────────────────────────────────
    print()
    print("═" * 62)
    print("  OPTIMIZATION COMPLETE")
    print("═" * 62)
    print(f"  Total evaluations : {eval_count}")
    print(f"  Best cost         : {best_cost:.6f}")
    if best_params:
        print(f"  Best MANEUVER_TIME: {best_params['MANEUVER_TIME']:.3f}s")
        print(f"  Best KP           : {np.round(best_params['KP'], 1).tolist()}")
        print(f"  Best KD           : {np.round(best_params['KD'], 1).tolist()}")
        print(f"  Best KI           : {np.round(best_params['KI'], 4).tolist()}")
    print(f"\n  Results saved to  : {LOG_PATH}")
    print(f"  Best params saved : {BEST_PATH}")
    print("═" * 62)


# ═════════════════════════════════════════════════════════════════════════════
# OPTIMIZER BACKENDS
# ═════════════════════════════════════════════════════════════════════════════

def _run_cma(z0, sigma0, objective, rng):
    """Run CMA-ES indefinitely (restarts on convergence)."""
    opts = cma.CMAOptions()
    opts["bounds"]       = [[0.0]*22, [1.0]*22]
    opts["seed"]         = int(rng.integers(1, 10000))
    opts["verbose"]      = -9   # suppress CMA's own output
    opts["tolx"]         = 1e-5
    opts["tolfun"]       = 1e-6
    opts["maxfevals"]    = int(1e9)  # we manage budget ourselves

    restart = 0
    while True:
        restart += 1
        # Warm-start from z0 on first run, perturb on restarts
        z_start = z0 if restart == 1 else np.clip(z0 + rng.normal(0, 0.1, size=22), 0, 1)
        print(f"\n[CMA] Restart #{restart}")
        es = cma.CMAEvolutionStrategy(z_start.tolist(), sigma0, opts)
        es.optimize(objective)


def _run_nelder(z0, objective):
    """Fallback: Nelder-Mead with restarts."""
    bounds = [(0.0, 1.0)] * 22
    restart = 0
    x_best  = z0.copy()
    while True:
        restart += 1
        print(f"\n[NM] Restart #{restart}")
        res = scipy_minimize(
            objective, x_best,
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-5, "fatol": 1e-6, "disp": False},
        )
        x_best = np.clip(res.x, 0, 1)


if __name__ == "__main__":
    main()