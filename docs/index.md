# Centrifuge Tube Transporter
### Alan, Zoey, Sarah, Sophia — 16299 Final Project

A feedback control algorithm for transporting a centrifuge tube containing separated liquid layers using a Franka Panda robotic arm — without mixing them.

---

## Motivation

Biology labs increasingly rely on robotic automation, but standard motion planning ignores the physics of sensitive payloads. Moving a centrifuge tube too aggressively mixes its separated layers, ruining the sample. This project tackles the core challenge: transport the tube as fast as possible while keeping mixing below a threshold using active feedback control.

---

## How It Works

### Mixing Metric

Rather than simulating fluid directly, we define a proxy metric:

- `tube_axis` — which way the tube is pointing (gripper Z-axis)
- `a_effective` — gravity + end-effector acceleration (what the liquid feels)
- `a_liquid` — `a_effective` with a 1-second lag, modeling liquid inertia
- `mix_angle` — angle between `tube_axis` and `a_liquid`

```
mix_angle = arccos(dot(tube_axis, a_liquid / |a_liquid|))   [degrees]
```

0° = tube perfectly aligned with liquid settlement = no mixing risk. The integrated mixing score `∫ mix_angle dt` (°·s) is the primary benchmark.

---

### System Architecture

```
Trajectory Planner → Wrist Orientation PID → Joint Controller → MuJoCo Physics
                             ↑                      ↑
                    [feedback loop]          [overrides default MuJoCo gains]

Feedback loop:
  MuJoCo Physics → end effector pose
      → IMU sensor: mix_angle = angle(tube_axis, a_liquid)
          → error = mix_angle - 0°
              → Wrist PID → correction factor
                  → null-space projection → modifies q_des for Joint Controller
```

**Three layers:**

1. **Trajectory Planner** — 5th-order minimum-jerk polynomial per phase. Zero velocity and acceleration at endpoints minimizes jerk. Phases: Hover → Descend → Grasp → Lift → Transport → Place → Release.

2. **Wrist Orientation PID** — closed-loop controller on `mix_angle`. Correction projected into Jacobian null-space to rotate the wrist toward alignment without disturbing end-effector position.

3. **Joint PID** — custom torque controller overriding MuJoCo's default gains: `τ = Kp*(q_des - q) - Kd*q̇`

---

### Key Implementation Details

**Minimum-jerk trajectory** (Flash & Hogan 1985, Macfarlane & Croft 2003):
```
x(t) = a₀ + a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵
a₃ = 10(xf-x₀)/T³,  a₄ = -15(xf-x₀)/T⁴,  a₅ = 6(xf-x₀)/T⁵
```

**Damped least-squares IK** (Buss 2009) converts Cartesian targets to joint angles without singularity blowup:
```
dq = Jᵀ(JJᵀ + λ²I)⁻¹ dx
```

**Null-space injection** (Liégeois 1977) lets wrist orientation control run without fighting the position task.

---

## Results

| Metric | No PID | With PID |
|---|---|---|
| Mean avg mix angle (°) | 34.41 | 14.19 |
| Mean max mix angle (°) | 43.80 | 42.09 |
| Mean integrated mix (°·s) | 96.42 | 39.75 |
| Mean avg EE speed (m/s) | 0.3726 | 0.4093 |

PID reduces integrated mixing score by ~59% while moving faster on average. It also resolves IK redundancy — for a given end-effector position, the null-space PID finds the joint configuration that best aligns the tube with the liquid.

---

## Replication

### Requirements
```bash
pip install mujoco numpy
mjpython main.py   # mjpython required on macOS Apple Silicon
```

Place your Franka Panda XML at `franka_emika_panda/scene.xml`. The script auto-generates a combined scene with the centrifuge tube injected.

### Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `LIQUID_TAU` | 1.0s | Liquid reorientation lag |
| `ACCEL_LPFILTER_ALPHA` | 0.03 | Acceleration smoothing |
| `USE_WRIST_PID` | True | Enable closed-loop mixing control |
| `wrist_pid kp` | 0.5 | Correction aggressiveness |
| `wrist_pid kd` | 0.1 | Wrist damping — increase if oscillating |

### Tuning the PID
Start with `ki=0, kd=0`. Increase `kp` until the arm visibly corrects wrist orientation during transport. Add `kd` if the wrist oscillates. Keep `ki` very small — the integral resets on every phase transition.

---

## Future Work

- **RL** — use reinforcement learning with a mixing penalty in the reward function to find optimal transport trajectories, combined with PID for fine-grained EE control
- **Real hardware** — transfer to a physical Franka Panda arm; primary challenge is noisy acceleration estimation and tube pickup calibration
- **Fluid simulation** — replace the proxy metric with particle-based simulation for ground-truth mixing measurement

---

## References

- Flash & Hogan (1985). The coordination of arm movements. *Journal of Neuroscience*, 5(7).
- Macfarlane & Croft (2003). Jerk-bounded manipulator trajectory planning. *IEEE T-RA*, 19(1).
- Buss (2009). Introduction to inverse kinematics with Jacobian transpose, pseudoinverse and damped least squares methods.
- Liégeois (1977). Automatic supervisory control of multibody mechanisms. *IEEE T-SMC*, 7(12).