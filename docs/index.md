
*Alan Ramirez, Zoey You, Sarah Huang, Sophia Zhao*

# 16299 Final Project: Centrifuge Tube Transporter

## Abstract

This project focuses on implementing an algorithm for the transportation of centrifuge tubes using robotic arms. The system is  developed and tested in simulation using a Mujoco gym environment and a Franka arm library in Python, which allows for modeling the physics of the robotic arm. The implementation is evaluated against benchmarks that estimate the mixing of separated component after a centrifuge.

## Motivation

Biology labs increasingly rely on robotic automation, but standard motion planning ignores the physics of sensitive payloads. Moving a centrifuge tube too aggressively mixes its separated layers, ruining the sample. This project considers these constraints to transport the tube as fast as possible while keeping mixing as low as possible using active feedback control.



## Methods : Kinematics

### Control Architecture and Trajectory
The kinematics approach utilizes a dual-loop PID control architecture to track a predefined sequence of spatial waypoints (Hover, Descend, Grasp, Lift, Transport, Place, Release). 

* **Trajectory Generation**: To prevent sudden acceleration spikes that would disturb the payload, transitions between waypoints are governed by a 5th-order minimum-jerk polynomial. For a normalized time $\tau = t/\text{duration}$, the position scaling $s$ is calculated as:
    $$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$
    The overall execution speed of these phases is parameterized by a tunable `time_scale`.
* **Task-Space and Joint-Space Control**: A Task-Space PID controller minimizes the Cartesian error between the minimum-jerk trajectory and the end-effector's actual position. This 3D correction is mapped into the 7-DOF joint space using a damped pseudo-inverse Jacobian. This is combined with a null-space projection to keep the robot near a safe home posture ($Q_{\text{HOME}}$) without disrupting the end-effector task. Finally, a Joint-Space PD controller computes the necessary motor torques.

### Active Slosh Compensation
To minimize liquid mixing, the system mathematically models the internal liquid surface by tracking the **effective gravity vector**. This vector accounts for both standard gravity and the low-pass filtered acceleration of the end-effector, simulating the delayed sloshing motion of a viscous fluid.

If active wrist compensation (`use_wrist_pid`) is enabled, the controller computes the cross product between the actual tube's Z-axis and the effective gravity vector. This rotational error is mapped directly to the wrist joints via the Jacobian transpose, allowing the robot to actively tilt the tube into curves to counteract lateral acceleration and keep the liquid surface flat relative to the tube opening.

### Optimization Methodology
To find the optimal balance between speed and stability, the system uses an automated grid sweep (`run_sweep`). It systematically evaluates combinations of PID gains (`kp_scale`, `kd_scale`, `task_kp`) and execution speeds (`time_scale`). Each configuration is evaluated using a weighted objective function:

$$\text{Score} = w_{\text{tilt}} \cdot \text{Tilt} + w_{\text{reach}} \cdot \text{Reach} + w_{\text{risk}} \cdot \text{Risk} + w_{\text{time}} \cdot \text{Time}$$

Lower scores indicate superior performance. The algorithm heavily penalizes liquid tilt and path deviation while rewarding faster simulation completion times.

## Files

- **`test_picking.py`**: Interactive simulator with real-time viewer. Runs one trial with configurable LIQUID_TAU and wrist PID gains.
- **`record_test_picking.py`**: Offline video recorder. Generates MP4 of trajectory with custom camera angles for visualization.
- **`trial_runs.py`**: Batch sweep over LIQUID_TAU (0.0→2.0s). Runs multiple trials and saves stats to CSV.
- **`kinematics.py`**: Automated grid sweep and trajectory simulation focusing on minimum-jerk motion and active slosh compensation.



## Methods : Partial PID

### Mixing Metric

Liquid simulation is tricky. It is instead dealt with by defining a proxy metric:

- `tube_axis`: which way the tube is pointing (gripper Z-axis)
- `a_effective`: gravity + end-effector acceleration (what the liquid feels)
- `a_liquid`: `a_effective` with a 1-second lag, modeling liquid inertia
- `mix_angle`: angle between `tube_axis` and `a_liquid`

```
mix_angle = arccos(dot(tube_axis, a_liquid / |a_liquid|))   [degrees]
```

0° = tube perfectly aligned with liquid settlement = no mixing risk. The integrated mixing score `∫ mix_angle dt` (°·s) is the primary benchmark.

---

### System Architecture
![system architecture](sys_arch_tube.png)

**Three layers:**

1. **Trajectory Planner**: 5th-order minimum-jerk polynomial per phase. Zero velocity and acceleration at endpoints minimizes jerk. Phases: Hover → Descend → Grasp → Lift → Transport → Place → Release.

2. **Wrist Orientation PID**: closed-loop controller on `mix_angle`. Correction projected into Jacobian null-space to rotate the wrist toward alignment without disturbing end-effector position.

3. **Joint PID**: custom torque controller overriding MuJoCo's default gains: `τ = Kp*(q_des - q) - Kd*q̇`

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


## Files

- **`test_picking.py`**: Interactive simulator with real-time viewer. Runs one trial with configurable LIQUID_TAU and wrist PID gains.
- **`record_test_picking.py`**: Offline video recorder. Generates MP4 of trajectory with custom camera angles for visualization.
- **`trial_runs.py`**: Batch sweep over LIQUID_TAU (0.0→2.0s). Runs multiple trials and saves stats to CSV.

---

## Results

<div style="display: flex; gap: 20px; justify-content: center; flex-wrap: wrap;">
  <div>
    <h3>No PID Control</h3>
    <p><i>[Insert kinematics_no_pid.gif]</i></p>
  </div>
  <div>
    <h3>Joint PD Only</h3>
    <p><i>[Insert kinematics_joint_pd_only.gif]</i></p>
  </div>
  <div>
    <h3>Full Body Only (No Wrist)</h3>
    <p><i>[Insert kinematics_full_body_only.gif]</i></p>
  </div>
  <div>
    <h3>Full Body + Wrist Compensation</h3>
    <p><i>[Insert kinematics_full_body.gif]</i></p>
  </div>
</div>


<div style="display: flex; gap: 20px; justify-content: center;">
  <div>
    <h3>Without PID</h3>
    <img src="no_pid_simulation(2).gif" width="400" />
  </div>
  <div>
    <h3>With Wrist Orientation PID</h3>
    <img src="pid_simulation(1).gif" width="400" />
  </div>
</div>

---

Linearly interpolated lag (`LIQUID_TAU`) from 0.0 to 2.0 seconds across 50 trials each with and without wrist orientation PID control.

| Metric | No Wrist PID | With Wrist PID |
|---|---|---|
| Mean avg EE speed (m/s) | 0.3726 | 0.4076 |
| Mean max EE speed (m/s) | 1.2977 | 1.4255 |
| Mean avg pos error (m) | 0.0781 | 0.0843 |
| Mean max pos error (m) | 0.3246 | 0.3338 |
| **Mean avg mix angle (°)** | **34.41** | **14.09** |
| **Mean max mix angle (°)** | **43.80** | **41.51** |
| **Mean integrated mix (°·s)** | **96.42** | **39.49** |

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
| `wrist_pid kd` | 0.1 | Wrist damping: increase if oscillating |

### Tuning the PID
Start with `ki=0, kd=0`. Increase `kp` until the arm visibly corrects wrist orientation during transport. Add `kd` if the wrist oscillates. 

---

## Future Work

- **RL**: use reinforcement learning with a mixing penalty in the reward function to find optimal transport trajectories, combined with PID for fine-grained EE control
- **Real hardware**: transfer to a physical Franka Panda arm; primary challenge is noisy acceleration estimation and tube pickup calibration
- **Fluid simulation**: replace the proxy metric with particle-based simulation for ground-truth mixing measurement

---

## References

- Flash & Hogan (1985). The coordination of arm movements. *Journal of Neuroscience*, 5(7).
- Macfarlane & Croft (2003). Jerk-bounded manipulator trajectory planning. *IEEE T-RA*, 19(1).
- Buss (2009). Introduction to inverse kinematics with Jacobian transpose, pseudoinverse and damped least squares methods.
- Liégeois (1977). Automatic supervisory control of multibody mechanisms. *IEEE T-SMC*, 7(12).