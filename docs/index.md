
*Alan Ramirez, Zoey You, Sarah Huang, Sophia Zhao*

# 16299 Final Project: Centrifuge Tube Transporter

## Abstract

Automated handling of biological samples is increasingly common in modern chemistry and biology laboratories, but standard robotic motion planners are largely agnostic to the fluid dynamics inside the containers they transport. For samples such as density-gradient centrifuge tubes, the jerk, acceleration, and orientation profile of the end effector during transport directly determines whether carefully separated liquid layers remain intact or re-mix. In this project, we implement a feedback control pipeline on a simulated Franka Panda 7-DOF arm in MuJoCo that transports a centrifuge tube while keeping its long axis aligned with the effective acceleration vector experienced by the liquid, so that the tube tilts with the trajectory rather than against it. The pipeline combines minimum-jerk 5th-order-polynomial trajectory planning, damped least-squares inverse (differential) kinematics, and a wrist-orientation PID controller injected through the null space of the IK problem. We define a mix angle, the error between the tube's z-axis and a lagged effective acceleration vector, as both the evaluation metric and the PID error signal. Across repeated trials, enabling the wrist PID reduced the mean average mix angle from 34.4 deg to 14.2 deg and the time-integrated mix from 96.4 deg-s to 39.7 deg-s, with comparable end-effector tracking error. These results are also compared against a no-PID inverse kinematics solution considering both end effector position and orientation, still showing improvement compared to inverse differential kinematics alone. These results show that a feedback layer exploiting kinematic redundancy is potentially useful to substantially reduce liquid disturbance during transport without modifying the underlying trajectory planner.

## Motivation

Robotic automation is expanding into chemistry and biology laboratories, where manipulators are increasingly used for tasks that were previously performed by hand: pipetting, plate handling, sample transfer, and tube transport between instruments such as centrifuges, incubators, and analyzers. Commercial systems are being developed for these purposes constantly. The promise of this automation is throughput and reproducibility, but it rests on an assumption that the robot can move sample containers without disturbing what is inside them.
However, this can be problematic. A centrifuge tube containing two separated liquid layers is highly sensitive to the dynamics of how it is moved. Sharp accelerations, abrupt direction changes, and orientation errors during transport induce sloshing that can re-mix layers and destroy the work the centrifuge has just done. Standard motion planners optimize for kinematic objectives such as path length, smoothness in joint space, or end-effector tracking error, do not necessarily have consideration of fluid mechanics within the end-effector object, of the direction of the effective acceleration vector inside the tube, or of how tube orientation should evolve in response to that vector.
The goal of this project is to explore how feedback control can help develop these systems with those constraints in mind. We implement a feedback controller that runs on top of a conventional trajectory planner and continuously corrects the tube orientation so that the tube's long axis remains aligned with the direction of the effective acceleration the liquid experiences. Intuitively, the controller tries to make the tube behave from the liquid's perspective like a tube that is simply standing in gravity, even while the arm is moving along a trajectory. We evaluate this idea in simulation on a Franka Panda 7-DOF arm in MuJoCo, comparing transport with and without the PID layer.

---
## Methods: Kinematics

### Trajectory Generation

Transitions between the 7 waypoints (Hover, Descend, Grasp, Lift,
Transport, Place, Release) are controlled by a 5th-order minimum-jerk polynomial (see Macfarlane & Croft). For normalized time $\tau = t/T$ within each phase, the position interpolant is:

$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5$$

which enforces zero velocity and zero acceleration at phase boundaries. Overall execution speed is controlled by a tunable `time_scale` parameter that uniformly scales all phase
durations.

### Trajectory Following via Inverse Differential Kinematics

At each timestep, the Cartesian position error $\Delta\mathbf{x}$ between the
trajectory and the actual end-effector position is resolved into joint velocities
via damped least-squares (Buss 2004):

$$\Delta\mathbf{q} = \mathbf{J}^\top(\mathbf{J}\mathbf{J}^\top + \lambda^2\mathbf{I})^{-1}\Delta\mathbf{x}$$

The damping term $\lambda^2$ prevents joint velocity blowup near singular
configurations. The 7-DOF arm tracking a 3D position target leaves a 4-dimensional
null space, which is exploited to simultaneously pull the arm toward a safe nominal
posture $Q_\text{HOME}$ without disturbing end-effector tracking (Liégeois 1977):

$$\Delta\mathbf{q} = \Delta\mathbf{q}_\text{position} + (\mathbf{I} - \mathbf{J}^+\mathbf{J})\cdot k(Q_\text{HOME} - \mathbf{q})$$

The resulting desired joint positions are then tracked by a joint-level PD controller
that converts position and velocity errors into motor torques.

## Methods : Partial PID

### Mixing Metric

We found great troubles in trying to simulate liquids in our environment. It is instead dealt with by defining a proxy metric:

- `tube_axis`: which way the tube is pointing (gripper Z-axis)
- `a_effective`: gravity + end-effector acceleration (what the liquid feels)
- `a_liquid`: `a_effective` with a 1-second lag, modeling liquid inertia
- `mix_angle`: angle between `tube_axis` and `a_liquid`

```
mix_angle = arccos(dot(tube_axis, a_liquid / |a_liquid|))   [degrees]
```

0° = tube perfectly aligned with liquid settlement = no mixing risk. The integrated mixing score `∫ mix_angle dt` (°·s) is the primary benchmark.

---

## System Architecture
![system architecture](sys_arch_tube.png)

**Three layers:**

1. **Trajectory Planner**: 5th-order minimum-jerk polynomial per phase. Zero velocity and acceleration at endpoints minimizes jerk. Phases: Hover → Descend → Grasp → Lift → Transport → Place → Release.

2. **Wrist Orientation PID**: closed-loop controller on `mix_angle`. Correction projected into Jacobian null-space to rotate the wrist toward alignment without disturbing end-effector position.

3. **Joint PID**: custom torque controller overriding MuJoCo's default gains: `τ = Kp*(q_des - q) - Kd*q̇`


### Implementation Choices

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

<div style="display: flex; gap: 20px; justify-content: center;">
  <div>
    <h3>Without PID/orientation tracking at all</h3>
    <img src="nopid.gif" width="400" />
  </div>
  <div>
    <h3>With regular IDK (position + orientation)</h3>
    <img src="ik.gif" width="400" />
  </div>
  <div>
    <h3>With Wrist Orientation PID feedback loop</h3>
    <img src="pid.gif" width="400" />
  </div>
</div>

---

Linearly interpolated lag (`LIQUID_TAU`) from 0.0 to 2.0 seconds across 50 trials each with and without wrist orientation PID control.

| Metric                        | No PID/orientation control | With Wrist PID |
| ----------------------------- | ------------ | -------------- |
| Mean avg EE speed (m/s)       | 0.3292       | 0.3648         |
| Mean max EE speed (m/s)       | 1.3071       | 1.4695         |
| Mean avg pos error (m)        | 0.0668       | 0.0783         |
| Mean max pos error (m)        | 0.3067       | 0.3480         |
| **Mean avg mix angle (°)**    | **34.8832**  | **14.8218**    |
| **Mean max mix angle (°)**    | **44.6225**  | **42.1592**    |
| **Mean integrated mix (°·s)** | **115.1844** | **48.9414**    |

---

## Replication

### Requirements

Set up the environment: 
```
conda create -n 16299_final_project python=3.10
conda activate 16299_final_project
pip install -r requirements.txt
```

Place your Franka Panda XML at `franka_emika_panda/scene.xml`. The script should auto-generate a combined scene with the centrifuge tube injected.

### Relevant Files

- **`test_picking.py`**: Interactive simulator with real-time viewer. Runs one trial with configurable LIQUID_TAU and wrist PID gains.
- **`record_test_picking.py`**: Offline video recorder. Generates MP4 of trajectory with custom camera angles for visualization.
- **`trial_runs.py`**: Does a linear interp. over LIQUID_TAU (0.0→2.0s). Runs multiple trials and saves stats to CSV.
- **`kinematics.py`**: Runs regular 6 element IDK as a baseline.


### Key Parameters

| Parameter | Default | Effect |
|---|---|---|
| `LIQUID_TAU` | 1.0s | Liquid reorientation lag |
| `ACCEL_LPFILTER_ALPHA` | 0.03 | Acceleration smoothing |
| `USE_WRIST_PID` | True | Enable closed-loop mixing control |
| `wrist_pid kp` | 0.5 | Correction aggressiveness |
| `wrist_pid kd` | 0.1 | Wrist damping: increase if oscillating |

---

## Reflections

### Changes Since Initial Presentation

Considerations arose during initial presentations of this project to the class, where suggestions were made to not reduce the arm's DOF from 7 to 6. We chose to, instead of replacing the wrist entirely, blend existing inverse differential kinematics with information (injected into the null space) from the other PID loop. We additionally added a comparison to a full 6-element vector (position and orientation instead of just position) goal to compare against pure inverse kinematics with no separate PID in response to these suggestions.

### Iteration Process

The general process involved identifying the problem (for the project proposal), deciding where feedback control could assist in solving the problem, identifying a simulation environment, and then designing the implementation of how error for the PID loop is calculated. 

### What Worked

The idea of adding a separate PID feedback loop to inject some information about desired wrist position proved to improve mixing score by quite a lot. This was the crux of our project, and it performed better (even if marginally) than using full inverse kinematics with orientation and position. 

Getting the Mujoco simulation to have the robot follow trajectories worked well.

### What Didn't Work

Lots of effort went into deciding 1) where to use feedback control in the project and 2) how to define liquid mixing. 

For the first point, choosing to apply a separate PID loop for the wrist angle was decided after realizing using PID to "minimize jerk" was hard to accomplish. We found that trying to adjust all the joints' PID controllers to "minimize jerk" made it really difficult to have stable movement. Thus, we relied on the trajectory planning following the Macfarlane & Croft paper to minimize jerk overall, and decided to make the PID more granular. 

For the second point, this is still an ongoing question. We decided to go with our metric because it was simpler to implement and allowed for a clear error signal for PID to correct. Having the liquid "lag" also introduced real-world inconsistencies as well. However, this representation is something that can be continually worked on and improved, as it is not necessarily the best representation of fluid physics.

---
## Conclusion

We find that using PID to control tuble angle can be a potentially useful addition to laboratory robots. Adding a PID layer for the wrist joint specifically that is injected into the null space of regular pose can yield less liquid mixing (according to metrics defined by angle of the tube and liquid) while simultaneously not affecting the PID of the joints themselves for trajectory following. These results imply that considering centrifuge tube transport could be a well-scoped task for robot manipulators even though these payloads can be sensitive. 

---

## References

## References

- Macfarlane, S., & Croft, E. A. (2003). [Jerk-bounded manipulator trajectory
  planning: Design for real-time applications](https://www.researchgate.net/publication/3299311_Jerk-bounded_manipulator_trajectory_planning_Design_for_real-time_applications).
- Buss, S. R. (2004). [Introduction to inverse kinematics with Jacobian transpose,
  pseudoinverse and damped least squares methods](https://www.cs.cmu.edu/~15464-s13/lectures/lecture6/iksurvey.pdf).

- Liégeois, A. (1977). [Automatic supervisory control of the configuration and
  behavior of multibody mechanisms](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=4309644).