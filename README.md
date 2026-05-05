# README for 16299 Final Project

## WEBSITE
https://violetwesting08.github.io/16299_final_project_S26/

## Abstract

Automated handling of biological samples is increasingly common in modern chemistry and biology laboratories, but standard robotic motion planners are largely agnostic to the fluid dynamics inside the containers they transport. For samples such as density-gradient centrifuge tubes, the jerk, acceleration, and orientation profile of the end effector during transport directly determines whether carefully separated liquid layers remain intact or re-mix. In this project, we implement a feedback control pipeline on a simulated Franka Panda 7-DOF arm in MuJoCo that transports a centrifuge tube while keeping its long axis aligned with the effective acceleration vector experienced by the liquid, so that the tube tilts with the trajectory rather than against it. The pipeline combines minimum-jerk 5th-order-polynomial trajectory planning, damped least-squares inverse (differential) kinematics, and a wrist-orientation PID controller injected through the null space of the IK problem. We define a mix angle, the error between the tube's z-axis and a lagged effective acceleration vector, as both the evaluation metric and the PID error signal. Across repeated trials, enabling the wrist PID reduced the mean average mix angle from 34.4 deg to 14.2 deg and the time-integrated mix from 96.4 deg-s to 39.7 deg-s, with comparable end-effector tracking error. These results are also compared against a no-PID inverse kinematics solution considering both end effector position and orientation, still showing improvement compared to inverse differential kinematics alone. These results show that a feedback layer exploiting kinematic redundancy is potentially useful to substantially reduce liquid disturbance during transport without modifying the underlying trajectory planner.

## Motivation

Robotic automation is expanding into chemistry and biology laboratories, where manipulators are increasingly used for tasks that were previously performed by hand: pipetting, plate handling, sample transfer, and tube transport between instruments such as centrifuges, incubators, and analyzers. Commercial systems are being developed for these purposes constantly. The promise of this automation is throughput and reproducibility, but it rests on an assumption that the robot can move sample containers without disturbing what is inside them.
However, this can be problematic. A centrifuge tube containing two separated liquid layers is highly sensitive to the dynamics of how it is moved. Sharp accelerations, abrupt direction changes, and orientation errors during transport induce sloshing that can re-mix layers and destroy the work the centrifuge has just done. Standard motion planners optimize for kinematic objectives such as path length, smoothness in joint space, or end-effector tracking error, do not necessarily have consideration of fluid mechanics within the end-effector object, of the direction of the effective acceleration vector inside the tube, or of how tube orientation should evolve in response to that vector.
The goal of this project is to explore how feedback control can help develop these systems with those constraints in mind. We implement a feedback controller that runs on top of a conventional trajectory planner and continuously corrects the tube orientation so that the tube's long axis remains aligned with the direction of the effective acceleration the liquid experiences. Intuitively, the controller tries to make the tube behave from the liquid's perspective like a tube that is simply standing in gravity, even while the arm is moving along a trajectory. We evaluate this idea in simulation on a Franka Panda 7-DOF arm in MuJoCo, comparing transport with and without the PID layer.

## Running the code
```
Set up the environment: 
conda create -n 16299_final_project python=3.10
conda activate 16299_final_project
pip install -r requirements.txt
```

## Results and Conclusion

Detailed results can be seen at the project website above. 
