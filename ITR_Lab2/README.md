# Introduction to Robotics — Lab 2

## Forward Kinematics using MuJoCo

This repository contains the implementation for Lab 2 of the Introduction to Robotics course.

The objective of this lab is to simulate robotic manipulators in MuJoCo, provide joint angles as input, and obtain the end-effector position using forward kinematics.

## Robots

The following robots were implemented:

- **HEAL** — 6-DOF robotic manipulator
- **Franka Emika Panda** — 7-DOF robotic manipulator

## Implementation

For both robots, the Python scripts:

1. Load the corresponding MuJoCo model.
2. Identify the robot joints.
3. Accept joint angles in degrees.
4. Convert the joint angles to radians.
5. Set the corresponding joint positions in the MuJoCo model.
6. Perform forward kinematics using MuJoCo.
7. Obtain and display the end-effector position.
8. Open the MuJoCo visualization window.

## Project Structure

```text
ITR_Lab2/
├── Python_code/
│   ├── heal_fk.py
│   └── franka_fk.py
│
├── Robot_descriptions/
│   ├── single_arm_heal_effort_actuation_rs.xml
│   ├── heal_meshes/
│   └── Franka/
│       ├── panda.xml
│       ├── scene.xml
│       └── assets/
│
├── requirements.txt
├── environment.yml
└── README.md
