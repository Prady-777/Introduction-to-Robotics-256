import mujoco
import mujoco.viewer
import numpy as np
import os
import time


# ============================================================
# FRANKA MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XML_PATH = os.path.join(
    BASE_DIR,
    "robot_descriptions",
    "franka",
    "scene.xml"
)

print("\nLoading Franka model:")
print(XML_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# ============================================================
# ALIGN FRANKA BASE WITH FLOOR
# ============================================================

# First calculate the initial geometry positions
mujoco.mj_forward(model, data)


# Find all robot root bodies.
# A root body has the world body (body 0) as its parent.

root_bodies = []

for body_id in range(1, model.nbody):

    if model.body_parentid[body_id] == 0:

        body_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id
        )

        root_bodies.append(body_id)

print("\n===================================")
print("ROOT BODIES")
print("===================================")

for body_id in root_bodies:

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )

    print(
        f"{body_id}: {body_name}"
    )

print("===================================")


# In this scene the robot should be the root body.
# We calculate the lowest geometry belonging to
# the robot, excluding the floor.

robot_geom_z = []

for geom_id in range(model.ngeom):

    geom_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        geom_id
    )

    # Ignore the scene floor
    if geom_name == "floor":
        continue

    body_id = model.geom_bodyid[geom_id]

    # Ignore world-body geometry
    if body_id == 0:
        continue

    z = data.geom_xpos[geom_id][2]

    robot_geom_z.append(z)


if not robot_geom_z:

    raise RuntimeError(
        "Could not find Franka robot geometry."
    )


lowest_z = min(robot_geom_z)

print("\nLowest Franka geometry:")
print(f"z = {lowest_z:.6f} m")


# ============================================================
# MOVE ENTIRE FRANKA ROBOT
# ============================================================

# If the lowest point is above z=0,
# shift the root body downward.

if lowest_z > 0:

    print(
        f"Moving Franka down by "
        f"{lowest_z:.6f} m"
    )

    for body_id in root_bodies:

        model.body_pos[body_id][2] -= lowest_z


# Recalculate the model after moving the robot
mujoco.mj_forward(model, data)


# ============================================================
# FIND HINGE JOINTS
# ============================================================

hinge_joints = []

for i in range(model.njnt):

    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:

        name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i
        )

        hinge_joints.append({
            "id": i,
            "name": name,
            "qpos": model.jnt_qposadr[i]
        })


# ============================================================
# PRINT JOINT INFORMATION
# ============================================================

print("\n===================================")
print("FRANKA JOINTS")
print("===================================")

for i, joint in enumerate(hinge_joints):

    print(
        f"J{i+1}: {joint['name']}"
    )

print("===================================")


# ============================================================
# PRINT BODIES
# ============================================================

print("\n===================================")
print("FRANKA BODIES")
print("===================================")

for i in range(model.nbody):

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    )

    print(
        f"{i}: {body_name}"
    )

print("===================================")


# ============================================================
# END-EFFECTOR
# ============================================================

EE_NAME = "hand"

ee_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    EE_NAME
)

if ee_id == -1:

    raise RuntimeError(
        f"Could not find body named '{EE_NAME}'."
    )


print("\nEnd-effector:")
print(f"Body ID   = {ee_id}")
print(f"Body name = {EE_NAME}")


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_kinematics(joint_angles_deg):

    """
    Calculate Franka end-effector position.

    Input:
        [q1,q2,q3,q4,q5,q6,q7]

    Angles are in degrees.

    Output:
        [x,y,z] in meters
    """

    if len(joint_angles_deg) != len(hinge_joints):

        raise ValueError(
            f"Expected {len(hinge_joints)} joint angles, "
            f"got {len(joint_angles_deg)}."
        )


    # Degrees → radians
    joint_angles_rad = np.deg2rad(
        joint_angles_deg
    )


    # Set joint positions
    for i, joint in enumerate(hinge_joints):

        data.qpos[joint["qpos"]] = (
            joint_angles_rad[i]
        )


    # Forward kinematics
    mujoco.mj_forward(
        model,
        data
    )


    # End-effector position
    position = data.xpos[ee_id].copy()

    return position


# ============================================================
# JOINT ANGLES
# ============================================================

q = [
    90,
    0,
    90,
    45,
    90,
    75,
    0
]


# ============================================================
# FORWARD KINEMATICS
# ============================================================

position = forward_kinematics(q)


# ============================================================
# PRINT RESULT
# ============================================================

print("\n===================================")
print("FORWARD KINEMATICS")
print("===================================")

print("\nJoint angles:")

for i, angle in enumerate(q):

    print(
        f"q{i+1} = {angle:8.3f}°"
    )


print("\nEnd-effector position:")

print(
    f"X = {position[0]: .6f} m"
)

print(
    f"Y = {position[1]: .6f} m"
)

print(
    f"Z = {position[2]: .6f} m"
)

print("===================================")


# ============================================================
# OPEN MUJOCO VIEWER
# ============================================================

print("\nOpening MuJoCo viewer...")
print("Close the window to exit.")

with mujoco.viewer.launch_passive(
    model,
    data
) as viewer:

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)