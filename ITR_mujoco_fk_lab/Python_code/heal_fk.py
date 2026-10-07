import mujoco
import mujoco.viewer
import numpy as np
import os
import time


# ============================================================
# PATH TO HEAL MODEL
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

XML_PATH = os.path.join(
    BASE_DIR,
    "robot_descriptions",
    "single_arm_heal_effort_actuation_rs_mj.xml"
)

print("\nLoading HEAL model:")
print(XML_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# ============================================================
# PRINT ALL JOINTS
# ============================================================

print("\n========================================")
print("HEAL JOINTS")
print("========================================")

for i in range(model.njnt):

    joint_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        i
    )

    joint_type = model.jnt_type[i]
    qpos_index = model.jnt_qposadr[i]

    print(
        f"{i}: {joint_name}"
        f" | type = {joint_type}"
        f" | qpos index = {qpos_index}"
    )

print("========================================")


# ============================================================
# PRINT ALL BODIES
# ============================================================

print("\n========================================")
print("HEAL BODIES")
print("========================================")

for i in range(model.nbody):

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        i
    )

    print(f"{i}: {body_name}")

print("========================================")


# ============================================================
# GET HINGE JOINTS
# ============================================================

hinge_joints = []

for i in range(model.njnt):

    if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:

        joint_name = mujoco.mj_id2name(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            i
        )

        hinge_joints.append({
            "id": i,
            "name": joint_name,
            "qpos": model.jnt_qposadr[i]
        })


print("\nNumber of hinge joints:", len(hinge_joints))


# ============================================================
# SET JOINT ANGLES
# ============================================================

def set_joint_angles(angles_deg):

    if len(angles_deg) != len(hinge_joints):

        raise ValueError(
            f"Expected {len(hinge_joints)} joint angles, "
            f"but received {len(angles_deg)}."
        )

    # Degrees → radians
    angles_rad = np.deg2rad(angles_deg)

    for i, joint in enumerate(hinge_joints):

        data.qpos[joint["qpos"]] = angles_rad[i]

    # Recalculate forward kinematics
    mujoco.mj_forward(model, data)


# ============================================================
# CHOOSE JOINT ANGLES HERE
# ============================================================

angles = [
    0,
    30,
    0,
    25,
    0,
    0
]


# ============================================================
# APPLY ANGLES
# ============================================================

set_joint_angles(angles)


# ============================================================
# PRINT JOINT ANGLES
# ============================================================

print("\n========================================")
print("CURRENT JOINT ANGLES")
print("========================================")

for i, joint in enumerate(hinge_joints):

    q = data.qpos[joint["qpos"]]

    print(
        f"{joint['name']:20s} : "
        f"{np.rad2deg(q):8.3f} deg"
    )


# ============================================================
# PRINT BODY POSITIONS
# ============================================================

print("\n========================================")
print("BODY POSITIONS")
print("========================================")

for body_id in range(model.nbody):

    body_name = mujoco.mj_id2name(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_id
    )

    pos = data.xpos[body_id]

    print(
        f"{body_id:2d} "
        f"{body_name:25s} "
        f"({pos[0]: .4f}, "
        f"{pos[1]: .4f}, "
        f"{pos[2]: .4f})"
    )


# ============================================================
# OPEN MUJOCO VIEWER
# ============================================================

print("\nOpening MuJoCo viewer...")
print("Close the window to exit.")

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        viewer.sync()

        time.sleep(0.01)