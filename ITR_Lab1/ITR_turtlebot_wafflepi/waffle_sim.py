import os
import time
import numpy as np
import glfw
import mujoco


# ==========================================================
# 1. LOAD TURTLEBOT3 WAFFLE PI
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "robotis_mujoco_menagerie",
    "robotis_tb3",
    "scene_turtlebot3_waffle_pi.xml"
)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

print("Loaded TurtleBot3 Waffle Pi")


# ==========================================================
# 2. FIND ACTUAL JOINTS
# ==========================================================

base_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "base_joint"
)

left_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "wheel_left"
)

right_joint = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_JOINT,
    "wheel_right"
)

base_qpos = model.jnt_qposadr[base_joint]
left_qpos = model.jnt_qposadr[left_joint]
right_qpos = model.jnt_qposadr[right_joint]

print("base qpos index :", base_qpos)
print("left wheel index :", left_qpos)
print("right wheel index:", right_qpos)


# ==========================================================
# 3. DIFFERENTIAL DRIVE PARAMETERS
# ==========================================================

WHEEL_RADIUS = 0.033      # m
WHEEL_BASE = 0.288       # m

LINEAR_SPEED = 0.15      # m/s
ANGULAR_SPEED = 0.8      # rad/s

DT = 0.01


# ==========================================================
# 4. ROBOT COMMANDS
# ==========================================================

v_cmd = 0.0
omega_cmd = 0.0


# ==========================================================
# 5. INITIAL POSE
# ==========================================================

initial_qpos = data.qpos.copy()

theta = 0.0


# ==========================================================
# 6. DIFFERENTIAL DRIVE
# ==========================================================

def differential_drive(v, omega):

    # Wheel linear velocities
    v_left = v - (omega * WHEEL_BASE / 2.0)
    v_right = v + (omega * WHEEL_BASE / 2.0)

    # Convert to wheel angular velocity
    omega_left = v_left / WHEEL_RADIUS
    omega_right = v_right / WHEEL_RADIUS

    return omega_left, omega_right


# ==========================================================
# 7. GLFW KEY STATE
# ==========================================================

def reset_robot():

    global theta
    global v_cmd
    global omega_cmd

    data.qpos[:] = initial_qpos
    data.qvel[:] = 0

    theta = 0.0

    v_cmd = 0.0
    omega_cmd = 0.0

    mujoco.mj_forward(model, data)


# ==========================================================
# 8. AXIS DRAWING
# ==========================================================

def add_arrow(scene, start, end, color):

    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]

    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.array([0.012, 0.012, 0.012]),
        np.array(start),
        np.eye(3).flatten(),
        np.array(color)
    )

    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.012,
        np.array(start),
        np.array(end)
    )

    scene.ngeom += 1


def draw_frames(scene):

    # Existing robot/environment geometry count is already
    # present after mjv_updateScene().
    original_ngeom = scene.ngeom

    L = 0.20

    # ------------------------------------------------------
    # WORLD FRAME
    # ------------------------------------------------------

    world_origin = np.array([0.0, 0.0, 0.02])

    # X = RED
    add_arrow(
        scene,
        world_origin,
        world_origin + np.array([L, 0.0, 0.0]),
        [1.0, 0.0, 0.0, 1.0]
    )

    # Y = GREEN
    add_arrow(
        scene,
        world_origin,
        world_origin + np.array([0.0, L, 0.0]),
        [0.0, 1.0, 0.0, 1.0]
    )

    # Z = BLUE
    add_arrow(
        scene,
        world_origin,
        world_origin + np.array([0.0, 0.0, L]),
        [0.0, 0.0, 1.0, 1.0]
    )

    # ------------------------------------------------------
    # BODY FRAME
    # ------------------------------------------------------

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base"
    )

    position = data.xpos[body_id].copy()

    R = data.xmat[body_id].reshape(3, 3)

    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    # Body X = RED
    add_arrow(
        scene,
        position,
        position + L * x_axis,
        [1.0, 0.0, 0.0, 1.0]
    )

    # Body Y = GREEN
    add_arrow(
        scene,
        position,
        position + L * y_axis,
        [0.0, 1.0, 0.0, 1.0]
    )

    # Body Z = BLUE
    add_arrow(
        scene,
        position,
        position + L * z_axis,
        [0.0, 0.0, 1.0, 1.0]
    )


# ==========================================================
# 9. ROTATION MATRIX TEXT
# ==========================================================

def matrix_string():

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base"
    )

    R = data.xmat[body_id].reshape(3, 3)

    return (
        "BODY ROTATION MATRIX\n"
        f"[ {R[0,0]: .3f}  {R[0,1]: .3f}  {R[0,2]: .3f} ]\n"
        f"[ {R[1,0]: .3f}  {R[1,1]: .3f}  {R[1,2]: .3f} ]\n"
        f"[ {R[2,0]: .3f}  {R[2,1]: .3f}  {R[2,2]: .3f} ]"
    )


# ==========================================================
# 10. GLFW INITIALIZATION
# ==========================================================

if not glfw.init():
    raise RuntimeError("Could not initialize GLFW")

window = glfw.create_window(
    1200,
    800,
    "MuJoCo - TurtleBot3 Waffle Pi",
    None,
    None
)

if not window:
    glfw.terminate()
    raise RuntimeError("Could not create GLFW window")

glfw.make_context_current(window)
glfw.swap_interval(1)


# ==========================================================
# 11. MUJOCO VISUALIZATION
# ==========================================================

cam = mujoco.MjvCamera()
opt = mujoco.MjvOption()
scn = mujoco.MjvScene(model, 2000)
ctx = mujoco.MjrContext(
    model,
    mujoco.mjtFontScale.mjFONTSCALE_150
)

mujoco.mjv_defaultCamera(cam)
mujoco.mjv_defaultOption(opt)

# Initial camera
cam.azimuth = 90
cam.elevation = -25
cam.distance = 2.5
cam.lookat[:] = [0.0, 0.0, 0.0]


# ==========================================================
# 12. MAIN LOOP
# ==========================================================

previous_enter = False

print()
print("==========================================")
print("TURTLEBOT3 WAFFLE PI CONTROLS")
print("==========================================")
print("UP ARROW     = Forward")
print("DOWN ARROW   = Backward")
print("LEFT ARROW   = Rotate Left")
print("RIGHT ARROW  = Rotate Right")
print("SPACE        = Stop")
print("ENTER        = Reset")
print("ESC          = Quit")
print("==========================================")


while not glfw.window_should_close(window):

    frame_start = time.time()

    glfw.poll_events()

    # ======================================================
    # KEYBOARD
    # ======================================================

    up = glfw.get_key(window, glfw.KEY_UP) == glfw.PRESS
    down = glfw.get_key(window, glfw.KEY_DOWN) == glfw.PRESS
    left = glfw.get_key(window, glfw.KEY_LEFT) == glfw.PRESS
    right = glfw.get_key(window, glfw.KEY_RIGHT) == glfw.PRESS
    space = glfw.get_key(window, glfw.KEY_SPACE) == glfw.PRESS
    enter = glfw.get_key(window, glfw.KEY_ENTER) == glfw.PRESS
    escape = glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS

    # ------------------------------------------------------
    # Reset only once when ENTER is pressed
    # ------------------------------------------------------

    if enter and not previous_enter:
        reset_robot()

    previous_enter = enter

    # ------------------------------------------------------
    # Quit
    # ------------------------------------------------------

    if escape:
        break

    # ------------------------------------------------------
    # Commands
    # ------------------------------------------------------

    if space:
        v_cmd = 0.0
        omega_cmd = 0.0

    elif up:
        v_cmd = LINEAR_SPEED
        omega_cmd = 0.0

    elif down:
        v_cmd = -LINEAR_SPEED
        omega_cmd = 0.0

    elif left:
        v_cmd = 0.0
        omega_cmd = ANGULAR_SPEED

    elif right:
        v_cmd = 0.0
        omega_cmd = -ANGULAR_SPEED

    else:
        v_cmd = 0.0
        omega_cmd = 0.0


    # ======================================================
    # DIFFERENTIAL DRIVE CALCULATION
    # ======================================================

    wheel_left, wheel_right = differential_drive(
        v_cmd,
        omega_cmd
    )


    # ======================================================
    # ROBOT KINEMATICS
    # ======================================================

    x = data.qpos[base_qpos + 0]
    y = data.qpos[base_qpos + 1]

    # Robot orientation
    theta += omega_cmd * DT

    # Position in WORLD FRAME
    x += v_cmd * np.cos(theta) * DT
    y += v_cmd * np.sin(theta) * DT


    # ======================================================
    # UPDATE FREE BASE JOINT
    # ======================================================

    data.qpos[base_qpos + 0] = x
    data.qpos[base_qpos + 1] = y

    # Keep original Z height
    data.qpos[base_qpos + 2] = initial_qpos[base_qpos + 2]

    # Quaternion: [w, x, y, z]
    data.qpos[base_qpos + 3] = np.cos(theta / 2.0)
    data.qpos[base_qpos + 4] = 0.0
    data.qpos[base_qpos + 5] = 0.0
    data.qpos[base_qpos + 6] = np.sin(theta / 2.0)


    # ======================================================
    # ROTATE WHEELS VISUALLY
    # ======================================================

    data.qpos[left_qpos] += wheel_left * DT
    data.qpos[right_qpos] += wheel_right * DT


    # ======================================================
    # UPDATE MUJOCO
    # ======================================================

    mujoco.mj_forward(model, data)


    # ======================================================
    # RENDER
    # ======================================================

    width, height = glfw.get_framebuffer_size(window)

    viewport = mujoco.MjrRect(
        0,
        0,
        width,
        height
    )

    mujoco.mjr_setBuffer(
        mujoco.mjtFramebuffer.mjFB_WINDOW,
        ctx
    )

    # Build scene from current MuJoCo state
    mujoco.mjv_updateScene(
        model,
        data,
        opt,
        None,
        cam,
        mujoco.mjtCatBit.mjCAT_ALL,
        scn
    )

    # Add coordinate frames
    draw_frames(scn)

    # Render
    mujoco.mjr_render(
        viewport,
        scn,
        ctx
    )


    # ======================================================
    # MATRIX OVERLAY
    # ======================================================

    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "base"
    )

    R = data.xmat[body_id].reshape(3, 3)

    text1 = (
        f"BODY ROTATION MATRIX\n"
        f"[ {R[0,0]: .3f}  {R[0,1]: .3f}  {R[0,2]: .3f} ]\n"
        f"[ {R[1,0]: .3f}  {R[1,1]: .3f}  {R[1,2]: .3f} ]\n"
        f"[ {R[2,0]: .3f}  {R[2,1]: .3f}  {R[2,2]: .3f} ]"
    )

    text2 = (
        "UP/DOWN = MOVE    LEFT/RIGHT = ROTATE\n"
        "SPACE = STOP      ENTER = RESET"
    )

    mujoco.mjr_overlay(
        mujoco.mjtFont.mjFONT_NORMAL,
        mujoco.mjtGridPos.mjGRID_TOPLEFT,
        viewport,
        text1,
        text2,
        ctx
    )


    # ======================================================
    # DISPLAY
    # ======================================================

    glfw.swap_buffers(window)

    elapsed = time.time() - frame_start

    if elapsed < DT:
        time.sleep(DT - elapsed)


# ==========================================================
# EXIT
# ==========================================================

ctx.free()
scn.free()

# Do not call glfw.terminate() here on this Linux/NVIDIA setup.