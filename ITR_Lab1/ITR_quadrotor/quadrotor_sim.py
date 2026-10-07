#!/usr/bin/env python3
"""
quadrotor_sim.py  --  ITR Lab 1 : Quadrotor 6-DOF pose, coordinate frames and
                      rotation matrix in MuJoCo (official `mujoco.viewer`).

Run:
    python3 quadrotor_sim.py              # interactive simulation
    python3 quadrotor_sim.py --selftest   # headless maths / model checks

Idea
----
The quadrotor is one rigid body ("base_link") with a FREE joint, so its whole
state is  qpos = [x y z | qw qx qy qz]  (position + unit quaternion, w first).
We do NOT simulate flight dynamics: keyboard commands edit the pose directly,
mj_forward() recomputes the kinematics, and the viewer shows the result.

Every frame we
    1. turn the quaternion into a 3x3 rotation matrix  R  (body -> world),
    2. draw the WORLD frame (identity axes at the origin),
    3. draw the BODY  frame (columns of R, starting at the quadrotor position),
    4. print R in the viewer overlay.

Controls  (letters are avoided on purpose: the MuJoCo viewer uses A-Z as its own shortcuts)
-------------------------------------------------------------------------------------------
    Up / Down        move +X / -X   (world axes)
    Left / Right     move +Y / -Y
    Page Up / Down   move +Z / -Z
    Home / End       yaw   +/-   (rotation about the BODY z axis)
    Insert / Delete  pitch +/-   (rotation about the BODY y axis)
    8 / 9            roll  +/-   (rotation about the BODY x axis)
    Enter            reset pose
    Esc or Q         quit
"""

import collections
import os
import sys
import time

import numpy as np
import mujoco
import mujoco.viewer

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(BASE_DIR, "quadrotor_scene.xml")   # works from any cwd

BODY_NAME = "base_link"
JOINT_NAME = "root"

MOVE_STEP = 0.10                 # metres per key press
ROT_STEP = np.deg2rad(10.0)      # radians per key press
Z_MIN = 0.05                     # keep the quadrotor above the floor

INIT_POS = np.array([0.0, 0.0, 0.5])
INIT_RPY = (0.0, 0.0, 0.0)       # roll, pitch, yaw  [rad]

WORLD_AXIS_LEN = 1.2             # length of the world-frame arrows  [m]
BODY_AXIS_LEN = 0.40             # length of the body-frame arrows   [m]
WORLD_AXIS_WIDTH = 0.010
BODY_AXIS_WIDTH = 0.013

# X = red, Y = green, Z = blue  (RGBA, float32 as MuJoCo expects)
AXIS_COLORS = (
    np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),   # X  red
    np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),   # Y  green
    np.array([0.0, 0.3, 1.0, 1.0], dtype=np.float32),   # Z  blue
)

# --------------------------------------------------------------------------- #
# Key map.
# IMPORTANT: in MuJoCo 3.x the viewer itself uses EVERY letter A-Z as a
# shortcut (W = wireframe, S = shadows, D = static bodies, J = joints, ...).
# Using WASD/IJKL would therefore also flip those display options, so we use
# keys the viewer does not use.  Values are GLFW key codes.
# action = (kind, axis_index, sign)    kind: "move" (world) | "rotate" (body)
# --------------------------------------------------------------------------- #
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_PAGE_UP, KEY_PAGE_DOWN = 266, 267
KEY_HOME, KEY_END, KEY_INSERT, KEY_DELETE = 268, 269, 260, 261
KEY_8, KEY_9 = 56, 57
KEY_ENTER, KEY_ESCAPE, KEY_Q = 257, 256, 81

ACTIONS = {
    KEY_UP:        ("move", 0, +1),
    KEY_DOWN:      ("move", 0, -1),
    KEY_LEFT:      ("move", 1, +1),
    KEY_RIGHT:     ("move", 1, -1),
    KEY_PAGE_UP:   ("move", 2, +1),
    KEY_PAGE_DOWN: ("move", 2, -1),
    KEY_HOME:      ("rotate", 2, +1),   # yaw
    KEY_END:       ("rotate", 2, -1),
    KEY_INSERT:    ("rotate", 1, +1),   # pitch
    KEY_DELETE:    ("rotate", 1, -1),
    KEY_8:         ("rotate", 0, +1),   # roll
    KEY_9:         ("rotate", 0, -1),
}

HELP_LEFT = (
    "Up/Down\n"
    "Left/Right\n"
    "PgUp/PgDn\n"
    "Home/End\n"
    "Insert/Delete\n"
    "8 / 9\n"
    "Enter\n"
    "Esc or Q"
)
HELP_RIGHT = (
    "move +X / -X\n"
    "move +Y / -Y\n"
    "move +Z / -Z\n"
    "yaw   +/-  (body z)\n"
    "pitch +/-  (body y)\n"
    "roll  +/-  (body x)\n"
    "reset pose\n"
    "quit"
)


# --------------------------------------------------------------------------- #
# Rotation maths  (MuJoCo quaternion convention: [w, x, y, z])
# --------------------------------------------------------------------------- #
def rpy_to_quat(roll, pitch, yaw):
    """Roll-pitch-yaw -> quaternion [w, x, y, z].

    Equivalent to  R = Rz(yaw) @ Ry(pitch) @ Rx(roll)  (verified by --selftest).
    """
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,    # w
        sr * cp * cy - cr * sp * sy,    # x
        cr * sp * cy + sr * cp * sy,    # y
        cr * cp * sy - sr * sp * cy,    # z
    ])


def quat_to_mat(quat):
    """Unit quaternion [w,x,y,z] -> 3x3 rotation matrix R (body -> world)."""
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    return R.reshape(3, 3)


def mat_to_rpy(R):
    """Inverse of rpy_to_quat for R = Rz(yaw) Ry(pitch) Rx(roll)  [rad]."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def rotate_about_body_axis(quat, axis_index, angle):
    """Rotate the body by `angle` about one of ITS OWN axes (x=0, y=1, z=2).

    q_new = q * dq  (quaternion product)  <=>  R_new = R @ R_delta
    Post-multiplication means the rotation is expressed in the body frame.
    """
    axis = np.zeros(3)
    axis[axis_index] = 1.0
    dq = np.zeros(4)
    mujoco.mju_axisAngle2Quat(dq, axis, angle)
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, quat, dq)
    mujoco.mju_normalize4(out)           # in place: keep it a unit quaternion
    return out


# --------------------------------------------------------------------------- #
# Drawing helpers (custom geometry in viewer.user_scn)
# --------------------------------------------------------------------------- #
def add_axis(scn, start, end, rgba, width):
    """Append one coordinate axis: a capsule shaft from `start` to `end`
    plus a small sphere at the tip (so the direction is easy to see).

    mjv_connector() builds a capsule that spans EXACTLY the two end points.
    (We avoid mjGEOM_ARROW: in MuJoCo 3.14 it renders only about half of the
    requested length, which would make the drawn axes misleading.)
    """
    if scn.ngeom + 2 > scn.maxgeom:
        return
    shaft = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(shaft, mujoco.mjtGeom.mjGEOM_CAPSULE,
                        np.zeros(3), np.zeros(3), np.zeros(9), rgba)
    mujoco.mjv_connector(shaft, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(start, dtype=np.float64),
                         np.asarray(end, dtype=np.float64))
    scn.ngeom += 1

    tip = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(tip, mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([2.4 * width, 0.0, 0.0]),
                        np.asarray(end, dtype=np.float64),
                        np.eye(3).flatten(), rgba)
    scn.ngeom += 1


def add_label(scn, position, text, rgba):
    """Append a floating 3D text label."""
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LABEL,
                        np.array([0.1, 0.0, 0.0]),
                        np.asarray(position, dtype=np.float64),
                        np.eye(3).flatten(), rgba)
    geom.label = text
    scn.ngeom += 1


def draw_frames(scn, body_pos, R):
    """Draw world frame (identity) and body frame (columns of R)."""
    scn.ngeom = 0                                   # clear previous frame
    names = ("X", "Y", "Z")
    for i in range(3):
        e_i = np.eye(3)[:, i]                       # unit axis i, e.g. [1,0,0]

        # World frame: origin at (0,0,0), axes are the unit vectors themselves.
        w_tip = WORLD_AXIS_LEN * e_i
        add_axis(scn, np.zeros(3), w_tip, AXIS_COLORS[i], WORLD_AXIS_WIDTH)
        add_label(scn, w_tip * 1.06, names[i], AXIS_COLORS[i])

        # Body frame: origin at the quadrotor, axis i expressed in the world
        # frame is  R @ e_i  == i-th COLUMN of R.
        b_axis_world = R @ e_i
        b_tip = body_pos + BODY_AXIS_LEN * b_axis_world
        add_axis(scn, body_pos, b_tip, AXIS_COLORS[i], BODY_AXIS_WIDTH)
        add_label(scn, body_pos + 1.12 * BODY_AXIS_LEN * b_axis_world,
                  names[i].lower() + "_b", AXIS_COLORS[i])


def clean(v, ndigits):
    """Round and drop the sign of negative zero so we never print '-0.000'."""
    return round(float(v), ndigits) + 0.0


def fmt_matrix_rows(R):
    return ["[ " + "  ".join(f"{clean(v, 3):+.3f}" for v in row) + " ]" for row in R]


def update_overlay(viewer, pos, R):
    """Show pose + rotation matrix inside the viewer window (official API)."""
    roll, pitch, yaw = np.rad2deg(mat_to_rpy(R))
    rows = fmt_matrix_rows(R)
    left = "Position x y z [m]\nRoll Pitch Yaw [deg]\nR  row 1\nR  row 2\nR  row 3"
    right = (f"{clean(pos[0], 2):+.2f}  {clean(pos[1], 2):+.2f}  {clean(pos[2], 2):+.2f}\n"
             f"{clean(roll, 1):+.1f}  {clean(pitch, 1):+.1f}  {clean(yaw, 1):+.1f}\n"
             f"{rows[0]}\n{rows[1]}\n{rows[2]}")
    viewer.set_texts([
        # (font scale, grid position, left column text, right column text)
        (mujoco.mjtFontScale.mjFONTSCALE_150, mujoco.mjtGridPos.mjGRID_TOPRIGHT,
         left, right),
        (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMRIGHT,
         HELP_LEFT, HELP_RIGHT),
    ])


# --------------------------------------------------------------------------- #
# Main simulation object
# --------------------------------------------------------------------------- #
class QuadrotorSim:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.data = mujoco.MjData(self.model)

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, JOINT_NAME)
        if self.body_id < 0 or joint_id < 0:
            raise RuntimeError(f"'{BODY_NAME}' / '{JOINT_NAME}' not found in {XML_PATH}")
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"joint '{JOINT_NAME}' is not a free joint")
        # qpos[adr : adr+7] = [x y z qw qx qy qz]
        self.qadr = int(self.model.jnt_qposadr[joint_id])

        self.pos = INIT_POS.copy()
        self.quat = rpy_to_quat(*INIT_RPY)

        self.pending = collections.deque(maxlen=256)   # key codes from the UI thread
        self.quit_requested = False

    # -- state -> MuJoCo -------------------------------------------------- #
    def write_pose_to_mujoco(self):
        self.data.qpos[self.qadr:self.qadr + 3] = self.pos
        self.data.qpos[self.qadr + 3:self.qadr + 7] = self.quat
        mujoco.mj_forward(self.model, self.data)       # update xpos / xmat / geoms

    def rotation_matrix(self):
        return quat_to_mat(self.quat)

    # -- keyboard --------------------------------------------------------- #
    def key_callback(self, keycode):
        """Runs in the viewer's UI thread: only queue the key, nothing else."""
        self.pending.append(keycode)

    def process_key(self, keycode):
        """Apply ONE queued key. Returns True if the pose changed."""
        if keycode in (KEY_ESCAPE, KEY_Q):
            self.quit_requested = True
            return False
        if keycode == KEY_ENTER:
            self.pos = INIT_POS.copy()
            self.quat = rpy_to_quat(*INIT_RPY)
            return True
        action = ACTIONS.get(keycode)
        if action is None:
            return False
        kind, axis, sign = action
        if kind == "move":
            self.pos[axis] += sign * MOVE_STEP
            self.pos[2] = max(self.pos[2], Z_MIN)
        else:
            self.quat = rotate_about_body_axis(self.quat, axis, sign * ROT_STEP)
        return True

    # -- one refresh of everything visible ------------------------------- #
    def refresh(self, viewer, announce=True):
        self.write_pose_to_mujoco()
        R = self.rotation_matrix()
        with viewer.lock():                            # protect user_scn while editing
            draw_frames(viewer.user_scn, self.pos, R)
        update_overlay(viewer, self.pos, R)
        viewer.sync()                                  # push data + user_scn to the window
        if announce:
            r, p, y = np.rad2deg(mat_to_rpy(R))
            print(f"pos=({clean(self.pos[0], 2):+.2f}, {clean(self.pos[1], 2):+.2f}, "
                  f"{clean(self.pos[2], 2):+.2f})  "
                  f"rpy=({clean(r, 0):+.0f}, {clean(p, 0):+.0f}, {clean(y, 0):+.0f}) deg")
            print("R =\n" + "\n".join("    " + row for row in fmt_matrix_rows(R)))

    def run(self):
        print(f"Running   : {os.path.abspath(__file__)}")
        print(f"Scene     : {XML_PATH}")
        print(f"MuJoCo    : {mujoco.__version__}")
        print("Controls  : arrows = move X/Y | PgUp/PgDn = move Z | Home/End = yaw |\n"
              "            Insert/Delete = pitch | 8/9 = roll | Enter = reset | Esc/Q = quit\n")

        with mujoco.viewer.launch_passive(
                self.model, self.data,
                key_callback=self.key_callback,
                show_left_ui=False, show_right_ui=False) as viewer:

            with viewer.lock():                        # initial camera
                viewer.cam.lookat[:] = (0.0, 0.0, 0.6)
                viewer.cam.distance = 3.8
                viewer.cam.azimuth = 30.0              # oblique view: X, Y and Z all visible
                viewer.cam.elevation = -25.0

            self.refresh(viewer)

            while viewer.is_running() and not self.quit_requested:
                changed = False
                while self.pending:
                    changed |= self.process_key(self.pending.popleft())
                if changed:
                    self.refresh(viewer)
                time.sleep(1.0 / 60.0)


# --------------------------------------------------------------------------- #
# Headless self-test (no window): proves the maths and the model are consistent
# --------------------------------------------------------------------------- #
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"[{'PASS' if cond else 'FAIL'}] {name}")

    def Rx(a): c, s = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    def Ry(a): c, s = np.cos(a), np.sin(a); return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    def Rz(a): c, s = np.cos(a), np.sin(a); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    sim = QuadrotorSim()
    m, d = sim.model, sim.data
    check("model: free joint -> nq == 7, nv == 6", m.nq == 7 and m.nv == 6)
    check("model: no actuators needed (nu == 0)", m.nu == 0)

    rng = np.random.default_rng(1)
    good_rpy = good_R = good_back = True
    for _ in range(500):
        r, p, y = rng.uniform(-3.0, 3.0, 3)
        p = np.clip(p, -1.4, 1.4)
        q = rpy_to_quat(r, p, y)
        R = quat_to_mat(q)
        good_rpy &= np.allclose(R, Rz(y) @ Ry(p) @ Rx(r), atol=1e-9)
        good_R &= np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
        good_back &= np.allclose(mat_to_rpy(R), (r, p, y), atol=1e-7) or \
                     np.allclose(np.sin(mat_to_rpy(R)), np.sin((r, p, y)), atol=1e-7)
    check("rpy_to_quat == Rz(yaw) Ry(pitch) Rx(roll)", good_rpy)
    check("R is orthonormal with det = +1", good_R)
    check("mat_to_rpy inverts rpy_to_quat", good_back)

    # Pose written to qpos -> MuJoCo's own xmat/xpos must equal our R / pos.
    sim.pos = np.array([0.3, -0.2, 0.8])
    sim.quat = rpy_to_quat(0.4, -0.3, 1.1)
    sim.write_pose_to_mujoco()
    check("MuJoCo xmat[base_link] == our R",
          np.allclose(d.xmat[sim.body_id].reshape(3, 3), sim.rotation_matrix(), atol=1e-9))
    check("MuJoCo xpos[base_link] == our position", np.allclose(d.xpos[sim.body_id], sim.pos))

    # Columns of R are the body axes in the world frame.
    sim.quat = rpy_to_quat(0, 0, np.pi / 2)            # yaw 90 deg
    R = sim.rotation_matrix()
    check("yaw 90 deg: body x axis (column 0) == world +Y", np.allclose(R[:, 0], [0, 1, 0], atol=1e-9))
    check("yaw 90 deg: body y axis (column 1) == world -X", np.allclose(R[:, 1], [-1, 0, 0], atol=1e-9))
    check("yaw 90 deg: body z axis (column 2) == world +Z", np.allclose(R[:, 2], [0, 0, 1], atol=1e-9))

    # Key handling.
    sim.process_key(KEY_ENTER)
    sim.process_key(KEY_UP)
    check("Up key: +0.1 m along world X", np.allclose(sim.pos, [0.1, 0.0, 0.5]))
    for _ in range(9):
        sim.process_key(KEY_HOME)                      # 9 x 10 deg = 90 deg yaw
    check("9 x Home = yaw +90 deg", np.allclose(sim.rotation_matrix(), Rz(np.pi / 2), atol=1e-9))
    sim.process_key(KEY_LEFT)                          # world +Y regardless of heading
    check("Left key moves along WORLD +Y (not body)", np.allclose(sim.pos, [0.1, 0.1, 0.5]))
    sim.process_key(KEY_ENTER)
    check("Enter resets pose", np.allclose(sim.pos, INIT_POS) and np.allclose(sim.rotation_matrix(), np.eye(3)))

    # Drawing code runs on an off-screen user scene (no window needed).
    scn = mujoco.MjvScene(m, 200)
    draw_frames(scn, sim.pos, np.eye(3))
    check("draw_frames adds 18 geoms (2 frames x 3 axes x shaft+tip+label)", scn.ngeom == 18)

    print("\nALL TESTS PASSED" if ok else "\nSOME TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        sys.exit(selftest())
    QuadrotorSim().run()