"""
Kinematics helpers shared by every IK method.

Two layers:
  1. Pure-NumPy rotation / quaternion maths (MuJoCo convention: q = [w, x, y, z]).
     These follow the tutorial's "DLS-IK (Quaternion Computations)" section and
     are unit-tested in tests/test_ik_math.py without MuJoCo.
  2. MjKinematics: wraps a *private* MjData so IK can run FK / Jacobians
     without disturbing the simulation state.

Every IK solver only needs this interface:
    kin.n                -> number of arm joints
    kin.lower, kin.upper -> joint limits (arrays)
    kin.fk(q)            -> (tcp_pos[3], tcp_R[3x3])
    kin.jacobian(q)      -> 6 x n geometric Jacobian [linear; angular], world frame
"""
import numpy as np

# ----------------------------------------------------------------------------
# Rotation / quaternion utilities (scalar-first, like MuJoCo)
# ----------------------------------------------------------------------------

def quat_mul(a, b):
    """Hamilton product a ⊗ b, both [w, x, y, z]."""
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_conj(q):
    """Inverse of a *unit* quaternion."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def mat_to_quat(R):
    """Rotation matrix -> unit quaternion [w, x, y, z] (Shepperd's method)."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def quat_to_mat(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_error(q_target, q_current):
    """
    Orientation error as an axis-angle vector (world frame), exactly as in the
    tutorial:  q_e = q_t ⊗ q_c^{-1};  flip sign if w_e < 0 (hemisphere continuity);
    eps_w = theta * v_e / sin(theta/2),  theta = 2*acos(clip(w_e, -1, 1)).
    """
    qe = quat_mul(q_target, quat_conj(q_current))
    if qe[0] < 0.0:
        qe = -qe
    v = qe[1:]
    s = np.linalg.norm(v)                       # = sin(theta/2) for a unit quaternion
    if s < 1e-12:
        return np.zeros(3)
    # theta = 2*acos(w) == 2*atan2(|v|, w); atan2 keeps precision for tiny angles
    theta = 2.0 * np.arctan2(s, np.clip(qe[0], -1.0, 1.0))
    return theta * v / s


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def top_down_rotation(yaw):
    """TCP orientation with the approach (z) axis pointing straight down and the
    finger-closing axis (x) rotated by `yaw` about world z."""
    return rot_z(yaw) @ rot_x(np.pi)


def tcp_yaw(R):
    """Yaw of the TCP x-axis projected on the world xy-plane."""
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def slerp_R(R0, R1, s):
    """Spherical interpolation between two rotation matrices, s in [0, 1]."""
    q0, q1 = mat_to_quat(R0), mat_to_quat(R1)
    if np.dot(q0, q1) < 0:
        q1 = -q1
    d = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if d > 0.9995:
        q = q0 + s * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - s) * th) * q0 + np.sin(s * th) * q1) / np.sin(th)
    return quat_to_mat(q / np.linalg.norm(q))


def pose_error(p_t, R_t, p_c, R_c):
    """6-vector [position error; orientation error] used by DLS/QP IK."""
    return np.concatenate([p_t - p_c, quat_error(mat_to_quat(R_t), mat_to_quat(R_c))])


# ----------------------------------------------------------------------------
# MuJoCo wrapper
# ----------------------------------------------------------------------------

class MjKinematics:
    """FK and Jacobian of the arm TCP site using a private MjData."""

    def __init__(self, model, joint_names, site_name="tcp"):
        import mujoco  # imported here so the maths above works without MuJoCo
        self.mj = mujoco
        self.model = model
        self.data = mujoco.MjData(model)
        self.joint_names = list(joint_names)
        jids = [model.joint(n).id for n in self.joint_names]
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in jids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in jids])
        self.lower = np.array([model.jnt_range[j][0] for j in jids])
        self.upper = np.array([model.jnt_range[j][1] for j in jids])
        self.site_id = model.site(site_name).id
        self.n = len(jids)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    def sync_full_state(self, qpos_full):
        """Copy the full simulation qpos (cube, fingers...) into the IK data."""
        self.data.qpos[:] = qpos_full

    def _update(self, q):
        self.data.qpos[self.qpos_adr] = q
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_comPos(self.model, self.data)

    def fk(self, q):
        self._update(q)
        p = self.data.site_xpos[self.site_id].copy()
        R = self.data.site_xmat[self.site_id].reshape(3, 3).copy()
        return p, R

    def jacobian(self, q):
        self._update(q)
        self.mj.mj_jacSite(self.model, self.data, self._jacp, self._jacr, self.site_id)
        return np.vstack([self._jacp[:, self.dof_adr], self._jacr[:, self.dof_adr]])
