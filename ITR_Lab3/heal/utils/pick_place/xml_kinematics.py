"""
MuJoCo-free forward kinematics of the HEAL arm, parsed directly from the scene XML.
Used by the unit tests and by `workspace_analysis.py --backend numpy`. It follows MuJoCo's
conventions (quat = wxyz, euler = intrinsic 'xyz', joint at the body origin), and its
Jacobian is checked against finite differences in tests/test_ik_math.py.
"""
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import quat_to_mat, rot_x, rot_z
from .config import SCENE_XML as XML


def _axang(ax, a):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _vec(s, default="0 0 0"):
    return np.array([float(v) for v in (s or default).split()])


class XmlChainKinematics:
    """Independent FK/Jacobian of the HEAL chain (MuJoCo conventions: quat wxyz,
    euler 'xyz' intrinsic, joint at body origin)."""

    def __init__(self, path=XML):
        body = ET.parse(path).getroot().find("worldbody").find("body[@name='base_link']")
        self.links = []
        for name in ["link_1", "link_2", "link_3", "link_4", "link_5", "end_effector"]:
            body = body.find(f"body[@name='{name}']")
            if body.get("quat"):
                R = quat_to_mat(_vec(body.get("quat")))
            elif body.get("euler"):
                e = _vec(body.get("euler"))
                R = rot_x(e[0]) @ _ry(e[1]) @ rot_z(e[2])
            else:
                R = np.eye(3)
            j = body.find("joint")
            rng = _vec(j.get("range"))
            self.links.append(dict(p=_vec(body.get("pos")), R=R, ax=_vec(j.get("axis")), rng=rng))
        grip = body.find("body[@name='gripper_base']")
        self.tcp = _vec(grip.find("site[@name='tcp']").get("pos")) + _vec(grip.get("pos"))
        self.n = 6
        self.lower = np.array([l["rng"][0] for l in self.links])
        self.upper = np.array([l["rng"][1] for l in self.links])

    def _frames(self, q):
        p, R, fr = np.zeros(3), np.eye(3), []
        for qi, l in zip(q, self.links):
            p = p + R @ l["p"]
            R = R @ l["R"]
            fr.append((p.copy(), R @ (l["ax"] / np.linalg.norm(l["ax"]))))
            R = R @ _axang(l["ax"], qi)
        return p + R @ self.tcp, R, fr

    def fk(self, q):
        p, R, _ = self._frames(q)
        return p, R

    def jacobian(self, q):
        p, _, fr = self._frames(q)
        J = np.zeros((6, 6))
        for i, (pi, a) in enumerate(fr):
            J[:3, i] = np.cross(a, p - pi)
            J[3:, i] = a
        return J


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


