"""
MuJoCo-free forward kinematics of the arm, parsed directly from the scene XML.
Used by the unit tests and by `workspace_analysis.py --backend numpy`.

Generic: walks the body tree from <worldbody> down to the body that holds the site
`tcp`, composes every body's fixed transform (pos + quat/euler) and the hinge joints
named in `joint_names`. Conventions follow MuJoCo: quat = [w x y z], euler = intrinsic
'xyz' in radians, joint rotation about `pos` of the joint.
The Jacobian is checked against finite differences in tests/test_ik_math.py.
"""
import xml.etree.ElementTree as ET

import numpy as np

from .kinematics import quat_to_mat, rot_x, rot_z
from .config import SCENE_XML as XML, Config


def _vec(s, default="0 0 0"):
    return np.array([float(v) for v in (s or default).split()])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _axang(ax, a):
    ax = np.asarray(ax, float) / np.linalg.norm(ax)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _body_rot(b):
    if b.get("quat"):
        return quat_to_mat(_vec(b.get("quat")))
    if b.get("euler"):
        e = _vec(b.get("euler"))
        return rot_x(e[0]) @ _ry(e[1]) @ rot_z(e[2])
    if b.get("axisangle"):
        v = _vec(b.get("axisangle"))
        return _axang(v[:3], v[3])
    return np.eye(3)


def _path_to_site(elem, site="tcp"):
    """List of <body> elements from `elem` down to the body containing the site."""
    for b in elem.findall("body"):
        if b.find(f"site[@name='{site}']") is not None:
            return [b]
        sub = _path_to_site(b, site)
        if sub:
            return [b] + sub
    return None


class XmlChainKinematics:
    """FK / geometric Jacobian of the serial chain base -> tcp site."""

    def __init__(self, path=XML, joint_names=None, site="tcp"):
        joint_names = list(joint_names or Config().arm_joints)
        world = ET.parse(path).getroot().find("worldbody")
        chain = _path_to_site(world, site)
        if chain is None:
            raise ValueError(f"site '{site}' not found in {path}")
        self.links, lo, hi = [], [], []
        for b in chain:
            j = None
            for jj in b.findall("joint"):
                if jj.get("name") in joint_names:
                    j = jj
            link = dict(p=_vec(b.get("pos")), R=_body_rot(b), ax=None, jp=np.zeros(3))
            if j is not None:
                link["ax"] = _vec(j.get("axis"), "0 0 1")
                link["jp"] = _vec(j.get("pos"))
                r = _vec(j.get("range"), "-3.14159 3.14159")
                lo.append(r[0]); hi.append(r[1])
            self.links.append(link)
        self.tcp = _vec(chain[-1].find(f"site[@name='{site}']").get("pos"))
        self.lower, self.upper = np.array(lo), np.array(hi)
        self.n = len(lo)
        assert self.n == len(joint_names), "joint names / XML mismatch"

    def _frames(self, q):
        p, R, fr, k = np.zeros(3), np.eye(3), [], 0
        for l in self.links:
            p = p + R @ l["p"]
            R = R @ l["R"]
            if l["ax"] is not None:
                a = R @ (l["ax"] / np.linalg.norm(l["ax"]))
                pj = p + R @ l["jp"]
                fr.append((pj, a))
                Rq = _axang(l["ax"], q[k])
                p = p + R @ (l["jp"] - Rq @ l["jp"])
                R = R @ Rq
                k += 1
        return p + R @ self.tcp, R, fr

    def fk(self, q):
        p, R, _ = self._frames(q)
        return p, R

    def jacobian(self, q):
        p, _, fr = self._frames(q)
        J = np.zeros((6, self.n))
        for i, (pi, a) in enumerate(fr):
            J[:3, i] = np.cross(a, p - pi)
            J[3:, i] = a
        return J


def scene_geometry(path=XML):
    """Table rectangle/top and tray pose read from the scene XML (for plots)."""
    w = ET.parse(path).getroot().find("worldbody")
    table = w.find("body[@name='table']")
    tp = _vec(table.get("pos"))
    tg = table.find("geom[@name='table_top']")
    ts = _vec(tg.get("size"))
    tgp = _vec(tg.get("pos"))
    c = tp + tgp
    tray = w.find("body[@name='tray']")
    yp = _vec(tray.get("pos"))
    yaw = _vec(tray.get("euler"))[2] if tray.get("euler") else 0.0
    fl = _vec(tray.find("geom[@name='tray_floor']").get("size"))
    return dict(x0=c[0] - ts[0], x1=c[0] + ts[0], y0=c[1] - ts[1], y1=c[1] + ts[1],
                top=c[2] + ts[2], tray_x=yp[0], tray_y=yp[1], tray_yaw=yaw, tray_half=fl[0])
