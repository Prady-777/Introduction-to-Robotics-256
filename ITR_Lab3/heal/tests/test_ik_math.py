"""
MuJoCo-free checks of the IK maths, run on the real HEAL kinematics parsed straight
from robot_descriptions/pick_place_scene.xml.

    python tests/test_ik_math.py        (or: pytest tests/)
"""
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from utils.pick_place.kinematics import (quat_error, mat_to_quat, quat_to_mat, rot_x, rot_z,  # noqa
                                         top_down_rotation, pose_error)
from utils.pick_place.ik_solvers import DLSIK, QPIK  # noqa
from utils.pick_place.xml_kinematics import XmlChainKinematics, _axang  # noqa

KIN = XmlChainKinematics()
TARGETS = [((0.45, 0.10, 0.30), 0.3), ((0.35, -0.30, 0.28), -1.2), ((0.30, 0.35, 0.37), 2.5)]


def test_quaternion_roundtrip_and_error():
    rng = np.random.default_rng(0)
    for _ in range(50):
        ax = rng.normal(size=3); ax /= np.linalg.norm(ax)
        ang = rng.uniform(0, np.pi - 1e-3)
        R0 = quat_to_mat(mat_to_quat(_axang(rng.normal(size=3), 1.0)))
        R1 = _axang(ax, ang) @ R0                     # rotate by (ax, ang) in world frame
        assert np.allclose(quat_to_mat(mat_to_quat(R1)), R1, atol=1e-9)
        err = quat_error(mat_to_quat(R1), mat_to_quat(R0))
        assert np.allclose(err, ax * ang, atol=1e-6), (err, ax * ang)


def test_jacobian_matches_finite_differences():
    q = np.array([0.3, -0.4, 0.5, 0.2, -0.6, 0.7])
    J = KIN.jacobian(q)
    p0, R0 = KIN.fk(q)
    for i in range(6):
        dq = np.zeros(6); dq[i] = 1e-6
        p1, R1 = KIN.fk(q + dq)
        assert np.allclose((p1 - p0) / 1e-6, J[:3, i], atol=1e-4)
        w = quat_error(mat_to_quat(R1), mat_to_quat(R0)) / 1e-6
        assert np.allclose(w, J[3:, i], atol=1e-4)


def _check_solution(res, p, R):
    pf, Rf = KIN.fk(res.q)
    e = pose_error(np.array(p), R, pf, Rf)
    assert res.success and np.linalg.norm(e[:3]) < 1.5e-3 and np.linalg.norm(e[3:]) < 1.5e-2


def test_dls_reaches_top_down_targets():
    ik = DLSIK(KIN, max_iters=300)
    for p, yaw in TARGETS:
        res = ik.solve(np.array(p), top_down_rotation(yaw), np.zeros(6))
        _check_solution(res, p, top_down_rotation(yaw))


def test_qp_reaches_targets_and_respects_limits():
    ik = QPIK(KIN, max_iters=300, v_max=2.0)
    for p, yaw in TARGETS:
        res = ik.solve(np.array(p), top_down_rotation(yaw), np.zeros(6))
        _check_solution(res, p, top_down_rotation(yaw))
        assert np.all(res.q >= KIN.lower - 1e-9) and np.all(res.q <= KIN.upper + 1e-9)
    # tight artificial limits + trajectory step bound must hold exactly
    tight = XmlChainKinematics()
    tight.lower, tight.upper = -0.5 * np.ones(6), 0.5 * np.ones(6)
    ik2 = QPIK(tight, max_iters=50)
    q_prev = np.zeros(6)
    res = ik2.solve(np.array([0.30, 0.30, 0.25]), top_down_rotation(1.0), q_prev,
                    q_prev=q_prev, max_step=0.04)
    assert np.all(np.abs(res.q - q_prev) <= 0.04 + 1e-7)
    assert np.all(np.abs(res.q) <= 0.5 + 1e-7)


def test_qp_step_equals_dls_closed_form():
    """With inactive constraints the QP minimiser must equal Δt·(JᵀJ+λI)⁻¹Jᵀẋ_d.
    This holds with c = -(2/Δt)Jᵀẋ_d; the tutorial's -(1/Δt) would give half the step."""
    q = np.array([0.2, -0.3, 0.4, 0.1, -0.2, 0.3])
    J = KIN.jacobian(q)
    xd = np.array([0.05, -0.02, 0.03, 0.1, -0.05, 0.02])
    lam, dt = 1e-3, 0.02
    H = 2 / dt**2 * (J.T @ J + lam * np.eye(6))
    dls = dt * np.linalg.solve(J.T @ J + lam * np.eye(6), J.T @ xd)
    assert np.allclose(np.linalg.solve(H, 2 / dt * J.T @ xd), dls)
    assert np.allclose(np.linalg.solve(H, 1 / dt * J.T @ xd), 0.5 * dls)


def test_home_pose_is_top_down():
    p, R = KIN.fk(np.zeros(6))
    assert R[2, 2] < -0.99 and abs(p[2] - 0.42) < 0.02


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS  {name}")
