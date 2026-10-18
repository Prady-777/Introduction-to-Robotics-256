"""
Three IK solvers behind ONE interface so the planner can swap them:

    solver.solve(p_target, R_target, q_init, q_prev=None, max_step=None) -> IKResult

  * MinkIK : off-the-shelf baseline (https://github.com/kevinzakka/mink)
  * DLSIK  : closed-form Damped Least Squares, closed loop on the pose error,
             with an optional null-space posture objective (tutorial eq. for q̇_null)
  * QPIK   : the same DLS cost written as a QP  ½Δqᵀ H Δq + cᵀ Δq  s.t.  G Δq ≤ h
             with joint-limit AND velocity-limit inequality constraints.

`q_prev` / `max_step` describe the previous trajectory knot and the largest joint
change allowed until this knot (v_max * Δt_knot). Only QPIK enforces it as a hard
constraint; the others ignore it, so velocity violations show up in the logs.
"""
import time
from dataclasses import dataclass, field

import numpy as np

from .kinematics import pose_error


@dataclass
class IKResult:
    q: np.ndarray
    success: bool
    iters: int
    pos_err: float
    rot_err: float
    solve_time: float
    limit_hits: int = 0          # iterations where a joint limit had to be enforced
    info: dict = field(default_factory=dict)


class _IKBase:
    name = "base"

    def __init__(self, kin, pos_tol=1e-3, rot_tol=1e-2, max_iters=200, dt=0.02,
                 gain=40.0, max_lin_step=0.05, max_ang_step=0.3, q_rest=None):
        self.kin = kin
        self.pos_tol, self.rot_tol = pos_tol, rot_tol
        self.max_iters = max_iters
        self.dt = dt                  # fictitious integration step of the IK loop
        self.gain = gain              # task-space feedback gain K: x_dot_d = K * e
        self.max_lin_step = max_lin_step
        self.max_ang_step = max_ang_step
        self.q_rest = np.zeros(kin.n) if q_rest is None else np.asarray(q_rest, float)

    # x_dot_d = K e, clipped so one iteration never asks for a huge jump
    def _desired_twist(self, e):
        xd = self.gain * e
        lin, ang = xd[:3], xd[3:]
        nl, na = np.linalg.norm(lin) * self.dt, np.linalg.norm(ang) * self.dt
        if nl > self.max_lin_step:
            lin = lin * self.max_lin_step / nl
        if na > self.max_ang_step:
            ang = ang * self.max_ang_step / na
        return np.concatenate([lin, ang])

    def _errors(self, q, p_t, R_t):
        p, R = self.kin.fk(q)
        e = pose_error(p_t, R_t, p, R)
        return e, np.linalg.norm(e[:3]), np.linalg.norm(e[3:])

    def _converged(self, pe, re):
        return pe < self.pos_tol and re < self.rot_tol


# -----------------------------------------------------------------------------
# 1) Damped Least Squares (closed form)
# -----------------------------------------------------------------------------
class DLSIK(_IKBase):
    """q_dot = (JᵀJ + λI)⁻¹ Jᵀ x_dot_d  +  (I - (JᵀJ + λI)⁻¹ JᵀJ) q_dot_null"""
    name = "dls"

    def __init__(self, kin, damping=1e-3, null_gain=0.05, **kw):
        super().__init__(kin, **kw)
        self.lam = damping
        self.null_gain = null_gain    # α in q_dot_null = α (q_rest - q)

    def solve(self, p_t, R_t, q_init, q_prev=None, max_step=None):
        t0 = time.perf_counter()
        q = np.array(q_init, dtype=float)
        n = self.kin.n
        hits = 0
        pe = re = np.inf
        for it in range(self.max_iters + 1):
            e, pe, re = self._errors(q, p_t, R_t)
            if self._converged(pe, re):
                break
            if it == self.max_iters:
                break
            J = self.kin.jacobian(q)
            A = J.T @ J + self.lam * np.eye(n)
            A_inv_JT = np.linalg.solve(A, J.T)
            qdot = A_inv_JT @ self._desired_twist(e)
            if self.null_gain > 0:
                N = np.eye(n) - A_inv_JT @ J
                qdot += N @ (self.null_gain / self.dt * (self.q_rest - q))
            q_new = q + qdot * self.dt
            q_clip = np.clip(q_new, self.kin.lower, self.kin.upper)
            if np.any(q_clip != q_new):
                hits += 1              # DLS itself knows nothing about limits
            q = q_clip
        return IKResult(q, self._converged(pe, re), it, pe, re,
                        time.perf_counter() - t0, hits)


# -----------------------------------------------------------------------------
# 2) QP-based DLS with joint-limit + velocity-limit constraints
# -----------------------------------------------------------------------------
def _box_qp(H, c, lb, ub):
    """min ½xᵀHx + cᵀx  s.t. lb ≤ x ≤ ub.
    Uses qpsolvers (daqp / quadprog / osqp...) with the explicit G x ≤ h form from
    the tutorial; falls back to SciPy's bounded least-squares if qpsolvers is absent."""
    n = len(c)
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.concatenate([ub, -lb])
    try:
        import qpsolvers
        for solver in ("daqp", "quadprog", "proxqp", "osqp", "clarabel", "cvxopt"):
            if solver in qpsolvers.available_solvers:
                x = qpsolvers.solve_qp(H, c, G, h, solver=solver)
                if x is not None:
                    return x, solver
    except ImportError:
        pass
    # Fallback: H = 2/dt²(JᵀJ+λI) ⇒ factor H = LᵀL and solve bounded least squares
    from scipy.optimize import lsq_linear
    L = np.linalg.cholesky(H).T
    b = -np.linalg.solve(L.T, c)
    return lsq_linear(L, b, bounds=(lb, ub)).x, "scipy_lsq_linear"


class QPIK(_IKBase):
    """
    min_Δq  ½ Δqᵀ H Δq + cᵀ Δq     s.t.  G Δq ≤ h
      H = 2/Δt² (JᵀJ + λI + μI)
      c = -2/Δt Jᵀ x_dot_d  - 2μ/Δt q_dot_null        (μ = posture weight, 0 = pure DLS)
    NOTE: expanding the tutorial's cost gives c = -(2/Δt) Jᵀ x_dot_d; the notes write
    -(1/Δt) which would halve every step. The factor is verified in tests/test_ik_math.py.
    Constraints (all stacked as G Δq ≤ h with G = [I; -I]):
      q_min ≤ q + Δq ≤ q_max                         (joint limits)
      |q + Δq - q_prev| ≤ max_step = v_max·Δt_knot    (velocity limits along the trajectory)
      |Δq| ≤ v_max·Δt                                (velocity limit of one IK iteration)
    """
    name = "qp"

    def __init__(self, kin, damping=1e-3, posture_weight=1e-3, null_gain=0.05,
                 v_max=2.0, **kw):
        super().__init__(kin, **kw)
        self.lam = damping
        self.mu = posture_weight
        self.null_gain = null_gain
        self.v_max = np.broadcast_to(np.asarray(v_max, float), (kin.n,)).copy()
        self.backend = None

    def solve(self, p_t, R_t, q_init, q_prev=None, max_step=None):
        t0 = time.perf_counter()
        q = np.array(q_init, dtype=float)
        n, dt = self.kin.n, self.dt
        lo, hi = self.kin.lower.copy(), self.kin.upper.copy()
        if q_prev is not None and max_step is not None:
            lo = np.maximum(lo, q_prev - max_step)
            hi = np.minimum(hi, q_prev + max_step)
        hits = 0
        pe = re = np.inf
        for it in range(self.max_iters + 1):
            e, pe, re = self._errors(q, p_t, R_t)
            if self._converged(pe, re) or it == self.max_iters:
                break
            J = self.kin.jacobian(q)
            xd = self._desired_twist(e)
            qd_null = self.null_gain / dt * (self.q_rest - q)
            H = 2.0 / dt**2 * (J.T @ J + (self.lam + self.mu) * np.eye(n))
            c = -2.0 / dt * (J.T @ xd) - 2.0 * self.mu / dt * qd_null
            step = self.v_max * dt
            lb = np.maximum(lo - q, -step)
            ub = np.minimum(hi - q, step)
            lb = np.minimum(lb, ub)          # keep the box feasible if q is outside it
            dq, self.backend = _box_qp(H, c, lb, ub)
            unconstrained = np.linalg.solve(H, -c)
            if np.any(np.abs(dq - unconstrained) > 1e-6):
                hits += 1                    # a constraint was active this iteration
            q = q + dq
        return IKResult(q, self._converged(pe, re), it, pe, re,
                        time.perf_counter() - t0, hits, {"backend": self.backend})


# -----------------------------------------------------------------------------
# 3) Off-the-shelf: Mink
# -----------------------------------------------------------------------------
class MinkIK(_IKBase):
    """Differential IK with Mink: FrameTask on the TCP site + small PostureTask,
    ConfigurationLimit and VelocityLimit. Mink works on the FULL qpos (cube free
    joint and fingers included); those DoFs have zero Jacobian columns, so the
    damping term keeps them still."""
    name = "mink"

    def __init__(self, kin, model, joint_names, site_name="tcp", v_max=2.0,
                 posture_cost=1e-4, damping=1e-3, **kw):
        super().__init__(kin, **kw)
        import mink
        import qpsolvers
        self.mink = mink
        self.model = model
        self.cfg = mink.Configuration(model)
        self.task = mink.FrameTask(frame_name=site_name, frame_type="site",
                                   position_cost=1.0, orientation_cost=1.0,
                                   lm_damping=1.0)
        self.posture = mink.PostureTask(model, cost=posture_cost)
        self.limits = [mink.ConfigurationLimit(model),
                       mink.VelocityLimit(model, {j: float(v_max) for j in joint_names})]
        self.damping = damping
        self.solver = next(s for s in ("daqp", "quadprog", "proxqp", "osqp", "clarabel")
                           if s in qpsolvers.available_solvers)
        self.full_q = None          # full qpos template, set by the planner

    def _target(self, p, R):
        SE3, SO3 = self.mink.SE3, self.mink.SO3
        try:
            return SE3.from_rotation_and_translation(SO3.from_matrix(R), p)
        except AttributeError:      # very old mink versions
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p
            return SE3.from_matrix(T)

    def solve(self, p_t, R_t, q_init, q_prev=None, max_step=None):
        t0 = time.perf_counter()
        adr = self.kin.qpos_adr
        q_full = (self.full_q if self.full_q is not None else self.model.qpos0).copy()
        q_full[adr] = q_init
        self.cfg.update(q_full)
        rest = q_full.copy(); rest[adr] = self.q_rest
        self.posture.set_target(rest)
        self.task.set_target(self._target(p_t, R_t))
        hits = 0
        pe = re = np.inf
        for it in range(self.max_iters + 1):
            q = self.cfg.q[adr].copy()
            _, pe, re = self._errors(q, p_t, R_t)
            if self._converged(pe, re) or it == self.max_iters:
                break
            v = self.mink.solve_ik(self.cfg, [self.task, self.posture], self.dt,
                                   self.solver, damping=self.damping, limits=self.limits)
            self.cfg.integrate_inplace(v, self.dt)
        q = self.cfg.q[adr].copy()
        if np.any(q < self.kin.lower - 1e-6) or np.any(q > self.kin.upper + 1e-6):
            hits += 1
        return IKResult(q, self._converged(pe, re), it, pe, re,
                        time.perf_counter() - t0, hits, {"backend": self.solver})


def make_solver(name, kin, model=None, joint_names=None, **kw):
    name = name.lower()
    if name == "dls":
        return DLSIK(kin, **kw)
    if name == "qp":
        return QPIK(kin, **kw)
    if name == "mink":
        return MinkIK(kin, model, joint_names, **kw)
    raise ValueError(f"unknown IK method '{name}' (use mink | dls | qp)")
