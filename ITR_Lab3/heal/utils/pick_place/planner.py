"""
Episode logic: approach -> descend -> grasp -> lift -> transit -> lower -> release -> retreat

Every motion is a straight Cartesian line (minimum-jerk timing) discretised into knots
at cfg.knot_dt. IK is solved at each knot, warm-started from the previous knot, which
gives the joint trajectory q*(t) that the PD + gravity-compensation controller tracks.

Collision avoidance between cube, gripper and tray ("smart" placement):
  1. Safe transit height: cube bottom and fingertips stay `clearance` above the tray walls.
  2. Place yaw: the gripper yaw is chosen so the HELD cube ends up aligned with the tray
     walls (accounting for any yaw slip measured after the lift), picking the
     candidate (k·90°) that needs the least wrist rotation.
  3. Before executing, the planned transit/lower/release motion is checked: cube
     corners and open fingertips must fit inside the tray with a margin; transit
     clearance is predicted from FK. If a check fails, the transit height is raised
     or the next yaw candidate is tried.
  4. During execution, contacts and signed distances (mj_geomDistance) are logged.
"""
import math
import time

import numpy as np

from .kinematics import top_down_rotation, tcp_yaw, wrap_angle, slerp_R


class PlanFailure(Exception):
    def __init__(self, reason, phase=""):
        super().__init__(reason)
        self.reason, self.phase = reason, phase


def min_jerk(t):
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class PickPlacePlanner:
    def __init__(self, env, solver, cfg):
        self.env, self.solver, self.cfg = env, solver, cfg
        self.kin = env.kin
        m = env.model
        # fingertip depth below the TCP and open-finger half span (for tray checks)
        site_z = m.site_pos[m.site("tcp").id][2]
        pad = m.geom("finger_right_pad").id
        fbody = m.body("finger_right").id
        pad_bottom = m.body_pos[fbody][2] + m.geom_pos[pad][2] + m.geom_size[pad][2]
        self.tip_below_tcp = float(pad_bottom - site_z)
        self.pad_outer_open = float(m.geom_pos[pad][0] + m.geom_size[pad][0] + cfg.finger_open)
        self.pad_half_width = float(m.geom_size[pad][1])

    # ------------------------------------------------------------ geometry
    def safe_height(self, extra=0.0):
        e = self.env
        lowest = max(e.cube_half, self.tip_below_tcp)
        top = max(e.tray_wall_top, e.table_top_z)
        return top + self.cfg.clearance + lowest + extra

    def to_tray_frame(self, p):
        e = self.env
        c, s = math.cos(-e.tray_yaw), math.sin(-e.tray_yaw)
        d = np.asarray(p[:2]) - e.tray_pos[:2]
        return np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])

    def cube_corners_in_tray(self, center, cube_yaw):
        h = self.env.cube_half
        rel = cube_yaw - self.env.tray_yaw
        c, s = math.cos(rel), math.sin(rel)
        local = self.to_tray_frame(center)
        return [local + np.array([c * x - s * y, s * x + c * y])
                for x, y in ((h, h), (h, -h), (-h, h), (-h, -h))]

    # ------------------------------------------------------------ planning
    def cart_poses(self, p0, R0, p1, R1, speed):
        dist = np.linalg.norm(p1 - p0)
        ang = math.acos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))
        T = max(dist / speed, ang / self.cfg.ang_speed, 0.2)
        n = max(1, math.ceil(T / self.cfg.knot_dt))
        out = []
        for i in range(1, n + 1):
            s = min_jerk(i / n)
            out.append((p0 + (p1 - p0) * s, slerp_R(R0, R1, s)))
        return out, T / n

    def plan_segment(self, q_start, p1, R1, speed, phase, st):
        cfg = self.cfg
        p0, R0 = self.kin.fk(q_start)
        poses, dt = self.cart_poses(p0, R0, p1, R1, speed)
        knots, q = [], np.array(q_start, float)
        for k, (p, R) in enumerate(poses):
            last = k == len(poses) - 1
            res = self.solver.solve(p, R, q, q_prev=q, max_step=cfg.v_max * dt)
            st["ik_calls"] += 1
            st["ik_iters"].append(res.iters)
            st["plan_time"] += res.solve_time
            st["limit_hits"] += res.limit_hits
            scale = 2.0 if last else 2.0 * cfg.knot_tol_scale
            if res.pos_err > cfg.ik_pos_tol * scale or res.rot_err > cfg.ik_rot_tol * scale:
                st["ik_fail_knots"] += 1
                raise PlanFailure(f"ik_infeasible", phase)
            if last:
                st["max_endpoint_err"] = max(st["max_endpoint_err"], res.pos_err)
            vel = np.abs(res.q - q) / dt
            st["max_planned_vel"] = max(st["max_planned_vel"], float(vel.max()))
            st["vel_violations"] += int(np.sum(vel > cfg.v_max * 1.001))
            st["jl_violations"] += int(np.sum((res.q < self.kin.lower - 1e-6) |
                                              (res.q > self.kin.upper + 1e-6)))
            q = res.q
            knots.append(q.copy())
        return np.array(knots), dt

    # ------------------------------------------------------------ execution
    def execute(self, knots, dt, q_start, phase):
        env, cfg = self.env, self.cfg
        env.status_text = phase
        n_sub = max(1, round(dt / env.model.opt.timestep))
        prev = np.array(q_start, float)
        for qk in knots:
            qd = (qk - prev) / dt
            for s in range(1, n_sub + 1):
                env.step(prev + (qk - prev) * s / n_sub, qd, phase)
            self.track_err = max(self.track_err, float(np.max(np.abs(qk - env.q))))
            fail = self.check_runtime()
            if fail:
                return fail
            prev = qk
        t_end = env.data.time + cfg.settle_timeout
        while np.max(np.abs(env.q - prev)) > cfg.settle_tol and env.data.time < t_end:
            env.step(prev, np.zeros(6), phase)
        return self.check_runtime()

    def check_runtime(self):
        env = self.env
        if self.cfg.abort_on_collision and env.events:
            return "collision_" + env.events[0][2]
        if env.carrying and np.linalg.norm(env.cube_pos() - env.tcp_pose()[0]) > self.cfg.slip_dist:
            return "grasp_slip"
        return None

    # ------------------------------------------------------------ episode
    def run_episode(self, cube_x, cube_y, cube_yaw):
        env, cfg = self.env, self.cfg
        t_wall = time.perf_counter()
        st = dict(ik_calls=0, ik_iters=[], plan_time=0.0, limit_hits=0, ik_fail_knots=0,
                  max_endpoint_err=0.0, max_planned_vel=0.0, vel_violations=0,
                  jl_violations=0, replans=0)
        self.track_err = 0.0
        nan = float("nan")
        log = dict(cube_x=cube_x, cube_y=cube_y, cube_yaw=cube_yaw, success=False,
                   failure_reason="", failure_phase="", z_safe=nan, grasp_yaw=nan,
                   place_yaw=nan, grasp_yaw_slip=nan, pred_clear_cube_tray=nan)

        def finish(reason="", phase=""):
            ok = reason == ""
            log.update(success=ok, failure_reason=reason, failure_phase=phase)
            it = st.pop("ik_iters")
            log.update(st)
            log.update(ik_iters_total=int(np.sum(it)) if it else 0,
                       ik_iters_mean=float(np.mean(it)) if it else 0.0,
                       ik_iters_max=int(np.max(it)) if it else 0,
                       max_track_err=self.track_err,
                       min_clear_cube_tray=_f(env.min_clear["cube_tray"]),
                       min_clear_robot_table=_f(env.min_clear["robot_table"]),
                       min_clear_gripper_tray=_f(env.min_clear["gripper_tray"]),
                       soft_contacts=env.soft_contacts,
                       collisions=";".join(f"{e[1]}:{e[2]}" for e in env.events),
                       sim_time=env.data.time, wall_time=time.perf_counter() - t_wall)
            cp = env.cube_pos()
            log.update(cube_final_x=cp[0], cube_final_y=cp[1], cube_final_z=cp[2],
                       dist_to_tray_center=float(np.linalg.norm(self.to_tray_frame(cp))))
            env.status_text = "SUCCESS" if ok else f"FAIL: {reason}"
            env.hold(env.q, 0.4, "end")
            return log

        env.reset(cube_x, cube_y, cube_yaw)
        env.sync_ik_state()
        if hasattr(self.solver, "full_q"):
            self.solver.full_q = env.data.qpos.copy()
        q0 = env.q
        cube_p, cyaw = env.cube_pos(), env.cube_yaw()
        z_safe = self.safe_height()
        log["z_safe"] = z_safe

        # ---------------- PICK: try grasp yaws closest to the current wrist yaw ----------------
        _, R_now = env.tcp_pose()
        yaw_now = tcp_yaw(R_now)
        cands = sorted((wrap_angle(cyaw + k * math.pi / 2) for k in range(4)),
                       key=lambda a: abs(wrap_angle(a - yaw_now)))
        plan = None
        for psi in cands:
            try:
                Rg = top_down_rotation(psi)
                p_pre = np.array([cube_p[0], cube_p[1], z_safe])
                p_grasp = cube_p.copy()
                k1, d1 = self.plan_segment(q0, p_pre, Rg, cfg.lin_speed, "approach", st)
                k2, d2 = self.plan_segment(k1[-1], p_grasp, Rg, cfg.slow_speed, "descend", st)
                k3, d3 = self.plan_segment(k2[-1], p_pre, Rg, cfg.slow_speed, "lift", st)
                plan = (k1, d1, k2, d2, k3, d3)
                log["grasp_yaw"] = psi
                break
            except PlanFailure as f:
                st["replans"] += 1
                last_fail = f
        if plan is None:
            return finish(last_fail.reason, "pick_" + last_fail.phase)
        k1, d1, k2, d2, k3, d3 = plan

        r = self.execute(k1, d1, q0, "approach")
        if r:
            return finish(r, "approach")
        r = self.execute(k2, d2, k1[-1], "descend")
        if r:
            return finish(r, "descend")
        env.status_text = "grasp"
        env.set_gripper(open_=False)
        env.hold(k2[-1], cfg.grasp_time, "grasp")
        env.carrying = True
        r = self.execute(k3, d3, k2[-1], "lift")
        if r:
            return finish("grasp_failed" if r == "grasp_slip" else r, "lift")
        rise = env.cube_pos()[2] - cube_p[2]
        if rise < cfg.lift_success_frac * (z_safe - cube_p[2]):
            return finish("grasp_failed", "lift")

        # ---------------- PLACE: align held cube with the tray walls ----------------
        q_now = env.q
        env.sync_ik_state()
        _, R_now = env.tcp_pose()
        yaw_now = tcp_yaw(R_now)
        delta = wrap_angle(env.cube_yaw() - yaw_now)            # cube yaw relative to gripper
        log["grasp_yaw_slip"] = float(((delta + math.pi / 4) % (math.pi / 2)) - math.pi / 4)
        tray_xy = env.tray_pos[:2]
        z_place = env.tray_floor_top + cfg.place_gap + env.cube_half
        cands = sorted((wrap_angle(env.tray_yaw - delta + k * math.pi / 2) for k in range(4)),
                       key=lambda a: abs(wrap_angle(a - yaw_now)))
        plan = None
        for psi in cands:
            for extra in (0.0, 0.05):                            # raise transit if needed
                try:
                    Rp = top_down_rotation(psi)
                    zs = z_safe + extra
                    p_above = np.array([tray_xy[0], tray_xy[1], zs])
                    p_place = np.array([tray_xy[0], tray_xy[1], z_place])
                    self.check_place_fit(p_place, psi, delta)
                    k4, d4 = self.plan_segment(q_now, p_above, Rp, cfg.lin_speed, "transit", st)
                    pred = self.predicted_transit_clearance(k4)
                    if pred < 0.5 * cfg.clearance:
                        raise PlanFailure("transit_clearance", "transit")
                    k5, d5 = self.plan_segment(k4[-1], p_place, Rp, cfg.slow_speed, "lower", st)
                    k6, d6 = self.plan_segment(k5[-1], p_above, Rp, cfg.slow_speed, "retreat", st)
                    plan = (k4, d4, k5, d5, k6, d6)
                    log["place_yaw"], log["pred_clear_cube_tray"] = psi, pred
                    break
                except PlanFailure as f:
                    st["replans"] += 1
                    last_fail = f
            if plan:
                break
        if plan is None:
            return finish(last_fail.reason, "place_" + last_fail.phase)
        k4, d4, k5, d5, k6, d6 = plan

        r = self.execute(k4, d4, q_now, "transit")
        if r:
            return finish(r, "transit")
        r = self.execute(k5, d5, k4[-1], "lower")
        if r:
            return finish(r, "lower")
        env.status_text = "release"
        env.set_gripper(open_=True)
        env.carrying = False
        env.hold(k5[-1], cfg.release_time, "release")
        r = self.execute(k6, d6, k5[-1], "retreat")
        if r:
            return finish(r, "retreat")
        env.hold(k6[-1], 0.3, "settle")

        # ---------------- success check ----------------
        cp = env.cube_pos()
        local = self.to_tray_frame(cp)
        inside = np.all(np.abs(local) <= env.tray_inner_half - 0.5 * env.cube_half)
        low = cp[2] < env.tray_floor_top + env.cube_half + 0.015
        if not (inside and low):
            return finish("place_missed", "release")
        return finish()

    # ------------------------------------------------------------ smart checks
    def check_place_fit(self, p_place, psi, delta, margin=0.005):
        """Cube corners and the opened fingers must fit inside the tray at release."""
        env = self.env
        corners = self.cube_corners_in_tray(p_place, psi + delta)
        lim = env.tray_inner_half - margin
        if max(np.max(np.abs(c)) for c in corners) > lim:
            raise PlanFailure("cube_does_not_fit_tray", "lower")
        rel = psi - env.tray_yaw
        c, s = abs(math.cos(rel)), abs(math.sin(rel))
        # extent of the open finger pads along the tray axes (rotated rectangle)
        ex = c * self.pad_outer_open + s * self.pad_half_width
        ey = s * self.pad_outer_open + c * self.pad_half_width
        off = np.abs(self.to_tray_frame(p_place))
        if off[0] + ex > lim or off[1] + ey > lim:
            raise PlanFailure("fingers_hit_tray", "release")

    def predicted_transit_clearance(self, knots):
        """Vertical gap between the lowest point of cube/fingers and the tray-wall top,
        for every knot whose footprint overlaps the tray. +inf if it never overlaps."""
        env = self.env
        lowest = max(env.cube_half, self.tip_below_tcp)
        reach = env.tray_outer_half + math.sqrt(2) * env.cube_half
        best = np.inf
        for q in knots:
            p, _ = self.kin.fk(q)
            if np.all(np.abs(self.to_tray_frame(p)) < reach):
                best = min(best, p[2] - lowest - env.tray_wall_top)
        return float(best)


def _f(x):
    return float(x) if np.isfinite(x) else float("nan")


def sample_cube_pose(env, cfg, rng, max_tries=10000):
    """Uniform (x, y) on the table, inside the dexterous annulus r∈[r_min, r_max], away from
    the edges and the tray; yaw uniform in cfg.yaw_range."""
    x0, x1, y0, y1 = env.table_xy
    m = cfg.table_margin
    keep = env.tray_outer_half + math.sqrt(2) * env.cube_half + cfg.tray_keepout
    c, s = math.cos(-env.tray_yaw), math.sin(-env.tray_yaw)
    for _ in range(max_tries):
        x, y = rng.uniform(x0 + m, x1 - m), rng.uniform(y0 + m, y1 - m)
        if not cfg.r_min <= math.hypot(x, y) <= cfg.r_max:
            continue
        dx, dy = x - env.tray_pos[0], y - env.tray_pos[1]
        if abs(c * dx - s * dy) < keep and abs(s * dx + c * dy) < keep:
            continue
        return x, y, float(rng.uniform(*cfg.yaw_range))
    raise RuntimeError("could not sample a cube pose - check table / r_min / r_max")
