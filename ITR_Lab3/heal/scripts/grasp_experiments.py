#!/usr/bin/env python3
"""Run the same episodes under different contact/gripper physics and measure in-hand slip.
   python scripts/grasp_experiments.py                  (defaults: methods dls mink, episodes 22 21)"""
import argparse, os, sys
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pick_place.config import Config, SCENE_XML
from utils.pick_place.env import PickPlaceEnv, ARM_JOINTS
from utils.pick_place.ik_solvers import make_solver
from utils.pick_place.planner import PickPlacePlanner, sample_cube_pose

VARIANTS = {
    "current":       {},
    "no_noslip":     {"noslip": 0},
    "soft_pads":     {"pad_solref": (0.02, 1.0)},
    "dt_1ms":        {"timestep": 0.001},
    "stiff_fingers": {"finger_kp": 800, "finger_force": 30},
    "condim6":       {"condim": 6},
    "combo":         {"timestep": 0.001, "pad_solref": (0.02, 1.0), "finger_kp": 800,
                      "finger_force": 30, "noslip": 10},
}

def apply(env, v):
    m = env.model
    pads = [m.geom("finger_left_pad").id, m.geom("finger_right_pad").id]
    if "noslip" in v: m.opt.noslip_iterations = v["noslip"]
    if "timestep" in v: m.opt.timestep = v["timestep"]
    if "pad_solref" in v:
        for g in pads: m.geom_solref[g] = v["pad_solref"]
    if "condim" in v:
        for g in pads + [env.cube_geom]: m.geom_condim[g] = v["condim"]
    if "finger_kp" in v:
        for a in env.finger_act:
            m.actuator_gainprm[a, 0] = v["finger_kp"]; m.actuator_biasprm[a, 1] = -v["finger_kp"]
    if "finger_force" in v:
        for a in env.finger_act: m.actuator_forcerange[a] = [-v["finger_force"], v["finger_force"]]

def run(method, ep, v):
    cfg = Config(); env = PickPlaceEnv(SCENE_XML, cfg); apply(env, v); m, d = env.model, env.data
    solver = make_solver(method, env.kin, model=m, joint_names=ARM_JOINTS, **cfg.solver_kwargs(method))
    pl = PickPlacePlanner(env, solver, cfg)
    fl = m.jnt_qposadr[m.joint("finger_left_joint").id]
    st = {"slip": 0.0, "open": 0.0}; orig = env.step
    def step(q_des, qd_des, phase=""):
        orig(q_des, qd_des, phase)
        if phase in ("lift", "transit", "lower"):
            p, R = env.tcp_pose(); rel = R.T @ (env.cube_pos() - p)
            st["slip"] = max(st["slip"], 1000 * np.hypot(rel[0], rel[1]))
            st["open"] = max(st["open"], 1000 * d.qpos[fl])
    env.step = step
    x, y, yaw = sample_cube_pose(env, cfg, np.random.default_rng(ep))
    log = pl.run_episode(x, y, yaw)
    return log, st, m

ap = argparse.ArgumentParser()
ap.add_argument("--methods", nargs="+", default=["dls", "mink"])
ap.add_argument("--eps", nargs="+", type=int, default=[22, 21])
ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
a = ap.parse_args()
first = True
print(f"{'variant':14s} {'method':6s} {'ep':>3s}  {'result':28s} {'max slip[mm]':>12s} {'max finger[mm]':>14s}")
for name in a.variants:
    for method in a.methods:
        for ep in a.eps:
            log, st, m = run(method, ep, VARIANTS[name])
            if first:
                print(f"(scene check: noslip_iterations={m.opt.noslip_iterations}, "
                      f"pad half-width={m.geom_size[m.geom('finger_left_pad').id][1]*1000:.0f} mm)")
                first = False
            res = "OK" if log["success"] else f"FAIL {log['failure_reason']}"
            print(f"{name:14s} {method:6s} {ep:3d}  {res:28s} {st['slip']:12.1f} {st['open']:14.1f}", flush=True)
