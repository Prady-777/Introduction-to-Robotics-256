#!/usr/bin/env python3
"""One window, all 12 episodes (mink, dls, qp; episodes 4-7), shown at PLAYBACK_SPEED x real time.
   Physics and trajectories are identical to the logged runs - only the display is faster.
   python scripts/demo_all.py"""
import os, sys, time
import numpy as np
import mujoco
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pick_place.config import Config, SCENE_XML
from utils.pick_place.env import PickPlaceEnv, ARM_JOINTS
from utils.pick_place.ik_solvers import make_solver
from utils.pick_place.planner import PickPlacePlanner, sample_cube_pose

PLAYBACK_SPEED = 1.5
METHODS = {"mink": "MINK (off-the-shelf IK)", "dls": "DLS-IK", "qp": "QP-IK"}
EPISODES = [4, 5, 6, 7]

cfg = Config()
env = PickPlaceEnv(SCENE_XML, cfg, viewer=True, realtime=False)
dt_sync = cfg.viewer_every * env.model.opt.timestep
clock = {"start": time.time(), "sim": 0.0}

orig_step = env.step
def paced_step(q, qd, phase=""):
    orig_step(q, qd, phase)
    if env._step_count % cfg.viewer_every == 0:
        clock["sim"] += dt_sync
        lag = clock["start"] + clock["sim"] / PLAYBACK_SPEED - time.time()
        if lag > 0:
            time.sleep(lag)
        elif lag < -0.2:                      # computer too slow: don't try to catch up
            clock["start"] -= lag
env.step = paced_step

def text(a, b=""):
    try:
        env.viewer.set_texts((mujoco.mjtFontScale.mjFONTSCALE_200,
                              mujoco.mjtGridPos.mjGRID_TOPLEFT, a, b))
    except Exception:
        pass

def wait(s):
    t = time.time() + s
    while time.time() < t and env.viewer.is_running():
        env.viewer.sync(); time.sleep(0.02)

for i in range(3, 0, -1):
    print(f"starting in {i} ...", flush=True); text(f"Starting in {i}"); wait(1)

t_begin = time.time()
for m, name in METHODS.items():
    planner = PickPlacePlanner(env, make_solver(m, env.kin, model=env.model, joint_names=ARM_JOINTS,
                                                **cfg.solver_kwargs(m)), cfg)
    print(f"\n===== {name} =====", flush=True)
    text(name, f"playback {PLAYBACK_SPEED}x"); wait(1.5)
    for ep in EPISODES:
        x, y, yaw = sample_cube_pose(env, cfg, np.random.default_rng(ep))
        text(name, f"episode {ep}   ({PLAYBACK_SPEED}x)")
        clock["start"], clock["sim"] = time.time(), 0.0
        log = planner.run_episode(x, y, yaw)
        res = "SUCCESS" if log["success"] else "FAIL " + log["failure_reason"]
        print(f"[{m}] episode {ep}: {res}", flush=True)
        text(name, f"episode {ep}: {res}"); wait(0.5)

print(f"\nDONE in {time.time() - t_begin:.0f} s - stop your screen recording.", flush=True)
text("DONE"); wait(2)
os._exit(0)
