#!/usr/bin/env python3
"""Run ONE episode and print what the cube does inside the gripper.
   python scripts/debug_episode.py dls 22"""
import os, sys
import numpy as np
import mujoco
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.pick_place.config import Config, SCENE_XML
from utils.pick_place.env import PickPlaceEnv, ARM_JOINTS
from utils.pick_place.ik_solvers import make_solver
from utils.pick_place.planner import PickPlacePlanner, sample_cube_pose

method, ep = sys.argv[1], int(sys.argv[2])
cfg = Config(); env = PickPlaceEnv(SCENE_XML, cfg); m, d = env.model, env.data
solver = make_solver(method, env.kin, model=m, joint_names=ARM_JOINTS, **cfg.solver_kwargs(method))
pl = PickPlacePlanner(env, solver, cfg)
fl = m.jnt_qposadr[m.joint("finger_left_joint").id]; fr = m.jnt_qposadr[m.joint("finger_right_joint").id]
pads = {m.geom("finger_left_pad").id, m.geom("finger_right_pad").id}
rows, orig = [], env.step
def step(q_des, qd_des, phase=""):
    orig(q_des, qd_des, phase)
    if env._step_count % 10: return
    p, R = env.tcp_pose(); rel = R.T @ (env.cube_pos() - p)
    n, F, f6 = 0, 0.0, np.zeros(6)
    for i in range(d.ncon):
        c = d.contact[i]
        if env.cube_geom in (c.geom1, c.geom2) and (c.geom1 in pads or c.geom2 in pads):
            mujoco.mj_contactForce(m, d, i, f6); n += 1; F += abs(f6[0])
    rows.append((d.time, phase, *(1000 * rel), np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))),
                 1000 * d.qpos[fl], 1000 * d.qpos[fr], n, F,
                 np.abs(d.qvel[env.dadr]).max(), np.degrees(env.q[-1])))
env.step = step
x, y, yaw = sample_cube_pose(env, cfg, np.random.default_rng(ep))
log = pl.run_episode(x, y, yaw)
print(f"{method} ep {ep}: success={log['success']} reason={log['failure_reason']} "
      f"grasp_yaw={np.degrees(log['grasp_yaw']):.1f}deg place_yaw={np.degrees(log['place_yaw']):.1f}deg")
print(" time  phase     cube_in_gripper[mm] x/y/z  tilt  fingerL/R[mm] nC  F[N]  max|qd|  q_last[deg]")
keep = [r for r in rows if r[1] in ("grasp", "lift", "transit", "lower")]
last_t = -1
for i, r in enumerate(keep):
    if r[0] - last_t >= 0.2 or i >= len(keep) - 8:
        last_t = r[0]
        print(f"{r[0]:5.2f} {r[1]:8s} {r[2]:7.1f} {r[3]:7.1f} {r[4]:7.1f}  {r[5]:5.1f}  "
              f"{r[6]:5.1f} {r[7]:5.1f}  {r[8]:2d} {r[9]:5.1f}  {r[10]:6.2f}  {r[11]:7.1f}")
