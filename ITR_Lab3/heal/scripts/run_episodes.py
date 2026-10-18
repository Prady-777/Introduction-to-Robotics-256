#!/usr/bin/env python3
"""
Run N randomised pick-and-place episodes with one IK method and log everything.

Examples
  python scripts/run_episodes.py --ik mink --n 25                    # headless batch
  python scripts/run_episodes.py --ik dls  --n 5 --viewer --realtime  # watch it
  python scripts/run_episodes.py --ik qp   --n 4 --record results/videos/qp.mp4

Episode i always uses seed (--seed + i), so the same cube poses are used for every IK
method (fair comparison). Outputs (default --out results/logs):
  results/logs/<ik>/episodes.csv      one row per episode (all metrics)
  results/logs/<ik>/episodes.jsonl    same, JSON lines
  results/logs/<ik>/summary.json      aggregate metrics
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)

from utils.pick_place.config import Config, SCENE_XML, RESULTS_DIR  # noqa: E402
from utils.pick_place.env import PickPlaceEnv, ARM_JOINTS  # noqa: E402
from utils.pick_place.ik_solvers import make_solver  # noqa: E402
from utils.pick_place.planner import PickPlacePlanner, sample_cube_pose  # noqa: E402


def summarize(rows):
    ok = np.array([r["success"] for r in rows], bool)
    reasons = {}
    for r in rows:
        if not r["success"]:
            reasons[r["failure_reason"]] = reasons.get(r["failure_reason"], 0) + 1

    def stat(key, mask=None):
        v = np.array([r[key] for r in rows], float)
        if mask is not None:
            v = v[mask]
        v = v[np.isfinite(v)]
        return {"mean": float(v.mean()), "std": float(v.std()), "min": float(v.min()),
                "max": float(v.max())} if len(v) else None

    return {
        "episodes": len(rows), "successes": int(ok.sum()), "success_rate": float(ok.mean()),
        "failure_reasons": reasons,
        "plan_time_s": stat("plan_time"), "ik_iters_mean": stat("ik_iters_mean"),
        "ik_iters_total": stat("ik_iters_total"),
        "min_clear_cube_tray": stat("min_clear_cube_tray"),
        "min_clear_robot_table": stat("min_clear_robot_table"),
        "min_clear_gripper_tray": stat("min_clear_gripper_tray"),
        "joint_limit_hits": stat("limit_hits"), "vel_violations": stat("vel_violations"),
        "max_planned_vel": stat("max_planned_vel"), "max_track_err": stat("max_track_err"),
        "replans": stat("replans"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ik", choices=["mink", "dls", "qp"], required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--start", type=int, default=0, help="first episode index")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="slow viewer to real time")
    ap.add_argument("--record", default=None, help="mp4 path (offscreen render)")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "logs"))
    ap.add_argument("--no-snapshots", action="store_true",
                    help="do not save a PNG for failed episodes")
    args = ap.parse_args()

    cfg = Config()
    env = PickPlaceEnv(SCENE_XML, cfg, viewer=args.viewer, record_path=args.record,
                       realtime=args.realtime, label=args.ik.upper())
    solver = make_solver(args.ik, env.kin, model=env.model, joint_names=ARM_JOINTS,
                         **cfg.solver_kwargs(args.ik))
    planner = PickPlacePlanner(env, solver, cfg)

    out_dir = os.path.join(args.out, args.ik)
    os.makedirs(out_dir, exist_ok=True)
    if args.record:
        os.makedirs(os.path.dirname(os.path.abspath(args.record)), exist_ok=True)
    csv_path = os.path.join(out_dir, "episodes.csv")
    jsonl_path = os.path.join(out_dir, "episodes.jsonl")
    rows, writer = [], None
    t0 = time.time()
    print(f"[{args.ik}] table top z={env.table_top_z:.3f}  tray at {np.round(env.tray_pos[:2], 3)} "
          f"yaw={env.tray_yaw:.2f}  safe transit TCP z={planner.safe_height():.3f}")
    with open(csv_path, "w", newline="") as fcsv, open(jsonl_path, "w") as fj:
        for ep in range(args.start, args.start + args.n):
            seed = args.seed + ep
            rng = np.random.default_rng(seed)
            x, y, yaw = sample_cube_pose(env, cfg, rng)
            env.label = f"{args.ik.upper()} | episode {ep} | cube ({x:.2f}, {y:.2f}, {np.degrees(yaw):.0f} deg)"
            log = planner.run_episode(x, y, yaw)
            log = {"method": args.ik, "episode": ep, "seed": seed, **log}
            log = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                   for k, v in log.items()}
            rows.append(log)
            if writer is None:
                writer = csv.DictWriter(fcsv, fieldnames=list(log.keys()), extrasaction="ignore")
                writer.writeheader()
            writer.writerow(log)
            fcsv.flush()
            fj.write(json.dumps(log) + "\n")
            fj.flush()
            if not log["success"] and not args.no_snapshots:
                try:
                    snap_dir = os.path.join(out_dir, "failures")
                    os.makedirs(snap_dir, exist_ok=True)
                    env.snapshot(os.path.join(snap_dir, f"ep{ep:03d}_{log['failure_reason']}.png"))
                except Exception as e:          # no OpenGL context -> skip silently
                    print(f"  (snapshot skipped: {e})")
            status = "OK  " if log["success"] else f"FAIL {log['failure_reason']} ({log['failure_phase']})"
            print(f"[{args.ik}] ep {ep:3d}  cube=({x:+.3f},{y:+.3f},{yaw:+.2f})  {status:45s} "
                  f"plan={log['plan_time']:.2f}s  iters/knot={log['ik_iters_mean']:.1f}  "
                  f"clear(cube-tray)={log['min_clear_cube_tray']:.3f}", flush=True)
            if args.viewer and env.viewer is not None and not env.viewer.is_running():
                break

    summary = summarize(rows)
    summary.update(method=args.ik, seed=args.seed, wall_time_s=time.time() - t0,
                   ik_backend=getattr(solver, "backend", None) or getattr(solver, "solver", None))
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[{args.ik}] success {summary['successes']}/{summary['episodes']} "
          f"({100 * summary['success_rate']:.0f} %)  failures: {summary['failure_reasons']}")
    print(f"logs -> {out_dir}")
    env.close()


if __name__ == "__main__":
    main()
