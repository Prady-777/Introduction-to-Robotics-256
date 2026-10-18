#!/usr/bin/env python3
"""
Task + dexterous workspace of the HEAL arm (TCP of the parallel gripper).

  python scripts/workspace_analysis.py                    # MuJoCo kinematics (default)
  python scripts/workspace_analysis.py --backend numpy    # same maths without MuJoCo
  python scripts/workspace_analysis.py --quick            # coarser grid, ~4x faster

Definitions
  Task workspace      : all TCP positions reachable within the joint limits
                        (Monte-Carlo FK over uniformly sampled joint vectors).
  Dexterous workspace : positions where a TOP-DOWN grasp (approach axis = -z) is feasible
                        for EVERY tested yaw in [-π, π)  -> needed for random cube yaw.
Joint 1 rotates about the world z axis through the base, so dexterity only depends on
(r, z) with r = sqrt(x² + y²). We therefore scan an (r, z) grid and revolve it.

Outputs -> results/workspace/
  task_workspace_3d.png, task_workspace_views.png, dexterous_rz.png,
  dexterous_xy_at_grasp_height.png, workspace_summary.json, dexterity_grid.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle, Polygon, Circle  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
from utils.pick_place.config import Config, SCENE_XML, RESULTS_DIR  # noqa: E402
from utils.pick_place.kinematics import top_down_rotation  # noqa: E402
from utils.pick_place.ik_solvers import DLSIK  # noqa: E402

# scene numbers used for overlays (must match pick_place_scene.xml)
TABLE = dict(x0=0.20, x1=0.60, y0=-0.45, y1=0.45, top=0.235)
TRAY = dict(x=0.40, y=-0.28, yaw=0.4, half=0.09)
CUBE_HALF = 0.025


def make_kinematics(backend):
    if backend == "numpy":
        from utils.pick_place.xml_kinematics import XmlChainKinematics
        return XmlChainKinematics(SCENE_XML)
    import mujoco
    from utils.pick_place.kinematics import MjKinematics
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    return MjKinematics(model, [f"joint_{i}" for i in range(1, 7)], "tcp")


def task_workspace(kin, n, rng):
    Q = rng.uniform(kin.lower, kin.upper, size=(n, kin.n))
    P = np.array([kin.fk(q)[0] for q in Q])
    return P[P[:, 2] > 0.0]          # below the floor is meaningless


def dexterity_grid(kin, rs, zs, yaws, rng, restarts=2):
    """F[i, j] = fraction of yaws for which a top-down pose at (r_j, 0, z_i) is solvable.
    Continuation: for each (r, yaw) start at the top and walk down, warm-starting the IK
    from the previous height; random restarts when warm-start fails."""
    ik = DLSIK(kin, max_iters=150, null_gain=0.0)
    F = np.zeros((len(zs), len(rs)))
    for j, r in enumerate(rs):
        for yaw in yaws:
            R = top_down_rotation(yaw)
            q = None
            for seed in [np.zeros(kin.n)] + [rng.uniform(-2.5, 2.5, kin.n) for _ in range(6)]:
                res = ik.solve(np.array([r, 0, zs[0]]), R, seed)
                if res.success:
                    q = res.q
                    break
            for i, z in enumerate(zs):
                ok = False
                if q is not None:
                    res = ik.solve(np.array([r, 0, z]), R, q)
                    ok = res.success
                if not ok:
                    for _ in range(restarts):
                        res = ik.solve(np.array([r, 0, z]), R, rng.uniform(-2.5, 2.5, kin.n))
                        if res.success:
                            ok = True
                            break
                if ok:
                    q = res.q
                F[i, j] += ok
        print(f"  r = {r:.3f} m done", flush=True)
    return F / len(yaws)


def draw_scene_xy(ax, cfg, show_band=True):
    ax.add_patch(Rectangle((TABLE["x0"], TABLE["y0"]), TABLE["x1"] - TABLE["x0"],
                           TABLE["y1"] - TABLE["y0"], fill=False, ec="saddlebrown", lw=2.5,
                           label="table"))
    c, s = np.cos(TRAY["yaw"]), np.sin(TRAY["yaw"])
    h = TRAY["half"]
    corners = [(TRAY["x"] + c * a - s * b, TRAY["y"] + s * a + c * b)
               for a, b in ((h, h), (-h, h), (-h, -h), (h, -h))]
    ax.add_patch(Polygon(corners, closed=True, fill=False, ec="tab:blue", lw=2, label="tray"))
    if show_band:
        for rr, ls in ((cfg.r_min, "--"), (cfg.r_max, "--")):
            ax.add_patch(Circle((0, 0), rr, fill=False, ec="k", ls=ls, lw=1))
    ax.plot(0, 0, "k^", ms=10, label="robot base")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["mujoco", "numpy"], default="mujoco")
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--replot", action="store_true", help="reuse saved dexterity_grid.npz")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "workspace"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = Config()
    rng = np.random.default_rng(0)
    kin = make_kinematics(args.backend)
    t0 = time.time()

    # ---------------- task workspace ----------------
    P = task_workspace(kin, args.samples, rng)
    r = np.hypot(P[:, 0], P[:, 1])
    print(f"task workspace: {len(P)} pts, max reach r = {r.max():.3f} m, "
          f"z in [{P[:, 2].min():.3f}, {P[:, 2].max():.3f}] m")

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    sub = P[rng.choice(len(P), min(8000, len(P)), replace=False)]
    sc = ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], c=sub[:, 2], s=1, cmap="viridis", alpha=0.5)
    xx, yy = np.meshgrid([TABLE["x0"], TABLE["x1"]], [TABLE["y0"], TABLE["y1"]])
    ax.plot_surface(xx, yy, np.full_like(xx, TABLE["top"]), color="saddlebrown", alpha=0.5)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title("Task workspace (TCP positions, uniform joint sampling)")
    fig.colorbar(sc, shrink=0.6, label="z [m]")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "task_workspace_3d.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    hb = axs[0].hexbin(P[:, 0], P[:, 1], gridsize=60, cmap="Greys", mincnt=1)
    draw_scene_xy(axs[0], cfg)
    axs[0].set_aspect("equal"); axs[0].set_xlabel("x [m]"); axs[0].set_ylabel("y [m]")
    axs[0].set_title("Top view: TCP sample density"); axs[0].legend(loc="lower left", fontsize=8)
    fig.colorbar(hb, ax=axs[0], label="samples")
    hb = axs[1].hexbin(r, P[:, 2], gridsize=60, cmap="Greys", mincnt=1)
    axs[1].axhline(TABLE["top"], color="saddlebrown", lw=2, label=f"table top H = {TABLE['top']} m")
    axs[1].axvspan(cfg.r_min, cfg.r_max, color="g", alpha=0.15, label="cube sampling band")
    axs[1].set_xlabel("r = √(x²+y²) [m]"); axs[1].set_ylabel("z [m]")
    axs[1].set_title("Side view (revolved about joint 1)"); axs[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "task_workspace_views.png"), dpi=150)
    plt.close(fig)

    # ---------------- dexterous workspace ----------------
    step = 0.05 if args.quick else 0.025
    rs = np.round(np.arange(0.10, 0.75 + 1e-9, 0.05), 3)
    zs = np.round(np.arange(0.55, 0.125 - 1e-9, -step), 3)
    yaws = np.linspace(-np.pi, np.pi, 8 if args.quick else 12, endpoint=False)
    npz = os.path.join(args.out, "dexterity_grid.npz")
    if args.replot and os.path.exists(npz):
        d = np.load(npz)
        F, rs, zs, yaws = d["F"], d["rs"], d["zs"], d["yaws"]
        step = float(abs(zs[0] - zs[1]))
        print("re-using", npz)
    else:
        print(f"dexterity grid: {len(rs)} r x {len(zs)} z x {len(yaws)} yaw ...")
        F = dexterity_grid(kin, rs, zs, yaws, rng)
        np.savez(npz, F=F, rs=rs, zs=zs, yaws=yaws)

    grasp_z = TABLE["top"] + CUBE_HALF
    transit_z = TABLE["top"] + 0.06 + cfg.clearance + CUBE_HALF   # tray wall top + clearance
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(F, origin="upper", aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                   extent=[rs[0] - 0.025, rs[-1] + 0.025, zs[-1] - step / 2, zs[0] + step / 2])
    ax.axhline(TABLE["top"], color="saddlebrown", lw=2.5, label=f"table top H = {TABLE['top']:.3f} m")
    ax.axhline(grasp_z, color="k", ls="--", label=f"grasp TCP z = {grasp_z:.3f} m")
    ax.axhline(transit_z, color="tab:blue", ls=":", label=f"transit TCP z = {transit_z:.3f} m")
    ax.axvline(cfg.r_min, color="k", lw=1); ax.axvline(cfg.r_max, color="k", lw=1)
    ax.set_xlabel("r [m]"); ax.set_ylabel("TCP z [m]")
    ax.set_title("Dexterous workspace: fraction of yaws with a feasible top-down grasp")
    fig.colorbar(im, label="fraction of yaws feasible (1 = dexterous)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dexterous_rz.png"), dpi=150)
    plt.close(fig)

    # XY map: a cube at radius r is pickable if EVERY height of the vertical approach
    # column (grasp height .. transit height) is dexterous -> min over that z-range.
    col = (zs >= grasp_z - step + 1e-9) & (zs <= transit_z + step / 2)
    prof = F[col].min(axis=0)
    i_g = int(np.argmin(np.abs(zs - grasp_z)))
    gx, gy = np.meshgrid(np.linspace(-0.8, 0.8, 500), np.linspace(-0.8, 0.8, 500))
    gr = np.hypot(gx, gy)
    M = np.interp(gr, rs, prof, left=prof[0], right=0.0)
    fig, ax = plt.subplots(figsize=(7, 8))
    im = ax.pcolormesh(gx, gy, M, cmap="RdYlGn", vmin=0, vmax=1, shading="auto")
    draw_scene_xy(ax, cfg)
    ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_title(f"Vertical top-down approach feasible for all yaws, z ∈ [{zs[col].min():.3f}, "
                 f"{zs[col].max():.3f}] m\n(dashed: cube sampling annulus r ∈ [r_min, r_max])")
    fig.colorbar(im, label="fraction of yaws feasible")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "dexterous_xy_at_grasp_height.png"), dpi=150)
    plt.close(fig)

    # ---------------- summary / table recommendation ----------------
    band = (rs >= cfg.r_min - 1e-9) & (rs <= cfg.r_max + 1e-9)
    full = np.all(F[:, band] >= 1.0 - 1e-9, axis=1)
    z_ok = zs[full]
    z_min_dex = float(z_ok.min()) if len(z_ok) else float("nan")
    dex_r = rs[prof >= 1.0 - 1e-9]
    summary = {
        "backend": args.backend,
        "task_workspace": {"max_reach_r": float(r.max()), "z_min": float(P[:, 2].min()),
                           "z_max": float(P[:, 2].max()), "samples": int(len(P))},
        "dexterous": {"lowest_tcp_z_fully_dexterous_over_band": z_min_dex,
                      "band_r": [cfg.r_min, cfg.r_max],
                      "dexterous_r_whole_approach_column": [float(dex_r.min()), float(dex_r.max())]
                      if len(dex_r) else None,
                      "yaws_tested": len(yaws)},
        "table_design": {
            "H_recommended_min": z_min_dex - CUBE_HALF if np.isfinite(z_min_dex) else None,
            "H_chosen": TABLE["top"], "L_x": TABLE["x1"] - TABLE["x0"],
            "W_y": TABLE["y1"] - TABLE["y0"], "grasp_tcp_z": grasp_z, "transit_tcp_z": transit_z},
        "runtime_s": time.time() - t0,
    }
    with open(os.path.join(args.out, "workspace_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"plots -> {args.out}")


if __name__ == "__main__":
    main()
