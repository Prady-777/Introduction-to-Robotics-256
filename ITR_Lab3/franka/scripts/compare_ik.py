#!/usr/bin/env python3
"""
Compare the three IK methods on the SAME seeds.

  python scripts/compare_ik.py                 # reads results/logs/{mink,dls,qp}/episodes.csv

Writes to results/comparison/:
  summary_table.md / summary_table.csv   metrics table for the report
  stats.md                               paired tests (McNemar on success, Wilcoxon on time/iters)
  success_rate.png, plan_time.png, ik_iterations.png, clearance.png,
  failure_reasons.png, violations.png, episode_grid.png, cube_positions.png
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(REPO_ROOT)
from utils.pick_place.config import RESULTS_DIR, SCENE_XML  # noqa: E402
from utils.pick_place.xml_kinematics import scene_geometry  # noqa: E402

METHODS = ["mink", "dls", "qp"]
LABELS = {"mink": "Mink (baseline)", "dls": "DLS-IK", "qp": "QP-IK"}
COLORS = {"mink": "#4C72B0", "dls": "#DD8452", "qp": "#55A868"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def to_markdown(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out) + "\n"


def load(log_dir):
    dfs = {}
    for m in METHODS:
        p = os.path.join(log_dir, m, "episodes.csv")
        if os.path.exists(p):
            dfs[m] = pd.read_csv(p).sort_values("episode").reset_index(drop=True)
    return dfs


def boxplot(ax, dfs, col, ylabel, success_only=False):
    data, labels = [], []
    for m, df in dfs.items():
        v = df[df.success] if success_only else df
        v = v[col].replace([np.inf, -np.inf], np.nan).dropna().values
        data.append(v)
        labels.append(LABELS[m])
    bp = ax.boxplot(data, patch_artist=True, showmeans=True)
    ax.set_xticks(range(1, len(labels) + 1), labels)
    for patch, m in zip(bp["boxes"], dfs):
        patch.set_facecolor(COLORS[m]); patch.set_alpha(0.6)
    for i, v in enumerate(data):
        ax.scatter(np.random.default_rng(0).normal(i + 1, 0.04, len(v)), v, s=8, c="k", alpha=0.5)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=os.path.join(RESULTS_DIR, "logs"))
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "comparison"))
    args = ap.parse_args()
    dfs = load(args.logs)
    if not dfs:
        sys.exit("no logs found - run scripts/run_episodes.py first")
    os.makedirs(args.out, exist_ok=True)

    # ---------------- summary table ----------------
    rows = []
    for m, df in dfs.items():
        n, k = len(df), int(df.success.sum())
        p, lo, hi = wilson(k, n)

        def ms(col, d=df):
            v = d[col].replace([np.inf, -np.inf], np.nan).dropna()
            return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) else "n/a"
        reasons = df[~df.success].failure_reason.value_counts().to_dict()
        rows.append({
            "Method": LABELS[m], "Episodes": n,
            "Success rate": f"{100 * p:.0f} % ({k}/{n}) [{100 * lo:.0f}-{100 * hi:.0f}]",
            "Plan time / ep [s]": ms("plan_time"),
            "IK iters / knot": ms("ik_iters_mean"),
            "IK iters / ep": ms("ik_iters_total"),
            "Min clear cube-tray [m]": ms("min_clear_cube_tray"),
            "Min clear robot-table [m]": ms("min_clear_robot_table"),
            "Joint-limit hits / ep": ms("limit_hits"),
            "Vel. violations / ep": ms("vel_violations"),
            "Max planned |q̇| [rad/s]": ms("max_planned_vel"),
            "Failure reasons": ", ".join(f"{r}: {c}" for r, c in reasons.items()) or "-",
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(args.out, "summary_table.csv"), index=False)
    with open(os.path.join(args.out, "summary_table.md"), "w") as f:
        f.write(to_markdown(tab))
    print(tab.to_string(index=False))

    # ---------------- paired statistics (same seeds) ----------------
    lines = ["# Paired comparison (identical episode seeds)\n"]
    try:
        from scipy.stats import wilcoxon, binomtest
    except ImportError:
        wilcoxon = binomtest = None
    ms_ = list(dfs)
    for i in range(len(ms_)):
        for j in range(i + 1, len(ms_)):
            a, b = dfs[ms_[i]], dfs[ms_[j]]
            common = sorted(set(a.episode) & set(b.episode))
            a = a.set_index("episode").loc[common]
            b = b.set_index("episode").loc[common]
            n01 = int((~a.success & b.success).sum())
            n10 = int((a.success & ~b.success).sum())
            line = f"## {LABELS[ms_[i]]} vs {LABELS[ms_[j]]}  (n={len(common)})\n"
            if binomtest and n01 + n10 > 0:
                pv = binomtest(n10, n01 + n10, 0.5).pvalue
                line += f"- McNemar exact (success): discordant {n10} vs {n01}, p = {pv:.3g}\n"
            else:
                line += f"- Success discordant pairs: {n10} vs {n01}\n"
            for col in ("plan_time", "ik_iters_mean"):
                d = (a[col] - b[col]).dropna()
                if wilcoxon and len(d) > 5 and np.any(d != 0):
                    pv = wilcoxon(a[col].loc[d.index], b[col].loc[d.index]).pvalue
                    line += f"- Wilcoxon {col}: median diff = {d.median():.4f}, p = {pv:.3g}\n"
            lines.append(line)
    with open(os.path.join(args.out, "stats.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))

    # ---------------- plots ----------------
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, (m, df) in enumerate(dfs.items()):
        p, lo, hi = wilson(int(df.success.sum()), len(df))
        ax.bar(i, 100 * p, color=COLORS[m], yerr=[[100 * (p - lo)], [100 * (hi - p)]], capsize=6)
        ax.text(i, 100 * p + 2, f"{100 * p:.0f}%", ha="center")
    ax.set_xticks(range(len(dfs)), [LABELS[m] for m in dfs])
    ax.set_ylabel("success rate [%] (95% Wilson CI)"); ax.set_ylim(0, 110); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "success_rate.png"), dpi=150); plt.close(fig)

    for col, fname, ylab in [("plan_time", "plan_time.png", "IK planning time per episode [s]"),
                             ("ik_iters_mean", "ik_iterations.png", "mean IK iterations per knot")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        boxplot(ax, dfs, col, ylab)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, fname), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, t in zip(axs, ["min_clear_cube_tray", "min_clear_gripper_tray", "min_clear_robot_table"],
                          ["cube ↔ tray walls", "gripper ↔ tray walls", "robot ↔ table"]):
        boxplot(ax, dfs, col, "min distance [m]")
        ax.set_title(t); ax.axhline(0, color="r", lw=1)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "clearance.png"), dpi=150); plt.close(fig)

    reasons = sorted({r for df in dfs.values() for r in df[~df.success].failure_reason.dropna()})
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(dfs))
    for r in reasons:
        cnt = np.array([(df.failure_reason == r).sum() for df in dfs.values()])
        ax.bar([LABELS[m] for m in dfs], cnt, bottom=bottom, label=r)
        bottom += cnt
    ax.set_ylabel("failed episodes"); ax.set_title("Failure reasons")
    if reasons:
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "failure_reasons.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, t in zip(axs, ["limit_hits", "vel_violations", "max_planned_vel"],
                          ["joint-limit activations / episode", "velocity-limit violations / episode",
                           "max planned joint speed [rad/s]"]):
        boxplot(ax, dfs, col, t)
    axs[2].axhline(2.0, color="r", ls="--", lw=1, label="v_max"); axs[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "violations.png"), dpi=150); plt.close(fig)

    eps = sorted(set().union(*[set(df.episode) for df in dfs.values()]))
    grid = np.full((len(dfs), len(eps)), np.nan)
    for i, df in enumerate(dfs.values()):
        for _, r in df.iterrows():
            grid[i, eps.index(r.episode)] = float(r.success)
    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(eps)), 2.2))
    ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(len(dfs)), [LABELS[m] for m in dfs]); ax.set_xlabel("episode (same seed per column)")
    ax.set_title("Per-episode outcome (green = success)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "episode_grid.png"), dpi=150); plt.close(fig)

    fig, axs = plt.subplots(1, len(dfs), figsize=(5 * len(dfs), 4.5), squeeze=False)
    for ax, (m, df) in zip(axs[0], dfs.items()):
        ok = df.success
        ax.scatter(df.cube_y[ok], df.cube_x[ok], c="g", label="success")
        ax.scatter(df.cube_y[~ok], df.cube_x[~ok], c="r", marker="x", label="failure")
        for _, r in df.iterrows():
            ax.plot([r.cube_y, r.cube_y + 0.03 * np.sin(r.cube_yaw)],
                    [r.cube_x, r.cube_x + 0.03 * np.cos(r.cube_yaw)], "k-", lw=0.8)
        g = scene_geometry(SCENE_XML)
        ax.add_patch(plt.Rectangle((g["y0"], g["x0"]), g["y1"] - g["y0"], g["x1"] - g["x0"],
                                   fill=False, ec="saddlebrown", lw=2))
        ax.set_title(LABELS[m]); ax.set_xlabel("y [m]"); ax.set_ylabel("x [m]")
        ax.set_aspect("equal"); ax.invert_xaxis(); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Initial cube poses (top view, robot base at origin)")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "cube_positions.png"), dpi=150); plt.close(fig)
    print(f"\nplots + tables -> {args.out}")


if __name__ == "__main__":
    main()
