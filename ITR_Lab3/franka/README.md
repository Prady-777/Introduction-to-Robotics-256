# ITR Lab 3 – Tutorial 4: Pick and Place with IK — **Franka Emika Panda**

The 7-DoF Franka Panda picks a randomly posed 5 cm cube from a table and places it in a tray,
using three interchangeable IK methods on identical random seeds:

| Method | Class in `utils/pick_place/ik_solvers.py` | Idea |
|---|---|---|
| **Mink** (baseline) | `MinkIK` | [kevinzakka/mink](https://github.com/kevinzakka/mink): FrameTask + PostureTask, ConfigurationLimit + VelocityLimit |
| **DLS-IK** | `DLSIK` | closed-loop q̇ = (JᵀJ+λI)⁻¹Jᵀẋ_d + null-space posture term |
| **QP-IK** | `QPIK` | the same cost as a QP ½ΔqᵀHΔq + cᵀΔq s.t. GΔq ≤ h with joint + velocity limits |

---

## 1. Setup (Ubuntu)

```bash
cd ~/Documents/ITR/<this folder>/Lab3          # wherever you unzipped it
sudo apt install python3.10-venv               # only if "ensurepip is not available"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python tests/test_ik_math.py                   # 6 x PASS, no MuJoCo needed
```
In every new terminal, `cd` into the folder and run `source .venv/bin/activate` again.

## 2. Workflow

| # | Command | Produces |
|---|---|---|
| 1 | `python tests/test_ik_math.py` | sanity check of the IK maths on the Panda |
| 2 | `python scripts/run_episodes.py --ik dls --n 3 --viewer --realtime` | watch 3 episodes (trail: blue empty, green carrying) |
| 3 | `python scripts/workspace_analysis.py` | `results/workspace/` plots + `workspace_summary.json` |
| 4 | `python scripts/run_episodes.py --ik mink --n 25` (then `dls`, `qp`) | `results/logs/<ik>/episodes.csv`, `summary.json`, `failures/*.png` |
| 5 | `python scripts/compare_ik.py` | `results/comparison/` tables, statistics, plots |
| 6 | start a screen recording, then `python scripts/demo_all.py` | one window, 12 episodes (4 per method), 1.5× playback, ~3 min |

Optional stress test (makes the IK comparison meaningful: the Panda's normal motions stay under 1.2 rad/s):
```bash
sed -i 's/v_max: float = 2.0 /v_max: float = 0.8 /' utils/pick_place/config.py
for m in mink dls qp; do python scripts/run_episodes.py --ik $m --n 25 --out results/stress_logs; done
python scripts/compare_ik.py --logs results/stress_logs --out results/stress_comparison
sed -i 's/v_max: float = 0.8 /v_max: float = 2.0 /' utils/pick_place/config.py
```

Debug helpers: `python scripts/debug_episode.py dls 22` (prints what the cube does inside the gripper),
`python scripts/grasp_experiments.py` (re-runs episodes with physics settings switched off one at a time).

Notes: the "Segmentation fault" / `GLXBadDrawable` message some Linux drivers print when a viewer window
closes is harmless (`run_episodes.py` and `demo_all.py` exit without closing the window for that reason).
Episode *i* always uses seed *i*, so every method sees the same cube poses.

## 3. What is different from the HEAL version

| Topic | HEAL (6-DoF) | Franka Panda (7-DoF) |
|---|---|---|
| Model | rebuilt from `arm_gripper.xml` + HEAL meshes | rebuilt from Franka's official kinematics (same body offsets / joint limits as MuJoCo Menagerie `panda.xml`); capsule links, approximate inertias — the starter `franka_scene.xml` referenced `mjx_panda.xml`, arena and plate files that were not in the zip |
| Redundancy | none (6 DoF for a 6-D pose) | 1 redundant DoF → null-space / posture term actually chooses the elbow pose |
| Joint limits | ±π on every joint | asymmetric and tight, e.g. joint 4 ∈ [−3.07, −0.07], joint 6 ∈ [−0.02, 3.75], joint 7 ∈ ±2.90 → q = 0 is *not* a valid pose |
| Home pose | q = 0 | q = (0, −π/4, 0, −3π/4, 0, π/2, π/4) (classic "ready" pose, TCP straight down at (0.31, 0, 0.52)) |
| Actuation / control | motors, joint damping 55 as D-term | motors with real Panda torque limits (±87 N·m joints 1–4, ±12 N·m joints 5–7); joint damping 40/8/5 as D-term, K_p = (400,400,400,400,100,100,60) |
| Gripper | primitive parallel gripper on the flange | same gripper, mounted 0.107 m from the flange with the Panda hand's 45° offset |
| Dexterous region | r ∈ [0.10, 0.50] m, z ≥ 0.25 m | r ∈ [0.30, 0.70] m, z from 0.05 to ≈ 0.55 m |
| Table (L × W × H) | 0.40 × 0.90 × 0.235 m | **0.48 × 1.10 × 0.20 m** |
| Cube sampling band | r ∈ [0.30, 0.52] m | r ∈ [0.32, 0.68] m |
| Tray | (0.40, −0.28), yaw 0.4 | (0.50, −0.32), yaw 0.4 |

All lessons from the HEAL runs are already included: Mink posture cost 10⁻⁴, DLS/QP null-space gain
0.005, 1 ms time step + noslip solver, stiff fingers (kp 800, 30 N), 40 mm finger pads.

## 4. Repository layout

```
Lab3/
├── robot_descriptions/pick_place_scene.xml   Panda + gripper + table + tray + cube (single file)
├── utils/pick_place/
│   ├── config.py           robot (joints, home, gains) + every tunable number
│   ├── kinematics.py       quaternion maths + MjKinematics (FK / Jacobian on a private MjData)
│   ├── xml_kinematics.py   MuJoCo-free FK of any serial arm in the XML + scene geometry reader
│   ├── ik_solvers.py       MinkIK, DLSIK, QPIK (one interface, any number of joints)
│   ├── env.py              simulation, PD + gravity compensation, contacts, clearances, video
│   └── planner.py          waypoints, Cartesian interpolation, tray-aware placement, logging
├── scripts/  workspace_analysis.py  run_episodes.py  compare_ik.py  demo_all.py
│             debug_episode.py  grasp_experiments.py  make_final_video.py
├── tests/test_ik_math.py
├── report/report.md        report draft (fill the ⟨…⟩ placeholders from results/)
└── results/                generated (workspace plots pre-computed with --backend numpy)
```
The original HEAL files from the starter zip (`heal_meshes/`, `arm_gripper.xml`, `load_heal_robot.py`, …)
are left untouched and are not used by this version.

## 5. How it works

**Control.** τ = qfrc_bias + K_p(q* − q) + b·q̇* with the actuator torque limits of the real Panda.
Gravity/Coriolis are cancelled by `qfrc_bias`; the joint damping b acts as the D-term (implicit in MuJoCo).

**Episode.** approach → descend → grasp → lift → transit → lower → release → retreat; straight Cartesian
lines with minimum-jerk timing sampled at 50 Hz; IK at every knot, warm-started; whole pick (and whole place)
planned before execution. Grasp yaw is the cube-symmetric (k·90°) option needing the least wrist rotation,
with the other three as fall-backs — important on the Panda because joint 7 only turns ±166°.

**Collision avoidance cube ↔ gripper ↔ tray (20 pts).** Transit height keeps cube bottom and fingertips
5 cm above the tray walls; place yaw aligns the held cube with the rotated tray (in-hand yaw re-measured
after the lift); before execution the cube corners and open pads must fit inside the tray and the transit
clearance is predicted from FK (raise height / try another yaw otherwise); at runtime contacts end the
episode with a reason and `mj_geomDistance` clearances are logged.

**Failure reasons:** `ik_infeasible`, `collision_<a>-<b>`, `grasp_failed`, `grasp_slip`, `place_missed`,
`cube_does_not_fit_tray`, `fingers_hit_tray`, `transit_clearance` (+ phase).

## 6. Troubleshooting (`utils/pick_place/config.py`)

| Symptom | Try |
|---|---|
| Arm lags / overshoots, torque saturation | `kp` (lower = softer), `lin_speed`; check `max_track_err` in the CSV |
| `collision_arm-table` | raise `r_min`, or the table height in the XML (feasible H range 0.025–0.415 m) |
| Many `ik_infeasible` | check `limit_hits` — joint 7 or joint 4 at a limit; narrow `r_min…r_max` |
| Cube slips | `python scripts/debug_episode.py dls <ep>` and `python scripts/grasp_experiments.py` |
