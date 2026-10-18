# ITR Lab 3 – Tutorial 4: Pick and Place in MuJoCo with IK

The HEAL 6-DoF arm picks a randomly posed cube from a table and places it in a tray. The task is
solved with three interchangeable IK methods on identical random seeds:

| Method | File | Idea |
|---|---|---|
| **Mink** (baseline) | `utils/pick_place/ik_solvers.py · MinkIK` | [kevinzakka/mink](https://github.com/kevinzakka/mink): FrameTask + PostureTask, ConfigurationLimit + VelocityLimit |
| **DLS-IK** | `… · DLSIK` | closed-loop q̇ = (JᵀJ+λI)⁻¹Jᵀẋ_d + null-space posture term |
| **QP-IK** | `… · QPIK` | same cost as a QP ½ΔqᵀHΔq + cᵀΔq s.t. GΔq ≤ h with joint + velocity limits |

---

## 1. Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python tests/test_ik_math.py                           # 6 × PASS, no MuJoCo needed
```

* **macOS:** use `mjpython` instead of `python` for anything with `--viewer`.
* **Headless Linux / SSH:** `export MUJOCO_GL=egl` (or `osmesa`) before `--record` or snapshots.

## 2. Workflow (run from the `Lab3/` folder)

| # | Command | Produces |
|---|---|---|
| 1 | `python tests/test_ik_math.py` | sanity check of quaternion error, Jacobian, DLS, QP |
| 2 | `python scripts/run_episodes.py --ik dls --n 3 --viewer --realtime` | watch 3 episodes (TCP trail: blue = empty, green = carrying) |
| 3 | `python scripts/workspace_analysis.py` (`--quick` for a fast pass) | `results/workspace/*.png`, `workspace_summary.json` |
| 4 | `python scripts/run_episodes.py --ik mink --n 25` | `results/logs/mink/episodes.csv`, `episodes.jsonl`, `summary.json`, `failures/*.png` |
| 5 | same with `--ik dls` and `--ik qp` | `results/logs/dls/…`, `results/logs/qp/…` |
| 6 | `python scripts/compare_ik.py` | `results/comparison/` tables, stats, 8 plots |
| 7 | `python scripts/run_episodes.py --ik mink --n 4 --record results/videos/mink.mp4` (+ dls, qp) | ~1 min video per method |
| 8 | `python scripts/make_final_video.py` | `results/videos/final_submission.mp4` (3 min, with title cards) |

Episode *i* always uses seed `--seed + i`, so all methods see the **same cube poses**.
To replay a single episode (e.g. a failure): `python scripts/run_episodes.py --ik qp --start 7 --n 1 --viewer --realtime`.

## 3. Repository layout

```
Lab3/
├── robot_descriptions/
│   ├── pick_place_scene.xml        ← NEW: HEAL arm + gripper + table + tray + cube (single file)
│   └── heal_meshes/                ← arm meshes (unchanged)
├── utils/pick_place/
│   ├── config.py                   all tunable numbers (gains, speeds, sampling region, tolerances)
│   ├── kinematics.py               quaternion maths + MjKinematics (FK / Jacobian on a private MjData)
│   ├── xml_kinematics.py           MuJoCo-free FK of the same XML (tests, --backend numpy)
│   ├── ik_solvers.py               MinkIK, DLSIK, QPIK (one interface)
│   ├── env.py                      simulation, PD + gravity compensation, contacts, clearance, video
│   └── planner.py                  waypoints, Cartesian interpolation, tray-aware placement, logging
├── scripts/
│   ├── workspace_analysis.py       task + dexterous workspace, table design
│   ├── run_episodes.py             randomised batch runs + logs
│   ├── compare_ik.py               statistics + plots across methods
│   ├── make_final_video.py         joins the three recordings
│   └── (original course examples, unchanged)
├── tests/test_ik_math.py
├── report/report.md                report (fill the ⟨…⟩ placeholders from results/)
└── results/                        everything generated
```

## 4. How it works

**Scene.** The original `single_arm_heal_effort_actuation_rs_mj.xml` and the Robotiq mesh assets were
not included in the starter zip, so `pick_place_scene.xml` rebuilds the HEAL arm from
`arm_gripper.xml` (same kinematics and inertias) and adds a primitive parallel-jaw gripper (two slide
joints, position actuators, high-friction pads). Collision bitmasks: robot geoms never collide with
each other; table, tray and cube collide with the robot and with each other.

**Control.** τ = qfrc_bias + K_p(q* − q) + b·q̇* on the six torque-controlled joints. The model’s joint
damping b = 55 N·m·s/rad is the D-term (integrated implicitly, therefore stable); b·q̇* removes its lag.

**Episode.** approach → descend → grasp → lift → transit → lower → release → retreat. Each motion is a
straight Cartesian line with minimum-jerk timing, sampled at 50 Hz. IK runs at every knot,
warm-started from the previous one, and the whole pick (or place) is planned before anything is executed.

**Collision avoidance between cube, gripper and tray (rubric item, 20 pts).**
1. *Safe transit height*: the lowest point of the cube/fingers stays `clearance` (5 cm) above the tray walls.
2. *Place yaw*: after the lift, the cube’s yaw relative to the gripper is measured, and the gripper yaw
   is chosen so the cube lands **aligned with the (rotated) tray**, using the k·90° option needing least wrist motion.
3. *Pre-execution checks*: cube corners and opened finger pads must fit inside the tray; the transit
   clearance is predicted from FK. If a check fails, the planner raises the transit height or tries the next yaw.
4. *Runtime monitoring*: contacts (cube–wall, gripper–wall, arm–table…) end the episode with a reason;
   signed distances from `mj_geomDistance` are logged as the minimum clearance.

**Failure reasons logged:** `ik_infeasible`, `collision_<a>-<b>`, `grasp_failed`, `grasp_slip`,
`place_missed`, `cube_does_not_fit_tray`, `fingers_hit_tray`, `transit_clearance`, each with the phase
in which it happened.

## 5. Tuning / troubleshooting (all in `utils/pick_place/config.py`)

| Symptom | Try |
|---|---|
| Cube slips during lift/transit | lower `finger_close` (more squeeze), lower `slow_speed`, raise pad friction in the XML |
| Fingers hit the cube on descent | check `grasp_yaw` in the log; reduce `finger_open` only if the gripper hits the table |
| Arm lags or overshoots | `kp` (higher = stiffer); `lin_speed` |
| `collision_arm-table` | raise `r_min` or the table/grasp height; inspect `failures/*.png` |
| Many `ik_infeasible` | narrow `r_min…r_max` to the green band of `dexterous_rz.png`; raise `ik_max_iters` |
| `mj_geomDistance` missing | upgrade MuJoCo ≥ 3.2 (clearances are then logged as NaN, nothing else breaks) |
