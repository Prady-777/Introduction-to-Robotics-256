"""All tunable numbers live here. Geometry of table/tray/cube is read from the XML."""
import os
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENE_XML = os.path.join(REPO_ROOT, "robot_descriptions", "pick_place_scene.xml")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")


@dataclass
class Config:
    # ---------------- robot / control ----------------
    q_home: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)  # TCP points straight down at (0.38, 0, 0.42)
    kp: tuple = (600.0, 600.0, 600.0, 300.0, 300.0, 200.0)   # N·m/rad, D-term = model damping
    v_max: float = 2.0            # rad/s joint velocity limit used by planner + QP + Mink
    finger_open: float = 0.040    # m per finger (gap 8.4 cm)
    finger_close: float = -0.020  # target below contact -> steady squeeze (~6 N per pad)

    # ---------------- workspace-derived cube sampling ----------------
    # Dexterous region (all 12 test yaws feasible top-down for EVERY height of the vertical
    # approach column z = 0.25 ... 0.375 m): r in [0.10, 0.50] m, partial (≥75 %) up to 0.55.
    # Lower bound raised to 0.30 m to keep the arm clear of its own base/turret.
    r_min: float = 0.30
    r_max: float = 0.52
    table_margin: float = 0.05    # keep cube centre this far from the table edges
    tray_keepout: float = 0.05    # extra free space around the tray for the fingers
    yaw_range: tuple = (-np.pi, np.pi)

    # ---------------- motion ----------------
    knot_dt: float = 0.02         # Cartesian path is discretised at 50 Hz
    lin_speed: float = 0.15       # m/s for free motions
    slow_speed: float = 0.05      # m/s for the final descent / lift
    ang_speed: float = 0.5        # rad/s
    clearance: float = 0.05       # required gap cube-bottom <-> tray-wall top during transit
    pregrasp_height: float = 0.10 # above cube centre
    place_gap: float = 0.010      # cube bottom above tray floor when released
    grasp_time: float = 0.6
    release_time: float = 0.5
    settle_time: float = 0.3
    settle_tol: float = 0.01      # rad
    settle_timeout: float = 1.0

    # ---------------- IK (shared by all methods) ----------------
    ik_dt: float = 0.02
    ik_gain: float = 40.0         # K in x_dot_d = K e   (K*dt = 0.8)
    ik_damping: float = 1e-3      # λ
    ik_pos_tol: float = 1e-3
    ik_rot_tol: float = 1e-2
    ik_max_iters: int = 200
    null_gain: float = 0.005      # α of the null-space / posture objective
    knot_tol_scale: float = 3.0   # intermediate knots may be 3x looser than endpoints

    # ---------------- success / failure checks ----------------
    lift_success_frac: float = 0.5   # cube must rise ≥ 50 % of the commanded lift
    slip_dist: float = 0.04          # cube-TCP distance that counts as a drop
    abort_on_collision: bool = True

    # ---------------- logging / video ----------------
    clearance_every: int = 10     # sim steps between clearance queries
    viewer_every: int = 8         # sim steps between viewer syncs
    trail_every: int = 15
    trail_len: int = 600
    video_w: int = 960
    video_h: int = 540
    video_fps: int = 30

    ik_kwargs: dict = field(default_factory=dict)

    def solver_kwargs(self, method):
        kw = dict(pos_tol=self.ik_pos_tol, rot_tol=self.ik_rot_tol, max_iters=self.ik_max_iters,
                  dt=self.ik_dt, gain=self.ik_gain, q_rest=np.asarray(self.q_home))
        if method == "dls":
            kw.update(damping=self.ik_damping, null_gain=self.null_gain)
        elif method == "qp":
            kw.update(damping=self.ik_damping, null_gain=self.null_gain, v_max=self.v_max)
        elif method == "mink":
            kw.update(damping=self.ik_damping, v_max=self.v_max)
        kw.update(self.ik_kwargs)
        return kw
