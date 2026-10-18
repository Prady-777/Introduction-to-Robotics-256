"""
PickPlaceEnv: everything that touches the MuJoCo simulation.

Control law for the 7 torque-controlled Panda joints (same idea as the course's
JointPositionController, plus a damping feed-forward):
    tau = qfrc_bias  +  Kp (q_des - q)  +  b · q_dot_des
qfrc_bias cancels gravity + Coriolis; the model's joint damping b (40/8/5 N·m·s/rad)
acts as the D-term and is integrated implicitly by MuJoCo, which keeps the loop
stable; b·q_dot_des cancels the lag that damping would otherwise create.
"""
import time

import numpy as np
import mujoco

from .kinematics import MjKinematics
from .config import Config

ARM_JOINTS = list(Config().arm_joints)
ARM_ACTUATORS = list(Config().arm_actuators)
FINGER_ACTUATORS = ["finger_left_act", "finger_right_act"]

# contact pairs (sorted category tuples) that end an episode as a collision
HARD_PAIRS = {("arm", "table"), ("arm", "tray_wall"), ("arm", "tray_floor"),
              ("arm", "cube"), ("gripper", "tray_wall")}
# contacts that are only counted (light scraping, e.g. a fingertip on the table)
SOFT_PAIRS = {("gripper", "table"), ("gripper", "tray_floor")}


class PickPlaceEnv:
    def __init__(self, xml_path, cfg, viewer=False, record_path=None, realtime=False,
                 label=""):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        m = self.model

        self.kin = MjKinematics(m, ARM_JOINTS, "tcp")
        self.arm_act = np.array([m.actuator(n).id for n in ARM_ACTUATORS])
        self.finger_act = np.array([m.actuator(n).id for n in FINGER_ACTUATORS])
        self.qadr, self.dadr = self.kin.qpos_adr, self.kin.dof_adr
        self.damping = m.dof_damping[self.dadr].copy()
        self.kp = np.broadcast_to(np.asarray(cfg.kp, float), (len(ARM_JOINTS),)).copy()
        self.q_home = np.asarray(cfg.q_home, float)
        self.tcp_site = m.site("tcp").id

        # cube free joint
        cj = m.joint("cube_free").id
        self.cube_qadr = m.jnt_qposadr[cj]
        self.cube_dadr = m.jnt_dofadr[cj]
        self.cube_body = m.body("cube").id
        self.cube_geom = m.geom("cube_geom").id
        self.cube_half = float(m.geom_size[self.cube_geom][0])

        # ---- scene geometry read from the XML (single source of truth) ----
        mujoco.mj_forward(m, self.data)
        tg = m.geom("table_top").id
        tpos, tsize = self.data.geom_xpos[tg], m.geom_size[tg]
        self.table_top_z = float(tpos[2] + tsize[2])
        self.table_xy = (tpos[0] - tsize[0], tpos[0] + tsize[0],
                         tpos[1] - tsize[1], tpos[1] + tsize[1])
        tray = m.body("tray").id
        self.tray_pos = self.data.xpos[tray].copy()
        tq = m.body_quat[tray]
        self.tray_yaw = float(2 * np.arctan2(tq[3], tq[0]))
        fl = m.geom("tray_floor").id
        self.tray_floor_top = float(self.data.geom_xpos[fl][2] + m.geom_size[fl][2])
        wn = m.geom("tray_wall_n").id
        self.tray_inner_half = float(abs(m.geom_pos[wn][1]) - m.geom_size[wn][1])
        walls = [m.geom(n).id for n in ("tray_wall_n", "tray_wall_s", "tray_wall_e", "tray_wall_w")]
        self.tray_wall_top = max(float(self.data.geom_xpos[g][2] + m.geom_size[g][2]) for g in walls)
        self.tray_outer_half = self.tray_inner_half + 2 * float(m.geom_size[walls[0]][1])

        # ---- geom categories for collision / clearance bookkeeping ----
        arm_root = m.body(cfg.arm_root_body).id
        grip_root = m.body("gripper_base").id
        self.cat = {}
        for g in range(m.ngeom):
            if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
                continue                                  # visual only
            b = m.geom_bodyid[g]
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if self._is_descendant(b, grip_root):
                self.cat[g] = "gripper"
            elif self._is_descendant(b, arm_root):
                self.cat[g] = "arm"
            elif g == self.cube_geom:
                self.cat[g] = "cube"
            elif name == "table_top":
                self.cat[g] = "table"
            elif name.startswith("tray_wall"):
                self.cat[g] = "tray_wall"
            elif name == "tray_floor":
                self.cat[g] = "tray_floor"
            elif name == "floor":
                self.cat[g] = "floor"
        self.robot_geoms = [g for g, c in self.cat.items() if c in ("arm", "gripper")]
        self.gripper_geoms = [g for g, c in self.cat.items() if c == "gripper"]
        self.wall_geoms = walls
        self.table_geom = tg
        self.has_geom_distance = hasattr(mujoco, "mj_geomDistance")

        # ---- visualisation ----
        self.realtime = realtime
        self.label = label
        self.viewer = mujoco.viewer.launch_passive(m, self.data) if viewer else None
        self.trail = []
        self.recorder = None
        if record_path:
            import imageio
            self.renderer = mujoco.Renderer(m, height=cfg.video_h, width=cfg.video_w)
            self.recorder = imageio.get_writer(record_path, fps=cfg.video_fps, macro_block_size=1)
            self._next_frame_t = 0.0
        self.status_text = ""
        self.reset_episode_stats()

    # ------------------------------------------------------------------ utils
    def _is_descendant(self, body, root):
        while body > 0:
            if body == root:
                return True
            body = self.model.body_parentid[body]
        return body == root

    @property
    def q(self):
        return self.data.qpos[self.qadr].copy()

    def tcp_pose(self):
        return (self.data.site_xpos[self.tcp_site].copy(),
                self.data.site_xmat[self.tcp_site].reshape(3, 3).copy())

    def cube_pos(self):
        return self.data.xpos[self.cube_body].copy()

    def cube_yaw(self):
        q = self.data.qpos[self.cube_qadr + 3:self.cube_qadr + 7]
        return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))

    def sync_ik_state(self):
        self.kin.sync_full_state(self.data.qpos)

    # ------------------------------------------------------------------ reset
    def reset(self, cube_x, cube_y, cube_yaw):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qadr] = self.q_home
        a = self.cube_qadr
        self.data.qpos[a:a + 3] = [cube_x, cube_y, self.table_top_z + self.cube_half + 1e-3]
        self.data.qpos[a + 3:a + 7] = [np.cos(cube_yaw / 2), 0, 0, np.sin(cube_yaw / 2)]
        self.set_gripper(open_=True)
        for fa, j in zip(self.finger_act, ("finger_left_joint", "finger_right_joint")):
            self.data.qpos[self.model.jnt_qposadr[self.model.joint(j).id]] = self.cfg.finger_open
        mujoco.mj_forward(self.model, self.data)
        self.trail.clear()
        self.reset_episode_stats()
        self._next_frame_t = 0.0          # sim clock restarts each episode
        self.hold(self.q_home, self.cfg.settle_time, phase="reset")

    def reset_episode_stats(self):
        self.events = []                 # (time, phase, pair) of hard contacts
        self._seen_events = set()
        self.soft_contacts = 0
        self.min_clear = {"cube_tray": np.inf, "robot_table": np.inf, "gripper_tray": np.inf}
        self._step_count = 0
        self.carrying = False

    # ----------------------------------------------------------------- control
    def set_gripper(self, open_=True):
        self.data.ctrl[self.finger_act] = self.cfg.finger_open if open_ else self.cfg.finger_close

    def step(self, q_des, qd_des, phase=""):
        d, m = self.data, self.model
        q = d.qpos[self.qadr]
        tau = d.qfrc_bias[self.dadr] + self.kp * (q_des - q) + self.damping * qd_des
        d.ctrl[self.arm_act] = tau
        t_wall = time.perf_counter()
        mujoco.mj_step(m, d)
        self._step_count += 1
        self._monitor(phase)
        self._render(t_wall)

    def hold(self, q_des, duration, phase=""):
        for _ in range(int(duration / self.model.opt.timestep)):
            self.step(q_des, np.zeros(len(q_des)), phase)

    # ----------------------------------------------------- collision monitoring
    def _monitor(self, phase):
        d = self.data
        for i in range(d.ncon):
            c = d.contact[i]
            a, b = self.cat.get(c.geom1), self.cat.get(c.geom2)
            if a is None or b is None:
                continue
            pair = tuple(sorted((a, b)))
            hard = pair in HARD_PAIRS or (pair == ("cube", "tray_wall") and self.carrying)
            if hard:
                ev = (phase, f"{pair[0]}-{pair[1]}")
                if ev not in self._seen_events:          # log each pair once per phase
                    self._seen_events.add(ev)
                    self.events.append((round(d.time, 3),) + ev)
            elif pair in SOFT_PAIRS:
                self.soft_contacts += 1
        if self._step_count % self.cfg.clearance_every == 0:
            self._update_clearance()

    def _dist(self, g1, g2, dmax=0.3):
        if self.has_geom_distance:
            return mujoco.mj_geomDistance(self.model, self.data, g1, g2, dmax, np.zeros(6))
        return np.nan

    def _update_clearance(self):
        mc = self.min_clear
        mc["robot_table"] = min(mc["robot_table"],
                                min(self._dist(g, self.table_geom) for g in self.robot_geoms))
        mc["gripper_tray"] = min(mc["gripper_tray"],
                                 min(self._dist(g, w) for g in self.gripper_geoms for w in self.wall_geoms))
        if self.carrying:
            mc["cube_tray"] = min(mc["cube_tray"],
                                  min(self._dist(self.cube_geom, w) for w in self.wall_geoms))

    # ---------------------------------------------------------- visualisation
    def _add_trail_geoms(self, scn):
        for p, rgba in self.trail[-self.cfg.trail_len:]:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                np.array([0.004, 0, 0]), p, np.eye(3).flatten(), rgba)
            scn.ngeom += 1

    def _render(self, t_wall):
        if self._step_count % self.cfg.trail_every == 0:      # TCP trail (blue empty, green carrying)
            col = np.array([0.1, 0.9, 0.2, 1.0]) if self.carrying else np.array([0.2, 0.5, 1.0, 1.0])
            self.trail.append((self.data.site_xpos[self.tcp_site].copy(), col))
        if self.viewer is None and self.recorder is None:
            return
        if self.viewer is not None and self._step_count % self.cfg.viewer_every == 0:
            with self.viewer.lock():
                self.viewer.user_scn.ngeom = 0
                self._add_trail_geoms(self.viewer.user_scn)
            self.viewer.sync()
            if self.realtime:
                lag = self.model.opt.timestep * self.cfg.viewer_every - (time.perf_counter() - t_wall)
                if lag > 0:
                    time.sleep(lag)
        if self.recorder is not None and self.data.time >= self._next_frame_t:
            self._next_frame_t += 1.0 / self.cfg.video_fps
            self.renderer.update_scene(self.data, camera="overview")
            self._add_trail_geoms(self.renderer.scene)
            frame = self.renderer.render()
            if self.status_text:
                frame = _put_text(frame, f"{self.label}  {self.status_text}")
            self.recorder.append_data(frame)

    def snapshot(self, path, camera="overview"):
        """Save a PNG of the current state (used for failure screenshots)."""
        if not hasattr(self, "_snap"):
            self._snap = mujoco.Renderer(self.model, height=self.cfg.video_h, width=self.cfg.video_w)
        self._snap.update_scene(self.data, camera=camera)
        self._add_trail_geoms(self._snap.scene)
        frame = _put_text(self._snap.render(), f"{self.label}  {self.status_text}")
        from PIL import Image
        Image.fromarray(frame).save(path)

    def close(self):
        if self.recorder is not None:
            self.recorder.close()
        if self.viewer is not None:
            self.viewer.close()


def _put_text(frame, text):
    try:
        from PIL import Image, ImageDraw
        img = Image.fromarray(frame)
        dr = ImageDraw.Draw(img)
        dr.rectangle([0, 0, img.width, 22], fill=(0, 0, 0))
        dr.text((6, 5), text, fill=(255, 255, 255))
        return np.asarray(img)
    except ImportError:
        return frame


# `mujoco.viewer` must be imported explicitly
import mujoco.viewer  # noqa: E402
