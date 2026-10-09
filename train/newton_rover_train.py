#!/usr/bin/env python3
"""DHWANI rover in NVIDIA Newton: an Ackermann JetRacer on rough ground, trained with PPO and
exported as a plain .npz that nodes/policy_runner.py runs with numpy on the Jetson Nano.

WHAT IS SIMULATED. One rover is 7 rigid bodies in a Featherstone articulation: a chassis box on
a free joint, two steering knuckles (revolute about Z, position-controlled, limited to +-0.38 rad,
which is the 0.5 m minimum turning radius of the real car), and four wheels (revolute about Y;
the rear pair velocity-driven through a torque limit, the front pair free-rolling). Wheels are
spheres of the real radius - the usual RC-car simplification, and it keeps every ground contact a
sphere-heightfield contact.

The numbers are the JetRacer's: wheelbase 0.20 m, track 0.19 m, wheel radius 32.5 mm, about
1.6 kg all up, 0.5 m/s commanded top speed (TEB is capped at 0.28 m/s on the robot).

THE TERRAIN is one 12 x 12 m heightfield whose roughness grows along X: near-flat on the left,
gentle undulation, 5 cm bumps, then rubble with 4 cm quantised steps on the right. A wheel of
32.5 mm radius gets stuck in the rubble - that is the point. Every world carries the same
terrain; rovers spawn anywhere on it with a random heading and a goal 1.5-4 m away.

THE POLICY sees only what the real robot can measure today: gravity direction in the body frame
(IMU), body linear velocity (odometry/EKF), body angular rate (gyro), goal direction and distance
in the body frame (map pose + waypoint), and its own previous command. 14 numbers. No terrain
samples, no wheel encoders (the JetRacer has none). It outputs throttle and steering in [-1, 1],
the two numbers the JetRacer takes.

WHAT IT IS FOR. A terrain-reactive driving reflex: slow down when tilted, steer around what it
cannot climb, back off and re-approach when stuck. The layer under move_base that move_base does
not have. It is not the mapper, the planner or the mission logic - those stay in ROS.

--smoke runs on a laptop CPU in a few minutes with 4 rovers. Training runs on a Kaggle T4.
Everything it produces goes to --out: policy.npz, train_log.json, terrain.png, tracks.png,
summary.json. Nothing here needs a GPU on the robot: the export is weights, not a framework.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---------------------------------------------------------------- the robot (JetRacer testbed)
WHEELBASE = 0.20
TRACK = 0.19
WHEEL_R = 0.0325
AXLE_DROP = 0.025                    # axle below the chassis centre -> 27.5 mm ground clearance
CHASSIS_H = AXLE_DROP + WHEEL_R      # chassis centre above ground when resting on flat ground
CHASSIS_HALF = (0.15, 0.06, 0.03)    # 0.30 x 0.12 x 0.06 m body; wheels sit outside it
MAX_STEER = 0.38                     # rad; atan(0.20 / 0.5)
MAX_SPEED = 0.5                      # m/s at throttle 1.0
STEER_KE, STEER_KD = 1.0, 0.04       # with the armature below: 44 rad/s, damping ratio 0.9
DRIVE_KD, DRIVE_EFFORT = 0.05, 0.25  # N.m.s/rad, N.m per rear wheel
WHEEL_ARMATURE = 5e-4                # reflected motor+gear inertia; a bare 43 g wheel (2.8e-5 kg.m2)
                                     # under a 0.05 damper is unstable at a 5 ms substep (dt*kd/I = 9)
CONTACT = dict(ke=2000.0, kd=20.0, kf=100.0)  # soft-contact gains sized for 40 g wheels at 5 ms

# ---------------------------------------------------------------- the task
CONTROL_HZ = 50                      # policy_runner.py runs at 50 Hz on the Nano
SUBSTEPS = 4                         # even, so the two-state swap lands back on state 0 (CUDA graph)
DT = 1.0 / (CONTROL_HZ * SUBSTEPS)
TERRAIN_M = 12.0
TERRAIN_N = 121                      # 0.1 m cells
SPAWN_HALF = 5.0
GOAL_MIN, GOAL_MAX, GOAL_TOL = 1.5, 4.0, 0.25
EPISODE_S = 12.0
OBS_DIM, ACT_DIM = 14, 2
OBS_SPEC = ("gravity_body_xyz(3) linvel_body_xyz(3) angvel_body_xyz(3) "
            "goal_dir_body_xy(2) goal_dist_over_5m(1) prev_action(2)")
ACT_SPEC = "throttle[-1,1]*0.5m/s  steering[-1,1]*0.38rad ; consumer clips to [-1,1]"


# ================================================================ terrain
def make_terrain(seed=0):
    """Height grid (rows = y, cols = x) whose roughness grows with x. Heights in metres."""
    import torch
    import torch.nn.functional as F
    g = torch.Generator().manual_seed(seed)

    def field(cells):
        coarse = torch.randn(1, 1, cells, cells, generator=g)
        return F.interpolate(coarse, size=(TERRAIN_N, TERRAIN_N), mode="bicubic",
                             align_corners=True)[0, 0]

    xs = torch.linspace(-TERRAIN_M / 2, TERRAIN_M / 2, TERRAIN_N)
    X = xs[None, :].expand(TERRAIN_N, TERRAIN_N)          # x varies along columns
    w = (X + TERRAIN_M / 2) / TERRAIN_M                   # 0 at the left edge, 1 at the right
    undulation = field(6).clamp(-2, 2) / 2 * (0.004 + 0.018 * w)      # 2 m features, up to 2 cm
    bumps = field(30).clamp(-2, 2) / 2 * 0.03 * (w - 0.25).clamp(0, 1)  # 0.4 m features, to 3 cm
    rubble = (field(20) * 0.8).round().clamp(-2, 2) * 0.015 * (w > 0.75)  # 1.5 cm steps, right quarter
    h = undulation + bumps + rubble
    h = h - h.min()
    return h.numpy().astype(np.float32)


# ================================================================ simulation
class RoverSim(object):
    """N rovers on the same terrain, one Newton model, torch views onto the state."""

    def __init__(self, n, device, terrain, seed=0, solver="mujoco", mujoco_cpu=False):
        import torch
        import warp as wp
        import newton
        self.torch, self.wp, self.newton = torch, wp, newton
        self.n, self.device = n, device
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.H = torch.tensor(terrain, device=device)

        hf = newton.Heightfield(data=terrain, nrow=TERRAIN_N, ncol=TERRAIN_N,
                                hx=TERRAIN_M / 2, hy=TERRAIN_M / 2,
                                min_z=0.0, max_z=float(terrain.max()))
        rover = newton.ModelBuilder()
        rover.add_shape_heightfield(heightfield=hf,
                                    cfg=newton.ModelBuilder.ShapeConfig(mu=0.9, **CONTACT))
        self._build_rover(rover)
        b = newton.ModelBuilder()
        b.replicate(rover, n)                     # one world per rover; worlds never collide
        self.model = b.finalize(device=device)
        self.solver_name = solver
        if solver == "mujoco":
            # MuJoCo (warp-accelerated) does its own collision and its own implicit contact
            # model - the one every wheeled-robot RL pipeline uses. Newton's soft-contact
            # solvers were tried first and inject energy through 40 g wheels at a 5 ms step.
            self.solver = newton.solvers.SolverMuJoCo(
                self.model, use_mujoco_contacts=True, use_mujoco_cpu=mujoco_cpu,
                solver="newton", integrator="implicitfast", cone="elliptic",
                iterations=100, ls_iterations=50)
            self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)
            self.pipeline = None
        else:
            self.solver = (newton.solvers.SolverXPBD(self.model, iterations=10) if solver == "xpbd"
                           else newton.solvers.SolverFeatherstone(self.model, angular_damping=0.05))
            self.pipeline = newton.CollisionPipeline(self.model)
            self.contacts = self.pipeline.contacts()
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.graph = None

        # index maps: 7 joints and 7 bodies per rover, in the order they were added
        jq = self.model.joint_q_start.numpy()
        jqd = self.model.joint_qd_start.numpy()
        J = 7
        ar = np.arange
        t = lambda a: torch.tensor(np.asarray(a), device=device, dtype=torch.long)
        self.free_q = t([jq[J * i] + ar(7) for i in range(n)])              # (n,7)
        self.steer_q = t([[jq[J * i + 1], jq[J * i + 3]] for i in range(n)])  # (n,2) fl, fr
        self.wheel_q = t([[jq[J * i + k] for k in (2, 4, 5, 6)] for i in range(n)])
        self.env_qd = t([jqd[J * i] + ar(12) for i in range(n)])            # (n,12)
        self.drive_qd = t([[jqd[J * i + 5], jqd[J * i + 6]] for i in range(n)])  # rear wheels
        self.chassis = torch.arange(n, device=device) * 7

        self.goal = torch.zeros(n, 2, device=device)
        self.prev_a = torch.zeros(n, ACT_DIM, device=device)
        self.steps = torch.zeros(n, device=device, dtype=torch.long)
        self.d_prev = torch.zeros(n, device=device)
        self.ep_ret = torch.zeros(n, device=device)
        self.ep_reached = torch.zeros(n, device=device)
        self.spawn_xy = torch.zeros(n, 2, device=device)
        self.reset(torch.arange(n, device=device))

    def _build_rover(self, b):
        wp, newton = self.wp, self.newton
        Cfg = newton.ModelBuilder.ShapeConfig
        body_cfg = Cfg(density=500.0, mu=0.6, **CONTACT)
        wheel_cfg = Cfg(density=300.0, mu=0.9, **CONTACT)
        knuckle_cfg = Cfg(density=1000.0, mu=0.5, **CONTACT)
        qi = wp.quat_identity()
        chassis = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, CHASSIS_H), qi),
                             com=wp.vec3(0.0, 0.0, -0.02))   # battery-low CoM, 2 cm under centre
        b.add_shape_box(chassis, hx=CHASSIS_HALF[0], hy=CHASSIS_HALF[1], hz=CHASSIS_HALF[2],
                        cfg=body_cfg)
        joints = [b.add_joint_free(chassis, label="base")]
        hubs = [("fl", WHEELBASE / 2, TRACK / 2), ("fr", WHEELBASE / 2, -TRACK / 2),
                ("rl", -WHEELBASE / 2, TRACK / 2), ("rr", -WHEELBASE / 2, -TRACK / 2)]
        for name, x, y in hubs:
            hub = wp.vec3(x, y, CHASSIS_H - AXLE_DROP)
            parent, parent_xf = chassis, wp.transform(wp.vec3(x, y, -AXLE_DROP), qi)
            if name[0] == "f":
                kn = b.add_link(xform=wp.transform(hub, qi))
                b.add_shape_sphere(kn, radius=0.015, cfg=knuckle_cfg)
                joints.append(b.add_joint_revolute(
                    chassis, kn, parent_xform=parent_xf, child_xform=wp.transform(),
                    axis=newton.Axis.Z, limit_lower=-MAX_STEER, limit_upper=MAX_STEER,
                    target_ke=STEER_KE, target_kd=STEER_KD, armature=WHEEL_ARMATURE,
                    actuator_mode=newton.JointTargetMode.POSITION, label="steer_" + name))
                parent, parent_xf = kn, wp.transform()
            wheel = b.add_link(xform=wp.transform(hub, qi))
            b.add_shape_sphere(wheel, radius=WHEEL_R, cfg=wheel_cfg)
            driven = name[0] == "r"
            joints.append(b.add_joint_revolute(
                parent, wheel, parent_xform=parent_xf, child_xform=wp.transform(),
                axis=newton.Axis.Y, armature=WHEEL_ARMATURE, damping=1e-4,
                target_kd=DRIVE_KD if driven else 0.0,
                effort_limit=DRIVE_EFFORT if driven else None,
                actuator_mode=(newton.JointTargetMode.VELOCITY if driven
                               else newton.JointTargetMode.NONE),
                label="wheel_" + name))
        b.add_articulation(joints, label="rover")

    # ------------------------------------------------------------ helpers
    def terrain_h(self, xy):
        """Bilinear height lookup. rows = y, cols = x (verified by the spawn-height self-check)."""
        torch = self.torch
        s = (TERRAIN_N - 1) / TERRAIN_M
        u = ((xy[:, 0] + TERRAIN_M / 2) * s).clamp(0, TERRAIN_N - 1.001)
        v = ((xy[:, 1] + TERRAIN_M / 2) * s).clamp(0, TERRAIN_N - 1.001)
        c0, r0 = u.floor().long(), v.floor().long()
        fu, fv = (u - c0.float())[:, None], (v - r0.float())[:, None]
        H = self.H
        h00, h01 = H[r0, c0], H[r0, c0 + 1]
        h10, h11 = H[r0 + 1, c0], H[r0 + 1, c0 + 1]
        top = h00 * (1 - fu[:, 0]) + h01 * fu[:, 0]
        bot = h10 * (1 - fu[:, 0]) + h11 * fu[:, 0]
        return top * (1 - fv[:, 0]) + bot * fv[:, 0]

    @staticmethod
    def rot_inv(q, v):
        """Rotate world vectors v into the body frame of unit quaternions q (xyzw)."""
        qv, w = q[:, :3], q[:, 3:4]
        c = torch.cross if False else None  # placeholder to keep linters quiet
        import torch
        t = torch.cross(qv, v, dim=1)
        return v - 2.0 * w * t + 2.0 * torch.cross(qv, t, dim=1)

    def views(self):
        wp = self.wp
        return (wp.to_torch(self.s0.joint_q), wp.to_torch(self.s0.joint_qd),
                wp.to_torch(self.s0.body_q), wp.to_torch(self.s0.body_qd))

    def uniform(self, shape, lo, hi):
        return self.torch.rand(shape, generator=self.rng, device=self.device) * (hi - lo) + lo

    # ------------------------------------------------------------ reset / observe / step
    def reset(self, ids):
        torch = self.torch
        m = ids.numel()
        if m == 0:
            return
        xy = self.uniform((m, 2), -SPAWN_HALF, SPAWN_HALF)
        yaw = self.uniform((m,), -math.pi, math.pi)
        z = self.terrain_h(xy) + CHASSIS_H + 0.01
        pose = torch.stack([xy[:, 0], xy[:, 1], z, torch.zeros_like(z), torch.zeros_like(z),
                            torch.sin(yaw / 2), torch.cos(yaw / 2)], dim=1)
        jq, jqd, _, _ = self.views()
        jq[self.free_q[ids]] = pose
        jq[self.steer_q[ids]] = 0.0
        jq[self.wheel_q[ids]] = 0.0
        jqd[self.env_qd[ids]] = 0.0
        self.newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0,
                            indices=self.wp.from_torch(ids.to(torch.int32).contiguous()))
        self.new_goal(ids, xy)
        self.spawn_xy[ids] = xy
        self.prev_a[ids] = 0.0
        self.steps[ids] = 0
        self.ep_ret[ids] = 0.0
        self.ep_reached[ids] = 0.0

    def new_goal(self, ids, xy):
        torch = self.torch
        m = ids.numel()
        ang = self.uniform((m,), -math.pi, math.pi)
        dist = self.uniform((m,), GOAL_MIN, GOAL_MAX)
        g = xy + torch.stack([torch.cos(ang), torch.sin(ang)], 1) * dist[:, None]
        self.goal[ids] = g.clamp(-SPAWN_HALF - 0.5, SPAWN_HALF + 0.5)
        self.d_prev[ids] = (self.goal[ids] - xy).norm(dim=1)

    def observe(self):
        torch = self.torch
        _, _, bq, bqd = self.views()
        p, q = bq[self.chassis, :3], bq[self.chassis, 3:7]
        v, w = bqd[self.chassis, :3], bqd[self.chassis, 3:6]
        g = self.rot_inv(q, torch.tensor([[0.0, 0.0, -1.0]], device=self.device).expand(self.n, 3))
        vb, wb = self.rot_inv(q, v), self.rot_inv(q, w)
        to_goal = torch.cat([self.goal - p[:, :2], torch.zeros(self.n, 1, device=self.device)], 1)
        gb = self.rot_inv(q, to_goal)
        dist = to_goal[:, :2].norm(dim=1)
        dirb = gb[:, :2] / dist.clamp(min=1e-3)[:, None]
        obs = torch.cat([g, vb, wb, dirb, (dist / 5.0).clamp(max=1.0)[:, None], self.prev_a], 1)
        obs = torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        dist = torch.nan_to_num(dist, nan=99.0, posinf=99.0)
        return obs, p, g, dist

    def apply(self, action):
        torch, wp = self.torch, self.wp
        a = action.clamp(-1.0, 1.0)
        v = a[:, 0] * MAX_SPEED
        d = a[:, 1] * MAX_STEER
        # Ackermann: inner wheel turns tighter than the outer one
        R = WHEELBASE / torch.tan(d).abs().clamp(min=1e-4)
        inner = torch.atan(WHEELBASE / (R - TRACK / 2).clamp(min=0.05))
        outer = torch.atan(WHEELBASE / (R + TRACK / 2))
        left = torch.where(d >= 0, inner, outer) * torch.sign(d)
        right = torch.where(d >= 0, outer, inner) * torch.sign(d)
        small = d.abs() < 1e-3
        left, right = torch.where(small, d, left), torch.where(small, d, right)
        tq = wp.to_torch(self.control.joint_target_q)
        tqd = wp.to_torch(self.control.joint_target_qd)
        tq[self.steer_q] = torch.stack([left, right], 1)
        tqd[self.drive_qd] = (v / WHEEL_R)[:, None].expand(self.n, 2)
        return a

    def physics(self):
        wp = self.wp
        if self.graph is not None:
            wp.capture_launch(self.graph)
            return
        for _ in range(SUBSTEPS):
            self.s0.clear_forces()
            if self.pipeline is not None:
                self.pipeline.collide(self.s0, self.contacts)
            self.solver.step(self.s0, self.s1, self.control, self.contacts, DT)
            self.s0, self.s1 = self.s1, self.s0

    def capture(self):
        """CUDA graph of one control step's physics: the single biggest speed-up on the GPU."""
        wp = self.wp
        if not wp.get_device(self.device).is_cuda:
            return
        self.physics()                       # compile every kernel before capturing
        with wp.ScopedCapture(device=self.device) as cap:
            self.physics()
        self.graph = cap.graph

    def step(self, action):
        """Apply, simulate one control period, score. Returns reward, done, info (all torch)."""
        torch = self.torch
        a = self.apply(action)
        self.physics()
        obs, p, g, dist = self.observe()
        upright = -g[:, 2]
        diverged = ~torch.isfinite(p).all(dim=1) | ~torch.isfinite(upright)
        progress = torch.nan_to_num(self.d_prev - dist, nan=0.0, posinf=0.0, neginf=0.0)
        reached = (dist < GOAL_TOL) & ~diverged
        flipped = (upright < 0.3) | diverged
        self.steps += 1
        timeout = self.steps >= int(EPISODE_S * CONTROL_HZ)
        clean_upright = torch.nan_to_num(upright, nan=1.0).clamp(0.0, 1.0)
        reward = (6.0 * progress + 0.02 - 0.01 * (a - self.prev_a).abs().sum(1)
                  - 0.08 * (1.0 - clean_upright) + 5.0 * reached.float()
                  - 3.0 * flipped.float())
        reward = torch.nan_to_num(reward, nan=-3.0, posinf=0.0, neginf=-3.0)
        self.ep_ret += reward
        self.ep_reached += reached.float()
        self.prev_a = a.clone()
        self.d_prev = torch.nan_to_num(dist, nan=99.0)
        done = flipped | timeout
        info = {"reached": reached, "flipped": flipped, "timeout": timeout, "diverged": diverged,
                "progress": progress, "upright": upright, "ep_ret": self.ep_ret.clone(),
                "ep_reached": self.ep_reached.clone()}
        # a reached goal is replaced without a reset, so one episode can chain several goals
        rid = torch.nonzero(reached & ~done).flatten()
        if rid.numel():
            self.new_goal(rid, p[rid, :2])
        did = torch.nonzero(done).flatten()
        if did.numel():
            self.reset(did)
        return reward, done, info


# ================================================================ policy
def mlp(sizes):
    import torch.nn as nn
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class RunningNorm(object):
    def __init__(self, dim, device):
        import torch
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = bm - self.mean
        tot = self.count + bc
        self.mean = self.mean + delta * bc / tot
        m2 = self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot
        self.var = m2 / tot
        self.count = tot

    def __call__(self, x):
        return ((x - self.mean) / (self.var.sqrt() + 1e-6)).clamp(-10, 10)


def export_npz(actor, norm, path, meta):
    """The contract policy_runner.py reads: W{i}, b{i} (in x out), obs_mean/std, act_scale."""
    import torch
    d = {}
    lin = [m for m in actor if isinstance(m, torch.nn.Linear)]
    for i, m in enumerate(lin):
        d["W%d" % i] = m.weight.detach().cpu().numpy().T.astype(np.float32)
        d["b%d" % i] = m.bias.detach().cpu().numpy().astype(np.float32)
    d["obs_mean"] = norm.mean.cpu().numpy().astype(np.float32)
    d["obs_std"] = (norm.var.sqrt() + 1e-6).cpu().numpy().astype(np.float32)
    d["act_scale"] = np.float32(1.0)
    d["activation"] = np.frombuffer(b"tanh", dtype=np.uint8)
    d["obs_spec"] = np.frombuffer(OBS_SPEC.encode(), dtype=np.uint8)
    d["act_spec"] = np.frombuffer(ACT_SPEC.encode(), dtype=np.uint8)
    d["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez(path, **d)


# ================================================================ evaluation
def rollout(sim, act_fn, seconds, track_n=24):
    """Run a controller from fresh spawns; return aggregate stats and the first rovers' tracks."""
    import torch
    sim.reset(torch.arange(sim.n, device=sim.device))
    steps = int(seconds * CONTROL_HZ)
    reached = torch.zeros(sim.n, device=sim.device)
    flips = torch.zeros(sim.n, device=sim.device)
    diverged = torch.zeros(sim.n, device=sim.device)
    prog = torch.zeros(sim.n, device=sim.device)
    tracks = []
    obs, p, _, _ = sim.observe()
    for _ in range(steps):
        tracks.append(p[:track_n, :2].cpu().numpy().copy())
        with torch.no_grad():
            a = act_fn(obs)
        r, done, info = sim.step(a)
        reached += info["reached"].float()
        flips += info["flipped"].float()
        diverged += info["diverged"].float()
        prog += torch.nan_to_num(info["progress"], nan=0.0).clamp(min=-1, max=1)
        obs, p, _, _ = sim.observe()
    return {"goals_per_rover": float(reached.mean()), "goals_total": int(reached.sum()),
            "flip_rate": float(flips.mean()), "diverged_events": int(diverged.sum()),
            "progress_m_per_rover": float(prog.mean()),
            "progress_speed_mps": float(prog.mean()) / seconds,
            "seconds": seconds, "rovers": sim.n}, np.stack(tracks)


def naive_controller(obs):
    """Drive at goal, steer proportionally to heading error. What a hand-written reflex does."""
    import torch
    heading = torch.atan2(obs[:, 10], obs[:, 9])
    steer = (heading / MAX_STEER).clamp(-1, 1)
    throttle = torch.full_like(steer, 0.7)
    return torch.stack([throttle, steer], 1)


# ================================================================ plots
def plot_terrain(terrain, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(terrain * 100, origin="lower", cmap="terrain",
                   extent=[-TERRAIN_M / 2, TERRAIN_M / 2, -TERRAIN_M / 2, TERRAIN_M / 2])
    for x, lab in ((-4.5, "near flat"), (-1.5, "undulating"), (1.5, "5 cm bumps"), (4.5, "rubble")):
        ax.text(x, 5.4, lab, ha="center", fontsize=9, color="k",
                bbox=dict(fc="w", ec="none", alpha=0.7))
    ax.set_xlabel("x [m]  (roughness grows to the right)")
    ax.set_ylabel("y [m]")
    ax.set_title("Training terrain - height in cm (wheel radius 3.25 cm)")
    fig.colorbar(im, ax=ax, label="cm")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tracks(terrain, tracks_naive, tracks_policy, goals_start, path, stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    for ax, tr, name, st in ((axes[0], tracks_naive, "hand-written reflex", stats[0]),
                             (axes[1], tracks_policy, "Newton-trained policy", stats[1])):
        ax.imshow(terrain * 100, origin="lower", cmap="Greys", alpha=0.6,
                  extent=[-TERRAIN_M / 2, TERRAIN_M / 2, -TERRAIN_M / 2, TERRAIN_M / 2])
        for i in range(tr.shape[1]):
            ax.plot(tr[:, i, 0], tr[:, i, 1], lw=1.2)
            ax.plot(tr[0, i, 0], tr[0, i, 1], "k.", ms=4)
        ax.set_title("%s\n%.2f goals/rover, %.3f m/s toward goal, flips %.2f" % (
            name, st["goals_per_rover"], st["progress_speed_mps"], st["flip_rate"]))
        ax.set_xlabel("x [m]")
    axes[0].set_ylabel("y [m]")
    fig.suptitle("%d s from identical spawns, %d rovers each (first %d tracks drawn)" % (
        stats[0]["seconds"], stats[0]["rovers"], tracks_naive.shape[1]))
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    it = [r["iter"] for r in log]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(it, [r["reward_per_step"] for r in log]); axes[0, 0].set_title("reward / step")
    axes[0, 1].plot(it, [r["goals_per_episode"] for r in log]); axes[0, 1].set_title("goals reached / episode")
    axes[1, 0].plot(it, [r["flip_frac"] for r in log]); axes[1, 0].set_title("episodes ended by a flip")
    axes[1, 1].plot(it, [r["entropy"] for r in log]); axes[1, 1].set_title("policy entropy")
    for ax in axes.flat:
        ax.set_xlabel("PPO iteration"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ================================================================ training
def train(args):
    import torch
    import warp as wp
    wp.init()
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tdev = torch.device(device)
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    T0 = time.time()
    log_lines = []

    def say(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        log_lines.append(s)

    terrain = make_terrain(args.seed)
    np.save(os.path.join(args.out, "terrain.npy"), terrain)
    plot_terrain(terrain, os.path.join(args.out, "terrain.png"))
    say("terrain: %dx%d cells, %.1f m, height 0-%.1f cm" % (TERRAIN_N, TERRAIN_N, TERRAIN_M,
                                                            terrain.max() * 100))

    t = time.time()
    sim = RoverSim(args.envs, device, terrain, args.seed, args.solver, args.mujoco_cpu)
    say("model: %d rovers, %d bodies, %d joints, %d shapes, built in %.1f s" % (
        args.envs, sim.model.body_count, sim.model.joint_count, sim.model.shape_count,
        time.time() - t))
    t = time.time()
    sim.physics()
    wp.synchronize()
    say("first physics step (JIT compile): %.1f s" % (time.time() - t))
    sim.capture()

    # ---- self-checks: the sim must behave like a car before anything is trained on it
    checks = {}
    jq, _, bq, _ = sim.views()
    zero = torch.zeros(sim.n, 2, device=tdev)
    for _ in range(CONTROL_HZ):                              # 1 s of settling with no command
        sim.apply(zero); sim.physics()
    _, p, g, _ = sim.observe()
    resid = (p[:, 2] - sim.terrain_h(p[:, :2]) - CHASSIS_H).abs()
    checks["spawn_height_residual_m_p95"] = float(resid.quantile(0.95))
    checks["upright_mean_after_1s"] = float((-g[:, 2]).mean())
    say("check: resting height residual p95 %.3f m (expect < 0.02), upright %.3f" % (
        checks["spawn_height_residual_m_p95"], checks["upright_mean_after_1s"]))
    fwd = torch.tensor([[1.0, 0.0]], device=tdev).expand(sim.n, 2)
    _, p0, _, _ = sim.observe(); p0 = p0.clone()
    for _ in range(3 * CONTROL_HZ):
        sim.apply(fwd); sim.physics()
    obs, p1, _, _ = sim.observe()
    vb = obs[:, 3:6]
    checks["forward_speed_mps_median"] = float(vb[:, 0].median())
    say("check: full throttle for 3 s -> body-frame speed median %.2f m/s (command %.2f)" % (
        checks["forward_speed_mps_median"], MAX_SPEED))
    left = torch.tensor([[0.8, 1.0]], device=tdev).expand(sim.n, 2)
    for _ in range(2 * CONTROL_HZ):
        sim.apply(left); sim.physics()
    obs, _, _, _ = sim.observe()
    checks["yaw_rate_full_left_median"] = float(obs[:, 8].median())
    say("check: full left -> yaw rate median %+.2f rad/s (expect about +%.1f)" % (
        checks["yaw_rate_full_left_median"], 0.8 * MAX_SPEED / 0.5))
    t = time.time()
    for _ in range(100):
        sim.apply(zero); sim.physics()
    wp.synchronize()
    step_ms = (time.time() - t) / 100 * 1000
    checks["physics_ms_per_control_step"] = step_ms
    checks["env_steps_per_sec"] = sim.n / step_ms * 1000
    say("speed: %.2f ms per 50 Hz control step for %d rovers = %.0f env-steps/s (%.0fx real time)" % (
        step_ms, sim.n, checks["env_steps_per_sec"], sim.n * (1000.0 / step_ms) / CONTROL_HZ))
    sim.reset(torch.arange(sim.n, device=tdev))

    # ---- baseline before training, same spawns the policy will get later
    sim.rng.manual_seed(args.seed + 1)
    naive_stats, naive_tracks = rollout(sim, naive_controller, args.eval_s)
    say("baseline (hand-written reflex): %.2f goals/rover, %.3f m/s toward goal, flips %.2f" % (
        naive_stats["goals_per_rover"], naive_stats["progress_speed_mps"], naive_stats["flip_rate"]))

    # ---- PPO
    actor = mlp([OBS_DIM, args.hidden, args.hidden, ACT_DIM]).to(tdev)
    critic = mlp([OBS_DIM, args.hidden, args.hidden, 1]).to(tdev)
    log_std = torch.nn.Parameter(torch.full((ACT_DIM,), -0.9, device=tdev))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std],
                           lr=args.lr)
    norm = RunningNorm(OBS_DIM, tdev)
    N, T = sim.n, args.steps
    buf_obs = torch.zeros(T, N, OBS_DIM, device=tdev)
    buf_act = torch.zeros(T, N, ACT_DIM, device=tdev)
    buf_logp = torch.zeros(T, N, device=tdev)
    buf_rew = torch.zeros(T, N, device=tdev)
    buf_done = torch.zeros(T, N, device=tdev)
    buf_val = torch.zeros(T, N, device=tdev)
    log = []
    sim.rng.manual_seed(args.seed + 2)
    sim.reset(torch.arange(N, device=tdev))
    obs_raw, _, _, _ = sim.observe()
    best = -1e9
    best_state = None
    say("training: %d envs x %d steps = %d samples/iter, %d iters, hidden %d" % (
        N, T, N * T, args.iters, args.hidden))
    for it in range(args.iters):
        t_it = time.time()
        ep_rets, ep_goals, ep_flips, n_eps = 0.0, 0.0, 0.0, 0
        t_sim = 0.0
        with torch.no_grad():
            for k in range(T):
                norm.update(obs_raw)
                o = norm(obs_raw)
                mu = actor(o)
                std = log_std.exp()
                a = mu + std * torch.randn_like(mu)
                logp = (-0.5 * ((a - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)).sum(1)
                v = critic(o).squeeze(1)
                ts = time.time()
                r, done, info = sim.step(a)
                t_sim += time.time() - ts
                buf_obs[k], buf_act[k], buf_logp[k], buf_rew[k], buf_done[k], buf_val[k] = \
                    o, a, logp, r, done.float(), v
                if done.any():
                    d = done
                    ep_rets += float(info["ep_ret"][d].sum())
                    ep_goals += float(info["ep_reached"][d].sum())
                    ep_flips += float(info["flipped"][d].sum())
                    n_eps += int(d.sum())
                obs_raw, _, _, _ = sim.observe()
            last_v = critic(norm(obs_raw)).squeeze(1)
            adv = torch.zeros(T, N, device=tdev)
            gae = torch.zeros(N, device=tdev)
            for k in reversed(range(T)):
                nv = last_v if k == T - 1 else buf_val[k + 1]
                nonterm = 1.0 - buf_done[k]
                delta = buf_rew[k] + args.gamma * nv * nonterm - buf_val[k]
                gae = delta + args.gamma * args.lam * nonterm * gae
                adv[k] = gae
            ret = adv + buf_val
        b_obs, b_act = buf_obs.reshape(-1, OBS_DIM), buf_act.reshape(-1, ACT_DIM)
        b_logp, b_adv, b_ret = buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        M = b_obs.shape[0]
        mb = min(args.minibatch, M)
        kls, ents, pls, vls = [], [], [], []
        for _ in range(args.epochs):
            perm = torch.randperm(M, device=tdev)
            for s in range(0, M, mb):
                idx = perm[s:s + mb]
                mu = actor(b_obs[idx])
                std = log_std.exp()
                logp = (-0.5 * ((b_act[idx] - mu) / std) ** 2 - log_std
                        - 0.5 * math.log(2 * math.pi)).sum(1)
                ratio = (logp - b_logp[idx]).exp()
                pl = -torch.min(ratio * b_adv[idx],
                                ratio.clamp(1 - args.clip, 1 + args.clip) * b_adv[idx]).mean()
                vl = ((critic(b_obs[idx]).squeeze(1) - b_ret[idx]) ** 2).mean()
                ent = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum()
                loss = pl + 0.5 * vl - args.ent_coef * ent
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters())
                                               + [log_std], 1.0)
                opt.step()
                with torch.no_grad():
                    kls.append(float((b_logp[idx] - logp).mean()))
                ents.append(float(ent)); pls.append(float(pl)); vls.append(float(vl))
        row = {"iter": it + 1, "reward_per_step": float(torch.nan_to_num(buf_rew).mean()),
               "diverged_frac": float((buf_rew <= -4.999).float().mean()),
               "episode_return": ep_rets / max(n_eps, 1), "goals_per_episode": ep_goals / max(n_eps, 1),
               "flip_frac": ep_flips / max(n_eps, 1), "episodes": n_eps,
               "policy_loss": float(np.mean(pls)), "value_loss": float(np.mean(vls)),
               "entropy": float(np.mean(ents)), "approx_kl": float(np.mean(kls)),
               "std": [float(x) for x in log_std.exp()], "iter_s": time.time() - t_it,
               "sim_s": t_sim, "samples": N * T, "wall_min": (time.time() - T0) / 60.0}
        log.append(row)
        if (it + 1) % max(1, args.iters // 40) == 0 or it == 0:
            say("it %4d | r/step %+.4f | ret %+7.2f | goals/ep %.2f | flip %.2f | eps %5d | "
                "ent %.2f | kl %.4f | %.1fs (sim %.1fs) | %.1f min" % (
                    it + 1, row["reward_per_step"], row["episode_return"], row["goals_per_episode"],
                    row["flip_frac"], n_eps, row["entropy"], row["approx_kl"], row["iter_s"],
                    t_sim, row["wall_min"]))
        import copy
        score = row["goals_per_episode"] if n_eps >= 20 else -1.0
        if score > best:
            best = score
            best_state = (copy.deepcopy(actor.state_dict()),
                          norm.mean.clone(), norm.var.clone(), it + 1)
        if (it + 1) % 10 == 0:
            with open(os.path.join(args.out, "train_log.json"), "w") as f:
                json.dump(log, f)
            export_npz(actor, norm, os.path.join(args.out, "policy_latest.npz"), {"iter": it + 1})
        if (time.time() - T0) / 60.0 > args.max_minutes:
            say("wall-clock cap reached at iteration %d" % (it + 1))
            break

    # ---- final export and evaluation against the same spawns the baseline had
    if best_state is not None:
        sd, bmean, bvar, bit = best_state
        actor.load_state_dict(sd); norm.mean, norm.var = bmean, bvar
        say("restored best checkpoint from iteration %d (goals/ep %.2f) for export + eval" % (bit, best))
    meta = {"iters": len(log), "envs": N, "steps": T, "hidden": args.hidden,
            "best_iter": (best_state[3] if best_state else len(log)), "best_goals_per_ep": best,
            "obs_spec": OBS_SPEC, "act_spec": ACT_SPEC, "control_hz": CONTROL_HZ,
            "newton": __import__("newton").__version__, "warp": wp.config.version,
            "device": device, "trained_on": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    export_npz(actor, norm, os.path.join(args.out, "policy.npz"), meta)
    with open(os.path.join(args.out, "train_log.json"), "w") as f:
        json.dump(log, f)
    if log:
        plot_curves(log, os.path.join(args.out, "curves.png"))

    def policy_act(o):
        return actor(norm(o)).clamp(-1, 1)
    sim.rng.manual_seed(args.seed + 1)
    pol_stats, pol_tracks = rollout(sim, policy_act, args.eval_s)
    say("policy after training: %.2f goals/rover, %.3f m/s toward goal, flips %.2f" % (
        pol_stats["goals_per_rover"], pol_stats["progress_speed_mps"], pol_stats["flip_rate"]))
    plot_tracks(terrain, naive_tracks, pol_tracks, None, os.path.join(args.out, "tracks.png"),
                (naive_stats, pol_stats))

    # numpy re-check of the exported file against torch, the same maths policy_runner.py does
    z = np.load(os.path.join(args.out, "policy.npz"))
    x = obs_raw[:8].cpu().numpy()
    xn = (x - z["obs_mean"]) / z["obs_std"]
    h = xn
    for i in range(2):
        h = np.tanh(h.dot(z["W%d" % i]) + z["b%d" % i])
    y_np = h.dot(z["W2"]) + z["b2"]
    with torch.no_grad():
        y_t = actor(norm(obs_raw[:8])).cpu().numpy()
    checks["npz_vs_torch_max_abs_diff"] = float(np.abs(y_np - y_t).max())
    say("export check: numpy vs torch max |diff| = %.2e" % checks["npz_vs_torch_max_abs_diff"])

    summary = {"checks": checks, "baseline": naive_stats, "policy": pol_stats, "meta": meta,
               "iterations": len(log), "final": log[-1] if log else None,
               "total_env_steps": sum(r["samples"] for r in log),
               "simulated_rover_hours": sum(r["samples"] for r in log) / CONTROL_HZ / 3600.0,
               "wall_minutes": (time.time() - T0) / 60.0, "log": log_lines}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    say("done in %.1f min; %.1f simulated rover-hours; outputs in %s" % (
        summary["wall_minutes"], summary["simulated_rover_hours"], args.out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--eval-s", type=float, default=20.0)
    ap.add_argument("--max-minutes", type=float, default=100.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="out")
    ap.add_argument("--solver", default="mujoco", choices=["mujoco", "xpbd", "featherstone"])
    ap.add_argument("--mujoco-cpu", action="store_true", help="C MuJoCo instead of mujoco_warp")
    ap.add_argument("--smoke", action="store_true", help="4 rovers, 2 iterations, CPU is fine")
    args = ap.parse_args()
    if args.smoke:
        args.envs, args.iters, args.steps, args.minibatch, args.eval_s = 4, 2, 16, 64, 3.0
    train(args)


if __name__ == "__main__":
    main()
