# -*- coding: utf-8 -*-
"""Simulated mine world for the brain: ground truth, what the rover has seen, and its physics.

Per 1 s step: the rover advances 0.3 m (NORMAL) along its planned path, the lidar (360 rays,
8 m) reveals the truth into an int8 OccupancyGrid-like map, the camera may detect persons, the
battery drains per metre and per second, the radio link is evaluated at the rover's position,
and, in rockfall episodes, a corridor the rover already passed may collapse.

MODELS (all parameters sampled per episode from SimConfig ranges):
  battery   drain = drain_m x metres + drain_s x seconds, true rates = nominal x U(0.85, 1.25)
            (the brain only knows the nominal rates and what it measures); obs noise 0.3 %.
  link      loss_dB = alpha x path_m + beta x corners, link up iff loss_dB <= budget. path_m is
            the geodesic distance from home; corners are direction changes > 35 deg of the
            shortest path from home measured over 2 m. alpha 0.15-0.4 dB/m, beta 8-16 dB,
            budget 25-45 dB. Evaluated on the original map (rubble does not block the radio).
  camera    a person within 4 m, in line of sight and within +-50 deg of the heading is hit
            with probability p per second: 0.30 moving, 0.80 dwelling, x (1 - 0.5 (d/4)^2).
            Hits are clustered into sightings (1.5 m); a sighting with >= 3 hits is confirmed.
            False positives appear at 0.002 per moving second 2-4 m ahead and gain further hits
            at only 0.05 per second in view, so they rarely confirm.
  persons   0/1/2/3 with probability 0.15/0.35/0.30/0.20; each lies in a chamber with prob
            0.45, a dead-end zone (last 4 m of a dead end) 0.35, a corridor 0.20 (renormalised
            over the region types this mine has), >= 6 m from home, >= 3 m from each other.
  rockfall  once odometry passes U(40, 150) m, a 1.6 m-radius disc of rubble fills the rover's
            own trail at a point 8+ m (straight line) behind it and 5+ m from home, in a corridor.
            With probability 0.7 the spot is chosen so another way home still exists (if the mine
            has one); otherwise it may cut the rover off completely.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np

from ..geodesic import Graph, distances, STEP
from ..mapview import MapView
from ..geodesic import descend
from .mapgen import CORRIDOR, CHAMBER, DEADEND


class SimConfig(object):
    def __init__(self, **kw):
        self.dt = 1.0
        self.speed = 0.30
        self.slow_speed = 0.18
        self.lidar_range = 8.0
        self.lidar_rays = 360
        self.cam_range = 4.0
        self.cam_half_fov = math.radians(50.0)
        self.p_move = 0.30
        self.p_dwell = 0.80
        self.confirm_hits = 3
        self.fp_rate = 0.002
        self.fp_hit = 0.05
        self.drain_m = 0.04
        self.drain_s = 0.012
        self.drain_jitter = (0.85, 1.25)
        self.battery0 = (55.0, 100.0)
        self.battery_noise = 0.3
        self.link_loss = True
        self.alpha = (0.15, 0.40)
        self.beta = (8.0, 16.0)
        self.budget = (25.0, 45.0)
        self.rockfall = False
        self.rockfall_odo = (40.0, 150.0)
        self.person_counts = (0.15, 0.35, 0.30, 0.20)
        self.person_region = (0.45, 0.35, 0.20)      # chamber, dead end, corridor
        self.t_max = 10800.0
        for k, v in kw.items():
            if not hasattr(self, k):
                raise KeyError(k)
            setattr(self, k, v)


class World(object):
    def __init__(self, mine, cfg, seed, speed=None):
        self.mine = mine
        self.cfg = cfg
        self.rng = np.random.RandomState(seed)
        rng = self.rng
        self.res = mine.res
        self.truth = mine.free.copy()
        H, W = self.truth.shape
        self.H, self.W = H, W
        self.known = np.full((H, W), -1, dtype=np.int8)
        self.x, self.y = float(mine.home[0]), float(mine.home[1])
        self.yaw = float(mine.start_yaw)
        self.home = (float(mine.home[0]), float(mine.home[1]))
        self.t = 0.0
        self.odo = 0.0
        self.speed = cfg.speed if speed is None else speed
        self.battery = rng.uniform(*cfg.battery0)
        j = rng.uniform(*cfg.drain_jitter)
        self.drain_m = cfg.drain_m * j
        self.drain_s = cfg.drain_s * j
        self.trail = [(self.x, self.y)]
        self.blocked = []
        self.events = []
        self.sightings = []
        self.dwelling = False
        # lidar geometry
        step = 0.7 * self.res
        ns = int(cfg.lidar_range / step)
        ang = np.linspace(0.0, 2 * math.pi, cfg.lidar_rays, endpoint=False)
        r = (np.arange(ns) + 1) * step
        self._dx = np.cos(ang)[:, None] * r[None, :]
        self._dy = np.sin(ang)[:, None] * r[None, :]
        self._ns = ns
        self._place_persons()
        self._link_setup()
        self.rock_odo = rng.uniform(*cfg.rockfall_odo) if cfg.rockfall else None
        self.rockfall_at = None
        self.rockfall_route_left = None
        self.link_ok = True
        self.last_contact_t = 0.0
        self.last_link_pose = (self.x, self.y)
        self.link_ever_lost = False
        self.update_link()
        self.reveal()

    # ---- setup -----------------------------------------------------------------------------
    def _place_persons(self):
        cfg, rng, m = self.cfg, self.rng, self.mine
        n = int(rng.choice(4, p=cfg.person_counts))
        regs = [CHAMBER, DEADEND, CORRIDOR]
        cells = [np.flatnonzero((m.region == r).ravel() & self.truth.ravel()) for r in regs]
        self.persons = []
        tries = 0
        while len(self.persons) < n and tries < 200:
            tries += 1
            w = np.array([p if c.size else 0.0 for p, c in zip(cfg.person_region, cells)])
            if w.sum() <= 0:
                break
            k = rng.choice(3, p=w / w.sum())
            idx = cells[k][rng.randint(cells[k].size)]
            i, jj = divmod(int(idx), self.W)
            x, y = (jj + 0.5) * self.res, (i + 0.5) * self.res
            if math.hypot(x - self.home[0], y - self.home[1]) < 6.0:
                continue
            if any(math.hypot(x - p["x"], y - p["y"]) < 3.0 for p in self.persons):
                continue
            self.persons.append({"x": x, "y": y, "region": ["chamber", "deadend", "corridor"][k],
                                 "found_t": None, "reported_t": None, "reported_radio": False})

    def _link_setup(self):
        """Radio loss field on a 0.2 m copy of the original map (see module doc)."""
        cfg, rng = self.cfg, self.rng
        f = max(1, int(round(0.2 / self.res)))
        H2, W2 = self.H // f, self.W // f
        fr = self.truth[:H2 * f, :W2 * f].reshape(H2, f, W2, f).mean(axis=3).mean(axis=1) >= 0.5
        self._lf = f
        g = Graph(fr)
        hi, hj = int(self.home[1] / (0.2)), int(self.home[0] / 0.2)
        hn = g.node((hi, hj))
        if hn < 0:
            d2 = (g.rr - hi) ** 2 + (g.cc - hj) ** 2
            hn = int(np.argmin(d2))
        D = distances(g, [hn])[0]
        N = g.N
        ext = np.append(D, np.inf)
        par = g.NB[np.arange(N), np.argmin(ext[g.NB[:N]] + STEP, axis=1)]
        par[hn] = hn
        par = np.where(np.isfinite(D), par, hn)
        anc = par.copy()
        for _ in range(9):                      # ancestor ~2 m back along the path
            anc = par[anc]
        dirn = np.arctan2((g.rr - g.rr[anc]).astype(np.float64), (g.cc - g.cc[anc]).astype(np.float64))
        ref = np.zeros(N)
        corners = np.zeros(N, dtype=np.int32)
        fin = np.isfinite(D)
        order = np.argsort(np.where(fin, D, np.inf))
        buckets = np.floor(D[order[fin[order]]]).astype(np.int64)
        idx = order[fin[order]]
        ref[hn] = dirn[hn] if hn != anc[hn] else 0.0
        cuts = np.flatnonzero(np.diff(buckets)) + 1
        for grp in np.split(idx, cuts):
            grp = grp[grp != hn]
            if grp.size == 0:
                continue
            p = par[grp]
            young = D[grp] < 10.0                     # within 2 m of home: direction undefined
            dd = np.abs((dirn[grp] - ref[p] + np.pi) % (2 * np.pi) - np.pi)
            turn = (dd > math.radians(35.0)) & ~young
            ref[grp] = np.where(turn | young, dirn[grp], ref[p])
            corners[grp] = corners[p] + turn
        self.link_alpha = rng.uniform(*cfg.alpha)
        self.link_beta = rng.uniform(*cfg.beta)
        self.link_budget = rng.uniform(*cfg.budget) if cfg.link_loss else np.inf
        loss = self.link_alpha * D * 0.2 + self.link_beta * corners
        img = np.full((H2, W2), np.inf)
        img[g.rr, g.cc] = loss
        self._link_img = img
        self._corners_img = np.full((H2, W2), -1, dtype=np.int32)
        self._corners_img[g.rr, g.cc] = corners

    # ---- sensing ---------------------------------------------------------------------------
    def link_loss_at(self, x, y):
        f = self._lf
        i, j = int(y / (self.res * f)), int(x / (self.res * f))
        a0, a1 = max(0, i - 1), min(self._link_img.shape[0], i + 2)
        b0, b1 = max(0, j - 1), min(self._link_img.shape[1], j + 2)
        win = self._link_img[a0:a1, b0:b1]
        return float(win.min()) if win.size else np.inf

    def update_link(self):
        ok = self.link_loss_at(self.x, self.y) <= self.link_budget
        changed = ok != self.link_ok
        self.link_ok = ok
        if ok:
            self.last_contact_t = self.t
            self.last_link_pose = (self.x, self.y)
            for s in self.sightings:
                if s["confirmed"] and not s["reported"]:
                    s["reported"] = True
            for p in self.persons:
                if p["found_t"] is not None and p["reported_t"] is None:
                    p["reported_t"] = self.t
                    p["reported_radio"] = math.hypot(self.x - self.home[0], self.y - self.home[1]) > 1.0
        else:
            self.link_ever_lost = True
        return changed

    def reveal(self):
        res = self.res
        xs = self.x + self._dx
        ys = self.y + self._dy
        ii = np.floor(ys / res).astype(np.int64)
        jj = np.floor(xs / res).astype(np.int64)
        inb = (ii >= 0) & (ii < self.H) & (jj >= 0) & (jj < self.W)
        ii = np.clip(ii, 0, self.H - 1)
        jj = np.clip(jj, 0, self.W - 1)
        blk = ~(self.truth[ii, jj] & inb)
        anyb = blk.any(axis=1)
        first = np.where(anyb, blk.argmax(axis=1), self._ns)
        vis = np.arange(self._ns)[None, :] < first[:, None]
        self.known[ii[vis], jj[vis]] = 0
        rays = np.flatnonzero(anyb & inb[np.arange(len(first)), np.minimum(first, self._ns - 1)])
        hi = ii[rays, first[rays]]
        hj = jj[rays, first[rays]]
        newly = self.known[hi, hj] == 0            # was seen free, now an obstacle: fresh rubble
        self.known[hi, hj] = 100
        ci, cj = int(self.y / res), int(self.x / res)
        self.known[ci, cj] = 0
        added = False
        if newly.any():
            # the executive marks exactly the rubble it saw (0.1 m discs on the new obstacle
            # cells, one per 0.4 m), not a circle around it: the cells are already obstacles in
            # the map, the discs only make them permanent. A partly blocked corridor must stay
            # passable if the gap left is wide enough for the robot
            px = (hj[newly] + 0.5) * res
            py = (hi[newly] + 0.5) * res
            for x, y in zip(px.tolist(), py.tolist()):
                if len(self.blocked) >= 300:
                    break
                if any(math.hypot(x - b[0], y - b[1]) < 0.4 for b in self.blocked):
                    continue
                self.blocked.append((round(x, 2), round(y, 2), 0.1))
                added = True
            if added:
                self.events.append((self.t, "rubble_seen", float(px.mean()), float(py.mean())))
        return added

    def los(self, x0, y0, x1, y1):
        L = math.hypot(x1 - x0, y1 - y0)
        n = max(2, int(L / (0.5 * self.res)) + 1)
        t = np.linspace(0, 1, n)
        ii = np.floor((y0 + (y1 - y0) * t) / self.res).astype(np.int64)
        jj = np.floor((x0 + (x1 - x0) * t) / self.res).astype(np.int64)
        if ii.min() < 0 or jj.min() < 0 or ii.max() >= self.H or jj.max() >= self.W:
            return False
        return bool(self.truth[ii, jj].all())

    def camera(self, dt):
        """Detections this step. Returns True if a sighting was created or confirmed."""
        cfg, rng = self.cfg, self.rng
        p0 = cfg.p_dwell if self.dwelling else cfg.p_move
        ev = False
        targets = [(p["x"], p["y"], ("P", i)) for i, p in enumerate(self.persons)]
        targets += [(s["tx"], s["ty"], ("F", k)) for k, s in enumerate(self.sightings) if s["false"]]
        for (tx, ty, tag) in targets:
            d = math.hypot(tx - self.x, ty - self.y)
            if d > cfg.cam_range:
                continue
            b = abs((math.atan2(ty - self.y, tx - self.x) - self.yaw + math.pi) % (2 * math.pi) - math.pi)
            if b > cfg.cam_half_fov or not self.los(self.x, self.y, tx, ty):
                continue
            if tag[0] == "P":
                p = min(0.95, p0 * (1.0 - 0.5 * (d / cfg.cam_range) ** 2) * dt)
            else:
                p = min(0.95, cfg.fp_hit * dt)
            if rng.rand() < p:
                ev |= self._hit(tx, ty, tag)
        if not self.dwelling and rng.rand() < cfg.fp_rate * dt:
            d = rng.uniform(2.0, 4.0)
            a = self.yaw + rng.uniform(-cfg.cam_half_fov, cfg.cam_half_fov)
            tx, ty = self.x + d * math.cos(a), self.y + d * math.sin(a)
            i, j = int(ty / self.res), int(tx / self.res)
            if 0 <= i < self.H and 0 <= j < self.W and self.truth[i, j] and self.los(self.x, self.y, tx, ty):
                self.sightings.append({"x": tx + rng.normal(0, 0.3), "y": ty + rng.normal(0, 0.3), "hits": 1,
                                       "confirmed": False, "reported": False, "false": True,
                                       "tx": tx, "ty": ty, "person": None})
                ev = True
        return ev

    def _hit(self, tx, ty, tag):
        rng = self.rng
        ox, oy = tx + rng.normal(0, 0.3), ty + rng.normal(0, 0.3)
        best = None
        for s in self.sightings:
            if math.hypot(s["x"] - ox, s["y"] - oy) < 1.5:
                best = s
                break
        if best is None:
            self.sightings.append({"x": ox, "y": oy, "hits": 1, "confirmed": False, "reported": False,
                                   "false": tag[0] == "F", "tx": tx, "ty": ty,
                                   "person": tag[1] if tag[0] == "P" else None})
            return True
        n = best["hits"]
        best["x"] = (best["x"] * n + ox) / (n + 1)
        best["y"] = (best["y"] * n + oy) / (n + 1)
        best["hits"] = n + 1
        if tag[0] == "P" and best["person"] is None and not best["false"]:
            best["person"] = tag[1]
        if not best["confirmed"] and best["hits"] >= self.cfg.confirm_hits:
            best["confirmed"] = True
            if best["person"] is not None:
                p = self.persons[best["person"]]
                if p["found_t"] is None:
                    p["found_t"] = self.t
            if self.link_ok:
                best["reported"] = True
                self.update_link()
            return True
        return False

    # ---- rockfall ----------------------------------------------------------------------------
    def maybe_rockfall(self):
        if self.rock_odo is None or self.rockfall_at is not None or self.odo < self.rock_odo:
            return False
        rng = self.rng
        tr = np.array(self.trail[: max(1, int(len(self.trail) * 0.85))])
        if tr.shape[0] < 5:
            return False
        dr = np.hypot(tr[:, 0] - self.x, tr[:, 1] - self.y)
        dh = np.hypot(tr[:, 0] - self.home[0], tr[:, 1] - self.home[1])
        ii = (tr[:, 1] / self.res).astype(int)
        jj = (tr[:, 0] / self.res).astype(int)
        reg = self.mine.region[ii, jj]
        ok = np.flatnonzero((dr >= 8.0) & (dh >= 5.0) & ((reg == CORRIDOR) | (reg == DEADEND)))
        if ok.size == 0:
            self.rock_odo += 20.0
            return False
        R = 1.6
        want_route = rng.rand() < 0.7
        picks = ok[rng.permutation(ok.size)[:8]]
        chosen = None
        for k in picks:
            cx, cy = float(tr[k, 0]), float(tr[k, 1])
            disc, sl = self._disc(cx, cy, R)
            saved = self.truth[sl].copy()
            self.truth[sl] &= ~disc
            conn = self.home_connected_truth()
            self.truth[sl] = saved
            if conn == want_route or chosen is None:
                chosen = (cx, cy, disc, sl, conn)
            if conn == want_route:
                break
        cx, cy, disc, sl, conn = chosen
        self.truth[sl] &= ~disc
        self.rockfall_route_left = bool(conn)
        self.rockfall_at = (cx, cy, self.t)
        self.events.append((self.t, "rockfall", cx, cy))
        return True

    def _disc(self, cx, cy, R):
        i0, i1 = max(0, int((cy - R) / self.res)), min(self.H, int((cy + R) / self.res) + 2)
        j0, j1 = max(0, int((cx - R) / self.res)), min(self.W, int((cx + R) / self.res) + 2)
        I, J = np.mgrid[i0:i1, j0:j1]
        disc = ((J + 0.5) * self.res - cx) ** 2 + ((I + 0.5) * self.res - cy) ** 2 <= R * R
        return disc, (slice(i0, i1), slice(j0, j1))

    def home_connected_truth(self):
        """Is there any path home in the TRUE map (coarse 0.2 m, robot-sized)?"""
        v = MapView(np.where(self.truth, 0, 100).astype(np.int8), self.res, (0.0, 0.0), 0.2, 0.15,
                    include=[(self.x, self.y), self.home])
        rc = v.snap(self.x, self.y, 1.0, v.passable)
        hc = v.snap(self.home[0], self.home[1], 1.5, v.passable)
        if rc is None or hc is None:
            return False
        g = Graph(v.passable)
        d = distances(g, [g.node(rc)])[0]
        n = g.node(hc)
        return n >= 0 and bool(np.isfinite(d[n]))

    # ---- motion ------------------------------------------------------------------------------
    def plan(self, gx, gy, cell_m=0.2, inflate_m=0.15):
        """Path on the KNOWN map (same planning grid as the brain) to (gx, gy); list of (x, y)."""
        v = MapView(self.known, self.res, (0.0, 0.0), cell_m, inflate_m, blocked=self.blocked,
                    include=[(self.x, self.y), (gx, gy)])
        v.open_escape(self.x, self.y, 1.0)
        rc = v.snap(self.x, self.y, 1.0, v.passable)
        if rc is None:
            i, j = v.cell_of(self.x, self.y)
            rc = (i, j) if v.force_passable(i, j) else None
        gc = v.snap(gx, gy, 1.0, v.passable)
        if rc is None or gc is None:
            return None
        g = Graph(v.passable)
        gn, rn = g.node(gc), g.node(rc)
        d = distances(g, [gn])[0]
        if rn < 0 or not np.isfinite(d[rn]):
            return None
        nodes = descend(g, d, rn)
        xs, ys = v.xy_of(g.rr[nodes], g.cc[nodes])
        pts = list(zip(xs.tolist(), ys.tolist()))
        if pts:
            pts[-1] = (gx, gy) if math.hypot(pts[-1][0] - gx, pts[-1][1] - gy) < 0.3 else pts[-1]
        return pts

    def path_blocked(self, path, k0, look=25):
        """Has the known map or a blocked region closed the next part of the path?"""
        res = self.res
        for (x, y) in path[k0:k0 + look]:
            i, j = int(y / res), int(x / res)
            if self.known[max(0, i - 1):i + 2, max(0, j - 1):j + 2].max() >= 65:
                return True
            for (bx, by, br) in self.blocked:
                if math.hypot(x - bx, y - by) <= br + 0.15:
                    return True
        return False

    def advance(self, path, k, dist):
        """Move up to `dist` along path starting at index k. Returns (new k, metres moved)."""
        moved = 0.0
        res = self.res
        while dist > 1e-9 and k < len(path):
            tx, ty = path[k]
            dx, dy = tx - self.x, ty - self.y
            L = math.hypot(dx, dy)
            if L < 1e-6:
                k += 1
                continue
            s = min(L, dist)
            nx, ny = self.x + dx / L * s, self.y + dy / L * s
            i, j = int(ny / res), int(nx / res)
            if not (0 <= i < self.H and 0 <= j < self.W) or not self.truth[i, j]:
                break                                   # would drive into rock: stop
            self.x, self.y = nx, ny
            self.yaw = math.atan2(dy, dx)
            moved += s
            dist -= s
            if s >= L - 1e-9:
                k += 1
        if moved > 0:
            self.odo += moved
            if math.hypot(self.x - self.trail[-1][0], self.y - self.trail[-1][1]) >= 1.0:
                self.trail.append((self.x, self.y))
        return k, moved

    def tick(self, dt, moved):
        self.t += dt
        self.battery -= self.drain_m * moved + self.drain_s * dt

    def obs(self):
        rng = self.rng
        return {"grid": self.known, "res": self.res, "origin": (0.0, 0.0),
                "pose": (self.x, self.y, self.yaw), "home": self.home,
                "battery_pct": float(max(0.0, self.battery + rng.normal(0, self.cfg.battery_noise))),
                "link_ok": bool(self.link_ok), "since_contact_s": float(self.t - self.last_contact_t),
                "elapsed_s": float(self.t), "odo_m": float(self.odo),
                "persons": [{"x": s["x"], "y": s["y"], "confirmed": s["confirmed"], "reported": s["reported"],
                             "hits": s["hits"]} for s in self.sightings],
                "blocked": list(self.blocked), "last_link_pose": self.last_link_pose,
                "trail": self.trail[-60:]}
