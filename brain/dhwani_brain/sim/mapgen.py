# -*- coding: utf-8 -*-
"""Procedural underground mines on an occupancy grid.

Two generator styles:

  drift    (TRAINING) a portal on the west edge; corridors grown as a random tree of straight
           segments 5-14 m long that bend up to +-35 deg, branch at 50-110 deg (junction degree
           2-4), end in dead ends; 0-3 extra segments close loops; 0.3-0.6 of dead ends and a few
           corridor sides get chambers (3-7 m rooms, some with boulders). Widths 1.2-2.5 m.
  lattice  (HELD OUT, never trained on) room-and-pillar: a rotated grid of 1.4-2.5 m corridors
           every 7-11 m, some segments collapsed (dead ends), some pillars removed (rooms).

Walls get +-0.1 m of smooth roughness so corridors are not perfect strips. Extents 30-80 m.
Every map carries a region label per cell (corridor / chamber / dead-end zone) for the person
prior, and the corridor graph it was drawn from.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np

ROCK, CORRIDOR, CHAMBER, DEADEND = 0, 1, 2, 3


class Mine(object):
    pass


def _seg_dist(ax, ay, bx, by, px, py):
    """Distance from points (px, py) to segment a-b."""
    vx, vy = bx - ax, by - ay
    L2 = vx * vx + vy * vy
    if L2 < 1e-12:
        return np.hypot(px - ax, py - ay)
    t = np.clip(((px - ax) * vx + (py - ay) * vy) / L2, 0.0, 1.0)
    return np.hypot(px - (ax + t * vx), py - (ay + t * vy))


def _seg_seg_dist(a, b, c, d):
    """Minimum distance between segments a-b and c-d (2-D)."""
    def pd(p, s0, s1):
        return float(_seg_dist(s0[0], s0[1], s1[0], s1[1], np.array(p[0]), np.array(p[1])))

    def cross(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return 0.0
    return min(pd(a, c, d), pd(b, c, d), pd(c, a, b), pd(d, a, b))


def _smooth_noise(rng, h, w, cell_px, amp):
    """Bilinearly upsampled Gaussian noise, amplitude ~amp, correlation ~cell_px pixels."""
    gh = h // cell_px + 3
    gw = w // cell_px + 3
    g = rng.randn(gh, gw) * amp
    yi = np.arange(h) / float(cell_px)
    xi = np.arange(w) / float(cell_px)
    y0 = np.floor(yi).astype(int)
    x0 = np.floor(xi).astype(int)
    fy = (yi - y0)[:, None]
    fx = (xi - x0)[None, :]
    a = g[y0][:, x0]
    b = g[y0][:, x0 + 1]
    c = g[y0 + 1][:, x0]
    d = g[y0 + 1][:, x0 + 1]
    return (a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy).astype(np.float32)


class _Builder(object):
    def __init__(self, rng, W, H):
        self.rng = rng
        self.W, self.H = W, H
        self.nodes = []          # (x, y)
        self.edges = []          # (a, b, width)
        self.adj = {}
        self.chambers = []       # (cx, cy, a, b, theta)

    def add_node(self, x, y):
        self.nodes.append((float(x), float(y)))
        self.adj[len(self.nodes) - 1] = []
        return len(self.nodes) - 1

    def add_edge(self, a, b, w):
        self.edges.append((a, b, float(w)))
        self.adj[a].append(b)
        self.adj[b].append(a)

    def inside(self, x, y, m=2.5):
        return m <= x <= self.W - m and m <= y <= self.H - m

    def clear(self, p, q, w, skip_nodes, gap=1.6):
        """Segment p-q of width w keeps a rock wall >= gap from every edge not touching skip_nodes
        and from every chamber."""
        for (a, b, wo) in self.edges:
            if a in skip_nodes or b in skip_nodes:
                continue
            if _seg_seg_dist(p, q, self.nodes[a], self.nodes[b]) < 0.5 * (w + wo) + gap:
                return False
        for (cx, cy, ca, cb, th) in self.chambers:
            r = 0.5 * math.hypot(ca, cb)
            if float(_seg_dist(p[0], p[1], q[0], q[1], np.array(cx), np.array(cy))) < r + 0.5 * w + gap:
                return False
        return True

    def angle_ok(self, n, heading, min_deg=40.0):
        x, y = self.nodes[n]
        for m in self.adj[n]:
            mx, my = self.nodes[m]
            h2 = math.atan2(my - y, mx - x)
            d = abs((heading - h2 + math.pi) % (2 * math.pi) - math.pi)
            if math.degrees(d) < min_deg:
                return False
        return True


def gen_drift(seed, res=0.1, extent=(30.0, 80.0), width=(1.2, 2.5)):
    rng = np.random.RandomState(seed)
    W = rng.uniform(*extent)
    H = rng.uniform(*extent)
    B = _Builder(rng, W, H)
    base_w = rng.uniform(width[0], width[1] - 0.4)
    wjit = lambda: float(np.clip(base_w + rng.uniform(-0.25, 0.45), width[0], width[1]))
    density = rng.uniform(0.06, 0.12)
    target = density * W * H
    y0 = rng.uniform(0.3, 0.7) * H
    portal = B.add_node(1.0, y0)
    first = B.add_node(rng.uniform(6.0, 9.0), y0 + rng.uniform(-1.0, 1.0))
    B.add_edge(portal, first, wjit())
    total = math.hypot(B.nodes[first][0] - 1.0, B.nodes[first][1] - y0)
    tips = [(first, 0.0)]
    tries = 0
    while total < target and tries < 4000:
        tries += 1
        if not tips:
            cands = [n for n in range(1, len(B.nodes)) if len(B.adj[n]) < 3]
            if not cands:
                break
            n = cands[rng.randint(len(cands))]
            a = B.adj[n][0]
            base = math.atan2(B.nodes[n][1] - B.nodes[a][1], B.nodes[n][0] - B.nodes[a][0])
            tips.append((n, base + rng.choice([-1, 1]) * math.radians(rng.uniform(55, 105))))
            continue
        k = rng.randint(len(tips))
        n, heading = tips.pop(k)
        grown = False
        for _ in range(6):
            h = heading + math.radians(float(np.clip(rng.normal(0, 18), -35, 35)))
            L = rng.uniform(5.0, 14.0)
            x, y = B.nodes[n]
            q = (x + L * math.cos(h), y + L * math.sin(h))
            if not B.inside(q[0], q[1]):
                continue
            w = wjit()
            if not B.angle_ok(n, h) or not B.clear((x, y), q, w, {n}):
                continue
            m = B.add_node(q[0], q[1])
            B.add_edge(n, m, w)
            total += L
            grown = True
            r = rng.rand()
            if r < 0.12:
                pass                                                   # dead end
            elif r < 0.62:
                tips.append((m, h))                                    # continue
            elif r < 0.94:
                tips.append((m, h))
                tips.append((m, h + rng.choice([-1, 1]) * math.radians(rng.uniform(55, 105))))
            else:
                tips.append((m, h + math.radians(rng.uniform(55, 100))))
                tips.append((m, h - math.radians(rng.uniform(55, 100))))
                tips.append((m, h))
            break
        if not grown and len(B.adj[n]) == 1 and n != portal:
            pass                                                        # stays a dead end
    # loops
    n_loops = rng.choice([0, 1, 1, 2, 2, 3])
    for _ in range(n_loops):
        for _t in range(60):
            a, b = rng.randint(1, len(B.nodes)), rng.randint(1, len(B.nodes))
            if a == b or len(B.adj[a]) >= 4 or len(B.adj[b]) >= 4 or b in B.adj[a]:
                continue
            pa, pb = B.nodes[a], B.nodes[b]
            L = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
            if not (6.0 <= L <= 18.0):
                continue
            if _hops(B, a, b) < 4:
                continue
            h = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
            w = wjit()
            if not B.angle_ok(a, h) or not B.angle_ok(b, h + math.pi):
                continue
            if not B.clear(pa, pb, w, {a, b}):
                continue
            B.add_edge(a, b, w)
            break
    # chambers
    dead = [n for n in range(1, len(B.nodes)) if len(B.adj[n]) == 1]
    p_ch = rng.uniform(0.3, 0.6)
    for n in dead:
        if rng.rand() > p_ch:
            continue
        m = B.adj[n][0]
        h = math.atan2(B.nodes[n][1] - B.nodes[m][1], B.nodes[n][0] - B.nodes[m][0])
        a_len, b_len = rng.uniform(3.0, 7.0), rng.uniform(3.0, 6.0)
        cx = B.nodes[n][0] + 0.5 * a_len * math.cos(h) * 0.8
        cy = B.nodes[n][1] + 0.5 * a_len * math.sin(h) * 0.8
        if not B.inside(cx, cy, 0.5 * max(a_len, b_len) + 1.0):
            continue
        if _chamber_clear(B, cx, cy, a_len, b_len, {n, m}):
            B.chambers.append((cx, cy, a_len, b_len, h))
    for _ in range(rng.poisson(1.0)):                                  # side chambers
        if not B.edges:
            break
        a, b, w = B.edges[rng.randint(len(B.edges))]
        pa, pb = B.nodes[a], B.nodes[b]
        L = math.hypot(pb[0] - pa[0], pb[1] - pa[1])
        if L < 7.0:
            continue
        h = math.atan2(pb[1] - pa[1], pb[0] - pa[0])
        side = h + rng.choice([-1, 1]) * math.pi / 2
        t = rng.uniform(0.35, 0.65)
        mx, my = pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1])
        a_len, b_len = rng.uniform(3.0, 5.5), rng.uniform(3.0, 5.0)
        off = 0.5 * w + 0.5 * b_len + 1.2
        cx, cy = mx + off * math.cos(side), my + off * math.sin(side)
        if not B.inside(cx, cy, 0.5 * max(a_len, b_len) + 1.0):
            continue
        if _chamber_clear(B, cx, cy, a_len, b_len, {a, b}):
            stub = B.add_node(mx, my)
            door = B.add_node(mx + (off - 0.5 * b_len + 0.3) * math.cos(side),
                              my + (off - 0.5 * b_len + 0.3) * math.sin(side))
            # the stub node splits nothing (it lies on the edge); it only anchors the door corridor
            B.edges.append((stub, door, min(w, 1.5)))
            B.adj[stub].append(door)
            B.adj[door].append(stub)
            B.chambers.append((cx, cy, b_len, a_len, side))
    return _raster(B, rng, res, "drift", seed, portal)


def _hops(B, a, b, limit=6):
    seen = {a: 0}
    frontier = [a]
    while frontier:
        nxt = []
        for n in frontier:
            for m in B.adj[n]:
                if m not in seen:
                    seen[m] = seen[n] + 1
                    if m == b:
                        return seen[m]
                    if seen[m] < limit:
                        nxt.append(m)
        frontier = nxt
    return limit + 1


def _chamber_clear(B, cx, cy, a_len, b_len, skip):
    r = 0.5 * math.hypot(a_len, b_len)
    for (a, b, w) in B.edges:
        if a in skip or b in skip:
            continue
        pa, pb = B.nodes[a], B.nodes[b]
        if float(_seg_dist(pa[0], pa[1], pb[0], pb[1], np.array(cx), np.array(cy))) < r + 0.5 * w + 1.5:
            return False
    for (x2, y2, a2, b2, _) in B.chambers:
        if math.hypot(cx - x2, cy - y2) < r + 0.5 * math.hypot(a2, b2) + 1.5:
            return False
    return True


def gen_lattice(seed, res=0.1, extent=(30.0, 80.0), width=(1.4, 2.5)):
    """Room-and-pillar: a rotated grid of corridors. HELD OUT - not used for training."""
    rng = np.random.RandomState(seed)
    W = rng.uniform(*extent)
    H = rng.uniform(*extent)
    B = _Builder(rng, W, H)
    sx, sy = rng.uniform(7.0, 11.0), rng.uniform(7.0, 11.0)
    th = math.radians(rng.uniform(-30.0, 30.0))
    cx0, cy0 = W / 2.0, H / 2.0
    nx = int((W * 0.8) / sx) + 3
    ny = int((H * 0.8) / sy) + 3
    ids = {}
    for i in range(-nx, nx + 1):
        for j in range(-ny, ny + 1):
            u, v = i * sx, j * sy
            x = cx0 + u * math.cos(th) - v * math.sin(th)
            y = cy0 + u * math.sin(th) + v * math.cos(th)
            if B.inside(x, y, 3.0):
                ids[(i, j)] = B.add_node(x, y)
    wline_x = dict((i, rng.uniform(*width)) for i in range(-nx, nx + 1))
    wline_y = dict((j, rng.uniform(*width)) for j in range(-ny, ny + 1))
    p_cut = rng.uniform(0.08, 0.22)
    for (i, j), n in ids.items():
        for (di, dj) in ((1, 0), (0, 1)):
            m = ids.get((i + di, j + dj))
            if m is None:
                continue
            if rng.rand() < p_cut:
                continue
            B.add_edge(n, m, wline_y[j] if di else wline_x[i])
    # keep the LARGEST connected piece of the lattice (cut segments can isolate corners) and
    # attach the portal to its westmost node
    comp, seen = set(), set()
    for n in ids.values():
        if n in seen:
            continue
        cc = _component(B, n)
        seen |= cc
        if len(cc) > len(comp):
            comp = cc
    west = min(comp, key=lambda n: B.nodes[n][0])
    B.edges = [e for e in B.edges if e[0] in comp and e[1] in comp]
    for n in list(B.adj):
        B.adj[n] = [m for m in B.adj[n] if m in comp]
    y = B.nodes[west][1]
    portal = B.add_node(1.0, y)
    B.add_edge(portal, west, rng.uniform(*width))
    # rooms: remove some pillars (cells between 4 nodes all present)
    p_room = rng.uniform(0.05, 0.14)
    for (i, j), n in ids.items():
        if n not in comp:
            continue
        quad = [ids.get((i + a, j + b)) for a, b in ((0, 0), (1, 0), (0, 1), (1, 1))]
        if any(q is None or q not in comp for q in quad):
            continue
        if rng.rand() < p_room:
            u, v = (i + 0.5) * sx, (j + 0.5) * sy
            x = cx0 + u * math.cos(th) - v * math.sin(th)
            y2 = cy0 + u * math.sin(th) + v * math.cos(th)
            B.chambers.append((x, y2, sx - 0.5, sy - 0.5, th))
    return _raster(B, rng, res, "lattice", seed, portal)


def _component(B, start):
    seen = set([start])
    st = [start]
    while st:
        n = st.pop()
        for m in B.adj[n]:
            if m not in seen:
                seen.add(m)
                st.append(m)
    return seen


def _raster(B, rng, res, style, seed, portal):
    W, H = B.W, B.H
    h, w = int(math.ceil(H / res)), int(math.ceil(W / res))
    ys = (np.arange(h) + 0.5) * res
    xs = (np.arange(w) + 0.5) * res
    free = np.zeros((h, w), dtype=bool)
    region = np.zeros((h, w), dtype=np.int8)
    noise = _smooth_noise(rng, h, w, max(2, int(1.5 / res)), 0.08)
    for (a, b, wd) in B.edges:
        pa, pb = B.nodes[a], B.nodes[b]
        r = 0.5 * wd + 0.3
        x0, x1 = min(pa[0], pb[0]) - r, max(pa[0], pb[0]) + r
        y0, y1 = min(pa[1], pb[1]) - r, max(pa[1], pb[1]) + r
        j0, j1 = max(0, int(x0 / res)), min(w, int(math.ceil(x1 / res)) + 1)
        i0, i1 = max(0, int(y0 / res)), min(h, int(math.ceil(y1 / res)) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        PX, PY = np.meshgrid(xs[j0:j1], ys[i0:i1])
        d = _seg_dist(pa[0], pa[1], pb[0], pb[1], PX, PY)
        m = d <= 0.5 * wd + noise[i0:i1, j0:j1]
        free[i0:i1, j0:j1] |= m
        region[i0:i1, j0:j1][m] = CORRIDOR
    for (cx, cy, ca, cb, th) in B.chambers:
        r = 0.5 * math.hypot(ca, cb) + 0.3
        j0, j1 = max(0, int((cx - r) / res)), min(w, int(math.ceil((cx + r) / res)) + 1)
        i0, i1 = max(0, int((cy - r) / res)), min(h, int(math.ceil((cy + r) / res)) + 1)
        if i0 >= i1 or j0 >= j1:
            continue
        PX, PY = np.meshgrid(xs[j0:j1], ys[i0:i1])
        u = (PX - cx) * math.cos(th) + (PY - cy) * math.sin(th)
        v = -(PX - cx) * math.sin(th) + (PY - cy) * math.cos(th)
        nz = noise[i0:i1, j0:j1] * 2.0
        if style == "drift":        # rounded room
            m = (u / (0.5 * ca + nz)) ** 2 + (v / (0.5 * cb + nz)) ** 2 <= 1.0
        else:
            m = (np.abs(u) <= 0.5 * ca + nz) & (np.abs(v) <= 0.5 * cb + nz)
        free[i0:i1, j0:j1] |= m
        region[i0:i1, j0:j1][m] = CHAMBER
        # boulders
        for _ in range(rng.randint(0, 3)):
            bu, bv = rng.uniform(-0.3, 0.3) * ca, rng.uniform(-0.3, 0.3) * cb
            br = rng.uniform(0.25, 0.5)
            bx = cx + bu * math.cos(th) - bv * math.sin(th)
            by = cy + bu * math.sin(th) + bv * math.cos(th)
            bm = (PX - bx) ** 2 + (PY - by) ** 2 <= br * br
            free[i0:i1, j0:j1] &= ~bm
    # dead-end zones: corridor cells within 4 m of a degree-1 node (not the portal)
    for n in range(len(B.nodes)):
        if n == portal or len(B.adj.get(n, [])) != 1:
            continue
        x, y = B.nodes[n]
        j0, j1 = max(0, int((x - 4) / res)), min(w, int(math.ceil((x + 4) / res)) + 1)
        i0, i1 = max(0, int((y - 4) / res)), min(h, int(math.ceil((y + 4) / res)) + 1)
        PX, PY = np.meshgrid(xs[j0:j1], ys[i0:i1])
        m = ((PX - x) ** 2 + (PY - y) ** 2 <= 16.0) & (region[i0:i1, j0:j1] == CORRIDOR) & free[i0:i1, j0:j1]
        region[i0:i1, j0:j1][m] = DEADEND
    region[~free] = ROCK
    # the portal is open to the west edge; home 1.5 m inside
    px, py = B.nodes[portal]
    mine = Mine()
    mine.style = style
    mine.seed = seed
    mine.res = res
    mine.free = free
    mine.region = region
    mine.W, mine.H = W, H
    first = B.adj[portal][0]
    fx, fy = B.nodes[first]
    yaw = math.atan2(fy - py, fx - px)
    mine.home = (px + 1.2 * math.cos(yaw), py + 1.2 * math.sin(yaw))
    mine.start_yaw = yaw
    mine.nodes = list(B.nodes)
    mine.edges = list(B.edges)
    mine.chambers = list(B.chambers)
    mine.portal = portal
    hi, hj = int(mine.home[1] / res), int(mine.home[0] / res)
    free[max(0, hi - 3):hi + 4, max(0, hj - 3):hj + 4] = True
    region[max(0, hi - 3):hi + 4, max(0, hj - 3):hj + 4] = CORRIDOR
    # drop free cells not connected to home (noise islands)
    mine.free = _connected(free, (hi, hj))
    mine.region[~mine.free] = ROCK
    return mine


def _connected(free, seed_cell):
    """4-connected component of free containing seed_cell (numpy flood fill by dilation, coarse
    to fine: done once per mine, so simplicity wins over speed)."""
    from ..geodesic import Graph, distances
    g = Graph(free)
    n = g.node(seed_cell)
    if n < 0:
        return free
    d = distances(g, [n])[0]
    out = np.zeros_like(free)
    ok = np.isfinite(d)
    out[g.rr[ok], g.cc[ok]] = True
    return out


def generate(style, seed, res=0.1, **kw):
    if style == "drift":
        return gen_drift(seed, res, **kw)
    if style == "lattice":
        return gen_lattice(seed, res, **kw)
    raise ValueError("unknown style %r" % style)
