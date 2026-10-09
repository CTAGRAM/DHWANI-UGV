# -*- coding: utf-8 -*-
"""Geodesic distance on the planning grid: a vectorised NumPy wavefront.

WHAT IS COMPUTED. Shortest-path length through passable cells, 8-connected, orthogonal step 1
cell and diagonal step sqrt(2), no corner cutting (a diagonal step needs both orthogonal cells
free). That is the octile metric: within 8 % of the Euclidean geodesic, and never shorter than it,
which is the conservative side for a battery reserve.

HOW. A breadth-first wavefront by iterative dilation, generalised to two step lengths:
  round k takes every open cell whose distance lies in [k, k+1) and dilates that front by one
  step through a precomputed 8-neighbour table, keeping the smallest arrival per cell.
Because the shortest step is 1, nothing in [k, k+1) can still improve once round k starts
(Dial/Dinitz bucketing), so the result equals Dijkstra's exactly while every round is a handful
of whole-array NumPy operations on the front only.

WHY THE FRONT AND NOT THE WHOLE GRID. A dense dilation touches every cell every round, so its
cost is (rounds x cells); in a mine the front is a few corridor cross-sections, so working on it
costs (rounds x front) plus one pass to build the table. `distances_dense` keeps the literal
whole-grid version for the benchmark (bench/bench_wavefront.py measures both on the rover).

SEVERAL SOURCES, ONE LOOP. The robot, home and last-link fields share the loop: the graph is
stacked S times and the round count is set by the farthest field, not the sum of them. On the
Nano the per-round Python overhead dominates, so this nearly halves the cost of two fields.
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np

SQ2 = math.sqrt(2.0)
OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
STEP = np.array([1.0, 1.0, 1.0, 1.0, SQ2, SQ2, SQ2, SQ2], dtype=np.float32)


class Graph(object):
    """8-neighbour table over the passable cells of a boolean grid.

    nodes are numbered in row-major order; NB[n, k] is the neighbour of node n in direction k,
    or the sentinel N (one past the last node) when that move is not allowed.
    """

    def __init__(self, passable):
        P = np.asarray(passable, dtype=bool)
        H, W = P.shape
        rr, cc = np.nonzero(P)
        N = int(rr.size)
        self.shape = (H, W)
        self.N = N
        self.rr = rr.astype(np.int32)
        self.cc = cc.astype(np.int32)
        idx = np.full((H + 2, W + 2), N, dtype=np.int32)
        idx[rr + 1, cc + 1] = np.arange(N, dtype=np.int32)
        Pp = np.zeros((H + 2, W + 2), dtype=bool)
        Pp[1:-1, 1:-1] = P
        r1 = rr + 1
        c1 = cc + 1
        NB = np.empty((N + 1, 8), dtype=np.int32)
        for k, (dr, dc) in enumerate(OFFSETS):
            nb = idx[r1 + dr, c1 + dc]
            if dr and dc:
                ok = Pp[r1 + dr, c1] & Pp[r1, c1 + dc]
                nb = np.where(ok, nb, N)
            NB[:N, k] = nb
        NB[N, :] = N
        self.NB = NB
        self.idx = idx[1:-1, 1:-1]          # node index per cell, N where not passable

    def node(self, cell):
        """Node index of a (row, col) cell, or -1 if the cell is not passable / not given."""
        if cell is None:
            return -1
        i, j = cell
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return -1
        n = int(self.idx[i, j])
        return -1 if n >= self.N else n

    def image(self, d):
        """Scatter a per-node array (length N) into an H x W float32 image; inf elsewhere."""
        img = np.full(self.shape, np.inf, dtype=np.float32)
        if self.N:
            img[self.rr, self.cc] = d[:self.N]
        return img


def distances(graph, sources, max_dist=None):
    """Distance (in cells) from each source node to every node.

    sources: list with one entry per field: a node index (-1 = none, the row stays inf) or a
    sequence of node indices (a multi-source field: distance to the nearest of them).
    Returns float32 array of shape (len(sources), N). Unreachable = inf.
    max_dist (cells) stops the wavefront early; cells beyond it stay inf.

    Each round: take the open cells in the lowest unit bucket, offer dist + step to their
    8 neighbours, keep the smallest offer per cell (sort by cell, then value), store the
    improvements and put them on the open list. A cell may sit on the open list twice; both
    copies leave together because they carry the same value. Measured on the Nano (py2.7,
    60 m mine, 560 rounds): 137 us per round for two fields.
    """
    S = len(sources)
    N = graph.N
    M = N + 1                                 # per-source block: N nodes + 1 sentinel
    dist = np.full(S * M, np.inf, dtype=np.float32)
    dist[N::M] = -1.0                         # sentinel: can never be improved
    starts = []
    for s, src in enumerate(sources):
        for n in (src if hasattr(src, "__len__") else (src,)):
            if n is not None and 0 <= int(n) < N:
                starts.append(s * M + int(n))
    if not starts or N == 0:
        return dist.reshape(S, M)[:, :N].copy()
    if S == 1:
        NB = graph.NB
    else:                                      # stack the table so no index arithmetic per round
        NB = np.empty((S * M, 8), dtype=np.int32)
        for s in range(S):
            NB[s * M:(s + 1) * M] = graph.NB + s * M
    open_ = np.unique(np.array(starts, dtype=np.int64))
    dist[open_] = 0.0
    limit = np.inf if max_dist is None else float(max_dist)
    while open_.size:
        d = dist[open_]
        k = math.floor(float(d.min()))
        if k >= limit:
            break
        sel = d < k + 1.0
        F = open_[sel]
        dF = d[sel]
        open_ = open_[~sel]
        nb = NB[F].ravel()
        cand = (dF[:, None] + STEP).ravel()
        better = cand < dist[nb]
        nb = nb[better]
        if not nb.size:
            continue
        cand = cand[better]
        if nb.size > 1:
            o = np.lexsort((cand, nb))       # group by target, smallest arrival first
            nb = nb[o]
            cand = cand[o]
            keep = np.empty(nb.size, dtype=bool)
            keep[0] = True
            np.not_equal(nb[1:], nb[:-1], out=keep[1:])
            nb = nb[keep]
            cand = cand[keep]
        dist[nb] = cand
        open_ = np.concatenate((open_, nb))
    return dist.reshape(S, M)[:, :N].copy()


def descend(graph, d, node, max_steps=100000):
    """Shortest path from `node` back to the source of field d (per-node, length N), following
    the steepest descent; exact on an exact field. Returns the list of node indices, source last."""
    if node < 0 or not np.isfinite(d[node]):
        return []
    N = graph.N
    ext = np.empty(N + 1, dtype=np.float64)
    ext[:N] = d
    ext[N] = np.inf
    NB = graph.NB
    path = [int(node)]
    cur = int(node)
    for _ in range(max_steps):
        if ext[cur] <= 0.0:
            break
        nbs = NB[cur]
        c = ext[nbs] + STEP
        j = int(np.argmin(c))
        nxt = int(nbs[j])
        if nxt >= N or ext[nxt] >= ext[cur]:
            break
        cur = nxt
        path.append(cur)
    return path


def distances_dense(passable, src_cells, max_rounds=100000):
    """Reference: the same bucketed octile wavefront, but dilating on the DENSE grid each round.

    Identical output to distances() for one source; kept for the benchmark and the tests. Its
    cost is rounds x all cells, which is what the front-only version avoids.
    """
    P = np.asarray(passable, dtype=bool)
    H, W = P.shape
    out = []
    for src in src_cells:
        dist = np.full((H + 2, W + 2), np.inf, dtype=np.float32)
        if src is None or not P[src[0], src[1]]:
            out.append(dist[1:-1, 1:-1].copy())
            continue
        Pp = np.zeros((H + 2, W + 2), dtype=bool)
        Pp[1:-1, 1:-1] = P
        dist[src[0] + 1, src[1] + 1] = 0.0
        done = np.zeros_like(Pp)
        k = 0
        for _ in range(max_rounds):
            front = (~done) & (dist < k + 1.0)
            if not front.any():
                rest = dist[~done & np.isfinite(dist)]
                if rest.size == 0:
                    break
                k = int(math.floor(float(rest.min())))
                continue
            done |= front
            fd = np.where(front, dist, np.inf)
            for kk, (dr, dc) in enumerate(OFFSETS):
                # every cell (r, c) of the front offers dist + step to (r + dr, c + dc)
                sr = slice(max(0, -dr), H + 2 - max(0, dr))      # source rows
                sc = slice(max(0, -dc), W + 2 - max(0, dc))      # source cols
                tr = slice(max(0, dr), H + 2 - max(0, -dr))      # target rows
                tc = slice(max(0, dc), W + 2 - max(0, -dc))      # target cols
                ok = Pp[tr, tc]
                if dr and dc:                                    # no corner cutting
                    ok = ok & Pp[tr, sc] & Pp[sr, tc]
                tgt = dist[tr, tc]
                np.minimum(tgt, np.where(ok, fd[sr, sc] + STEP[kk], np.inf), out=tgt)
            k += 1
        out.append(dist[1:-1, 1:-1].copy())
    return out
