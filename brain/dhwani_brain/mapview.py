# -*- coding: utf-8 -*-
"""The coarse planning grid the brain reasons on, built from a ROS OccupancyGrid.

WHY A SECOND GRID. gmapping publishes 5 cm cells over a map that is mostly unknown (1984 x 1984
on this rover). Decisions need topology, not 5 cm detail, and the wavefront cost scales with
cell count, so the map is cropped to what has actually been seen and pooled to ~0.2 m cells.

POOLING RULE, per coarse cell of f x f fine cells:
    occupied  if ANY fine cell is occupied (a wall is never averaged away)
    free      if not occupied and at least half the fine cells are known free
    unknown   otherwise
Then occupied cells are inflated by inflate_m (a disc, measured centre to cell edge), and the
executive's blocked regions (failed plans, fresh rubble) are painted in with the same margin.
`passable` = free, not inflated, not blocked: where the robot CENTRE may be.

Coordinates follow nav_msgs/OccupancyGrid: row 0 is the minimum-y edge (unlike a PGM file).
"""
from __future__ import division, print_function, absolute_import

import math

import numpy as np


def disc_offsets(radius_cells):
    """Integer offsets (di, dj) != (0, 0) inside a disc of the given radius (in cells)."""
    R = int(math.floor(radius_cells + 1e-9))
    out = []
    r2 = radius_cells * radius_cells + 1e-9
    for di in range(-R, R + 1):
        for dj in range(-R, R + 1):
            if (di or dj) and di * di + dj * dj <= r2:
                out.append((di, dj))
    return out


def dilate(mask, offsets):
    """Binary dilation by an explicit offset list (a few shifted ORs; no scipy on the rover)."""
    if not offsets:
        return mask.copy()
    R = max(max(abs(a), abs(b)) for a, b in offsets)
    h, w = mask.shape
    pad = np.zeros((h + 2 * R, w + 2 * R), dtype=bool)
    pad[R:R + h, R:R + w] = mask
    out = mask.copy()
    for di, dj in offsets:
        out |= pad[R + di:R + di + h, R + dj:R + dj + w]
    return out


def dilate4(mask):
    """Cells that are, or are 4-adjacent to, a set cell."""
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def integral(mask):
    """Summed-area table with a zero row/column in front: box sums in O(1)."""
    h, w = mask.shape
    s = np.zeros((h + 1, w + 1), dtype=np.int32)
    s[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0, dtype=np.int32), axis=1, dtype=np.int32)
    return s


def box_count(ii, i, j, r):
    """Number of set cells in the (2r+1)^2 box around (i, j), clipped to the grid.
    Vectorised over arrays i, j. Returns (count, cells_in_box)."""
    h = ii.shape[0] - 1
    w = ii.shape[1] - 1
    a0 = np.clip(i - r, 0, h)
    a1 = np.clip(i + r + 1, 0, h)
    b0 = np.clip(j - r, 0, w)
    b1 = np.clip(j + r + 1, 0, w)
    cnt = ii[a1, b1] - ii[a0, b1] - ii[a1, b0] + ii[a0, b0]
    area = (a1 - a0) * (b1 - b0)
    return cnt, area


class MapView(object):
    """Cropped, pooled, inflated view of an occupancy grid.

    Attributes: occ, free, unknown, infl (inflated obstacles), blocked, passable, near_blocked
    (all h x w bool), cell (m), x0/y0 (map-frame corner of coarse cell (0, 0)).
    """

    def __init__(self, grid, res, origin, cell_m=0.2, inflate_m=0.15, occ_thresh=65,
                 free_thresh=25, blocked=None, include=None, margin_cells=3):
        g = np.asarray(grid)
        if g.ndim != 2:
            raise ValueError("obs['grid'] must be 2-D (H x W), got shape %r" % (g.shape,))
        if g.dtype != np.int8:
            g = g.astype(np.int16)
        H, W = g.shape
        res = float(res)
        ox, oy = float(origin[0]), float(origin[1])
        f = max(1, int(round(float(cell_m) / res)))
        cell = f * res
        HC = (H + f - 1) // f
        WC = (W + f - 1) // f

        # --- crop to the known bounding box (plus robot, home, ... ) -------------------------
        known = g >= 0
        rows = np.flatnonzero(known.any(axis=1))
        box = None
        if rows.size:
            cols = np.flatnonzero(known[rows[0]:rows[-1] + 1].any(axis=0))
            box = [int(rows[0]) // f, int(rows[-1]) // f + 1, int(cols[0]) // f, int(cols[-1]) // f + 1]
        for p in include or ():
            ci = int(math.floor((float(p[1]) - oy) / cell))
            cj = int(math.floor((float(p[0]) - ox) / cell))
            if 0 <= ci < HC and 0 <= cj < WC:
                if box is None:
                    box = [ci, ci + 1, cj, cj + 1]
                else:
                    box = [min(box[0], ci), max(box[1], ci + 1), min(box[2], cj), max(box[3], cj + 1)]
        if box is None:
            box = [0, 1, 0, 1]
        i0 = max(0, box[0] - margin_cells)
        i1 = min(HC, box[1] + margin_cells)
        j0 = max(0, box[2] - margin_cells)
        j1 = min(WC, box[3] + margin_cells)
        h, w = i1 - i0, j1 - j0
        sub = g[i0 * f:i1 * f, j0 * f:j1 * f]
        if sub.shape != (h * f, w * f):          # the grid's last row/col block is partial
            pad = np.full((h * f, w * f), -1, dtype=sub.dtype)
            pad[:sub.shape[0], :sub.shape[1]] = sub
            sub = pad

        # --- pool ----------------------------------------------------------------------------
        # Viewed as uint8, unknown (-1) becomes 255, so "free" and "occupied" are one compare
        # each, and OR/adding the f x f sub-slices beats numpy reductions over a length-4
        # axis by ~3x on the Nano (32 ms -> 11.5 ms for a 60 m map at 0.05 m).
        if sub.dtype == np.int8:
            u = sub.view(np.uint8)
            freef = u <= free_thresh
            occf = (u - np.uint8(occ_thresh)) <= (100 - occ_thresh)
        else:
            occf = (sub >= occ_thresh) & (sub <= 100)
            freef = (sub >= 0) & (sub <= free_thresh)
        if f > 1:
            o4 = occf.reshape(h, f, w, f)
            f4 = freef.reshape(h, f, w, f)
            oc = o4[:, :, :, 0].copy()
            fc = f4[:, :, :, 0].astype(np.uint8)
            for k in range(1, f):
                oc |= o4[:, :, :, k]
                fc += f4[:, :, :, k]
            occ = oc[:, 0, :].copy()
            nfree = fc[:, 0, :].copy()
            for k in range(1, f):
                occ |= oc[:, k, :]
                nfree += fc[:, k, :]
            free = (~occ) & (2 * nfree.astype(np.int16) >= f * f)
        else:
            occ = occf
            free = freef & ~occf
        self.occ = occ
        self.free = free
        self.unknown = ~(occ | free)

        self.f = f
        self.res_in = res
        self.cell = cell
        self.i0, self.j0 = i0, j0
        self.shape = (h, w)
        self.x0 = ox + j0 * cell
        self.y0 = oy + i0 * cell
        self.inflate_m = float(inflate_m)

        # --- inflate + blocked regions ------------------------------------------------------
        self.infl = dilate(occ, disc_offsets((inflate_m + 0.5 * cell) / cell))
        self.blocked = np.zeros((h, w), dtype=bool)
        self.near_blocked = np.zeros((h, w), dtype=bool)
        self.blocked_list = []
        for b in blocked or ():
            try:
                bx, by, br = float(b[0]), float(b[1]), float(b[2])
            except (TypeError, IndexError, ValueError):
                continue
            self.blocked_list.append((bx, by, br))
            self._paint_disc(self.blocked, bx, by, br + inflate_m + 0.5 * cell)
            self._paint_disc(self.near_blocked, bx, by, br + inflate_m + 0.5 * cell + 0.4)
        self.passable = free & ~self.infl & ~self.blocked
        self._ii = {}

    # ---- geometry --------------------------------------------------------------------------
    def cell_of(self, x, y):
        return (int(math.floor((y - self.y0) / self.cell)), int(math.floor((x - self.x0) / self.cell)))

    def inside(self, i, j):
        return 0 <= i < self.shape[0] and 0 <= j < self.shape[1]

    def xy_of(self, i, j):
        """Centre of cell(s) (i, j) in the map frame. Works on scalars or arrays."""
        return (self.x0 + (np.asarray(j) + 0.5) * self.cell, self.y0 + (np.asarray(i) + 0.5) * self.cell)

    def _paint_disc(self, mask, x, y, r):
        i, j = self.cell_of(x, y)
        R = int(math.ceil(r / self.cell)) + 1
        a0, a1 = max(0, i - R), min(self.shape[0], i + R + 1)
        b0, b1 = max(0, j - R), min(self.shape[1], j + R + 1)
        if a0 >= a1 or b0 >= b1:
            return
        ii = np.arange(a0, a1)[:, None]
        jj = np.arange(b0, b1)[None, :]
        cx = self.x0 + (jj + 0.5) * self.cell
        cy = self.y0 + (ii + 0.5) * self.cell
        mask[a0:a1, b0:b1] |= (cx - x) ** 2 + (cy - y) ** 2 <= r * r

    def snap(self, x, y, radius_m, mask):
        """Nearest cell of `mask` to (x, y) within radius_m, or None."""
        i, j = self.cell_of(x, y)
        R = int(math.ceil(radius_m / self.cell))
        a0, a1 = max(0, i - R), min(self.shape[0], i + R + 1)
        b0, b1 = max(0, j - R), min(self.shape[1], j + R + 1)
        if a0 >= a1 or b0 >= b1:
            return None
        ii, jj = np.nonzero(mask[a0:a1, b0:b1])
        if ii.size == 0:
            return None
        cx = self.x0 + (jj + b0 + 0.5) * self.cell
        cy = self.y0 + (ii + a0 + 0.5) * self.cell
        d2 = (cx - x) ** 2 + (cy - y) ** 2
        k = int(np.argmin(d2))
        if d2[k] > radius_m * radius_m + 1e-9:
            return None
        return (int(ii[k] + a0), int(jj[k] + b0))

    def open_escape(self, x, y, radius_m):
        """Within radius_m of (x, y), let free cells inside the inflation band count as passable.

        Lidar keeps refining walls, so the band can close BEHIND the robot after it squeezed past a
        boulder: the planning grid then shows it sealed in a pocket it physically drove into.
        move_base's costmap lets a robot leave the inflation band at high cost; this is the same
        allowance, kept local so it never creates a shortcut elsewhere. Occupied cells and blocked
        regions (rubble, failed plans) stay closed."""
        i, j = self.cell_of(x, y)
        R = int(math.ceil(radius_m / self.cell))
        a0, a1 = max(0, i - R), min(self.shape[0], i + R + 1)
        b0, b1 = max(0, j - R), min(self.shape[1], j + R + 1)
        if a0 >= a1 or b0 >= b1:
            return
        ii = np.arange(a0, a1)[:, None]
        jj = np.arange(b0, b1)[None, :]
        near = (ii - i) ** 2 + (jj - j) ** 2 <= R * R
        ok = self.free[a0:a1, b0:b1] & ~self.occ[a0:a1, b0:b1] & ~self.blocked[a0:a1, b0:b1] & near
        self.passable[a0:a1, b0:b1] |= ok

    def force_passable(self, i, j):
        """Make (i, j) and its free 4-neighbours passable. Used for the robot's own cell: the robot
        IS there, even if the map says the cell is unknown or inside an inflation band."""
        if not self.inside(i, j) or self.blocked[i, j]:
            return False
        self.passable[i, j] = True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if self.inside(a, b) and (self.free[a, b] or self.unknown[a, b]) and not self.blocked[a, b] \
                    and not self.occ[a, b]:
                self.passable[a, b] = True
        return True

    def integral(self, name):
        """Cached summed-area table of one of the masks."""
        if name not in self._ii:
            self._ii[name] = integral(getattr(self, name))
        return self._ii[name]
