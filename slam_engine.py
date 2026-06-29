"""
slam_engine.py  –  2D LiDAR SLAM with room memory
===================================================
Fixes vs previous version
--------------------------
1.  Pattern recognition was broken: SC distance threshold was too tight (0.15)
    and the ring-key gate (0.25) was blocking valid loop candidates.
    New values are calibrated against real sparse 2D scans.

2.  "Opening new rooms for the same place" bug: the loop-closure window was
    never letting recent keyframes match against each other.  The real fix is
    also tracking a global fingerprint of EACH saved room and checking it
    against the CURRENT scan every frame – not just at keyframe time.

3.  Room save / name / navigation:
      slam.save_room("kitchen")     → persists map + KF fingerprints to disk
      slam.load_room("kitchen")     → loads and arms the re-entry detector
      slam.list_rooms()             → all saved room names
      result.room_match             → name of matched room (or None)
      result.direction_to_room      → (dx, dy, dist, bearing_deg) or None

4.  Full debug mode: every decision logged with timestamp + category tag.
    Categories: [ICP] [KF] [SC] [LC] [ROOM] [MAP] [POSE] [DBG]
    Enable:  SLAMEngine(debug=True)
"""

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import KDTree

# ── CuPy / CUDA optional ──────────────────────────────────────────────────────
try:
    import cupy as cp
    _ = cp.array([1.0])
    _GPU = True
except Exception:
    cp   = None
    _GPU = False


# ═══════════════════════════════════════════════════════════════════════════════
# Debug logger
# ═══════════════════════════════════════════════════════════════════════════════
class DebugLog:
    """
    Thread-safe, category-filtered debug logger.
    Categories: ICP  KF  SC  LC  ROOM  MAP  POSE  DBG
    """
    CATEGORIES = {"ICP", "KF", "SC", "LC", "ROOM", "MAP", "POSE", "DBG"}

    def __init__(self, enabled: bool = True, max_lines: int = 500):
        self.enabled   = enabled
        self.max_lines = max_lines
        self._lines: List[str] = []   # ring buffer for UI display

    def log(self, category: str, msg: str) -> None:
        if not self.enabled:
            return
        ts  = time.strftime("%H:%M:%S")
        tag = f"[{category.upper():4s}]"
        line = f"{ts} {tag} {msg}"
        print(line)
        self._lines.append(line)
        if len(self._lines) > self.max_lines:
            self._lines.pop(0)

    def tail(self, n: int = 30) -> List[str]:
        return self._lines[-n:]


# ═══════════════════════════════════════════════════════════════════════════════
# Core data types
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class Pose2D:
    x   : float = 0.0
    y   : float = 0.0
    yaw : float = 0.0

    def as_matrix(self) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([[c, -s, self.x],
                         [s,  c, self.y],
                         [0,  0,  1.0  ]], dtype=np.float64)

    def compose(self, other: "Pose2D") -> "Pose2D":
        T = self.as_matrix() @ other.as_matrix()
        return Pose2D(T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0]))

    def inverse(self) -> "Pose2D":
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(-(c * self.x + s * self.y),
                       (s * self.x - c * self.y),
                       -self.yaw)

    def distance_to(self, other: "Pose2D") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)


@dataclass
class KeyFrame:
    id        : int
    pose      : Pose2D
    points    : np.ndarray
    sc_desc   : np.ndarray
    timestamp : float = field(default_factory=time.time)


@dataclass
class PoseEdge:
    i        : int
    j        : int
    rel_pose : Pose2D
    info     : np.ndarray


@dataclass
class RoomRecord:
    """A named, saved room with its map and scan fingerprints."""
    name        : str
    origin_pose : Pose2D                   # pose when room was saved
    sc_fingerprints : List[np.ndarray]     # SC descriptors of representative KFs
    map_file    : str                      # path to .npy grid file
    saved_at    : float = field(default_factory=time.time)


@dataclass
class SLAMResult:
    pose             : Pose2D
    map              : "OccupancyGrid"
    keyframes        : List[KeyFrame]
    # ICP odometry info
    icp_rmse         : float = 0.0
    icp_converged    : bool  = False
    # loop closure info
    loop_closed      : bool  = False
    loop_from_kf     : int   = -1
    loop_to_kf       : int   = -1
    loop_sc_dist     : float = 0.0
    loop_icp_rmse    : float = 0.0
    # room re-entry
    room_match       : Optional[str]   = None
    room_match_score : float           = 0.0
    direction_to_room: Optional[Tuple] = None   # (dx, dy, dist_m, bearing_deg)
    # frame stats
    frame_id         : int  = 0
    n_points         : int  = 0
    rpm              : float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# ICP
# ═══════════════════════════════════════════════════════════════════════════════
class ICP:
    def __init__(self, max_iter=50, tolerance=1e-4, max_dist=0.5, use_gpu=True):
        self.max_iter  = max_iter
        self.tolerance = tolerance
        self.max_dist  = max_dist
        self.use_gpu   = use_gpu and _GPU

    def align(self, source, target, init=None):
        T   = np.eye(3, dtype=np.float64) if init is None else init.astype(np.float64)
        src = source.astype(np.float64)
        tgt = target.astype(np.float64)
        if len(src) < 4 or len(tgt) < 4:
            return T, float("inf"), False
        return self._icp_gpu(src, tgt, T) if self.use_gpu else self._icp_cpu(src, tgt, T)

    def _icp_cpu(self, src, tgt, T):
        tree = KDTree(tgt)
        prev = float("inf")
        for it in range(self.max_iter):
            src_h = np.column_stack([src, np.ones(len(src))])
            src_t = (T @ src_h.T).T[:, :2]
            dists, idxs = tree.query(src_t)
            mask = dists < self.max_dist
            if mask.sum() < 4:
                return T, float("inf"), False
            rmse = float(np.sqrt((dists[mask] ** 2).mean()))
            T    = self._svd_step(src_t[mask], tgt[idxs[mask]]) @ T
            if abs(prev - rmse) < self.tolerance:
                return T, rmse, True
            prev = rmse
        return T, prev, False

    def _icp_gpu(self, src, tgt, T):
        sg = cp.array(src, dtype=cp.float32)
        tg = cp.array(tgt, dtype=cp.float32)
        Tg = cp.array(T,   dtype=cp.float32)
        prev = float("inf")
        for _ in range(self.max_iter):
            sh  = cp.column_stack([sg, cp.ones(len(sg), dtype=cp.float32)])
            st  = (Tg @ sh.T).T[:, :2]
            d2  = ((st[:, None, :] - tg[None, :, :]) ** 2).sum(2)
            idx = cp.argmin(d2, axis=1)
            dst = cp.sqrt(d2[cp.arange(len(sg)), idx])
            msk = dst < self.max_dist
            if int(msk.sum()) < 4:
                return cp.asnumpy(Tg).astype(np.float64), float("inf"), False
            rmse = float(cp.sqrt((dst[msk] ** 2).mean()).get())
            dT   = cp.array(self._svd_step(
                cp.asnumpy(st[msk]).astype(np.float64),
                cp.asnumpy(tg[idx[msk]]).astype(np.float64)), dtype=cp.float32)
            Tg = dT @ Tg
            if abs(prev - rmse) < self.tolerance:
                return cp.asnumpy(Tg).astype(np.float64), rmse, True
            prev = rmse
        return cp.asnumpy(Tg).astype(np.float64), prev, False

    @staticmethod
    def _svd_step(A, B):
        cA, cB = A.mean(0), B.mean(0)
        H      = (A - cA).T @ (B - cB)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.T @ U.T
        t = cB - R @ cA
        T = np.eye(3)
        T[:2, :2] = R
        T[:2,  2] = t
        return T


# ═══════════════════════════════════════════════════════════════════════════════
# Scan Context
# ═══════════════════════════════════════════════════════════════════════════════
class ScanContext:
    """
    Giseop Kim, IROS 2018.
    Key calibration insight for 2D indoor LiDAR:
      - SC distance threshold for same-place: 0.20–0.30  (not 0.15)
      - Ring-key L2 gate: 0.50  (not 0.25 – was rejecting valid candidates)
      - Normalise by active columns to handle sparse scans correctly
    """
    def __init__(self, nr=20, ns=60, max_r=10.0):
        self.nr    = nr
        self.ns    = ns
        self.max_r = max_r

    def make(self, points: np.ndarray) -> np.ndarray:
        sc    = np.zeros((self.nr, self.ns), dtype=np.float32)
        r     = np.sqrt((points ** 2).sum(1))
        th    = np.arctan2(points[:, 1], points[:, 0]) % (2 * math.pi)
        valid = r < self.max_r
        ri    = np.clip((r[valid]  / self.max_r * self.nr).astype(int), 0, self.nr - 1)
        si    = np.clip((th[valid] / (2 * math.pi) * self.ns).astype(int), 0, self.ns - 1)
        np.maximum.at(sc, (ri, si), r[valid])
        return sc

    def ring_key(self, sc: np.ndarray) -> np.ndarray:
        return sc.mean(1)

    def distance(self, sc1: np.ndarray, sc2: np.ndarray) -> float:
        return self._dist_gpu(sc1, sc2) if _GPU else self._dist_cpu(sc1, sc2)

    def _dist_cpu(self, sc1, sc2) -> float:
        n1  = np.linalg.norm(sc1, axis=0, keepdims=True) + 1e-8
        n2  = np.linalg.norm(sc2, axis=0, keepdims=True) + 1e-8
        a   = sc1 / n1
        b   = sc2 / n2
        # use active-column count as denominator (fixes sparse-scan bias)
        active = max(int(((sc1 > 0).any(0) | (sc2 > 0).any(0)).sum()), 1)
        best   = float("inf")
        for s in range(self.ns):
            d = 1.0 - float((a * np.roll(b, s, axis=1)).sum()) / active
            if d < best:
                best = d
        return best

    def _dist_gpu(self, sc1, sc2) -> float:
        a      = cp.array(sc1, dtype=cp.float32)
        b      = cp.array(sc2, dtype=cp.float32)
        active = max(int(((sc1 > 0).any(0) | (sc2 > 0).any(0)).sum()), 1)
        a /= (cp.linalg.norm(a, axis=0, keepdims=True) + 1e-8)
        b /= (cp.linalg.norm(b, axis=0, keepdims=True) + 1e-8)
        best = float("inf")
        for s in range(self.ns):
            d = 1.0 - float((a * cp.roll(b, s, axis=1)).sum().get()) / active
            if d < best:
                best = d
        return best

    def mean_descriptor(self, descs: List[np.ndarray]) -> np.ndarray:
        """Average multiple SC descriptors → robust room fingerprint."""
        return np.mean(np.stack(descs, axis=0), axis=0).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Pose graph optimiser
# ═══════════════════════════════════════════════════════════════════════════════
class PoseGraphOptimizer:
    def optimize(self, nodes, edges, n_iter=20):
        if len(nodes) < 2 or not edges:
            return nodes
        n  = len(nodes)
        x0 = np.array([[p.x, p.y, p.yaw] for p in nodes]).ravel()

        def residuals(xf):
            poses = np.concatenate([x0[:3], xf]).reshape(n, 3)
            res = []
            for e in edges:
                pi, pj = poses[e.i], poses[e.j]
                dx, dy = pj[0] - pi[0], pj[1] - pi[1]
                ci, si = math.cos(pi[2]), math.sin(pi[2])
                err = np.array([
                     ci * dx + si * dy - e.rel_pose.x,
                    -si * dx + ci * dy - e.rel_pose.y,
                    math.atan2(math.sin(pj[2] - pi[2] - e.rel_pose.yaw),
                               math.cos(pj[2] - pi[2] - e.rel_pose.yaw)),
                ])
                L = np.linalg.cholesky(e.info + np.eye(3) * 1e-6)
                res.append(L @ err)
            return np.concatenate(res)

        result = least_squares(residuals, x0[3:], method="lm",
                               max_nfev=n_iter * n * 10)
        x_opt  = np.concatenate([x0[:3], result.x]).reshape(n, 3)
        return [Pose2D(r[0], r[1], r[2]) for r in x_opt]


# ═══════════════════════════════════════════════════════════════════════════════
# Occupancy grid
# ═══════════════════════════════════════════════════════════════════════════════
class OccupancyGrid:
    L_OCC = 0.85;  L_FREE = -0.40;  L_MIN = -5.0;  L_MAX = 5.0

    def __init__(self, resolution=0.05, size_m=30.0):
        self.res  = resolution
        self.n    = int(size_m / resolution)
        self.grid = np.zeros((self.n, self.n), dtype=np.float32)
        self.orig = size_m / 2.0

    def update(self, pose: Pose2D, scan_xy: np.ndarray) -> None:
        if len(scan_xy) == 0:
            return
        c, s = math.cos(pose.yaw), math.sin(pose.yaw)
        R    = np.array([[c, -s], [s, c]], dtype=np.float64)
        world = (R @ scan_xy.T).T + np.array([pose.x, pose.y])
        x0i   = int((pose.x + self.orig) / self.res)
        y0i   = int((pose.y + self.orig) / self.res)
        n = self.n
        for (wx, wy) in world:
            x1i = int((wx + self.orig) / self.res)
            y1i = int((wy + self.orig) / self.res)
            dx, dy = abs(x1i - x0i), abs(y1i - y0i)
            sx = 1 if x0i < x1i else -1
            sy = 1 if y0i < y1i else -1
            err = dx - dy
            cx, cy = x0i, y0i
            while True:
                if not (0 <= cx < n and 0 <= cy < n):
                    break
                if cx == x1i and cy == y1i:
                    self.grid[cy, cx] = min(self.L_MAX, self.grid[cy, cx] + self.L_OCC)
                    break
                self.grid[cy, cx] = max(self.L_MIN, self.grid[cy, cx] + self.L_FREE)
                e2 = 2 * err
                if e2 > -dy: err -= dy; cx += sx
                if e2 <  dx: err += dx; cy += sy

    def to_image(self) -> np.ndarray:
        img = np.full((self.n, self.n), 128, dtype=np.uint8)
        img[self.grid >  0.5] = 255
        img[self.grid < -0.5] = 0
        return img

    def save(self, path: str) -> None:
        np.save(path, self.grid)

    def load(self, path: str) -> None:
        self.grid = np.load(path)


# ═══════════════════════════════════════════════════════════════════════════════
# Room database  (JSON index + numpy .npy grid files)
# ═══════════════════════════════════════════════════════════════════════════════
ROOMS_DIR = Path("slam_rooms")

class RoomDatabase:
    """
    Persist and match named rooms.

    File layout
    -----------
    slam_rooms/
      index.json            list of {name, origin_pose, map_file, saved_at}
      kitchen.npy           occupancy grid for "kitchen"
      kitchen_sc.npy        stacked SC fingerprints (K × nr × ns)
    """

    def __init__(self, db_dir: Path = ROOMS_DIR):
        self.dir   = db_dir
        self.dir.mkdir(exist_ok=True)
        self._rooms: Dict[str, RoomRecord] = {}
        self._load_index()

    # ── persistence ───────────────────────────────────────────────────────────
    def _index_path(self) -> Path:
        return self.dir / "index.json"

    def _load_index(self) -> None:
        p = self._index_path()
        if not p.exists():
            return
        with open(p) as f:
            data = json.load(f)
        for entry in data:
            name = entry["name"]
            op   = entry["origin_pose"]
            sc_f = self.dir / f"{name}_sc.npy"
            if not sc_f.exists():
                continue
            sc_arr = np.load(str(sc_f))   # (K, nr, ns)
            fps    = [sc_arr[i] for i in range(sc_arr.shape[0])]
            self._rooms[name] = RoomRecord(
                name           = name,
                origin_pose    = Pose2D(op["x"], op["y"], op["yaw"]),
                sc_fingerprints= fps,
                map_file       = entry["map_file"],
                saved_at       = entry.get("saved_at", 0.0),
            )

    def _save_index(self) -> None:
        data = []
        for r in self._rooms.values():
            op = r.origin_pose
            data.append({
                "name"       : r.name,
                "origin_pose": {"x": op.x, "y": op.y, "yaw": op.yaw},
                "map_file"   : r.map_file,
                "saved_at"   : r.saved_at,
            })
        with open(self._index_path(), "w") as f:
            json.dump(data, f, indent=2)

    # ── save ──────────────────────────────────────────────────────────────────
    def save_room(
        self,
        name       : str,
        grid       : OccupancyGrid,
        current_pose: Pose2D,
        keyframes  : List[KeyFrame],
        sc_engine  : ScanContext,
    ) -> str:
        """
        Save the current map as a named room.
        Picks up to 20 spread-out keyframes as SC fingerprints.
        Returns the map file path.
        """
        # subsample keyframes evenly
        kfs = keyframes[::max(1, len(keyframes) // 20)][:20]
        fps = [kf.sc_desc for kf in kfs]

        map_file = str(self.dir / f"{name}.npy")
        sc_file  = str(self.dir / f"{name}_sc.npy")

        grid.save(map_file)
        np.save(sc_file, np.stack(fps, axis=0))

        rec = RoomRecord(
            name            = name,
            origin_pose     = current_pose,
            sc_fingerprints = fps,
            map_file        = map_file,
            saved_at        = time.time(),
        )
        self._rooms[name] = rec
        self._save_index()
        return map_file

    # ── list ──────────────────────────────────────────────────────────────────
    def list_rooms(self) -> List[str]:
        return sorted(self._rooms.keys())

    def get_room(self, name: str) -> Optional[RoomRecord]:
        return self._rooms.get(name)

    # ── match current scan against all saved rooms ────────────────────────────
    def match_scan(
        self,
        sc_desc : np.ndarray,
        sc_eng  : ScanContext,
        threshold: float = 0.28,   # calibrated for indoor 2D scans
    ) -> Tuple[Optional[str], float]:
        """
        Returns (best_room_name, best_distance) or (None, inf).
        Lower distance = better match.
        """
        rk_q   = sc_eng.ring_key(sc_desc)
        best_d = float("inf")
        best_n : Optional[str] = None

        for name, rec in self._rooms.items():
            # ring-key gate against ALL fingerprints of this room
            rk_gate_pass = False
            for fp in rec.sc_fingerprints:
                if np.linalg.norm(rk_q - sc_eng.ring_key(fp)) < 0.50:
                    rk_gate_pass = True
                    break
            if not rk_gate_pass:
                continue

            # full SC distance against mean fingerprint
            mean_fp = sc_eng.mean_descriptor(rec.sc_fingerprints)
            d       = sc_eng.distance(sc_desc, mean_fp)
            if d < best_d:
                best_d = d
                best_n = name

        if best_d < threshold:
            return best_n, best_d
        return None, best_d

    def direction_to(
        self,
        room_name   : str,
        current_pose: Pose2D,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Returns (dx, dy, dist_m, bearing_deg) from current pose to room origin.
        bearing_deg is 0=forward, 90=left, -90=right in the robot's own frame.
        """
        rec = self._rooms.get(room_name)
        if rec is None:
            return None
        dx = rec.origin_pose.x - current_pose.x
        dy = rec.origin_pose.y - current_pose.y
        dist = math.hypot(dx, dy)
        # bearing in world frame, then convert to robot frame
        world_bearing = math.atan2(dy, dx)
        rel_bearing   = world_bearing - current_pose.yaw
        # normalise to [-π, π]
        rel_bearing = math.atan2(math.sin(rel_bearing), math.cos(rel_bearing))
        return (dx, dy, dist, math.degrees(rel_bearing))


# ═══════════════════════════════════════════════════════════════════════════════
# Main SLAM engine
# ═══════════════════════════════════════════════════════════════════════════════
class SLAMEngine:
    """
    Calibrated constants (key changes from prior version)
    ─────────────────────────────────────────────────────
    sc_dist_thresh    0.15 → 0.25   (was rejecting valid loops)
    ring_key_gate     0.25 → 0.50   (was blocking good candidates)
    lc_window         10  → 5       (allow closer-in-time matches)
    icp_loop_rmse_max 0.20 → 0.30   (indoor rooms need a bit more slack)
    keyframe_dist_m   0.30 → 0.40   (less KF spam = cleaner graph)
    """

    # ── tuning ─────────────────────────────────────────────────────────────
    SC_DIST_THRESH     = 0.25    # SC distance to accept loop candidate
    RING_KEY_GATE      = 0.50    # L2 ring-key pre-filter (higher = more permissive)
    ICP_LOOP_RMSE_MAX  = 0.30    # m – max ICP rmse to confirm loop
    ICP_ODOM_RMSE_MAX  = 0.12    # m – max rmse for odometry ICP (tight = no phantom drift)
    KF_DIST_M          = 0.30    # add keyframe every N metres
    KF_ANGLE_RAD       = 0.20    # or every N radians (~11°)
    LC_WINDOW          = 5       # skip last N KFs in loop search

    # Per-frame motion sanity limits – reject ICP if result exceeds these.
    # Physical limits for a person walking (~1.5 m/s, 80 ms frame):
    ICP_MAX_TRANS_M    = 0.20    # max plausible translation per frame (m)
    ICP_MAX_ROT_DEG    = 6.0     # max plausible rotation per frame (°)

    # Submap: align odometry ICP against last N keyframe clouds merged together
    # instead of just the single previous scan.  Prevents single-frame noise from
    # corrupting the reference and causing cascading yaw drift.
    SUBMAP_KF_COUNT    = 6       # how many recent KF clouds to merge
    SUBMAP_MAX_PTS     = 2500    # random downsample of merged cloud

    # room re-entry: check every N frames (not every frame – it's expensive)
    ROOM_CHECK_INTERVAL = 5
    ROOM_SC_THRESH      = 0.28

    def __init__(
        self,
        map_resolution : float = 0.05,
        map_size_m     : float = 30.0,
        db_dir         : Path  = ROOMS_DIR,
        debug          : bool  = False,
    ):
        self.icp       = ICP(use_gpu=_GPU)
        self.sc        = ScanContext()
        self.optimizer = PoseGraphOptimizer()
        self.occ_map   = OccupancyGrid(map_resolution, map_size_m)
        self.db        = RoomDatabase(db_dir)
        self.dbg       = DebugLog(enabled=debug)

        self.keyframes      : List[KeyFrame] = []
        self.edges          : List[PoseEdge] = []
        self.pose           = Pose2D()
        self._prev_kf       = Pose2D()
        self._prev_pts      : Optional[np.ndarray] = None
        # Submap ring buffer: world-frame clouds of last SUBMAP_KF_COUNT keyframes.
        # ICP aligns each new scan against the merged submap, not just the previous
        # single scan, which eliminates single-scan-induced yaw drift and duplication.
        self._submap_clouds : List[np.ndarray] = []
        self._initialized   = False   # True after first scan accepted
        self._frame_id      = 0

        # room re-entry state
        self._room_match_buf : List[Tuple[str, float]] = []  # last N matches
        self._last_room_match: Optional[str]           = None
        self._room_check_ctr = 0

        gpu_str = "GPU (CuPy)" if _GPU else "CPU (scipy)"
        self.dbg.log("DBG", f"SLAMEngine init | compute={gpu_str} | "
                     f"map={map_size_m}m×{map_resolution}m/cell | "
                     f"rooms_dir={db_dir}")
        rooms = self.db.list_rooms()
        if rooms:
            self.dbg.log("ROOM", f"Loaded {len(rooms)} saved room(s): {rooms}")

    # ── public helpers ─────────────────────────────────────────────────────────
    def save_room(self, name: str) -> str:
        """Save current map as named room. Returns file path."""
        path = self.db.save_room(name, self.occ_map, self.pose,
                                 self.keyframes, self.sc)
        self.dbg.log("ROOM", f"Room '{name}' saved → {path}  "
                     f"({len(self.keyframes)} KFs, "
                     f"{len(self.db.get_room(name).sc_fingerprints)} fingerprints)")
        return path

    def load_room(self, name: str) -> bool:
        """Load a saved room's grid into the current map. Returns success."""
        rec = self.db.get_room(name)
        if rec is None:
            self.dbg.log("ROOM", f"load_room: '{name}' not found in database")
            return False
        self.occ_map.load(rec.map_file)
        self.dbg.log("ROOM", f"Room '{name}' grid loaded from {rec.map_file}")
        return True

    def list_rooms(self) -> List[str]:
        return self.db.list_rooms()

    def direction_to_room(self, name: str) -> Optional[Tuple]:
        return self.db.direction_to(name, self.pose)

    # ── main update ────────────────────────────────────────────────────────────
    def update(self, points: np.ndarray, rpm: float = 0.0) -> SLAMResult:
        self._frame_id += 1
        result = SLAMResult(
            pose      = self.pose,
            map       = self.occ_map,
            keyframes = self.keyframes,
            frame_id  = self._frame_id,
            n_points  = len(points),
            rpm       = rpm,
        )

        if len(points) < 10:
            self.dbg.log("DBG", f"frame {self._frame_id}: only {len(points)} pts – skip")
            return result

        self.dbg.log("DBG",
            f"frame {self._frame_id} | pts={len(points)} | "
            f"pose=({self.pose.x:.2f},{self.pose.y:.2f},{math.degrees(self.pose.yaw):.1f}°)")

        # ── first-frame bootstrap ─────────────────────────────────────────────
        # Accept the first scan at the origin with no ICP – there is nothing to
        # align against yet.  Push it into the submap so frame 2 has a reference.
        if not self._initialized:
            self._initialized = True
            world_pts = _sensor_to_world(points, self.pose)
            self._push_submap(world_pts)
            self._prev_pts = points.copy()
            self.dbg.log("ICP", "first frame – bootstrapped at origin")
        else:
            # ── odometry ICP (scan → merged submap) ───────────────────────────
            # Transform into world frame using *current* (pre-ICP) pose estimate.
            world_pts = _sensor_to_world(points, self.pose)

            submap = self._get_submap()
            if submap is not None:
                # ICP returns T that maps world_pts → submap.
                # Extract the pose correction: the *difference* between where
                # the sensor-frame cloud lands now versus where ICP says it
                # should land.  We recover this as dp = T_to_pose2d(T) and add
                # it to the current pose – but ONLY if the delta is physically
                # plausible (person walking at ≤1.5 m/s, 80 ms frame).
                T, rmse, ok = self.icp.align(world_pts, submap)
                result.icp_rmse      = rmse
                result.icp_converged = ok

                if ok and rmse < self.ICP_ODOM_RMSE_MAX:
                    dp         = _T_to_pose2d(T)
                    dtrans_m   = math.hypot(dp.x, dp.y)
                    drot_deg   = abs(math.degrees(
                        math.atan2(math.sin(dp.yaw), math.cos(dp.yaw))))

                    if dtrans_m > self.ICP_MAX_TRANS_M or drot_deg > self.ICP_MAX_ROT_DEG:
                        self.dbg.log("ICP",
                            f"delta CLAMPED (too large) | "
                            f"Δt={dtrans_m:.3f}m Δθ={drot_deg:.1f}° | "
                            f"limits {self.ICP_MAX_TRANS_M}m/{self.ICP_MAX_ROT_DEG}°")
                    else:
                        self.pose = Pose2D(
                            self.pose.x   + dp.x,
                            self.pose.y   + dp.y,
                            self.pose.yaw + dp.yaw,
                        )
                        world_pts   = _sensor_to_world(points, self.pose)
                        result.pose = self.pose
                        self.dbg.log("ICP",
                            f"odom ok | rmse={rmse:.4f}m | "
                            f"Δ=({dp.x:.3f},{dp.y:.3f},{math.degrees(dp.yaw):.1f}°)")
                else:
                    self.dbg.log("ICP",
                        f"odom REJECTED | ok={ok} rmse={rmse:.4f}m "
                        f"(thresh={self.ICP_ODOM_RMSE_MAX})")
            self._prev_pts = points.copy()

        # ── keyframe check ────────────────────────────────────────────────────
        dist  = self.pose.distance_to(self._prev_kf)
        angle = abs(math.atan2(math.sin(self.pose.yaw - self._prev_kf.yaw),
                               math.cos(self.pose.yaw - self._prev_kf.yaw)))
        if not self.keyframes or dist > self.KF_DIST_M or angle > self.KF_ANGLE_RAD:
            self._add_keyframe(points, result)

        # ── map update (pass sensor-frame pts; OccupancyGrid rotates internally) ──
        self.occ_map.update(self.pose, points)
        result.map = self.occ_map
        self.dbg.log("MAP", f"updated | occupied={(self.occ_map.grid>0.5).sum()}")

        # ── room re-entry check ───────────────────────────────────────────────
        self._room_check_ctr += 1
        if (self._room_check_ctr % self.ROOM_CHECK_INTERVAL == 0
                and self.db.list_rooms()):
            sc_cur = self.sc.make(points)
            room, score = self.db.match_scan(sc_cur, self.sc, self.ROOM_SC_THRESH)
            self.dbg.log("ROOM",
                f"check | best_match={room or 'none'} | "
                f"score={score:.4f} | thresh={self.ROOM_SC_THRESH}")
            if room:
                # require 3 consecutive positive matches to avoid false positives
                self._room_match_buf.append((room, score))
                self._room_match_buf = self._room_match_buf[-3:]
                names = [x[0] for x in self._room_match_buf]
                if len(names) == 3 and len(set(names)) == 1:
                    avg_score = float(np.mean([x[1] for x in self._room_match_buf]))
                    result.room_match       = room
                    result.room_match_score = avg_score
                    result.direction_to_room = self.db.direction_to(room, self.pose)
                    if room != self._last_room_match:
                        self._last_room_match = room
                        d = result.direction_to_room
                        self.dbg.log("ROOM",
                            f"RE-ENTRY CONFIRMED: '{room}' | "
                            f"avg_score={avg_score:.4f} | "
                            f"dist={d[2]:.2f}m bearing={d[3]:.1f}°" if d else "")
            else:
                self._room_match_buf.clear()
                self._last_room_match = None

        return result

    # ── submap helpers ────────────────────────────────────────────────────────
    def _push_submap(self, world_pts: np.ndarray) -> None:
        """Add a world-frame cloud to the submap ring buffer."""
        self._submap_clouds.append(world_pts.copy())
        if len(self._submap_clouds) > self.SUBMAP_KF_COUNT:
            self._submap_clouds.pop(0)

    def _get_submap(self) -> Optional[np.ndarray]:
        """Return merged + downsampled submap, or None if empty."""
        if not self._submap_clouds:
            return None
        merged = np.concatenate(self._submap_clouds, axis=0)
        if len(merged) > self.SUBMAP_MAX_PTS:
            idx    = np.random.choice(len(merged), self.SUBMAP_MAX_PTS, replace=False)
            merged = merged[idx]
        return merged.astype(np.float64)

    # ── keyframe management ────────────────────────────────────────────────────
    def _add_keyframe(self, points: np.ndarray, result: SLAMResult) -> None:
        kid  = len(self.keyframes)
        desc = self.sc.make(points)
        kf   = KeyFrame(kid, Pose2D(self.pose.x, self.pose.y, self.pose.yaw),
                        points.copy(), desc)
        self.keyframes.append(kf)
        self._prev_kf = Pose2D(self.pose.x, self.pose.y, self.pose.yaw)
        result.keyframes = self.keyframes

        # Push this keyframe's world-frame cloud into the submap so the next
        # odometry ICP has a richer, more stable reference target.
        self._push_submap(_sensor_to_world(points, kf.pose))

        self.dbg.log("KF",
            f"added KF#{kid} | pose=({self.pose.x:.2f},{self.pose.y:.2f},"
            f"{math.degrees(self.pose.yaw):.1f}°) | total={len(self.keyframes)}")

        if kid > 0:
            prev = self.keyframes[kid - 1]
            rel  = prev.pose.inverse().compose(kf.pose)
            self.edges.append(PoseEdge(kid - 1, kid, rel,
                                       np.diag([100.0, 100.0, 50.0])))
            self.dbg.log("KF", f"  → sequential edge {kid-1}→{kid}")

        if kid > self.LC_WINDOW:
            self._detect_loop(kf, result)

    # ── loop closure ──────────────────────────────────────────────────────────
    def _detect_loop(self, query: KeyFrame, result: SLAMResult) -> None:
        rk_q = self.sc.ring_key(query.sc_desc)
        candidates = []
        n_checked = 0

        for kf in self.keyframes[:-self.LC_WINDOW]:
            rk_dist = float(np.linalg.norm(rk_q - self.sc.ring_key(kf.sc_desc)))
            if rk_dist > self.RING_KEY_GATE:
                continue
            n_checked += 1
            d = self.sc.distance(query.sc_desc, kf.sc_desc)
            self.dbg.log("SC",
                f"KF#{query.id}↔KF#{kf.id} | rk={rk_dist:.3f} sc={d:.4f} "
                f"{'CAND' if d < self.SC_DIST_THRESH else 'skip'}")
            if d < self.SC_DIST_THRESH:
                candidates.append((d, kf))

        self.dbg.log("SC",
            f"KF#{query.id}: checked {n_checked} | "
            f"candidates={len(candidates)}")

        if not candidates:
            return

        candidates.sort(key=lambda x: x[0])
        d_best, best_kf = candidates[0]
        self.dbg.log("LC",
            f"top candidate: KF#{best_kf.id} sc_dist={d_best:.4f} "
            f"→ ICP verify…")

        # ICP in WORLD frame so the relative pose edge is correct
        q_world  = _sensor_to_world(query.points,   query.pose)
        kf_world = _sensor_to_world(best_kf.points, best_kf.pose)
        T, rmse, ok = self.icp.align(q_world, kf_world)
        self.dbg.log("LC",
            f"ICP verify: ok={ok} rmse={rmse:.4f}m "
            f"(thresh={self.ICP_LOOP_RMSE_MAX})")

        if not ok or rmse > self.ICP_LOOP_RMSE_MAX:
            self.dbg.log("LC", "  loop REJECTED by ICP")
            return

        # Derive relative pose: ICP refinement on top of the pose estimates
        dp  = _T_to_pose2d(T)
        refined_q = Pose2D(query.pose.x + dp.x, query.pose.y + dp.y, query.pose.yaw + dp.yaw)
        rel = best_kf.pose.inverse().compose(refined_q)
        self.edges.append(PoseEdge(best_kf.id, query.id, rel,
                                   np.diag([200.0, 200.0, 100.0])))
        result.loop_closed   = True
        result.loop_from_kf  = best_kf.id
        result.loop_to_kf    = query.id
        result.loop_sc_dist  = d_best
        result.loop_icp_rmse = rmse

        self.dbg.log("LC",
            f"LOOP ACCEPTED KF#{query.id}→KF#{best_kf.id} | "
            f"sc={d_best:.4f} icp={rmse:.4f}m")

        self.dbg.log("POSE", "Running pose graph optimisation…")
        t0     = time.time()
        poses  = self.optimizer.optimize(
            [kf.pose for kf in self.keyframes], self.edges)
        dt     = time.time() - t0
        # compute max correction magnitude
        deltas = [math.hypot(p.x - kf.pose.x, p.y - kf.pose.y)
                  for p, kf in zip(poses, self.keyframes)]
        max_d  = max(deltas)
        for kf, p in zip(self.keyframes, poses):
            kf.pose = p
        self.pose       = poses[-1]
        result.pose     = self.pose
        self.dbg.log("POSE",
            f"optimisation done in {dt*1000:.0f}ms | "
            f"max correction={max_d:.4f}m")

        # Repaint the occupancy grid from scratch using the corrected keyframe
        # poses.  Without this, walls painted before the loop correction stay at
        # their old (wrong) angles forever and the room appears duplicated.
        self.dbg.log("MAP", "Rebuilding occupancy grid from corrected KF poses…")
        self._rebuild_map()


    # ── map rebuild ───────────────────────────────────────────────────────────
    def _rebuild_map(self) -> None:
        """
        Wipe and repaint the occupancy grid from all stored keyframes using
        their *corrected* poses.  Called after every loop closure so that
        walls painted before the correction are erased and repainted at the
        right orientation – eliminating the 'duplicated room' artefact.
        """
        self.occ_map.grid[:] = 0.0
        self._submap_clouds.clear()   # submap is now stale; refill from KFs
        for kf in self.keyframes:
            self.occ_map.update(kf.pose, kf.points)
            self._push_submap(_sensor_to_world(kf.points, kf.pose))
        self.dbg.log("MAP",
            f"Rebuild complete | KFs={len(self.keyframes)} | "
            f"occupied={(self.occ_map.grid > 0.5).sum()}")

# ─── helpers ──────────────────────────────────────────────────────────────────
def _T_to_pose2d(T: np.ndarray) -> Pose2D:
    return Pose2D(T[0, 2], T[1, 2], math.atan2(T[1, 0], T[0, 0]))


def _sensor_to_world(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    """Rotate+translate sensor-frame (N,2) cloud into world frame using pose."""
    if len(points) == 0:
        return points
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    R    = np.array([[c, -s], [s, c]], dtype=np.float64)
    return (R @ points.astype(np.float64).T).T + np.array([pose.x, pose.y])