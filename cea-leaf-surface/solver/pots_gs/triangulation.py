"""Calibrated pinhole projection and robust triangulation (NumPy only).

Convention: X_cam = R @ X_world + t; camera axes x right, y down, z forward.
Pixel observations MUST already be undistorted for their supplied K.
Translations and output coordinates share the caller's declared unit.
"""
from dataclasses import dataclass
from itertools import combinations
import numpy as np


@dataclass(frozen=True)
class Camera:
    view_id: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    unit: str = "unknown"

    def __post_init__(self):
        for name, shape in (("K", (3, 3)), ("R", (3, 3)), ("t", (3,))):
            a = np.asarray(getattr(self, name), dtype=float)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f"{name} must have finite shape {shape}")
            object.__setattr__(self, name, a)
        if not np.allclose(self.R.T @ self.R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(self.R), 1):
            raise ValueError("R must be a proper rotation")
        if self.K[0, 0] <= 0 or self.K[1, 1] <= 0 or not np.allclose(self.K[2], [0, 0, 1]):
            raise ValueError("K must be a conventional pinhole intrinsic matrix")

    @property
    def center(self):
        return -self.R.T @ self.t

    @property
    def extrinsic(self):
        return np.column_stack((self.R, self.t))

    def project(self, point):
        p = self.R @ np.asarray(point, dtype=float) + self.t
        if abs(p[2]) < 1e-12:
            return np.array([np.inf, np.inf]), float(p[2])
        q = self.K @ p
        return q[:2] / q[2], float(p[2])


@dataclass(frozen=True)
class TriangulationResult:
    point: np.ndarray
    inlier_views: tuple[str, ...]
    rejected_views: tuple[str, ...]
    residuals_px: dict[str, float]
    max_ray_angle_deg: float
    unit: str
    status: str = "estimated_from_observations"


def _dlt(cameras, pixels, weights):
    rows = []
    for c, xy, weight in zip(cameras, pixels, weights):
        p = np.linalg.solve(c.K, [*xy, 1.0])
        e = c.extrinsic
        rows.extend(np.sqrt(weight) * np.array([p[0] * e[2] - e[0], p[1] * e[2] - e[1]]))
    _, _, vh = np.linalg.svd(np.asarray(rows))
    h = vh[-1]
    if abs(h[3]) < 1e-12:
        raise ValueError("point at infinity / degenerate camera geometry")
    return h[:3] / h[3]


def triangulate(cameras, pixels, confidence=None, *, threshold_px=3.0, min_angle_deg=1.0):
    """Exhaustive pair consensus, confidence-weighted DLT, then Huber IRLS.

    Requires at least two independent, visible, reviewed physical correspondences.
    The caller must exclude inferred points and view-dependent contour samples.
    Enumerating all pairs is intentional for the target 3--12 input views.
    """
    cameras = list(cameras)
    xy = np.asarray(pixels, dtype=float)
    n = len(cameras)
    w = np.ones(n) if confidence is None else np.asarray(confidence, dtype=float)
    if n < 2 or xy.shape != (n, 2) or not np.isfinite(xy).all():
        raise ValueError("at least two finite pixel observations are required")
    if w.shape != (n,) or not np.isfinite(w).all() or np.any(w <= 0) or np.any(w > 1):
        raise ValueError("confidence must be in (0, 1]")
    if len({c.view_id for c in cameras}) != n or len({c.unit for c in cameras}) != 1:
        raise ValueError("view IDs must be unique and camera units identical")
    if threshold_px <= 0 or not 0 < min_angle_deg < 90:
        raise ValueError("invalid residual or angle threshold")

    def residuals(point):
        projected = [c.project(point) for c in cameras]
        return np.linalg.norm(np.asarray([p[0] for p in projected]) - xy, axis=1), np.asarray([p[1] for p in projected])

    best = None
    for pair in combinations(range(n), 2):
        try:
            p = _dlt([cameras[i] for i in pair], xy[list(pair)], w[list(pair)])
        except (ValueError, np.linalg.LinAlgError):
            continue
        err, z = residuals(p)
        inliers = (err <= threshold_px) & (z > 1e-9)
        score = (int(inliers.sum()), float(w[inliers].sum()), -float(np.sum(w * np.minimum(err, threshold_px))))
        if best is None or score > best[0]:
            best = score, inliers
    if best is None or best[1].sum() < 2:
        raise ValueError("no positive-depth multiview consensus")
    indices = np.flatnonzero(best[1])
    selected = [cameras[i] for i in indices]
    rw = w[indices].copy()
    for _ in range(8):
        point = _dlt(selected, xy[indices], rw)
        errors, depths = residuals(point)
        rw = w[indices] * np.minimum(1.0, (threshold_px / 2) / np.maximum(errors[indices], 1e-12))
    inliers = (errors <= threshold_px) & (depths > 1e-9)
    if not np.all(inliers[indices]):
        raise ValueError("refinement invalidated consensus")
    rays = np.asarray([point - c.center for c in selected])
    norms = np.linalg.norm(rays, axis=1)
    if np.any(norms < 1e-12):
        raise ValueError("point coincides with camera center")
    rays /= norms[:, None]
    # An angle near 180 degrees is also ill-conditioned for triangulation.
    angle = max(float(np.degrees(np.arccos(np.clip(abs(a @ b), 0, 1)))) for a, b in combinations(rays, 2))
    if angle < min_angle_deg:
        raise ValueError(f"insufficient triangulation angle: {angle:.6f} deg")
    return TriangulationResult(point, tuple(cameras[i].view_id for i in indices),
        tuple(cameras[i].view_id for i in range(n) if i not in indices),
        {c.view_id: float(e) for c, e in zip(cameras, errors)}, angle, cameras[0].unit)
