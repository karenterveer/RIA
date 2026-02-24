"""
Timing quality control for air-shower radio reconstruction.

Provides outlier detection, iterative pruning, and per-antenna
uncertainty estimation for timing data.  All functions operate on
plain NumPy arrays and are independent of the forward model.

Functions
---------
detect_timing_outliers
    Per-station k-nearest-neighbour outlier detection.
iterative_timing_pruning
    Iterative worst-residual removal until convergence.
get_local_timing_uncertainties
    Per-antenna uncertainty from local cluster residuals.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.spatial.distance import cdist

from . import config

logger = logging.getLogger(__name__)


# ===================================================================
#  Per-station outlier detection
# ===================================================================

def detect_timing_outliers(
    positions: np.ndarray,
    times: np.ndarray,
    station_ids: np.ndarray,
    neighbors_k: int | None = None,
    std_factor: float | None = None,
    min_neighbors: int | None = None,
) -> np.ndarray:
    """Flag timing outliers within each station.

    For every station a local median comparison is performed: each
    point's timing residual relative to its k nearest neighbours is
    compared against the station-wide RMS.  Points exceeding
    ``std_factor`` times the station RMS are flagged.

    Parameters
    ----------
    positions : ndarray, shape ``(2 or 3, N)``
        Ground-plane positions of all timing data points.
    times : ndarray, shape ``(N,)``
        Measured arrival times [s].
    station_ids : ndarray, shape ``(N,)``
        Integer station label for each data point.
    neighbors_k : int, optional
        Number of neighbours per query.  Defaults to
        ``config.OUTLIER_NEIGHBORS_K``.
    std_factor : float, optional
        Threshold in units of station RMS.  Defaults to
        ``config.OUTLIER_STD_FACTOR``.
    min_neighbors : int, optional
        Minimum cluster size to attempt detection.  Defaults to
        ``config.MIN_ABSOLUTE_NEIGHBORS``.

    Returns
    -------
    ndarray of bool, shape ``(N,)``
        ``True`` where the data point is *kept* (not an outlier).
    """
    if neighbors_k is None:
        neighbors_k = config.OUTLIER_NEIGHBORS_K
    if std_factor is None:
        std_factor = config.OUTLIER_STD_FACTOR
    if min_neighbors is None:
        min_neighbors = config.MIN_ABSOLUTE_NEIGHBORS

    n_total = times.size
    keep_mask = np.ones(n_total, dtype=bool)
    unique_stations = np.unique(station_ids)

    logger.info(
        "Running per-station outlier detection on %d timing candidates ...",
        n_total,
    )

    for station in unique_stations:
        station_mask = station_ids == station
        station_indices = np.where(station_mask)[0]
        n_pts = len(station_indices)
        if n_pts <= 1:
            continue

        station_pos = positions[:, station_mask]
        station_times = times[station_mask]

        k = min(neighbors_k, n_pts - 1)
        if k < min_neighbors:
            continue

        time_std_ns = np.std(
            (station_times - np.mean(station_times)) * 1e9
        )
        threshold_ns = max(std_factor * time_std_ns, 15.0)

        dist_matrix = cdist(station_pos.T, station_pos.T)
        is_outlier = np.zeros(n_pts, dtype=bool)

        for i in range(n_pts):
            nn = np.argsort(dist_matrix[i, :])[1 : k + 1]
            diff_ns = (
                np.abs(station_times[i] - np.median(station_times[nn]))
                * 1e9
            )
            if diff_ns > threshold_ns:
                is_outlier[i] = True

        keep_mask[station_indices] = ~is_outlier

    n_outliers = n_total - int(np.sum(keep_mask))
    logger.info("  Identified %d timing outliers for demotion.", n_outliers)
    return keep_mask


# ===================================================================
#  Iterative pruning
# ===================================================================

def iterative_timing_pruning(
    positions: np.ndarray,
    times: np.ndarray,
    initial_mask: np.ndarray,
    min_timing_points: int | None = None,
    uncert_threshold: float | None = None,
) -> tuple:
    """Iteratively remove timing outliers until convergence.

    Fits a polynomial surface ``(offset, x, y, r^2)`` at each iteration
    and removes the single worst residual until the residual standard
    deviation drops below the threshold.

    Parameters
    ----------
    positions : ndarray, shape ``(2 or 3, N)``
        Ground-plane positions.
    times : ndarray, shape ``(N,)``
        Measured arrival times [s].
    initial_mask : ndarray of bool, shape ``(N,)``
        Mask of points surviving the first outlier pass.
    min_timing_points : int, optional
        Stop if fewer points remain.  Default ``config.MIN_TIMING_POINTS``.
    uncert_threshold : float, optional
        Convergence threshold [s].  Default ``config.TIMING_UNCERT_THRESHOLD``.

    Returns
    -------
    mask : ndarray of bool
        Updated mask after pruning.
    final_std : float
        Residual standard deviation of the converged fit [s].
    coeffs : ndarray or None
        Polynomial coefficients ``(offset, kx, ky, kr2)``.
    mean_pos : ndarray or None
        2-D centroid of the retained positions [m].
    scale : float
        Position normalisation scale used in the fit.
    inv_AtA : ndarray or None
        Inverse of the design-matrix Gram matrix (for uncertainty propagation).
    """
    if min_timing_points is None:
        min_timing_points = config.MIN_TIMING_POINTS
    if uncert_threshold is None:
        uncert_threshold = config.TIMING_UNCERT_THRESHOLD

    current_mask = initial_mask.copy()
    max_removals = int(np.sum(initial_mask)) - min_timing_points + 1

    logger.info(
        "Starting iterative timing pruning.  Initial points: %d",
        int(np.sum(current_mask)),
    )

    for _ in range(max_removals):
        active = np.where(current_mask)[0]
        if len(active) < min_timing_points:
            break

        pos_a = positions[:, active]
        t_a = times[active]

        x = pos_a[0] - np.mean(pos_a[0])
        y = pos_a[1] - np.mean(pos_a[1])
        scale = np.std(np.sqrt(x ** 2 + y ** 2))
        if scale < 1e-6:
            scale = 1.0
        xn, yn = x / scale, y / scale
        r2n = xn ** 2 + yn ** 2

        A = np.column_stack([np.ones(len(t_a)), xn, yn, r2n])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, t_a, rcond=None)
            residuals = t_a - A @ coeffs
        except Exception:
            residuals = t_a - np.mean(t_a)
            coeffs = None

        std_res = np.std(residuals)
        if std_res < uncert_threshold:
            logger.info(
                "Timing pruning converged.  Std: %.2f ns.  Points: %d",
                std_res * 1e9, len(active),
            )
            try:
                inv_AtA = np.linalg.inv(A.T @ A)
            except np.linalg.LinAlgError:
                inv_AtA = None
            return (
                current_mask, std_res, coeffs,
                np.mean(pos_a[:2], axis=1), scale, inv_AtA,
            )

        worst = np.argmax(np.abs(residuals))
        current_mask[active[worst]] = False

    # Final computation with remaining points
    active = np.where(current_mask)[0]
    if len(active) >= min_timing_points:
        pos_a = positions[:, active]
        t_a = times[active]
        mean_pos = np.mean(pos_a[:2], axis=1)
        x = pos_a[0] - mean_pos[0]
        y = pos_a[1] - mean_pos[1]
        scale = np.std(np.sqrt(x ** 2 + y ** 2))
        if scale < 1e-6:
            scale = 1.0
        xn, yn = x / scale, y / scale
        r2n = xn ** 2 + yn ** 2
        A = np.column_stack([np.ones(len(t_a)), xn, yn, r2n])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, t_a, rcond=None)
            residuals = t_a - A @ coeffs
            final_std = np.std(residuals)
            try:
                inv_AtA = np.linalg.inv(A.T @ A)
            except np.linalg.LinAlgError:
                inv_AtA = None
            return current_mask, final_std, coeffs, mean_pos, scale, inv_AtA
        except Exception:
            pass

    return current_mask, 15e-9, None, None, 1.0, None


# ===================================================================
#  Local per-antenna uncertainty
# ===================================================================

def get_local_timing_uncertainties(
    positions: np.ndarray,
    times: np.ndarray,
    min_uncertainty_s: float = 1.5e-9,
    cluster_size: int | None = None,
) -> np.ndarray:
    """Per-antenna timing uncertainty from local cluster residuals.

    For each antenna the *k* nearest neighbours are identified and a
    polynomial surface ``(offset, x, y, r^2)`` is fit to their timing
    data.  The residual standard deviation of that fit provides the
    local timing uncertainty.

    Parameters
    ----------
    positions : ndarray, shape ``(2 or 3, N)``
        Ground-plane positions.
    times : ndarray, shape ``(N,)``
        Measured arrival times [s].
    min_uncertainty_s : float, optional
        Floor value returned when the fit residual is very small [s].
    cluster_size : int, optional
        Number of nearest neighbours per query.  Default
        ``config.LOCAL_FIT_NEIGHBORS_K``.

    Returns
    -------
    ndarray, shape ``(N,)``
        Estimated timing uncertainty per antenna [s].
    """
    from scipy.spatial import KDTree

    if cluster_size is None:
        cluster_size = config.LOCAL_FIT_NEIGHBORS_K

    n_antennas = positions.shape[1]
    uncertainties = np.zeros(n_antennas)
    fallback_std = 5.0e-9
    min_cluster = 8

    pos_2d = positions[:2].T
    tree = KDTree(pos_2d)

    for i in range(n_antennas):
        k = min(cluster_size, n_antennas)
        _, nn_idx = tree.query(pos_2d[i], k=k)

        if len(nn_idx) < min_cluster:
            uncertainties[i] = fallback_std
            continue

        pos_c = positions[:, nn_idx]
        t_c = times[nn_idx]
        n_pts = len(nn_idx)

        x = pos_c[0] - np.mean(pos_c[0])
        y = pos_c[1] - np.mean(pos_c[1])
        scale = np.std(np.sqrt(x ** 2 + y ** 2))
        if scale < 1e-6:
            scale = 1.0
        xn, yn = x / scale, y / scale
        r2n = xn ** 2 + yn ** 2

        A = np.column_stack([np.ones(n_pts), xn, yn, r2n])
        try:
            coeffs, _, _, _ = np.linalg.lstsq(A, t_c, rcond=None)
            std_val = np.std(t_c - A @ coeffs)
        except Exception:
            std_val = fallback_std

        uncertainties[i] = max(std_val, min_uncertainty_s)

    return uncertainties
