#!/usr/bin/env python3
"""Calculate fusion index with a nucleus-aware MF20/actin hybrid method.

The workflow consumes nucleus masks and predictions produced by MyoFuse, creates
candidate myotube regions from MF20 and phalloidin/actin, and counts nuclei only
when a connected candidate region is associated with at least two nuclei.

This is a custom experimental workflow that combines the released MyoFuse
outputs with independently implemented MF20- and Actin-structure refinement.
It is not an official MyoFuse release.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from aicsimageio import AICSImage
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import cKDTree
from skimage.measure import regionprops
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import find_boundaries
from tifffile import imread, imwrite

from czi_tiles import open_analysis_scene, read_channel


# Calibrated defaults used for the current experimental workflow. Keep these
# values fixed across experimental groups unless a new independent validation
# is performed.
MF20_THRESHOLD = 75
ACTIN_THRESHOLD = 30
ACTIN_SUPPORT_RADIUS = 3
NUCLEUS_SUPPORT_RADIUS = 2
NUCLEUS_SUPPORT_OVERLAP = 0.15
MIN_NUCLEI_PER_GROUP = 2
CONSERVATIVE_HYBRID_ONLY_PROBABILITY_MIN = 0.024

# Experimental guard for small candidates. A two-nucleus candidate is sent to
# review when the local fiber directions around its nuclei disagree in either
# actin or MF20. The values below improved both manually reviewed validation
# fields, but they should remain optional until more biological conditions have
# been validated.
DIRECTION_RADIUS = 25
DIRECTION_MAX_ANGLE = 15.0

# Experimental per-nucleus graph guard. Two nuclei form an edge only when they
# are close, share a continuous MF20 path, and have compatible local fiber
# directions. A signal-supported nucleus without any such neighbor becomes a
# yellow review candidate. These fixed exploratory values improved F1 on the
# three currently reviewed fields, but independent validation is still needed.
GRAPH_MAX_DISTANCE = 90.0
GRAPH_MF20_PATH_MIN = 0.65
GRAPH_MAX_PAIR_ANGLE = 25.0
GRAPH_MAX_LINE_ANGLE = 90.0
GRAPH_PATH_RADIUS = 3
GRAPH_PATH_TRIM = 0.16

# Rescue very weak/fragmented myotubes only when two nearby nuclei independently
# receive an In prediction from the published MyoFuse classifier.
WEAK_SUPPORT_RADIUS = 20
WEAK_SUPPORT_OVERLAP = 0.20
WEAK_CLUSTER_DISTANCE = 35.0
MIN_MYOFUSE_IN_PER_WEAK_CLUSTER = 2

# Experimental recall-oriented review queue. This deliberately does not change
# the automatic FI. It uses a lower MF20 threshold to find nuclei connected by
# a continuous, direction-compatible local graph, then sends the strongest
# candidates to manual review. The fixed cap prevents dim fields from creating
# an impractically large review queue.
ADAPTIVE_REVIEW_MF20_THRESHOLD = 30
ADAPTIVE_REVIEW_MAX_DISTANCE = 70.0
ADAPTIVE_REVIEW_MIN_MF20_PATH = 0.60
ADAPTIVE_REVIEW_MAX_PAIR_ANGLE = 35.0
ADAPTIVE_REVIEW_MAX_CANDIDATES = 50

# Experimental recall-oriented rescue based on local actin architecture. A
# nucleus is rescued only when it belongs to a graph component containing a
# MyoFuse/Hybrid In anchor. Edges require nearby nuclei, coherent and aligned
# actin fibers, a continuous actin path, and supporting MF20 along that path.
# These values were calibrated on nine fields from one control well and must be
# validated on independent wells and treatment groups before routine use.
ACTIN_STRUCTURE_MAX_DISTANCE = 60.0
ACTIN_STRUCTURE_DIRECTION_RADIUS = 28
ACTIN_STRUCTURE_MIN_COHERENCE = 0.20
ACTIN_STRUCTURE_MAX_PAIR_ANGLE = 35.0
ACTIN_STRUCTURE_MAX_LINE_ANGLE = 30.0
ACTIN_STRUCTURE_ACTIN_THRESHOLD = 25
ACTIN_STRUCTURE_MF20_THRESHOLD = 30
ACTIN_STRUCTURE_MIN_ACTIN_PATH = 0.55
ACTIN_STRUCTURE_MIN_MF20_PATH = 0.40

# Adaptive safeguards for the current Actin-structure method. When the
# published MyoFuse classifier finds very few In nuclei, a broad Hybrid anchor
# is unreliable because mononucleated MyHC-positive cells can form large
# connected masks. In that case, automatic graph rescue must connect to an
# original MyoFuse In nucleus and ambiguous Hybrid-anchored components are sent
# to manual review. Low-overlap automatic In calls are also reviewed.
ACTIN_STRUCTURE_LOW_MYOFUSE_FI_THRESHOLD = 0.10
ACTIN_STRUCTURE_LOW_SUPPORT_REVIEW_OVERLAP = 0.25
ACTIN_STRUCTURE_NEIGHBOR_REVIEW_PROBABILITY = 0.25
ACTIN_STRUCTURE_NEIGHBOR_REVIEW_DISTANCE = 35.0
MASK_QC_MAX_PROCESSED_FRACTION = 0.80
MASK_QC_MAX_RAW_FRACTION = 0.50

# Structure segmentation v2 evaluates an oriented corridor between nearby
# nuclei. Unlike the original Actin-structure rescue, the corridor uses locally
# normalized intensity and measures fiber alignment along the complete path.
# Short signal gaps are allowed because nuclei often appear as holes in MF20.
# Unanchored components require substantially stronger evidence than components
# connected to an existing MyoFuse/Hybrid In nucleus.
STRUCTURE_V2_MAX_DISTANCE = 85.0
STRUCTURE_V2_LOCAL_SIGMA = 24.0
STRUCTURE_V2_TENSOR_SIGMA = 4.0
STRUCTURE_V2_CORRIDOR_RADIUS = 4
STRUCTURE_V2_MIN_COHERENCE = 0.12
STRUCTURE_V2_MAX_LINE_ANGLE = 38.0
STRUCTURE_V2_MIN_ACTIN_PATH = 0.0
STRUCTURE_V2_MIN_MF20_PATH = 0.10
STRUCTURE_V2_MIN_ALIGNED_PATH = 0.20
STRUCTURE_V2_MAX_DARK_GAP = 0.55
STRUCTURE_V2_STRONG_ACTIN_PATH = 0.0
STRUCTURE_V2_STRONG_MF20_PATH = 0.45
STRUCTURE_V2_STRONG_ALIGNED_PATH = 0.60
STRUCTURE_V2_STRONG_MAX_DARK_GAP = 0.20
STRUCTURE_V2_REVIEW_MAX_CANDIDATES = 50


def scale(image: np.ndarray) -> np.ndarray:
    """Scale a channel to 8-bit using robust whole-image percentiles."""
    low, high = np.percentile(image, [1, 99.7])
    if high <= low:
        raise ValueError("Image channel has insufficient intensity range")
    return (np.clip((image.astype(np.float32) - low) / (high - low), 0, 1) * 255).astype(np.uint8)


def cleaned_threshold_mask(channel: np.ndarray, threshold: int) -> np.ndarray:
    """Return the conservative morphology-cleaned threshold mask."""
    kernel = np.ones((4, 4), np.uint8)
    _, low = cv2.threshold(channel, threshold, 255, cv2.THRESH_BINARY)
    low = cv2.morphologyEx(low, cv2.MORPH_OPEN, kernel)
    low = cv2.morphologyEx(low, cv2.MORPH_CLOSE, kernel)
    return low.astype(bool)


def hole_preserving_mask(conservative: np.ndarray, nuclei: np.ndarray) -> np.ndarray:
    """Apply the original contour rule that preserves nuclear signal holes."""
    processed = conservative.astype(np.uint8) * 255
    processed = cv2.bitwise_not(processed)
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        region = np.zeros_like(processed)
        cv2.drawContours(region, (contour,), -1, 255, -1)
        pixels = region == 255
        if pixels.any() and nuclei[pixels].mean() > 50:
            cv2.drawContours(processed, (contour,), -1, 0, -1)
    return cv2.bitwise_not(processed).astype(bool)


def make_mask(channel: np.ndarray, nuclei: np.ndarray, threshold: int) -> np.ndarray:
    """Create a signal mask with a guard against morphology inversion failure."""
    conservative = cleaned_threshold_mask(channel, threshold)
    result = hole_preserving_mask(conservative, nuclei)
    # In high-background images the outer background contour can be mistaken
    # for a signal object, turning a sparse MF20 mask into almost the full
    # image. Fall back to the conservative threshold mask only for this clear
    # contradiction. Routine images retain the original validated behavior.
    if (
        result.mean() > MASK_QC_MAX_PROCESSED_FRACTION
        and conservative.mean() < MASK_QC_MAX_RAW_FRACTION
    ):
        return conservative
    return result


def mask_qc_fallback_active(
    channel: np.ndarray, nuclei: np.ndarray, threshold: int
) -> bool:
    """Report whether make_mask used its conservative saturation fallback."""
    conservative = cleaned_threshold_mask(channel, threshold)
    original = hole_preserving_mask(conservative, nuclei)
    return bool(
        original.mean() > MASK_QC_MAX_PROCESSED_FRACTION
        and conservative.mean() < MASK_QC_MAX_RAW_FRACTION
    )


def adjacent_pixel_correlation(channel: np.ndarray) -> float:
    """Return mean horizontal/vertical one-pixel spatial correlation."""
    horizontal = np.corrcoef(channel[:, 1:].ravel(), channel[:, :-1].ravel())[0, 1]
    vertical = np.corrcoef(channel[1:, :].ravel(), channel[:-1, :].ravel())[0, 1]
    values = np.asarray([horizontal, vertical], dtype=float)
    return float(np.nanmean(values))


def channel_quality_table(
    image: AICSImage,
    tile_index: Optional[int],
    dapi: np.ndarray,
    channel_names: list[str],
    nuclei_channel: int,
    actin_channel: int,
    mf20_channel: int,
    threshold: int,
) -> tuple[pd.DataFrame, Optional[str]]:
    """Measure spatial continuity and stop on a strongly noise-like MF20 channel."""
    records = []
    for channel_index, channel_name in enumerate(channel_names):
        try:
            scaled = scale(read_channel(image, channel_index, tile_index))
            mask = make_mask(scaled, dapi, threshold)
            foreground_fraction = float(mask.mean())
            component_count, _, stats, _ = cv2.connectedComponentsWithStats(
                mask.astype(np.uint8), 8
            )
            sizes = stats[1:, cv2.CC_STAT_AREA]
            largest_share = (
                float(sizes.max() / sizes.sum()) if component_count > 1 and sizes.sum() else 0.0
            )
            correlation = adjacent_pixel_correlation(scaled)
        except ValueError:
            foreground_fraction, largest_share, correlation, component_count = 0.0, 0.0, 0.0, 1
        records.append({
            "Channel index": channel_index,
            "Channel name": channel_name,
            "Selected role": (
                "Nuclei" if channel_index == nuclei_channel else
                "Actin" if channel_index == actin_channel else
                "MF20" if channel_index == mf20_channel else "Other"
            ),
            "Adjacent-pixel correlation": correlation,
            "Threshold foreground fraction": foreground_fraction,
            "Connected components": max(0, int(component_count) - 1),
            "Largest component share": largest_share,
        })
    table = pd.DataFrame(records)
    selected = table.loc[table["Channel index"].eq(mf20_channel)].iloc[0]
    looks_like_noise = (
        selected["Adjacent-pixel correlation"] < 0.10
        and selected["Threshold foreground fraction"] < 0.002
    )
    if not looks_like_noise:
        return table, None
    alternatives = table[
        ~table["Channel index"].isin([nuclei_channel, actin_channel, mf20_channel])
        & table["Adjacent-pixel correlation"].gt(0.25)
        & table["Threshold foreground fraction"].between(0.002, 0.70)
    ].sort_values("Adjacent-pixel correlation", ascending=False)
    suggestion = ""
    if not alternatives.empty:
        best = alternatives.iloc[0]
        suggestion = (
            f" C{int(best['Channel index'])} ({best['Channel name']}) has substantially more "
            "spatially continuous signal and should be checked."
        )
    warning = (
        f"Selected MF20 channel C{mf20_channel} ({channel_names[mf20_channel]}) failed automatic "
        "quality control: its signal is nearly spatially random and produced no usable foreground."
        + suggestion
    )
    return table, warning


def nucleus_metrics(
    labels: np.ndarray,
    nucleus_ids: np.ndarray,
    mask: np.ndarray,
    group_labels: np.ndarray,
) -> pd.DataFrame:
    """Measure overlap and connected-component membership for each nucleus."""
    properties = {int(item.label): item for item in regionprops(labels)}
    records = []
    for nucleus_id in nucleus_ids:
        nucleus_id = int(nucleus_id)
        item = properties[nucleus_id]
        coordinates = item.coords
        rows, columns = coordinates[:, 0], coordinates[:, 1]
        supported = mask[rows, columns]
        overlap = float(supported.mean())
        values, counts = np.unique(
            group_labels[rows[supported], columns[supported]], return_counts=True
        )
        positive = values > 0
        group = int(values[positive][np.argmax(counts[positive])]) if positive.any() else 0
        y, x = item.centroid
        records.append((nucleus_id, float(x), float(y), overlap, group))
    return pd.DataFrame(records, columns=["Label ID", "X", "Y", "Overlap", "Group"])


def connected_nucleus_clusters(frame: pd.DataFrame, distance: float) -> list[set[int]]:
    """Return local nucleus clusters within each broad image-support group."""
    clusters = []
    for _, group_frame in frame.groupby("Weak group"):
        indices = list(group_frame.index)
        neighbors = {index: set() for index in indices}
        coordinates = group_frame[["X", "Y"]].to_numpy(dtype=float)
        for left_position, right_position in cKDTree(coordinates).query_pairs(distance):
            left, right = indices[left_position], indices[right_position]
            neighbors[left].add(right)
            neighbors[right].add(left)
        remaining = set(indices)
        while remaining:
            seed = remaining.pop()
            cluster = {seed}
            stack = [seed]
            while stack:
                current = stack.pop()
                new = neighbors[current] & remaining
                remaining -= new
                cluster |= new
                stack.extend(new)
            clusters.append(cluster)
    return clusters


def orientation_gradients(channel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return smoothed horizontal and vertical gradients for fiber direction."""
    smoothed = cv2.GaussianBlur(channel.astype(np.float32), (0, 0), 1.2)
    horizontal = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    vertical = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    return horizontal, vertical


def local_fiber_orientation(
    gradients: tuple[np.ndarray, np.ndarray],
    x: float,
    y: float,
    radius: int,
    allowed: np.ndarray,
) -> tuple[float, float]:
    """Estimate local fiber angle (0-180 degrees) and tensor coherence."""
    horizontal, vertical = gradients
    y0, y1 = max(0, int(y - radius)), min(horizontal.shape[0], int(y + radius + 1))
    x0, x1 = max(0, int(x - radius)), min(horizontal.shape[1], int(x + radius + 1))
    yy, xx = np.ogrid[y0:y1, x0:x1]
    local = (xx - x) ** 2 + (yy - y) ** 2 <= radius ** 2
    local &= allowed[y0:y1, x0:x1]
    gx = horizontal[y0:y1, x0:x1]
    gy = vertical[y0:y1, x0:x1]
    magnitude = gx * gx + gy * gy
    values = magnitude[local]
    if len(values) < 10:
        return np.nan, np.nan
    local &= magnitude >= np.percentile(values, 40)
    xx_sum = float((gx[local] ** 2).sum())
    yy_sum = float((gy[local] ** 2).sum())
    xy_sum = float((gx[local] * gy[local]).sum())
    angle = (np.degrees(0.5 * np.arctan2(2 * xy_sum, xx_sum - yy_sum)) + 90) % 180
    coherence = np.hypot(xx_sum - yy_sum, 2 * xy_sum) / (xx_sum + yy_sum + 1e-9)
    return float(angle), float(coherence)


def axial_angle_difference(left: float, right: float) -> float:
    """Return the smallest difference between two unoriented 0-180 degree axes."""
    return float(abs(((left - right + 90) % 180) - 90))


def locally_normalized_score(channel: np.ndarray, sigma: float) -> np.ndarray:
    """Combine robust global intensity with positive local contrast."""
    image = channel.astype(np.float32) / 255.0
    low, high = np.percentile(image, [2.0, 99.5])
    global_score = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
    background = cv2.GaussianBlur(image, (0, 0), sigma)
    residual = np.maximum(image - background, 0)
    local_high = float(np.percentile(residual, 99.5))
    local_score = np.clip(residual / max(local_high, 1e-6), 0, 1)
    return np.clip(0.58 * global_score + 0.42 * local_score, 0, 1)


def structure_tensor_maps(
    channel: np.ndarray, tensor_sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pixel fiber angle and structure-tensor coherence."""
    smoothed = cv2.GaussianBlur(channel.astype(np.float32), (0, 0), 1.2)
    gx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3)
    xx = cv2.GaussianBlur(gx * gx, (0, 0), tensor_sigma)
    yy = cv2.GaussianBlur(gy * gy, (0, 0), tensor_sigma)
    xy = cv2.GaussianBlur(gx * gy, (0, 0), tensor_sigma)
    angle = (np.degrees(0.5 * np.arctan2(2 * xy, xx - yy)) + 90) % 180
    coherence = np.hypot(xx - yy, 2 * xy) / (xx + yy + 1e-6)
    return angle.astype(np.float32), coherence.astype(np.float32)


def longest_false_run_fraction(values: np.ndarray) -> float:
    """Return the longest consecutive False run as a fraction of the vector."""
    if len(values) == 0:
        return 1.0
    longest = current = 0
    for value in values:
        if bool(value):
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return float(longest / len(values))


def oriented_corridor_metrics(
    actin_score: np.ndarray,
    mf20_score: np.ndarray,
    angle_map: np.ndarray,
    coherence_map: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: int,
    min_coherence: float,
    max_line_angle: float,
) -> dict[str, float]:
    """Measure signal strength, alignment and gaps in a nucleus-to-nucleus corridor."""
    dx, dy = float(x2 - x1), float(y2 - y1)
    distance = float(np.hypot(dx, dy))
    line_angle = float(np.degrees(np.arctan2(dy, dx)) % 180)
    samples = max(14, int(distance * 1.25))
    fractions = np.linspace(0.14, 0.86, samples)
    # Sample several pixels across the proposed myotube axis. This tolerates a
    # slightly curved fiber while still requiring a continuous axial corridor.
    perpendicular_x = -dy / max(distance, 1e-6)
    perpendicular_y = dx / max(distance, 1e-6)
    offsets = np.arange(-radius, radius + 1, dtype=float)
    height, width = actin_score.shape
    actin_values = []
    mf20_values = []
    aligned_values = []
    for fraction in fractions:
        center_x = x1 + dx * fraction
        center_y = y1 + dy * fraction
        xs = np.clip(
            np.rint(center_x + perpendicular_x * offsets).astype(int), 0, width - 1
        )
        ys = np.clip(
            np.rint(center_y + perpendicular_y * offsets).astype(int), 0, height - 1
        )
        actin_line = actin_score[ys, xs]
        mf20_line = mf20_score[ys, xs]
        # Use the location with the strongest combined evidence, rather than
        # requiring the exact straight centreline to hit a mildly curved fiber.
        best = int(np.argmax(0.58 * actin_line + 0.42 * mf20_line))
        actin_values.append(float(actin_line[best]))
        mf20_values.append(float(mf20_line[best]))
        local_angle = float(angle_map[ys[best], xs[best]])
        local_coherence = float(coherence_map[ys[best], xs[best]])
        aligned_values.append(
            local_coherence >= min_coherence
            and axial_angle_difference(local_angle, line_angle) <= max_line_angle
        )
    actin_values = np.asarray(actin_values)
    mf20_values = np.asarray(mf20_values)
    aligned_values = np.asarray(aligned_values, dtype=bool)
    actin_supported = actin_values >= 0.30
    mf20_supported = mf20_values >= 0.20
    visible = actin_supported | mf20_supported
    return {
        "Distance (px)": distance,
        "Actin path fraction": float(actin_supported.mean()),
        "MF20 path fraction": float(mf20_supported.mean()),
        "Aligned path fraction": float(aligned_values.mean()),
        "Maximum dark gap fraction": longest_false_run_fraction(visible),
        "Median Actin score": float(np.median(actin_values)),
        "Median MF20 score": float(np.median(mf20_values)),
    }


def structure_v2_rescue(
    nuclei: pd.DataFrame,
    current_in: set[int],
    anchor_ids: set[int],
    actin: np.ndarray,
    mf20: np.ndarray,
    max_distance: float,
    local_sigma: float,
    tensor_sigma: float,
    corridor_radius: int,
    min_coherence: float,
    max_line_angle: float,
    min_actin_path: float,
    min_mf20_path: float,
    min_aligned_path: float,
    max_dark_gap: float,
    strong_only: bool,
    allow_unanchored_auto: bool,
) -> tuple[set[int], set[int], dict[int, dict[str, float]], pd.DataFrame, np.ndarray]:
    """Classify nuclei with locally normalized, direction-continuous corridors.

    Components connected to a published/Hybrid In anchor use the regular
    corridor thresholds. Components without an anchor are automatically rescued
    only when their evidence satisfies the stricter strong-corridor rule;
    otherwise they are sent to manual review.
    """
    actin_score = locally_normalized_score(actin, local_sigma)
    mf20_score = locally_normalized_score(mf20, local_sigma)
    angle_map, coherence_map = structure_tensor_maps(actin, tensor_sigma)
    parent: dict[int, int] = {}

    def find(nucleus_id: int) -> int:
        parent.setdefault(nucleus_id, nucleus_id)
        while parent[nucleus_id] != nucleus_id:
            parent[nucleus_id] = parent[parent[nucleus_id]]
            nucleus_id = parent[nucleus_id]
        return nucleus_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    records: list[dict[str, float | int]] = []
    compatible_edges: list[tuple[int, int, dict[str, float]]] = []
    rows = list(nuclei.iterrows())
    coordinates = nuclei[["X", "Y"]].to_numpy(dtype=float)
    for left_position, right_position in sorted(
        cKDTree(coordinates).query_pairs(max_distance)
    ):
        _, left = rows[left_position]
        _, right = rows[right_position]
        left_id, right_id = int(left["Label ID"]), int(right["Label ID"])
        metrics = oriented_corridor_metrics(
            actin_score,
            mf20_score,
            angle_map,
            coherence_map,
            float(left["X"]),
            float(left["Y"]),
            float(right["X"]),
            float(right["Y"]),
            corridor_radius,
            min_coherence,
            max_line_angle,
        )
        compatible = bool(
            metrics["Actin path fraction"] >= min_actin_path
            and metrics["MF20 path fraction"] >= min_mf20_path
            and metrics["Aligned path fraction"] >= min_aligned_path
            and metrics["Maximum dark gap fraction"] <= max_dark_gap
        )
        strong = bool(
            metrics["Actin path fraction"] >= STRUCTURE_V2_STRONG_ACTIN_PATH
            and metrics["MF20 path fraction"] >= STRUCTURE_V2_STRONG_MF20_PATH
            and metrics["Aligned path fraction"] >= STRUCTURE_V2_STRONG_ALIGNED_PATH
            and metrics["Maximum dark gap fraction"]
            <= STRUCTURE_V2_STRONG_MAX_DARK_GAP
        )
        record = {
            "Nucleus A": left_id,
            "Nucleus B": right_id,
            **metrics,
            "Compatible edge": int(compatible),
            "Strong edge": int(strong),
        }
        records.append(record)
        edge_used = bool(compatible and (strong or not strong_only))
        record["Edge used for component"] = int(edge_used)
        if edge_used:
            union(left_id, right_id)
            compatible_edges.append((left_id, right_id, metrics | {"Strong": float(strong)}))

    components: dict[int, set[int]] = {}
    for nucleus_id in parent:
        components.setdefault(find(nucleus_id), set()).add(nucleus_id)
    edges_by_component: dict[int, list[tuple[int, int, dict[str, float]]]] = {}
    for left_id, right_id, metrics in compatible_edges:
        edges_by_component.setdefault(find(left_id), []).append(
            (left_id, right_id, metrics)
        )

    rescued: set[int] = set()
    review: set[int] = set()
    measurements: dict[int, dict[str, float]] = {
        int(nucleus_id): {
            "Structure v2 component": 0,
            "Structure v2 compatible neighbors": 0,
            "Structure v2 best Actin path fraction": np.nan,
            "Structure v2 best MF20 path fraction": np.nan,
            "Structure v2 best aligned path fraction": np.nan,
            "Structure v2 best maximum dark gap fraction": np.nan,
            "Structure v2 unanchored strong component": 0,
        }
        for nucleus_id in nuclei["Label ID"].astype(int)
    }
    corridor_mask = np.zeros(actin.shape, dtype=np.uint8)
    component_number = 0
    coordinates_by_id = nuclei.set_index("Label ID")[["X", "Y"]]
    for root, component in components.items():
        if len(component) < 2:
            continue
        component_edges = edges_by_component.get(root, [])
        strong_edges = [edge for edge in component_edges if edge[2]["Strong"] > 0]
        anchored = bool(component & anchor_ids)
        # Unanchored two-nucleus pairs remain review candidates because two
        # adjacent mononuclear myoblasts can closely mimic a short myotube.
        # Automatic unanchored rescue requires a redundant strong graph among
        # at least three nuclei.
        strong_unanchored = bool(
            allow_unanchored_auto
            and len(component) >= 3
            and len(strong_edges) >= len(component)
        )
        if anchored or strong_unanchored:
            rescued |= component - current_in
        else:
            review |= component - current_in
        component_number += 1
        for nucleus_id in component:
            item = measurements[nucleus_id]
            item["Structure v2 component"] = component_number
            item["Structure v2 unanchored strong component"] = int(
                strong_unanchored and not anchored
            )
        for left_id, right_id, metrics in component_edges:
            left_row = coordinates_by_id.loc[left_id]
            right_row = coordinates_by_id.loc[right_id]
            cv2.line(
                corridor_mask,
                (int(round(left_row["X"])), int(round(left_row["Y"]))),
                (int(round(right_row["X"])), int(round(right_row["Y"]))),
                255,
                max(3, corridor_radius * 2 + 1),
            )
            for nucleus_id in (left_id, right_id):
                item = measurements[nucleus_id]
                item["Structure v2 compatible neighbors"] += 1
                best = item["Structure v2 best Actin path fraction"]
                if not np.isfinite(best) or metrics["Actin path fraction"] > best:
                    item.update({
                        "Structure v2 best Actin path fraction": metrics["Actin path fraction"],
                        "Structure v2 best MF20 path fraction": metrics["MF20 path fraction"],
                        "Structure v2 best aligned path fraction": metrics["Aligned path fraction"],
                        "Structure v2 best maximum dark gap fraction": metrics["Maximum dark gap fraction"],
                    })
    return rescued, review, measurements, pd.DataFrame(records), corridor_mask.astype(bool)


def direction_guard(
    candidates: pd.DataFrame,
    group_labels: np.ndarray,
    actin: np.ndarray,
    mf20: np.ndarray,
    radius: int,
    max_angle: float,
) -> tuple[set[int], dict[int, dict[str, float]]]:
    """Flag two-nucleus groups with inconsistent local actin or MF20 direction."""
    actin_gradients = orientation_gradients(actin)
    mf20_gradients = orientation_gradients(mf20)
    rejected: set[int] = set()
    measurements: dict[int, dict[str, float]] = {}
    for group_id, group in candidates.groupby("Support group"):
        if len(group) != 2:
            continue
        allowed = group_labels == int(group_id)
        per_nucleus = []
        for _, row in group.iterrows():
            actin_angle, actin_coherence = local_fiber_orientation(
                actin_gradients, float(row["X"]), float(row["Y"]), radius, allowed
            )
            mf20_angle, mf20_coherence = local_fiber_orientation(
                mf20_gradients, float(row["X"]), float(row["Y"]), radius, allowed
            )
            per_nucleus.append((actin_angle, actin_coherence, mf20_angle, mf20_coherence))
        actin_difference = axial_angle_difference(per_nucleus[0][0], per_nucleus[1][0])
        mf20_difference = axial_angle_difference(per_nucleus[0][2], per_nucleus[1][2])
        finite = np.isfinite(actin_difference) and np.isfinite(mf20_difference)
        is_rejected = bool(finite and (actin_difference > max_angle or mf20_difference > max_angle))
        for (_, row), values in zip(group.iterrows(), per_nucleus):
            nucleus_id = int(row["Label ID"])
            measurements[nucleus_id] = {
                "Direction Actin angle difference (deg)": actin_difference,
                "Direction MF20 angle difference (deg)": mf20_difference,
                "Direction Actin coherence": values[1],
                "Direction MF20 coherence": values[3],
            }
            if is_rejected:
                rejected.add(nucleus_id)
    return rejected, measurements


def path_support_fraction(
    mask: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    radius: int,
    trim: float,
) -> float:
    """Measure signal continuity along the segment between two nuclei."""
    distance = float(np.hypot(x2 - x1, y2 - y1))
    samples = max(12, int(distance * 1.5))
    fractions = np.linspace(trim, 1 - trim, samples)
    height, width = mask.shape
    hits = []
    for fraction in fractions:
        x = int(round(x1 + (x2 - x1) * fraction))
        y = int(round(y1 + (y2 - y1) * fraction))
        x0, x1p = max(0, x - radius), min(width, x + radius + 1)
        y0, y1p = max(0, y - radius), min(height, y + radius + 1)
        hits.append(bool(mask[y0:y1p, x0:x1p].any()))
    return float(np.mean(hits))


def graph_guard(
    candidates: pd.DataFrame,
    group_labels: np.ndarray,
    actin: np.ndarray,
    mf20: np.ndarray,
    mf20_mask: np.ndarray,
    hybrid_mask: np.ndarray,
    direction_radius: int,
    max_distance: float,
    min_mf20_path: float,
    max_pair_angle: float,
    max_line_angle: float,
) -> tuple[set[int], dict[int, dict[str, float]], pd.DataFrame]:
    """Flag nuclei that have no signal-continuous, direction-compatible neighbor."""
    actin_gradients = orientation_gradients(actin)
    mf20_gradients = orientation_gradients(mf20)
    mf20_path_mask = binary_dilation(mf20_mask, disk(4))
    hybrid_path_mask = binary_dilation(hybrid_mask, disk(3))
    linked: set[int] = set()
    measurements: dict[int, dict[str, float]] = {}
    edge_records = []

    for group_id, group in candidates.groupby("Support group"):
        allowed = group_labels == int(group_id)
        local = {}
        for _, row in group.iterrows():
            nucleus_id = int(row["Label ID"])
            actin_angle, actin_coherence = local_fiber_orientation(
                actin_gradients, float(row["X"]), float(row["Y"]), direction_radius, allowed
            )
            mf20_angle, mf20_coherence = local_fiber_orientation(
                mf20_gradients, float(row["X"]), float(row["Y"]), direction_radius, allowed
            )
            local[nucleus_id] = (actin_angle, actin_coherence, mf20_angle, mf20_coherence)
            measurements[nucleus_id] = {
                "Graph compatible neighbor count": 0,
                "Graph best MF20 path fraction": np.nan,
                "Graph best hybrid path fraction": np.nan,
                "Graph best neighbor distance (px)": np.nan,
                "Graph best Actin pair angle difference (deg)": np.nan,
                "Graph best MF20 pair angle difference (deg)": np.nan,
                "Graph best Actin line angle difference (deg)": np.nan,
                "Graph best MF20 line angle difference (deg)": np.nan,
            }

        rows = list(group.iterrows())
        coordinates = group[["X", "Y"]].to_numpy(dtype=float)
        nearby_pairs = sorted(cKDTree(coordinates).query_pairs(max_distance))
        for left_position, right_position in nearby_pairs:
            _, left = rows[left_position]
            _, right = rows[right_position]
            left_id, right_id = int(left["Label ID"]), int(right["Label ID"])
            dx, dy = float(right["X"] - left["X"]), float(right["Y"] - left["Y"])
            distance = float(np.hypot(dx, dy))
            line_angle = float(np.degrees(np.arctan2(dy, dx)) % 180)
            left_values, right_values = local[left_id], local[right_id]
            actin_pair = axial_angle_difference(left_values[0], right_values[0])
            mf20_pair = axial_angle_difference(left_values[2], right_values[2])
            actin_line = max(
                axial_angle_difference(left_values[0], line_angle),
                axial_angle_difference(right_values[0], line_angle),
            )
            mf20_line = max(
                axial_angle_difference(left_values[2], line_angle),
                axial_angle_difference(right_values[2], line_angle),
            )
            mf20_path = path_support_fraction(
                mf20_path_mask, left["X"], left["Y"], right["X"], right["Y"],
                GRAPH_PATH_RADIUS, GRAPH_PATH_TRIM,
            )
            hybrid_path = path_support_fraction(
                hybrid_path_mask, left["X"], left["Y"], right["X"], right["Y"],
                GRAPH_PATH_RADIUS, GRAPH_PATH_TRIM,
            )
            direction_compatible = actin_pair <= max_pair_angle or mf20_pair <= max_pair_angle
            line_compatible = actin_line <= max_line_angle or mf20_line <= max_line_angle
            compatible = bool(
                mf20_path >= min_mf20_path and direction_compatible and line_compatible
            )
            edge_records.append({
                "Support group": int(group_id),
                "Nucleus A": left_id,
                "Nucleus B": right_id,
                "Distance (px)": distance,
                "MF20 path fraction": mf20_path,
                "Hybrid path fraction": hybrid_path,
                "Actin pair angle difference (deg)": actin_pair,
                "MF20 pair angle difference (deg)": mf20_pair,
                "Actin line angle difference (deg)": actin_line,
                "MF20 line angle difference (deg)": mf20_line,
                "Compatible edge": int(compatible),
            })
            if not compatible:
                continue
            linked.update((left_id, right_id))
            for nucleus_id, other_id in ((left_id, right_id), (right_id, left_id)):
                current = measurements[nucleus_id]
                current["Graph compatible neighbor count"] += 1
                best_path = current["Graph best MF20 path fraction"]
                if not np.isfinite(best_path) or mf20_path > best_path:
                    current.update({
                        "Graph best MF20 path fraction": mf20_path,
                        "Graph best hybrid path fraction": hybrid_path,
                        "Graph best neighbor distance (px)": distance,
                        "Graph best Actin pair angle difference (deg)": actin_pair,
                        "Graph best MF20 pair angle difference (deg)": mf20_pair,
                        "Graph best Actin line angle difference (deg)": actin_line,
                        "Graph best MF20 line angle difference (deg)": mf20_line,
                    })
    rejected = set(candidates["Label ID"].astype(int)) - linked
    return rejected, measurements, pd.DataFrame(edge_records)


def actin_structure_rescue(
    nuclei: pd.DataFrame,
    current_in: set[int],
    anchor_ids: set[int],
    review_anchor_ids: Optional[set[int]],
    actin: np.ndarray,
    mf20: np.ndarray,
    direction_radius: int,
    max_distance: float,
    min_coherence: float,
    max_pair_angle: float,
    max_line_angle: float,
    actin_threshold: int,
    mf20_threshold: int,
    min_actin_path: float,
    min_mf20_path: float,
) -> tuple[set[int], set[int], dict[int, dict[str, float]], pd.DataFrame]:
    """Rescue nuclei in anchored, direction-aligned actin graph components."""
    actin_gradients = orientation_gradients(actin)
    mf20_gradients = orientation_gradients(mf20)
    actin_allowed = binary_dilation(actin >= 12, disk(2))
    mf20_allowed = binary_dilation(mf20 >= 12, disk(2))
    actin_path_mask = binary_dilation(actin >= actin_threshold, disk(3))
    mf20_path_mask = binary_dilation(mf20 >= mf20_threshold, disk(3))
    local: dict[int, tuple[float, float, float, float]] = {}
    measurements: dict[int, dict[str, float]] = {}
    for _, row in nuclei.iterrows():
        nucleus_id = int(row["Label ID"])
        actin_angle, actin_coherence = local_fiber_orientation(
            actin_gradients,
            float(row["X"]),
            float(row["Y"]),
            direction_radius,
            actin_allowed,
        )
        mf20_angle, mf20_coherence = local_fiber_orientation(
            mf20_gradients,
            float(row["X"]),
            float(row["Y"]),
            direction_radius,
            mf20_allowed,
        )
        local[nucleus_id] = (
            actin_angle,
            actin_coherence,
            mf20_angle,
            mf20_coherence,
        )
        measurements[nucleus_id] = {
            "Actin structure component": 0,
            "Actin structure compatible neighbors": 0,
            "Actin structure local coherence": actin_coherence,
            "Actin structure best Actin path fraction": np.nan,
            "Actin structure best MF20 path fraction": np.nan,
            "Actin structure best neighbor distance (px)": np.nan,
        }

    parent: dict[int, int] = {}

    def find(nucleus_id: int) -> int:
        parent.setdefault(nucleus_id, nucleus_id)
        while parent[nucleus_id] != nucleus_id:
            parent[nucleus_id] = parent[parent[nucleus_id]]
            nucleus_id = parent[nucleus_id]
        return nucleus_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    edge_records = []
    rows = list(nuclei.iterrows())
    coordinates = nuclei[["X", "Y"]].to_numpy(dtype=float)
    for left_position, right_position in sorted(
        cKDTree(coordinates).query_pairs(max_distance)
    ):
        _, left = rows[left_position]
        _, right = rows[right_position]
        left_id, right_id = int(left["Label ID"]), int(right["Label ID"])
        dx, dy = float(right["X"] - left["X"]), float(right["Y"] - left["Y"])
        distance = float(np.hypot(dx, dy))
        line_angle = float(np.degrees(np.arctan2(dy, dx)) % 180)
        left_values, right_values = local[left_id], local[right_id]
        actin_pair = axial_angle_difference(left_values[0], right_values[0])
        actin_line = max(
            axial_angle_difference(left_values[0], line_angle),
            axial_angle_difference(right_values[0], line_angle),
        )
        coherence_values = [
            value for value in (left_values[1], right_values[1])
            if np.isfinite(value)
        ]
        actin_coherence = (
            float(min(coherence_values)) if coherence_values else np.nan
        )
        actin_path = path_support_fraction(
            actin_path_mask,
            left["X"], left["Y"], right["X"], right["Y"],
            GRAPH_PATH_RADIUS, 0.12,
        )
        mf20_path = path_support_fraction(
            mf20_path_mask,
            left["X"], left["Y"], right["X"], right["Y"],
            GRAPH_PATH_RADIUS, 0.12,
        )
        compatible = bool(
            np.isfinite(actin_coherence)
            and np.isfinite(actin_pair)
            and np.isfinite(actin_line)
            and actin_coherence >= min_coherence
            and actin_pair <= max_pair_angle
            and actin_line <= max_line_angle
            and actin_path >= min_actin_path
            and mf20_path >= min_mf20_path
        )
        edge_records.append({
            "Nucleus A": left_id,
            "Nucleus B": right_id,
            "Distance (px)": distance,
            "Actin coherence minimum": actin_coherence,
            "Actin pair angle difference (deg)": actin_pair,
            "Actin line angle difference (deg)": actin_line,
            "Actin path fraction": actin_path,
            "MF20 path fraction": mf20_path,
            "Compatible edge": int(compatible),
        })
        if not compatible:
            continue
        union(left_id, right_id)
        for nucleus_id in (left_id, right_id):
            item = measurements[nucleus_id]
            item["Actin structure compatible neighbors"] += 1
            best = item["Actin structure best Actin path fraction"]
            if not np.isfinite(best) or actin_path > best:
                item.update({
                    "Actin structure best Actin path fraction": actin_path,
                    "Actin structure best MF20 path fraction": mf20_path,
                    "Actin structure best neighbor distance (px)": distance,
                })

    components: dict[int, set[int]] = {}
    for nucleus_id in parent:
        components.setdefault(find(nucleus_id), set()).add(nucleus_id)
    rescued: set[int] = set()
    anchor_review_candidates: set[int] = set()
    component_number = 0
    for component in components.values():
        if len(component) < 2:
            continue
        if not (component & anchor_ids):
            if review_anchor_ids and component & review_anchor_ids:
                anchor_review_candidates |= component - current_in
            continue
        component_number += 1
        for nucleus_id in component:
            measurements[nucleus_id]["Actin structure component"] = component_number
        rescued |= component - current_in
    return (
        rescued,
        anchor_review_candidates,
        measurements,
        pd.DataFrame(edge_records),
    )


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def save_preview(
    path: Path,
    overlay: np.ndarray,
    include_weak_rescue: bool = False,
    include_actin_structure: bool = False,
) -> None:
    """Save a presentation-ready PNG with an explicit color legend."""
    height, width = overlay.shape[:2]
    canvas = Image.new("RGB", (width, height + 58), "white")
    canvas.paste(Image.fromarray(overlay), (0, 58))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    legend = [
        (
            (0, 255, 0),
            (
                "automatic In (main or Actin-structure)"
                if include_actin_structure
                else "main >=2-nucleus myotube"
            ),
        ),
        (
            (255, 220, 0),
            "weak-signal included" if include_weak_rescue else "excluded review candidate",
        ),
        ((255, 0, 0), "Out"),
        ((0, 210, 255), "retained myotube boundary"),
    ]
    x = 12
    for color, text in legend:
        draw.line((x, 19, x + 24, 19), fill=color, width=4)
        draw.text((x + 30, 12), text, fill="black", font=font)
        x += 30 + draw.textlength(text, font=font) + 18
    yellow_note = (
        "Yellow nuclei are included in FI." if include_weak_rescue
        else "Yellow nuclei are review candidates and are not counted in FI."
    )
    draw.text((12, 37), yellow_note, fill="black", font=font)
    canvas.save(path)


def run(args: argparse.Namespace) -> int:
    source = args.input.expanduser().resolve()
    results = args.myofuse_results.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    image, tile_index = open_analysis_scene(source, args.scene)
    channel_names = list(image.channel_names)
    for channel in (args.nuclei_channel, args.actin_channel, args.mf20_channel):
        if channel < 0 or channel >= len(channel_names):
            raise ValueError(f"Channel {channel} out of range; available: {channel_names}")
    if image.dims.T != 1 or image.dims.Z != 1:
        raise ValueError("T/Z stacks require an explicit projection before hybrid analysis")

    dapi = scale(read_channel(image, args.nuclei_channel, tile_index))
    actin = scale(read_channel(image, args.actin_channel, tile_index))
    mf20 = scale(read_channel(image, args.mf20_channel, tile_index))
    channel_quality, channel_warning = channel_quality_table(
        image,
        tile_index,
        dapi,
        channel_names,
        args.nuclei_channel,
        args.actin_channel,
        args.mf20_channel,
        args.mf20_threshold,
    )
    if channel_warning and not args.allow_mf20_qc_warning:
        raise ValueError(channel_warning)
    if channel_warning:
        print(f"WARNING: {channel_warning}", file=sys.stderr)
    stem = safe_name(source.stem + "__" + args.scene)
    label_path = args.label_mask or (results / f"{stem}_mask.tif")
    labels = imread(label_path).astype(np.int32)
    if labels.shape != dapi.shape:
        raise ValueError(f"Label mask shape {labels.shape} differs from image {dapi.shape}")

    predictions = pd.read_excel(results / "Nucleus_predictions.xlsx")
    if {"File", "Scene"}.issubset(predictions.columns):
        selected = predictions[(predictions["File"] == source.name) & (predictions["Scene"] == args.scene)]
        if selected.empty:
            raise ValueError(
                f"Nucleus_predictions.xlsx contains no rows for File={source.name!r}, "
                f"Scene={args.scene!r}"
            )
        predictions = selected.copy()
    required = {"Label ID", "Original Class"}
    if not required.issubset(predictions.columns):
        raise ValueError("Nucleus_predictions.xlsx must contain Label ID and Original Class")
    if predictions["Label ID"].duplicated().any():
        duplicates = sorted(
            predictions.loc[predictions["Label ID"].duplicated(keep=False), "Label ID"]
            .astype(int).unique().tolist()
        )
        raise ValueError(f"Nucleus_predictions.xlsx contains duplicate Label IDs: {duplicates[:10]}")
    if args.hybrid_only_probability_min > 0 and "Model Probability (In)" not in predictions.columns:
        raise ValueError(
            "Conservative mode requires Model Probability (In) in Nucleus_predictions.xlsx"
        )
    nucleus_ids = predictions["Label ID"].astype(int).to_numpy()
    label_ids = set(np.unique(labels)) - {0}
    if set(nucleus_ids) != label_ids:
        raise ValueError("Prediction IDs do not match the selected label mask and scene")

    mf20_mask = make_mask(mf20, dapi, args.mf20_threshold)
    actin_mask = make_mask(actin, dapi, args.actin_threshold)
    mf20_mask_qc_fallback = mask_qc_fallback_active(
        mf20, dapi, args.mf20_threshold
    )
    actin_mask_qc_fallback = mask_qc_fallback_active(
        actin, dapi, args.actin_threshold
    )
    hybrid = mf20_mask | (actin_mask & binary_dilation(mf20_mask, disk(args.actin_support_radius)))

    support = binary_dilation(hybrid, disk(args.nucleus_support_radius))
    _, support_groups = cv2.connectedComponents(support.astype(np.uint8), 8)
    main = nucleus_metrics(labels, nucleus_ids, support, support_groups).rename(
        columns={"Overlap": "Support overlap", "Group": "Support group"}
    )
    main_candidates = main[
        (main["Support group"] > 0) & (main["Support overlap"] >= args.nucleus_support_overlap)
    ]
    main_counts = main_candidates.groupby("Support group")["Label ID"].count()
    retained_main_groups = set(main_counts[main_counts >= args.min_nuclei].index.astype(int))
    signal_supported_in = set(
        main_candidates.loc[main_candidates["Support group"].isin(retained_main_groups), "Label ID"].astype(int)
    )
    direction_guard_candidates: set[int] = set()
    direction_measurements: dict[int, dict[str, float]] = {}
    graph_guard_candidates: set[int] = set()
    graph_measurements: dict[int, dict[str, float]] = {}
    graph_edges = pd.DataFrame()
    direction_guard_enabled = (
        args.direction_aware or args.actin_structure_rescue or args.structure_v2
    )
    retained_candidates = main_candidates[
        main_candidates["Support group"].isin(retained_main_groups)
    ]
    if direction_guard_enabled:
        direction_guard_candidates, direction_measurements = direction_guard(
            retained_candidates,
            support_groups,
            actin,
            mf20,
            args.direction_radius,
            args.direction_max_angle,
        )
    if args.graph_aware:
        graph_guard_candidates, graph_measurements, graph_edges = graph_guard(
            retained_candidates,
            support_groups,
            actin,
            mf20,
            mf20_mask,
            hybrid,
            args.direction_radius,
            args.graph_max_distance,
            args.graph_min_mf20_path,
            args.graph_max_pair_angle,
            args.graph_max_line_angle,
        )
    guarded_in = signal_supported_in - direction_guard_candidates - graph_guard_candidates
    prediction_by_id = predictions.set_index("Label ID")
    original_in_ids = set(
        predictions.loc[predictions["Original Class"].eq(1), "Label ID"].astype(int)
    )
    myofuse_original_fraction = len(original_in_ids) / len(nucleus_ids)
    if args.hybrid_only_probability_min > 0:
        main_in = {
            nucleus_id for nucleus_id in guarded_in
            if int(prediction_by_id.at[nucleus_id, "Original Class"]) == 1
            or float(prediction_by_id.at[nucleus_id, "Model Probability (In)"])
            >= args.hybrid_only_probability_min
        }
    else:
        main_in = set(guarded_in)
    probability_guard_candidates = guarded_in - main_in

    weak_support = binary_dilation(hybrid, disk(args.weak_support_radius))
    _, weak_groups = cv2.connectedComponents(weak_support.astype(np.uint8), 8)
    weak = nucleus_metrics(labels, nucleus_ids, weak_support, weak_groups).rename(
        columns={"Overlap": "Weak support overlap", "Group": "Weak group"}
    )
    audit = main.merge(weak[["Label ID", "Weak support overlap", "Weak group"]], on="Label ID")
    prediction_columns = [
        name for name in ("Label ID", "Original Class", "Final Class", "Model Probability (In)")
        if name in predictions.columns
    ]
    audit = audit.merge(predictions[prediction_columns], on="Label ID", how="left")
    weak_candidates = audit[
        (audit["Weak group"] > 0) & (audit["Weak support overlap"] >= args.weak_support_overlap)
    ].copy()
    rescued_ids = set()
    rescue_cluster_id = {}
    cluster_counter = 0
    for cluster in connected_nucleus_clusters(weak_candidates, args.weak_cluster_distance):
        cluster_frame = weak_candidates.loc[list(cluster)]
        myofuse_in_count = int((cluster_frame["Original Class"] == 1).sum())
        if len(cluster_frame) >= args.min_nuclei and myofuse_in_count >= args.min_myofuse_in:
            cluster_counter += 1
            for nucleus_id in cluster_frame["Label ID"].astype(int):
                rescued_ids.add(nucleus_id)
                rescue_cluster_id[nucleus_id] = cluster_counter

    weak_only = rescued_ids - main_in
    final_in = main_in | weak_only if args.include_weak_rescue else set(main_in)
    final_in -= direction_guard_candidates
    final_in -= graph_guard_candidates

    actin_structure_rescued: set[int] = set()
    actin_structure_anchor_review_candidates: set[int] = set()
    low_support_review_candidates: set[int] = set()
    neighbor_review_candidates: set[int] = set()
    nearest_by_id: dict[int, float] = {}
    actin_structure_measurements: dict[int, dict[str, float]] = {}
    actin_structure_edges = pd.DataFrame()
    structure_v2_rescued: set[int] = set()
    structure_v2_review_candidates: set[int] = set()
    structure_v2_measurements: dict[int, dict[str, float]] = {}
    structure_v2_edges = pd.DataFrame()
    structure_v2_corridor_mask = np.zeros(actin.shape, dtype=bool)
    structure_v2_low_myofuse_guard_candidate = bool(
        args.structure_v2
        and myofuse_original_fraction
        < args.actin_structure_low_myofuse_fi_threshold
        and len(main_in) / len(nucleus_ids)
        < args.actin_structure_low_myofuse_fi_threshold
    )
    structure_v2_low_myofuse_guard_active = False
    structure_rescue_share = 0.0
    low_myofuse_guard_active = bool(
        args.actin_structure_rescue
        and myofuse_original_fraction
        < args.actin_structure_low_myofuse_fi_threshold
    )
    if args.actin_structure_rescue:
        actin_anchor_ids = (
            original_in_ids
            if low_myofuse_guard_active
            else final_in | original_in_ids
        )
        review_anchor_ids = (
            final_in | original_in_ids if low_myofuse_guard_active else None
        )
        (
            actin_structure_rescued,
            actin_structure_anchor_review_candidates,
            actin_structure_measurements,
            actin_structure_edges,
        ) = actin_structure_rescue(
            audit,
            final_in,
            actin_anchor_ids,
            review_anchor_ids,
            actin,
            mf20,
            args.actin_structure_direction_radius,
            args.actin_structure_max_distance,
            args.actin_structure_min_coherence,
            args.actin_structure_max_pair_angle,
            args.actin_structure_max_line_angle,
            args.actin_structure_actin_threshold,
            args.actin_structure_mf20_threshold,
            args.actin_structure_min_actin_path,
            args.actin_structure_min_mf20_path,
        )
        final_in |= actin_structure_rescued
        if low_myofuse_guard_active:
            support_by_id = audit.set_index("Label ID")["Support overlap"]
            low_support_review_candidates = {
                nucleus_id
                for nucleus_id in final_in
                if float(support_by_id.get(nucleus_id, 0.0))
                < args.actin_structure_low_support_review_overlap
            }
            final_in -= low_support_review_candidates
        elif len(audit) > 1:
            nearest_distances = cKDTree(
                audit[["X", "Y"]].to_numpy(dtype=float)
            ).query(audit[["X", "Y"]].to_numpy(dtype=float), k=2)[0][:, 1]
            nearest_by_id = dict(
                zip(audit["Label ID"].astype(int), nearest_distances)
            )
            neighbor_review_candidates = set(
                audit.loc[
                    ~audit["Label ID"].isin(final_in)
                    & audit["Model Probability (In)"].ge(
                        args.actin_structure_neighbor_review_probability
                    )
                    & audit["Label ID"].map(nearest_by_id).le(
                        args.actin_structure_neighbor_review_distance
                    ),
                    "Label ID",
                ].astype(int)
            )

    if args.structure_v2:
        (
            structure_v2_rescued,
            structure_v2_review_candidates,
            structure_v2_measurements,
            structure_v2_edges,
            structure_v2_corridor_mask,
        ) = structure_v2_rescue(
            audit,
            final_in,
            final_in | original_in_ids,
            actin,
            mf20,
            args.structure_v2_max_distance,
            args.structure_v2_local_sigma,
            args.structure_v2_tensor_sigma,
            args.structure_v2_corridor_radius,
            args.structure_v2_min_coherence,
            args.structure_v2_max_line_angle,
            args.structure_v2_min_actin_path,
            args.structure_v2_min_mf20_path,
            args.structure_v2_min_aligned_path,
            args.structure_v2_max_dark_gap,
            False,
            True,
        )
        structure_candidate_total = (
            len(structure_v2_rescued) + len(structure_v2_review_candidates)
        )
        structure_rescue_share = (
            len(structure_v2_rescued) / structure_candidate_total
            if structure_candidate_total else 0.0
        )
        structure_v2_low_myofuse_guard_active = bool(
            structure_v2_low_myofuse_guard_candidate
            and structure_rescue_share < 0.50
        )
        if structure_v2_low_myofuse_guard_active:
            # In images where neither MyoFuse nor the confidence-guarded main
            # stage finds a reliable anchor population, aligned mononuclear
            # cells can imitate a myotube graph. Keep Structure v2 findings in
            # the yellow review queue instead of changing automatic FI.
            structure_v2_review_candidates |= structure_v2_rescued
            structure_v2_rescued = set()
            structure_v2_corridor_mask[:] = False
            support_by_id = audit.set_index("Label ID")["Support overlap"]
            structure_low_support = {
                nucleus_id
                for nucleus_id in final_in
                if float(support_by_id.get(nucleus_id, 0.0))
                < args.actin_structure_low_support_review_overlap
            }
            low_support_review_candidates |= structure_low_support
            final_in -= structure_low_support
        else:
            final_in |= structure_v2_rescued
        structure_v2_review_candidates -= final_in
        if len(structure_v2_review_candidates) > args.structure_v2_review_max_candidates:
            review_rank = audit[
                audit["Label ID"].isin(structure_v2_review_candidates)
            ].copy()
            review_rank["Structure aligned rank"] = review_rank["Label ID"].map(
                {
                    nucleus_id: values[
                        "Structure v2 best aligned path fraction"
                    ]
                    for nucleus_id, values in structure_v2_measurements.items()
                }
            ).fillna(0)
            review_rank["Structure MF20 rank"] = review_rank["Label ID"].map(
                {
                    nucleus_id: values[
                        "Structure v2 best MF20 path fraction"
                    ]
                    for nucleus_id, values in structure_v2_measurements.items()
                }
            ).fillna(0)
            review_rank = review_rank.sort_values(
                [
                    "Model Probability (In)",
                    "Structure aligned rank",
                    "Structure MF20 rank",
                    "Support overlap",
                ],
                ascending=False,
            )
            structure_v2_review_candidates = set(
                review_rank.head(args.structure_v2_review_max_candidates)[
                    "Label ID"
                ].astype(int)
            )

    adaptive_review_candidates: set[int] = set()
    adaptive_measurements: dict[int, dict[str, float]] = {}
    adaptive_edges = pd.DataFrame()
    if args.adaptive_review:
        adaptive_mf20_mask = make_mask(mf20, dapi, args.adaptive_review_mf20_threshold)
        adaptive_hybrid = adaptive_mf20_mask | (
            actin_mask
            & binary_dilation(adaptive_mf20_mask, disk(args.actin_support_radius))
        )
        adaptive_support = binary_dilation(
            adaptive_hybrid, disk(args.nucleus_support_radius)
        )
        _, adaptive_groups = cv2.connectedComponents(
            adaptive_support.astype(np.uint8), 8
        )
        adaptive_main = nucleus_metrics(
            labels, nucleus_ids, adaptive_support, adaptive_groups
        ).rename(
            columns={
                "Overlap": "Adaptive support overlap",
                "Group": "Adaptive support group",
            }
        )
        adaptive_pool = adaptive_main[
            (adaptive_main["Adaptive support group"] > 0)
            & (
                adaptive_main["Adaptive support overlap"]
                >= args.nucleus_support_overlap
            )
        ].copy()
        adaptive_counts = adaptive_pool.groupby("Adaptive support group")[
            "Label ID"
        ].count()
        adaptive_groups_with_multiple_nuclei = set(
            adaptive_counts[
                adaptive_counts >= args.min_nuclei
            ].index.astype(int)
        )
        adaptive_pool = adaptive_pool[
            adaptive_pool["Adaptive support group"].isin(
                adaptive_groups_with_multiple_nuclei
            )
        ].rename(columns={"Adaptive support group": "Support group"})
        adaptive_rejected, adaptive_measurements, adaptive_edges = graph_guard(
            adaptive_pool,
            adaptive_groups,
            actin,
            mf20,
            adaptive_mf20_mask,
            adaptive_hybrid,
            args.direction_radius,
            args.adaptive_review_max_distance,
            args.adaptive_review_min_mf20_path,
            args.adaptive_review_max_pair_angle,
            args.graph_max_line_angle,
        )
        adaptive_linked = (
            set(adaptive_pool["Label ID"].astype(int)) - adaptive_rejected - final_in
        )
        # Rank within each image rather than treating the published MyoFuse
        # probability as an absolute cross-image calibration. Probability is
        # used only to prioritize the bounded manual review queue.
        adaptive_rank = audit[audit["Label ID"].isin(adaptive_linked)].copy()
        adaptive_rank["Adaptive graph neighbor count"] = adaptive_rank[
            "Label ID"
        ].map(
            {
                nucleus_id: values["Graph compatible neighbor count"]
                for nucleus_id, values in adaptive_measurements.items()
            }
        ).fillna(0)
        adaptive_rank = adaptive_rank.sort_values(
            [
                "Model Probability (In)",
                "Adaptive graph neighbor count",
                "Weak support overlap",
            ],
            ascending=False,
        )
        adaptive_review_candidates = set(
            adaptive_rank.head(args.adaptive_review_max_candidates)[
                "Label ID"
            ].astype(int)
        )
    audit["Main group nucleus count"] = audit["Support group"].map(main_counts).fillna(0).astype(int)
    audit["Signal-supported hybrid class"] = audit["Label ID"].isin(signal_supported_in).astype(int)
    for column in (
        "Direction Actin angle difference (deg)",
        "Direction MF20 angle difference (deg)",
        "Direction Actin coherence",
        "Direction MF20 coherence",
    ):
        audit[column] = audit["Label ID"].map(
            {nucleus_id: values[column] for nucleus_id, values in direction_measurements.items()}
        )
    audit["Direction-aware review candidate"] = (
        audit["Label ID"].isin(direction_guard_candidates).astype(int)
    )
    for column in (
        "Graph compatible neighbor count",
        "Graph best MF20 path fraction",
        "Graph best hybrid path fraction",
        "Graph best neighbor distance (px)",
        "Graph best Actin pair angle difference (deg)",
        "Graph best MF20 pair angle difference (deg)",
        "Graph best Actin line angle difference (deg)",
        "Graph best MF20 line angle difference (deg)",
    ):
        audit[column] = audit["Label ID"].map(
            {nucleus_id: values[column] for nucleus_id, values in graph_measurements.items()}
        )
    audit["Graph-aware review candidate"] = (
        audit["Label ID"].isin(graph_guard_candidates).astype(int)
    )
    audit["Probability-guard rejected"] = (
        audit["Label ID"].isin(probability_guard_candidates).astype(int)
    )
    audit["Main-stage In"] = audit["Label ID"].isin(main_in).astype(int)
    audit["Weak rescue cluster"] = audit["Label ID"].map(rescue_cluster_id).fillna(0).astype(int)
    audit["Weak-signal review candidate"] = audit["Label ID"].isin(weak_only).astype(int)
    audit["Weak candidate included in FI"] = (
        audit["Label ID"].isin(weak_only).astype(int) if args.include_weak_rescue else 0
    )
    audit["Adaptive weak-signal review candidate"] = (
        audit["Label ID"].isin(adaptive_review_candidates).astype(int)
    )
    audit["Adaptive graph compatible neighbor count"] = audit["Label ID"].map(
        {
            nucleus_id: values["Graph compatible neighbor count"]
            for nucleus_id, values in adaptive_measurements.items()
        }
    )
    for column in (
        "Actin structure component",
        "Actin structure compatible neighbors",
        "Actin structure local coherence",
        "Actin structure best Actin path fraction",
        "Actin structure best MF20 path fraction",
        "Actin structure best neighbor distance (px)",
    ):
        audit[column] = audit["Label ID"].map(
            {
                nucleus_id: values[column]
                for nucleus_id, values in actin_structure_measurements.items()
            }
        )
    audit["Actin structure rescued"] = (
        audit["Label ID"].isin(actin_structure_rescued).astype(int)
    )
    audit["Actin structure Hybrid-anchor review candidate"] = (
        audit["Label ID"]
        .isin(actin_structure_anchor_review_candidates)
        .astype(int)
    )
    audit["Low-MyoFuse support review candidate"] = (
        audit["Label ID"].isin(low_support_review_candidates).astype(int)
    )
    audit["Nearest nucleus distance (px)"] = audit["Label ID"].map(nearest_by_id)
    audit["High-probability neighbor review candidate"] = (
        audit["Label ID"].isin(neighbor_review_candidates).astype(int)
    )
    for column in (
        "Structure v2 component",
        "Structure v2 compatible neighbors",
        "Structure v2 best Actin path fraction",
        "Structure v2 best MF20 path fraction",
        "Structure v2 best aligned path fraction",
        "Structure v2 best maximum dark gap fraction",
        "Structure v2 unanchored strong component",
    ):
        audit[column] = audit["Label ID"].map(
            {
                nucleus_id: values[column]
                for nucleus_id, values in structure_v2_measurements.items()
            }
        )
    audit["Structure v2 rescued"] = (
        audit["Label ID"].isin(structure_v2_rescued).astype(int)
    )
    audit["Structure v2 review candidate"] = (
        audit["Label ID"].isin(structure_v2_review_candidates).astype(int)
    )
    audit["Automatic final class"] = audit["Label ID"].isin(final_in).astype(int)
    audit["Decision"] = np.select(
        [
            audit["Low-MyoFuse support review candidate"].eq(1),
            audit["Actin structure Hybrid-anchor review candidate"].eq(1),
            audit["Graph-aware review candidate"].eq(1),
            audit["Direction-aware review candidate"].eq(1),
            audit["Probability-guard rejected"].eq(1),
            audit["Adaptive weak-signal review candidate"].eq(1),
            audit["High-probability neighbor review candidate"].eq(1),
            audit["Structure v2 review candidate"].eq(1),
            audit["Main-stage In"].eq(1) & audit["Automatic final class"].eq(1),
            audit["Structure v2 rescued"].eq(1)
            & audit["Automatic final class"].eq(1),
            audit["Actin structure rescued"].eq(1)
            & audit["Automatic final class"].eq(1),
            audit["Weak-signal review candidate"].eq(1),
        ],
        [
            "Review: low support in a low-MyoFuse image, excluded from FI",
            "Review: Actin component has only a Hybrid anchor, excluded from FI",
            "Review: no compatible signal-continuous graph neighbor, excluded from FI",
            "Review: inconsistent local fiber direction, excluded from FI",
            "Out: hybrid-only nucleus below the MyoFuse confidence guard",
            "Review: adaptive weak-signal graph candidate, excluded from FI",
            "Review: nearby high-probability MyoFuse candidate, excluded from FI",
            "Review: Structure v2 corridor without sufficient anchor/strength, excluded from FI",
            "In: >=2-nucleus supported myotube",
            "In: Structure v2 oriented-corridor rescue",
            "In: anchored Actin-structure graph rescue",
            (
                "In: validated weak-signal rule explicitly included"
                if args.include_weak_rescue
                else "Review: weak-signal candidate, excluded from FI"
            ),
        ],
        default="Out",
    )

    overlay = np.repeat(mf20[..., None], 3, axis=2)
    nucleus_boundary = find_boundaries(labels, mode="inner")
    lookup = np.zeros(labels.max() + 1, np.uint8)
    review_candidates = (
        weak_only
        | direction_guard_candidates
        | graph_guard_candidates
        | adaptive_review_candidates
        | actin_structure_anchor_review_candidates
        | low_support_review_candidates
        | neighbor_review_candidates
        | structure_v2_review_candidates
    ) - final_in
    lookup[list(final_in)] = 1
    lookup[list(review_candidates)] = 2
    nucleus_class = lookup[labels]
    overlay[nucleus_boundary & (nucleus_class == 0)] = (255, 0, 0)
    overlay[nucleus_boundary & (nucleus_class == 1)] = (0, 255, 0)
    overlay[nucleus_boundary & (nucleus_class == 2)] = (255, 220, 0)
    counted_main_groups = set(
        main.loc[main["Label ID"].isin(main_in), "Support group"].astype(int)
    )
    retained_support = np.isin(support_groups, list(counted_main_groups))
    if args.structure_v2:
        retained_support |= structure_v2_corridor_mask
    overlay[find_boundaries(retained_support, mode="inner")] = (0, 210, 255)

    tif_path = output / f"{stem}__hybrid_nucleus_aware_overlay.tif"
    png_path = output / f"{stem}__hybrid_nucleus_aware_preview.png"
    mask_path = output / f"{stem}__hybrid_nucleus_aware_mask.tif"
    excel_path = output / f"{stem}__hybrid_nucleus_aware_results.xlsx"
    imwrite(tif_path, overlay, photometric="rgb")
    imwrite(mask_path, retained_support.astype(np.uint8) * 255)
    save_preview(
        png_path,
        overlay,
        args.include_weak_rescue,
        args.actin_structure_rescue,
    )

    summary = pd.DataFrame([{
        "File": source.name,
        "Scene": args.scene,
        "Total nuclei": len(nucleus_ids),
        "MyoFuse original nuclei In": int(predictions["Original Class"].eq(1).sum()),
        "MyoFuse original Fusion Index (%)": 100 * float(predictions["Original Class"].eq(1).mean()),
        "MF20 mask QC fallback active": mf20_mask_qc_fallback,
        "Actin mask QC fallback active": actin_mask_qc_fallback,
        "MyoFuse final nuclei In": (
            int(predictions["Final Class"].eq(1).sum())
            if "Final Class" in predictions.columns else np.nan
        ),
        "MyoFuse final Fusion Index (%)": (
            100 * float(predictions["Final Class"].eq(1).mean())
            if "Final Class" in predictions.columns else np.nan
        ),
        "Signal-supported hybrid nuclei In": len(signal_supported_in),
        "Direction-aware review candidates": len(direction_guard_candidates),
        "Graph-aware review candidates": len(graph_guard_candidates),
        "Probability-guard rejected nuclei": len(probability_guard_candidates),
        "Main-stage nuclei In": len(main_in),
        "Weak-signal review candidates": len(weak_only),
        "Adaptive weak-signal review candidates": len(adaptive_review_candidates),
        "Actin structure rescued nuclei": len(actin_structure_rescued),
        "Actin structure Hybrid-anchor review candidates": len(
            actin_structure_anchor_review_candidates
        ),
        "Low-MyoFuse support review candidates": len(
            low_support_review_candidates
        ),
        "High-probability neighbor review candidates": len(
            neighbor_review_candidates
        ),
        "Structure v2 rescued nuclei": len(structure_v2_rescued),
        "Structure v2 review candidates": len(structure_v2_review_candidates),
        "Structure v2 low-MyoFuse guard active": (
            structure_v2_low_myofuse_guard_active
        ),
        "Structure v2 rescue share before guard": structure_rescue_share,
        "Low-MyoFuse adaptive guard active": low_myofuse_guard_active,
        "Weak candidates included in FI": len(weak_only) if args.include_weak_rescue else 0,
        "Automatic final nuclei In": len(final_in),
        "Automatic nuclei Out": len(nucleus_ids) - len(final_in),
        "Automatic Fusion Index (%)": 100 * len(final_in) / len(nucleus_ids),
    }])
    parameters = pd.DataFrame([
        ("Nuclei channel", f"C{args.nuclei_channel}: {channel_names[args.nuclei_channel]}"),
        ("Actin channel", f"C{args.actin_channel}: {channel_names[args.actin_channel]}"),
        ("MF20 channel", f"C{args.mf20_channel}: {channel_names[args.mf20_channel]}"),
        ("MF20 threshold (0-255)", args.mf20_threshold),
        ("MF20 mask QC fallback active", mf20_mask_qc_fallback),
        ("MF20 channel QC warning", channel_warning or "None"),
        ("Actin threshold (0-255)", args.actin_threshold),
        ("Actin mask QC fallback active", actin_mask_qc_fallback),
        ("Mask QC maximum processed fraction", MASK_QC_MAX_PROCESSED_FRACTION),
        ("Mask QC maximum conservative fraction", MASK_QC_MAX_RAW_FRACTION),
        ("Actin support radius (px)", args.actin_support_radius),
        ("Main nucleus support radius (px)", args.nucleus_support_radius),
        ("Main nucleus overlap threshold", args.nucleus_support_overlap),
        ("Minimum nuclei per group", args.min_nuclei),
        ("Direction-aware guard enabled", direction_guard_enabled),
        ("Direction radius (px)", args.direction_radius),
        ("Direction maximum angle difference (deg)", args.direction_max_angle),
        ("Graph-aware guard enabled", args.graph_aware),
        ("Graph maximum neighbor distance (px)", args.graph_max_distance),
        ("Graph minimum MF20 path fraction", args.graph_min_mf20_path),
        ("Graph maximum pair angle difference (deg)", args.graph_max_pair_angle),
        ("Graph maximum line angle difference (deg)", args.graph_max_line_angle),
        ("Hybrid-only probability minimum", args.hybrid_only_probability_min),
        ("Include weak candidates in FI", args.include_weak_rescue),
        ("Weak support radius (px)", args.weak_support_radius),
        ("Weak support overlap threshold", args.weak_support_overlap),
        ("Weak local cluster distance (px)", args.weak_cluster_distance),
        ("Minimum MyoFuse In per weak cluster", args.min_myofuse_in),
        ("Adaptive weak-signal review enabled", args.adaptive_review),
        ("Adaptive review MF20 threshold (0-255)", args.adaptive_review_mf20_threshold),
        ("Adaptive review maximum neighbor distance (px)", args.adaptive_review_max_distance),
        ("Adaptive review minimum MF20 path fraction", args.adaptive_review_min_mf20_path),
        ("Adaptive review maximum pair angle difference (deg)", args.adaptive_review_max_pair_angle),
        ("Adaptive review maximum candidates", args.adaptive_review_max_candidates),
        ("Actin structure rescue enabled", args.actin_structure_rescue),
        ("Structure v2 enabled", args.structure_v2),
        ("Structure v2 maximum distance (px)", args.structure_v2_max_distance),
        ("Structure v2 local normalization sigma", args.structure_v2_local_sigma),
        ("Structure v2 tensor sigma", args.structure_v2_tensor_sigma),
        ("Structure v2 corridor radius (px)", args.structure_v2_corridor_radius),
        ("Structure v2 minimum coherence", args.structure_v2_min_coherence),
        ("Structure v2 maximum line angle (deg)", args.structure_v2_max_line_angle),
        ("Structure v2 minimum Actin path", args.structure_v2_min_actin_path),
        ("Structure v2 minimum MF20 path", args.structure_v2_min_mf20_path),
        ("Structure v2 minimum aligned path", args.structure_v2_min_aligned_path),
        ("Structure v2 maximum dark gap", args.structure_v2_max_dark_gap),
        (
            "Structure v2 maximum review candidates",
            args.structure_v2_review_max_candidates,
        ),
        (
            "Structure v2 low-MyoFuse guard active",
            structure_v2_low_myofuse_guard_active,
        ),
        ("Structure v2 rescue share before guard", structure_rescue_share),
        ("Actin structure direction radius (px)", args.actin_structure_direction_radius),
        ("Actin structure maximum neighbor distance (px)", args.actin_structure_max_distance),
        ("Actin structure minimum coherence", args.actin_structure_min_coherence),
        ("Actin structure maximum pair angle difference (deg)", args.actin_structure_max_pair_angle),
        ("Actin structure maximum line angle difference (deg)", args.actin_structure_max_line_angle),
        ("Actin structure Actin threshold (0-255)", args.actin_structure_actin_threshold),
        ("Actin structure MF20 threshold (0-255)", args.actin_structure_mf20_threshold),
        ("Actin structure minimum Actin path fraction", args.actin_structure_min_actin_path),
        ("Actin structure minimum MF20 path fraction", args.actin_structure_min_mf20_path),
        (
            "Actin structure low-MyoFuse FI threshold",
            args.actin_structure_low_myofuse_fi_threshold,
        ),
        (
            "Actin structure low-support review overlap threshold",
            args.actin_structure_low_support_review_overlap,
        ),
        (
            "Actin structure neighbor-review probability threshold",
            args.actin_structure_neighbor_review_probability,
        ),
        (
            "Actin structure neighbor-review distance (px)",
            args.actin_structure_neighbor_review_distance,
        ),
        ("Low-MyoFuse adaptive guard active", low_myofuse_guard_active),
    ], columns=["Parameter", "Value"])
    with pd.ExcelWriter(excel_path) as writer:
        summary.to_excel(writer, index=False, sheet_name="Summary")
        parameters.to_excel(writer, index=False, sheet_name="Parameters")
        audit.sort_values("Label ID").to_excel(writer, index=False, sheet_name="Per nucleus")
        channel_quality.to_excel(writer, index=False, sheet_name="Channel QC")
        if args.graph_aware:
            graph_edges.to_excel(writer, index=False, sheet_name="Graph edges")
        if args.adaptive_review:
            adaptive_edges.to_excel(writer, index=False, sheet_name="Adaptive review edges")
        if args.actin_structure_rescue:
            actin_structure_edges.to_excel(
                writer, index=False, sheet_name="Actin structure edges"
            )
        if args.structure_v2:
            structure_v2_edges.to_excel(
                writer, index=False, sheet_name="Structure v2 edges"
            )

    print(summary.to_string(index=False))
    print(f"Results: {excel_path}")
    print(f"Preview: {png_path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Original CZI file")
    parser.add_argument("--myofuse-results", required=True, type=Path,
                        help="Folder containing MyoFuse mask TIFF and Nucleus_predictions.xlsx")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene", default="Image:0")
    parser.add_argument("--label-mask", type=Path, help="Override the auto-detected Cellpose label TIFF")
    parser.add_argument("--nuclei-channel", type=int, default=0)
    parser.add_argument("--actin-channel", type=int, default=1)
    parser.add_argument("--mf20-channel", type=int, default=2)
    parser.add_argument("--mf20-threshold", type=int, default=MF20_THRESHOLD)
    parser.add_argument(
        "--allow-mf20-qc-warning",
        action="store_true",
        help="Continue even when the selected MF20 channel strongly resembles random noise",
    )
    parser.add_argument("--actin-threshold", type=int, default=ACTIN_THRESHOLD)
    parser.add_argument("--actin-support-radius", type=int, default=ACTIN_SUPPORT_RADIUS)
    parser.add_argument("--nucleus-support-radius", type=int, default=NUCLEUS_SUPPORT_RADIUS)
    parser.add_argument("--nucleus-support-overlap", type=float, default=NUCLEUS_SUPPORT_OVERLAP)
    parser.add_argument("--min-nuclei", type=int, default=MIN_NUCLEI_PER_GROUP)
    parser.add_argument(
        "--direction-aware",
        action="store_true",
        help=(
            "Experimental: send two-nucleus groups to review when local actin or "
            "MF20 fiber directions differ beyond the configured angle"
        ),
    )
    parser.add_argument("--direction-radius", type=int, default=DIRECTION_RADIUS)
    parser.add_argument("--direction-max-angle", type=float, default=DIRECTION_MAX_ANGLE)
    parser.add_argument(
        "--graph-aware",
        action="store_true",
        help=(
            "Experimental: require each signal-supported nucleus to have at least one "
            "nearby, MF20-continuous, direction-compatible graph neighbor"
        ),
    )
    parser.add_argument("--graph-max-distance", type=float, default=GRAPH_MAX_DISTANCE)
    parser.add_argument("--graph-min-mf20-path", type=float, default=GRAPH_MF20_PATH_MIN)
    parser.add_argument("--graph-max-pair-angle", type=float, default=GRAPH_MAX_PAIR_ANGLE)
    parser.add_argument("--graph-max-line-angle", type=float, default=GRAPH_MAX_LINE_ANGLE)
    parser.add_argument(
        "--hybrid-only-probability-min", type=float, default=0.0,
        help=("Exclude hybrid-only nuclei below this MyoFuse In probability and "
              "mark them for review; 0 disables the conservative guard")
    )
    parser.add_argument("--weak-support-radius", type=int, default=WEAK_SUPPORT_RADIUS)
    parser.add_argument("--weak-support-overlap", type=float, default=WEAK_SUPPORT_OVERLAP)
    parser.add_argument("--weak-cluster-distance", type=float, default=WEAK_CLUSTER_DISTANCE)
    parser.add_argument("--min-myofuse-in", type=int, default=MIN_MYOFUSE_IN_PER_WEAK_CLUSTER)
    parser.add_argument(
        "--include-weak-rescue",
        action="store_true",
        help=(
            "Include weak-signal candidates in FI. Off by default; use only after "
            "independent validation across experimental groups."
        ),
    )
    parser.add_argument(
        "--adaptive-review",
        action="store_true",
        help=(
            "Experimental: add a bounded queue of low-MF20, graph-supported nuclei "
            "to manual review without changing the automatic FI"
        ),
    )
    parser.add_argument(
        "--adaptive-review-mf20-threshold",
        type=int,
        default=ADAPTIVE_REVIEW_MF20_THRESHOLD,
    )
    parser.add_argument(
        "--adaptive-review-max-distance",
        type=float,
        default=ADAPTIVE_REVIEW_MAX_DISTANCE,
    )
    parser.add_argument(
        "--adaptive-review-min-mf20-path",
        type=float,
        default=ADAPTIVE_REVIEW_MIN_MF20_PATH,
    )
    parser.add_argument(
        "--adaptive-review-max-pair-angle",
        type=float,
        default=ADAPTIVE_REVIEW_MAX_PAIR_ANGLE,
    )
    parser.add_argument(
        "--adaptive-review-max-candidates",
        type=int,
        default=ADAPTIVE_REVIEW_MAX_CANDIDATES,
    )
    parser.add_argument(
        "--actin-structure-rescue",
        action="store_true",
        help=(
            "Experimental: include nuclei connected to a MyoFuse/Hybrid In anchor "
            "by a continuous, direction-aligned Actin and MF20 graph"
        ),
    )
    parser.add_argument(
        "--actin-structure-max-distance",
        type=float,
        default=ACTIN_STRUCTURE_MAX_DISTANCE,
    )
    parser.add_argument(
        "--actin-structure-direction-radius",
        type=int,
        default=ACTIN_STRUCTURE_DIRECTION_RADIUS,
    )
    parser.add_argument(
        "--actin-structure-min-coherence",
        type=float,
        default=ACTIN_STRUCTURE_MIN_COHERENCE,
    )
    parser.add_argument(
        "--actin-structure-max-pair-angle",
        type=float,
        default=ACTIN_STRUCTURE_MAX_PAIR_ANGLE,
    )
    parser.add_argument(
        "--actin-structure-max-line-angle",
        type=float,
        default=ACTIN_STRUCTURE_MAX_LINE_ANGLE,
    )
    parser.add_argument(
        "--actin-structure-actin-threshold",
        type=int,
        default=ACTIN_STRUCTURE_ACTIN_THRESHOLD,
    )
    parser.add_argument(
        "--actin-structure-mf20-threshold",
        type=int,
        default=ACTIN_STRUCTURE_MF20_THRESHOLD,
    )
    parser.add_argument(
        "--actin-structure-min-actin-path",
        type=float,
        default=ACTIN_STRUCTURE_MIN_ACTIN_PATH,
    )
    parser.add_argument(
        "--actin-structure-min-mf20-path",
        type=float,
        default=ACTIN_STRUCTURE_MIN_MF20_PATH,
    )
    parser.add_argument(
        "--actin-structure-low-myofuse-fi-threshold",
        type=float,
        default=ACTIN_STRUCTURE_LOW_MYOFUSE_FI_THRESHOLD,
        help=(
            "Use original-MyoFuse-only graph anchors below this original FI "
            "fraction (default: 0.10)"
        ),
    )
    parser.add_argument(
        "--actin-structure-low-support-review-overlap",
        type=float,
        default=ACTIN_STRUCTURE_LOW_SUPPORT_REVIEW_OVERLAP,
        help=(
            "In low-MyoFuse images, send automatic In nuclei below this support "
            "overlap to review (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--actin-structure-neighbor-review-probability",
        type=float,
        default=ACTIN_STRUCTURE_NEIGHBOR_REVIEW_PROBABILITY,
        help="MyoFuse probability threshold for the close-neighbor review queue",
    )
    parser.add_argument(
        "--actin-structure-neighbor-review-distance",
        type=float,
        default=ACTIN_STRUCTURE_NEIGHBOR_REVIEW_DISTANCE,
        help="Maximum nearest-nucleus distance for the close-neighbor review queue",
    )
    parser.add_argument(
        "--structure-v2",
        action="store_true",
        help=(
            "Experimental: use locally normalized, direction-continuous corridors "
            "to rescue weak myotubes"
        ),
    )
    parser.add_argument(
        "--structure-v2-max-distance", type=float, default=STRUCTURE_V2_MAX_DISTANCE
    )
    parser.add_argument(
        "--structure-v2-local-sigma", type=float, default=STRUCTURE_V2_LOCAL_SIGMA
    )
    parser.add_argument(
        "--structure-v2-tensor-sigma", type=float, default=STRUCTURE_V2_TENSOR_SIGMA
    )
    parser.add_argument(
        "--structure-v2-corridor-radius", type=int,
        default=STRUCTURE_V2_CORRIDOR_RADIUS,
    )
    parser.add_argument(
        "--structure-v2-min-coherence", type=float,
        default=STRUCTURE_V2_MIN_COHERENCE,
    )
    parser.add_argument(
        "--structure-v2-max-line-angle", type=float,
        default=STRUCTURE_V2_MAX_LINE_ANGLE,
    )
    parser.add_argument(
        "--structure-v2-min-actin-path", type=float,
        default=STRUCTURE_V2_MIN_ACTIN_PATH,
    )
    parser.add_argument(
        "--structure-v2-min-mf20-path", type=float,
        default=STRUCTURE_V2_MIN_MF20_PATH,
    )
    parser.add_argument(
        "--structure-v2-min-aligned-path", type=float,
        default=STRUCTURE_V2_MIN_ALIGNED_PATH,
    )
    parser.add_argument(
        "--structure-v2-max-dark-gap", type=float,
        default=STRUCTURE_V2_MAX_DARK_GAP,
    )
    parser.add_argument(
        "--structure-v2-review-max-candidates", type=int,
        default=STRUCTURE_V2_REVIEW_MAX_CANDIDATES,
    )
    args = parser.parse_args()
    if args.min_nuclei < 2 or args.min_myofuse_in < 2:
        parser.error("min-nuclei and min-myofuse-in must be at least 2")
    if not 0 <= args.nucleus_support_overlap <= 1 or not 0 <= args.weak_support_overlap <= 1:
        parser.error("overlap thresholds must be in [0, 1]")
    if not 0 <= args.hybrid_only_probability_min <= 1:
        parser.error("hybrid-only-probability-min must be in [0, 1]")
    if args.direction_radius < 1:
        parser.error("direction-radius must be at least 1 pixel")
    if not 0 <= args.direction_max_angle <= 90:
        parser.error("direction-max-angle must be in [0, 90]")
    if args.graph_max_distance <= 0:
        parser.error("graph-max-distance must be positive")
    if not 0 <= args.graph_min_mf20_path <= 1:
        parser.error("graph-min-mf20-path must be in [0, 1]")
    if not 0 <= args.graph_max_pair_angle <= 90 or not 0 <= args.graph_max_line_angle <= 90:
        parser.error("graph angle thresholds must be in [0, 90]")
    if not 0 <= args.adaptive_review_mf20_threshold <= 255:
        parser.error("adaptive-review-mf20-threshold must be in [0, 255]")
    if args.adaptive_review_max_distance <= 0 or args.adaptive_review_max_candidates < 1:
        parser.error("adaptive review distance and candidate cap must be positive")
    if not 0 <= args.adaptive_review_min_mf20_path <= 1:
        parser.error("adaptive-review-min-mf20-path must be in [0, 1]")
    if not 0 <= args.adaptive_review_max_pair_angle <= 90:
        parser.error("adaptive-review-max-pair-angle must be in [0, 90]")
    if args.actin_structure_max_distance <= 0 or args.actin_structure_direction_radius < 1:
        parser.error("actin structure distance and direction radius must be positive")
    if not 0 <= args.actin_structure_min_coherence <= 1:
        parser.error("actin-structure-min-coherence must be in [0, 1]")
    if (
        not 0 <= args.actin_structure_max_pair_angle <= 90
        or not 0 <= args.actin_structure_max_line_angle <= 90
    ):
        parser.error("actin structure angle thresholds must be in [0, 90]")
    if (
        not 0 <= args.actin_structure_actin_threshold <= 255
        or not 0 <= args.actin_structure_mf20_threshold <= 255
    ):
        parser.error("actin structure intensity thresholds must be in [0, 255]")
    if (
        not 0 <= args.actin_structure_min_actin_path <= 1
        or not 0 <= args.actin_structure_min_mf20_path <= 1
    ):
        parser.error("actin structure path fractions must be in [0, 1]")
    if not 0 <= args.actin_structure_low_myofuse_fi_threshold <= 1:
        parser.error("actin-structure-low-myofuse-fi-threshold must be in [0, 1]")
    if not 0 <= args.actin_structure_low_support_review_overlap <= 1:
        parser.error(
            "actin-structure-low-support-review-overlap must be in [0, 1]"
        )
    if not 0 <= args.actin_structure_neighbor_review_probability <= 1:
        parser.error(
            "actin-structure-neighbor-review-probability must be in [0, 1]"
        )
    if args.actin_structure_neighbor_review_distance <= 0:
        parser.error("actin-structure-neighbor-review-distance must be positive")
    if (
        args.structure_v2_max_distance <= 0
        or args.structure_v2_local_sigma <= 0
        or args.structure_v2_tensor_sigma <= 0
        or args.structure_v2_corridor_radius < 1
        or args.structure_v2_review_max_candidates < 1
    ):
        parser.error("Structure v2 distances, sigmas and corridor radius must be positive")
    for name, value in (
        ("structure-v2-min-coherence", args.structure_v2_min_coherence),
        ("structure-v2-min-actin-path", args.structure_v2_min_actin_path),
        ("structure-v2-min-mf20-path", args.structure_v2_min_mf20_path),
        ("structure-v2-min-aligned-path", args.structure_v2_min_aligned_path),
        ("structure-v2-max-dark-gap", args.structure_v2_max_dark_gap),
    ):
        if not 0 <= value <= 1:
            parser.error(f"{name} must be in [0, 1]")
    if not 0 <= args.structure_v2_max_line_angle <= 90:
        parser.error("structure-v2-max-line-angle must be in [0, 90]")
    return run(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise
