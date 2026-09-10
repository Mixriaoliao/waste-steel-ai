"""Coordinate helpers for the static PCA demonstration image."""

from __future__ import annotations

import numpy as np


def map_pc_to_pixel(
    pc_coordinates: np.ndarray, image_width: int, image_height: int
) -> tuple[int, int]:
    """Map PCA coordinates to the plotting area of the bundled static image."""

    pc1_min, pc1_max = -30.0, 50.0
    pc2_min, pc2_max = -4.0, 4.0
    pc1_normalized = (pc_coordinates[0] - pc1_min) / (pc1_max - pc1_min)
    pc2_normalized = (pc_coordinates[1] - pc2_min) / (pc2_max - pc2_min)

    margin_left, margin_right = 0.12, 0.08
    margin_top, margin_bottom = 0.16, 0.12
    x = int(
        (margin_left + pc1_normalized * (1 - margin_left - margin_right))
        * image_width
    )
    y = int(
        (margin_top + (1.0 - pc2_normalized) * (1 - margin_top - margin_bottom))
        * image_height
    )
    return x, y
