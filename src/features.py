"""Deterministic heuristic image features used by the demonstration UI.

These values are visualization proxies. They are not measurements of physical
thickness, corrosion, or material purity.
"""

from __future__ import annotations

import hashlib

import numpy as np
from PIL import Image


def stable_image_seed(image_bytes: bytes) -> int:
    """Return a process-independent random seed derived from image content."""

    digest = hashlib.sha256(image_bytes).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def extract_heuristic_features(image: Image.Image) -> np.ndarray:
    """Map simple pixel statistics to deterministic demonstration proxies.

    The returned vector has three components named thickness proxy, corrosion
    proxy, and purity proxy for compatibility with the original visualization.
    None is a calibrated estimate of a real physical property.
    """

    image_gray = image.convert("L")
    image_array = np.asarray(image_gray, dtype=np.float64)
    brightness = image_array.mean() / 255.0
    pixel_std = image_array.std() / 255.0

    rng = np.random.default_rng(stable_image_seed(image.tobytes()))
    thickness_proxy = np.clip(3.0 + (pixel_std * 15.0), 1.0, 15.0)
    corrosion_proxy = np.clip(
        (1.0 - brightness) * 80.0 + rng.uniform(-5.0, 5.0), 5.0, 85.0
    )
    purity_proxy = np.clip(brightness * 1.1 - 0.05, 0.4, 0.98)

    return np.array([thickness_proxy, corrosion_proxy, purity_proxy])
