import hashlib

import numpy as np
from PIL import Image

from src.features import extract_heuristic_features, stable_image_seed


def test_stable_seed_uses_sha256_content_digest():
    content = b"same image content"
    expected = int.from_bytes(hashlib.sha256(content).digest()[:8], "big")

    assert stable_image_seed(content) == expected


def test_feature_extraction_is_deterministic_for_same_image():
    pixels = np.arange(64, dtype=np.uint8).reshape(8, 8)
    image = Image.fromarray(pixels, mode="L")

    first = extract_heuristic_features(image)
    second = extract_heuristic_features(image)

    np.testing.assert_array_equal(first, second)


def test_feature_proxies_stay_in_documented_ranges():
    image = Image.new("RGB", (10, 10), color=(120, 80, 40))
    thickness, corrosion, purity = extract_heuristic_features(image)

    assert 1.0 <= thickness <= 15.0
    assert 5.0 <= corrosion <= 85.0
    assert 0.4 <= purity <= 0.98
