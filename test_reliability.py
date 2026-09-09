"""Focused regression tests for the engineering reliability layer."""

import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, ".")

from utils.image_quality import assess_image_quality
from utils.reliability import calculate_reliability


def checkerboard(size=(480, 352), block=8):
    rows, cols = size[1], size[0]
    pattern = ((np.indices((rows, cols)).sum(axis=0) // block) % 2) * 255
    return Image.fromarray(pattern.astype("uint8"), mode="L")


def main():
    good = assess_image_quality(checkerboard())
    assert good["quality_level"] == "GOOD"
    assert good["resolution"]["acceptable"] is True

    poor = assess_image_quality(Image.new("L", (64, 64), 128))
    assert poor["quality_level"] == "POOR"
    assert any("resolution" in warning.lower() for warning in poor["warnings"])

    dark = assess_image_quality(Image.new("L", (480, 352), 0))
    bright = assess_image_quality(Image.new("L", (480, 352), 255))
    assert any("dark" in warning.lower() for warning in dark["warnings"])
    assert any("bright" in warning.lower() for warning in bright["warnings"])

    blurry = assess_image_quality(checkerboard().filter(ImageFilter.GaussianBlur(20)))
    assert any("blur" in warning.lower() for warning in blurry["warnings"])

    reliability = calculate_reliability(0.77, good)
    assert 0 <= reliability["score"] <= 100
    assert reliability["level"] in {"HIGH", "MODERATE", "LOW"}
    assert reliability["ood"]["status"] == "NOT_EVALUATED"
    assert reliability["calibration"]["status"] == "NOT_EVALUATED"
    print("ALL RELIABILITY TESTS PASSED")


if __name__ == "__main__":
    main()
