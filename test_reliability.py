"""Focused regression tests for the engineering reliability layer."""

import sys

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, ".")

from utils.image_quality import assess_image_quality
from utils.reliability import calculate_reliability_with_embedding, prediction_certainty


def checkerboard(size=(480, 352), block=8):
    rows, cols = size[1], size[0]
    pattern = ((np.indices((rows, cols)).sum(axis=0) // block) % 2) * 255
    return Image.fromarray(pattern.astype("uint8"), mode="L")


def main():
    assert prediction_certainty(0.50)["score"] == 0.0
    assert prediction_certainty(0.75)["score"] == 0.5
    assert prediction_certainty(0.99)["score"] == 0.98

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

    with np.load("model artifacts/ood_reference.npz") as artifact:
        reference_mean = artifact["mean"]
        far_embedding = reference_mean + np.ones_like(reference_mean) * 1000
    reliability = calculate_reliability_with_embedding(0.77, good, reference_mean)
    assert 0 <= reliability["score"] <= 100
    assert reliability["level"] in {"HIGH", "MODERATE", "LOW"}
    assert reliability["input_similarity"]["status"] == "NORMAL"
    assert reliability["calibration"]["status"] == "CALIBRATED"
    assert reliability["components_used"] == ["model_certainty", "image_quality", "input_similarity", "calibration"]
    assert reliability["model_certainty"]["percent"] == 54
    assert reliability["image_quality"]["percent"] == round(good["quality_score"] * 100)
    assert 0 <= reliability["calibration"]["calibrated_probability"] <= 1

    ood_reliability = calculate_reliability_with_embedding(0.77, good, far_embedding)
    assert ood_reliability["input_similarity"]["ood"] is True
    assert ood_reliability["score"] <= 49
    print("ALL RELIABILITY TESTS PASSED")


if __name__ == "__main__":
    main()
