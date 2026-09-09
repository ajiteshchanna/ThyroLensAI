"""Transparent, engineering-level checks for uploaded image quality."""

import numpy as np
from PIL import Image, ImageFilter

from utils.config import IMAGE_QUALITY_CONFIG

try:
    import cv2
except ImportError:  # pragma: no cover - requirements include OpenCV, fallback is retained.
    cv2 = None


def _clamp(value, minimum=0.0, maximum=1.0):
    return max(minimum, min(maximum, float(value)))


def _sharpness(gray_array):
    if cv2 is not None:
        return float(cv2.Laplacian(gray_array, cv2.CV_64F).var())
    edges = np.asarray(Image.fromarray(gray_array).filter(ImageFilter.FIND_EDGES), dtype=np.float32)
    return float(edges.var())


def assess_image_quality(image):
    """Return deterministic image metrics; this is not a clinical ultrasound assessment."""
    rgb_image = image.convert("RGB") if image.mode != "RGB" else image
    width, height = rgb_image.size
    gray_image = rgb_image.convert("L")
    gray = np.asarray(gray_image, dtype=np.float32) / 255.0
    mean_brightness = float(gray.mean())
    contrast = float(gray.std())
    sharpness = _sharpness(np.asarray(gray_image, dtype=np.uint8))

    resolution_score = _clamp(min(width / IMAGE_QUALITY_CONFIG["minimum_width"],
                                 height / IMAGE_QUALITY_CONFIG["minimum_height"]))
    brightness_score = _clamp(1.0 - max(
        (IMAGE_QUALITY_CONFIG["brightness_low"] - mean_brightness) /
        IMAGE_QUALITY_CONFIG["brightness_low"],
        (mean_brightness - IMAGE_QUALITY_CONFIG["brightness_high"]) /
        (1.0 - IMAGE_QUALITY_CONFIG["brightness_high"]),
        0.0,
    ))
    contrast_score = _clamp(contrast / 0.25)
    sharpness_score = _clamp(sharpness / (IMAGE_QUALITY_CONFIG["sharpness_low"] * 5.0))

    warnings = []
    if width < IMAGE_QUALITY_CONFIG["minimum_width"] or height < IMAGE_QUALITY_CONFIG["minimum_height"]:
        warnings.append("Image resolution is below the recommended minimum")
    if mean_brightness < IMAGE_QUALITY_CONFIG["brightness_low"]:
        warnings.append("Image appears excessively dark")
    elif mean_brightness > IMAGE_QUALITY_CONFIG["brightness_high"]:
        warnings.append("Image appears excessively bright")
    if contrast < IMAGE_QUALITY_CONFIG["contrast_low"]:
        warnings.append("Image has very low contrast")
    if sharpness < IMAGE_QUALITY_CONFIG["sharpness_low"]:
        warnings.append("Image appears excessively blurred")

    quality_score = _clamp(0.25 * resolution_score + 0.25 * brightness_score
                           + 0.20 * contrast_score + 0.30 * sharpness_score)
    if quality_score >= 0.8 and not warnings:
        quality_level = "GOOD"
    elif quality_score >= 0.6:
        quality_level = "FAIR"
    else:
        quality_level = "POOR"

    return {
        "quality_score": quality_score,
        "quality_level": quality_level,
        "resolution": {
            "width": width,
            "height": height,
            "acceptable": width >= IMAGE_QUALITY_CONFIG["minimum_width"] and
            height >= IMAGE_QUALITY_CONFIG["minimum_height"],
        },
        "brightness_score": brightness_score,
        "contrast_score": contrast_score,
        "sharpness_score": sharpness_score,
        "warnings": warnings,
    }