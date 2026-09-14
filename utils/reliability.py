"""Technical reliability assessment backed by persisted OOD/calibration artifacts."""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from utils.config import (
    CALIBRATION_FILENAME,
    OOD_REFERENCE_FILENAME,
    RELIABILITY_ARTIFACT_DIR,
    RELIABILITY_CONFIG,
)
from utils.logger import logger


_ARTIFACTS = None


def _load_artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is not None:
        return _ARTIFACTS
    artifact_dir = Path(RELIABILITY_ARTIFACT_DIR)
    ood_path = artifact_dir / OOD_REFERENCE_FILENAME
    calibration_path = artifact_dir / CALIBRATION_FILENAME
    if not ood_path.exists() or not calibration_path.exists():
        missing = [str(path) for path in (ood_path, calibration_path) if not path.exists()]
        logger.error("Reliability artifacts unavailable; missing files: %s", ", ".join(missing))
        _ARTIFACTS = None
        return None
    try:
        with np.load(ood_path) as ood:
            ood_data = {
                "mean": ood["mean"],
                "inverse_covariance": ood["inverse_covariance"],
                "threshold": float(ood["threshold"]),
            }
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("Reliability artifacts failed to load from %s", artifact_dir)
        _ARTIFACTS = None
        raise
    _ARTIFACTS = {"ood": ood_data, "calibration": calibration}
    return _ARTIFACTS


def extract_feature_embedding(model, processed_image):
    """Extract the existing model's pre-classifier global-average-pooled vector."""
    embedding_model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-2].output)
    return np.asarray(embedding_model(processed_image, training=False)).reshape(-1).astype(np.float64)


def _input_similarity(embedding, artifacts):
    centered = embedding - artifacts["ood"]["mean"]
    distance = float(np.sqrt(centered @ artifacts["ood"]["inverse_covariance"] @ centered))
    threshold = artifacts["ood"]["threshold"]
    similarity = float(np.exp(-distance / max(threshold, 1e-8)))
    return {
        "score": similarity,
        "percent": round(similarity * 100),
        "status": "OUT_OF_DISTRIBUTION" if distance > threshold else "NORMAL",
        "ood": bool(distance > threshold),
        "distance": distance,
        "threshold": threshold,
    }


def _calibration(model_score, artifacts):
    calibration = artifacts["calibration"]
    raw_score = float(np.clip(model_score, 1e-6, 1 - 1e-6))
    raw_logit = np.log(raw_score / (1.0 - raw_score))
    calibrated_probability = float(1.0 / (1.0 + np.exp(-np.clip(
        calibration["slope"] * raw_logit + calibration["intercept"], -40, 40
    ))))
    return {
        "status": "CALIBRATED",
        "method": calibration["method"],
        "raw_score": raw_score,
        "calibrated_probability": calibrated_probability,
        "ece": calibration["ece"],
        "brier_score": calibration["brier_score"],
        "sample_count": calibration["sample_count"],
        "split_independence": calibration["split_independence"],
    }


def _level_for_score(score):
    if score >= RELIABILITY_CONFIG["levels"]["high"]:
        return "HIGH"
    if score >= RELIABILITY_CONFIG["levels"]["moderate"]:
        return "MODERATE"
    return "LOW"


def prediction_certainty(model_score):
    certainty = min(1.0, abs(float(model_score) - 0.5) * 2.0)
    thresholds = RELIABILITY_CONFIG["confidence"]
    if certainty >= thresholds["high"]:
        level = "HIGH"
    elif certainty >= thresholds["moderate"]:
        level = "MODERATE"
    else:
        level = "LOW"
    return {"score": certainty, "percent": round(certainty * 100), "level": level}


def calculate_reliability(model_score, image_quality):
    """Combine available technical signals; weights are engineering choices."""
    return calculate_reliability_with_embedding(model_score, image_quality, None)


def calculate_reliability_with_embedding(model_score, image_quality, embedding):
    certainty = prediction_certainty(model_score)
    factors = {
        "model_certainty": certainty["score"],
        "image_quality": float(image_quality["quality_score"]),
    }
    input_similarity = {"status": "NOT_AVAILABLE"}
    calibration = {"status": "NOT_AVAILABLE"}
    try:
        artifacts = _load_artifacts()
        if artifacts is None:
            raise FileNotFoundError("Reliability artifacts are unavailable")
        if embedding is None:
            raise ValueError("Model embedding is unavailable")
        input_similarity = _input_similarity(embedding, artifacts)
    except Exception as exc:
        status = "ARTIFACT_UNAVAILABLE" if isinstance(exc, FileNotFoundError) else "NOT_AVAILABLE"
        input_similarity = {"status": status, "error": str(exc)}
    try:
        artifacts = _ARTIFACTS or _load_artifacts()
        if artifacts is None:
            raise FileNotFoundError("Calibration artifact is unavailable")
        calibration = _calibration(model_score, artifacts)
    except Exception as exc:
        calibration = {"status": "ARTIFACT_UNAVAILABLE", "error": str(exc)}

    if input_similarity.get("score") is not None:
        factors["input_similarity"] = input_similarity["score"]
    if calibration.get("ece") is not None:
        factors["calibration"] = max(0.0, 1.0 - float(calibration["ece"]))

    weights = RELIABILITY_CONFIG["weights"]
    active_weights = {key: weights[key] for key in factors}
    weight_total = sum(active_weights.values())
    score = round(100 * sum(active_weights[key] * value for key, value in factors.items()) / weight_total)
    if input_similarity.get("ood"):
        score = min(score, 49)
    level = _level_for_score(score)
    recommendations = {
        "HIGH": "AI output appears technically reliable enough to be considered as supporting information during clinical review.",
        "MODERATE": "Interpret the AI result cautiously. Clinical review is recommended.",
        "LOW": "AI reliability is low. Do not rely on the automated result alone. Clinical review is required.",
    }
    if input_similarity.get("ood"):
        recommendation = "Input is outside the model's expected data distribution. Clinical review is required."
    else:
        recommendation = recommendations[level]
    result = {
        "score": max(0, min(100, score)),
        "level": level,
        "model_certainty": certainty,
        "image_quality": {
            "score": float(image_quality["quality_score"]),
            "percent": round(float(image_quality["quality_score"]) * 100),
            "level": image_quality["quality_level"],
            "warnings": image_quality["warnings"],
        },
        "input_similarity": input_similarity,
        "ood": input_similarity.copy(),
        "calibration": calibration,
        "components_used": list(factors),
        "recommendation": recommendation,
    }
    if calibration.get("calibrated_probability") is not None:
        result["calibrated_probability"] = calibration["calibrated_probability"]
    return result