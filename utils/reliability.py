"""Technical reliability assessment with explicit future-signal statuses."""

from utils.config import RELIABILITY_CONFIG


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
    return {"score": certainty, "level": level}


def calculate_reliability(model_score, image_quality):
    """Combine only implemented signals; weights are engineering choices."""
    certainty = prediction_certainty(model_score)
    weights = RELIABILITY_CONFIG["weights"]
    weight_total = weights["model_certainty"] + weights["image_quality"]
    score = round(100 * (weights["model_certainty"] * certainty["score"]
                         + weights["image_quality"] * float(image_quality["quality_score"]))
                  / weight_total)
    level = _level_for_score(score)
    recommendations = {
        "HIGH": "AI result may be considered as supporting information during clinical review.",
        "MODERATE": "Interpret the AI result cautiously. Clinical review is recommended.",
        "LOW": "AI reliability is low. Do not rely on the automated result alone. Clinical review is required.",
    }
    return {
        "score": max(0, min(100, score)),
        "level": level,
        "model_certainty": certainty,
        "image_quality": {
            "score": float(image_quality["quality_score"]),
            "level": image_quality["quality_level"],
            "warnings": image_quality["warnings"],
        },
        "ood": {"status": "NOT_EVALUATED"},
        "calibration": {"status": "NOT_EVALUATED"},
        "recommendation": recommendations[level],
    }