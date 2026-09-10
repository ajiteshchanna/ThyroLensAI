"""Metric helpers for reproducible binary-classification evaluation.

This module only computes metrics from caller-provided labels and scores. It does
not contain reported results or silently substitute placeholder data.
"""

import json
from pathlib import Path

import numpy as np


def _confusion(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    return {
        "true_negative": int(np.sum((y_true == 0) & (y_pred == 0))),
        "false_positive": int(np.sum((y_true == 0) & (y_pred == 1))),
        "false_negative": int(np.sum((y_true == 1) & (y_pred == 0))),
        "true_positive": int(np.sum((y_true == 1) & (y_pred == 1))),
    }


def binary_metrics(y_true, scores, threshold=0.5):
    """Return threshold and threshold-free metrics for binary predictions."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if y_true.shape != scores.shape or y_true.ndim != 1:
        raise ValueError("y_true and scores must be one-dimensional arrays of equal length")
    if len(y_true) == 0 or not np.isfinite(scores).all() or not np.isin(y_true, [0, 1]).all():
        raise ValueError("labels and scores must be finite, non-empty binary data")

    y_pred = (scores >= threshold).astype(int)
    matrix = _confusion(y_true, y_pred)
    tn, fp = matrix["true_negative"], matrix["false_positive"]
    fn, tp = matrix["false_negative"], matrix["true_positive"]
    safe = lambda numerator, denominator: float(numerator / denominator) if denominator else None
    order = np.argsort(scores, kind="stable")
    sorted_labels = y_true[order]
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    auc = None
    if positives and negatives:
        ranks = np.arange(1, len(scores) + 1)
        positive_rank_sum = np.sum(ranks[sorted_labels == 1])
        auc = float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))

    precision = safe(tp, tp + fp)
    recall = safe(tp, tp + fn)
    specificity = safe(tn, tn + fp)
    return {
        "sample_count": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": specificity,
        "f1_score": safe(2 * tp, 2 * tp + fp + fn),
        "roc_auc": auc,
        "npv": safe(tn, tn + fn),
        "fpr": safe(fp, fp + tn),
        "fnr": safe(fn, fn + tp),
        "confusion_matrix": matrix,
    }


def calibration_metrics(labels, probabilities, bins=10):
    """Return Brier score and equal-width expected calibration error."""
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if labels.shape != probabilities.shape or labels.ndim != 1:
        raise ValueError("labels and probabilities must be one-dimensional arrays of equal length")
    if not len(labels) or not np.isfinite(probabilities).all():
        raise ValueError("calibration inputs must be finite and non-empty")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    ece = 0.0
    bin_rows = []
    for index in range(bins):
        lower, upper = index / bins, (index + 1) / bins
        mask = (probabilities >= lower) & (probabilities <= upper if index == bins - 1 else probabilities < upper)
        if not np.any(mask):
            continue
        confidence = float(probabilities[mask].mean())
        accuracy = float(labels[mask].mean())
        ece += float(mask.mean()) * abs(confidence - accuracy)
        bin_rows.append({"lower": lower, "upper": upper, "count": int(mask.sum()), "confidence": confidence, "accuracy": accuracy})
    return {"sample_count": int(len(labels)), "brier_score": float(np.mean((probabilities - labels) ** 2)), "ece": float(ece), "bins": bin_rows}


def write_json(data, path):
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")
