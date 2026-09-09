"""Build real OOD and calibration artifacts from the original Kaggle dataset.

The training notebook did not persist its split, and upsampled before splitting.
This builder removes exact byte duplicates before making deterministic reference
and calibration partitions. Split independence from the historical model fit
cannot be proven without the original split manifest, and is recorded as such.
"""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import tensorflow as tf
import kagglehub

from utils.config import (
    CALIBRATION_FILENAME,
    MODEL_FILENAME,
    OOD_REFERENCE_FILENAME,
    RELIABILITY_ARTIFACT_DIR,
    REPO_ID,
)
from utils.model_architecture import Avg2MaxPooling, DepthwiseSeparableConv
from utils.processing import preprocess_image


SEED = 42
CALIBRATION_FRACTION = 0.20
BATCH_SIZE = 32


def dataset_files():
    root = Path(kagglehub.dataset_download(
        "diveshzz/thyroid-cancer-classification-ultrasound-dataset"
    )) / "Thyroid Data"
    records = []
    seen = set()
    for label_dir in (root / "0", root / "1"):
        label = int(label_dir.name)
        for path in sorted(label_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            records.append((path, label, digest))
    return records


def stratified_partition(records):
    rng = np.random.default_rng(SEED)
    reference, calibration = [], []
    for label in (0, 1):
        group = [record for record in records if record[1] == label]
        rng.shuffle(group)
        count = max(1, round(len(group) * CALIBRATION_FRACTION))
        calibration.extend(group[:count])
        reference.extend(group[count:])
    rng.shuffle(reference)
    rng.shuffle(calibration)
    return reference, calibration


def load_model():
    path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    return tf.keras.models.load_model(
        path,
        custom_objects={
            "Avg2MaxPooling": Avg2MaxPooling,
            "DepthwiseSeparableConv": DepthwiseSeparableConv,
        },
        compile=False,
    )


def model_outputs(model, records):
    embedding_model = tf.keras.Model(model.inputs, model.layers[-2].output)
    embeddings, scores, labels = [], [], []
    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]
        images = [Image.open(path).convert("RGB") for path, _, _ in batch]
        inputs = np.concatenate([preprocess_image(image) for image in images], axis=0)
        embeddings.extend(embedding_model.predict(inputs, verbose=0))
        scores.extend(np.asarray(model.predict(inputs, verbose=0)).reshape(-1))
        labels.extend(label for _, label, _ in batch)
    return np.asarray(embeddings, dtype=np.float64), np.asarray(scores), np.asarray(labels)


def fit_platt(scores, labels):
    logits = np.log(np.clip(scores, 1e-6, 1 - 1e-6) / np.clip(1 - scores, 1e-6, 1))
    weights = np.array([1.0, 0.0], dtype=np.float64)
    design = np.column_stack([logits, np.ones_like(logits)])
    for _ in range(100):
        values = np.clip(design @ weights, -40, 40)
        probabilities = 1.0 / (1.0 + np.exp(-values))
        gradient = design.T @ (probabilities - labels)
        curvature = design.T @ (design * (probabilities * (1 - probabilities))[:, None])
        update = np.linalg.solve(curvature + np.eye(2) * 1e-8, gradient)
        weights -= update
        if np.max(np.abs(update)) < 1e-8:
            break
    return float(weights[0]), float(weights[1])


def calibration_metrics(probabilities, labels):
    brier = float(np.mean((probabilities - labels) ** 2))
    ece = 0.0
    for lower in np.linspace(0, 1, 11)[:-1]:
        upper = lower + 0.1
        mask = (probabilities >= lower) & (probabilities < upper if upper < 1 else probabilities <= upper)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(probabilities[mask].mean()) - float(labels[mask].mean()))
    return brier, ece


def main():
    records = dataset_files()
    reference_records, calibration_records = stratified_partition(records)
    model = load_model()
    reference_embeddings, _, _ = model_outputs(model, reference_records)
    calibration_embeddings, raw_scores, labels = model_outputs(model, calibration_records)

    mean = reference_embeddings.mean(axis=0)
    centered = reference_embeddings - mean
    covariance = np.cov(centered, rowvar=False)
    regularization = max(float(np.trace(covariance)) / covariance.shape[0] * 1e-3, 1e-6)
    inverse_covariance = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * regularization)
    reference_distances = np.sqrt(np.einsum("ij,jk,ik->i", centered, inverse_covariance, centered))
    threshold = float(np.quantile(reference_distances, 0.99))

    artifact_dir = Path(RELIABILITY_ARTIFACT_DIR)
    artifact_dir.mkdir(exist_ok=True)
    np.savez_compressed(
        artifact_dir / OOD_REFERENCE_FILENAME,
        mean=mean,
        inverse_covariance=inverse_covariance,
        threshold=threshold,
        reference_count=len(reference_records),
    )

    slope, intercept = fit_platt(raw_scores, labels)
    logits = np.clip(slope * np.log(np.clip(raw_scores, 1e-6, 1 - 1e-6) /
                                    np.clip(1 - raw_scores, 1e-6, 1)) + intercept, -40, 40)
    calibrated = 1.0 / (1.0 + np.exp(-logits))
    brier, ece = calibration_metrics(calibrated, labels)
    calibration = {
        "version": 1,
        "method": "Platt scaling",
        "slope": slope,
        "intercept": intercept,
        "brier_score": brier,
        "ece": ece,
        "sample_count": len(calibration_records),
        "class_counts": {str(label): int(np.sum(labels == label)) for label in (0, 1)},
        "source": "deduplicated original Kaggle dataset",
        "split_independence": "UNVERIFIABLE_FROM_HISTORICAL_TRAINING_MANIFEST",
        "note": "Engineering calibration; historical model split was not persisted.",
    }
    (artifact_dir / CALIBRATION_FILENAME).write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(json.dumps({"unique_images": len(records), "reference_images": len(reference_records),
                      "calibration_images": len(calibration_records), "ood_threshold": threshold,
                      "calibration": calibration}, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    main()