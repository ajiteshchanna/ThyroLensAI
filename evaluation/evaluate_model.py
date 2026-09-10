"""Evaluate a saved prediction table without embedding reported metrics.

Input CSV columns: ``label,score``. The caller must provide the actual held-out
set; the script intentionally fails when that evidence is absent.
"""

import argparse
import csv
from pathlib import Path

from evaluation.metrics import binary_metrics, calibration_metrics, write_json
from evaluation.plots import save_confusion_matrix, save_reliability_diagram


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"label", "score"}.issubset(rows[0]):
        raise ValueError("CSV must contain label and score columns")
    return [int(row["label"]) for row in rows], [float(row["score"]) for row in rows]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions_csv")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()
    labels, scores = load_csv(args.predictions_csv)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metrics = binary_metrics(labels, scores)
    calibration = calibration_metrics(labels, scores)
    write_json(metrics, output / "metrics.json")
    write_json(calibration, output / "calibration_metrics.json")
    save_confusion_matrix(metrics["confusion_matrix"], output / "confusion_matrix.png")
    save_reliability_diagram(calibration, output / "reliability_diagram.png")
    print(f"Wrote evaluation artifacts to {output}")


if __name__ == "__main__":
    main()
