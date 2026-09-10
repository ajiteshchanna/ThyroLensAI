# ThyroCheck AI

ThyroCheck AI is a research and educational AI decision-support prototype for binary thyroid image classification. It combines a custom FibonacciNet model with Grad-CAM, deterministic image-quality checks, feature-space input similarity, probability calibration, and an engineering-level AI technical reliability score.

## Overview

The system accepts readable JPEG and PNG images, preprocesses them to 224x224 RGB, and returns a benign/malignant classification. The result distinguishes three quantities:

- **Raw model score:** the sigmoid output produced by FibonacciNet.
- **Calibrated probability:** a Platt-scaled version of the raw score when the persisted calibration artifact is available.
- **AI technical reliability:** an engineering assessment of whether the available technical signals support cautious use of the result. It is not cancer probability, patient risk, diagnostic probability, or probability of correctness.

This is not a clinically validated diagnostic system.

## Key Features

- FastAPI web application and optional Streamlit interface.
- FibonacciNet binary classification without architecture changes or retraining.
- Grad-CAM visualization using the final depthwise-separable convolution layer.
- Deterministic resolution, brightness, contrast, and sharpness checks.
- Mahalanobis-distance input similarity/OOD screening.
- Persisted Platt calibration with Brier score and ECE metadata.
- Configurable technical reliability score and human-in-the-loop recommendation.
- DOCX decision-support report generation.
- Reusable evaluation module for actual held-out predictions.

## Architecture

```
Upload (JPEG/PNG)
  -> validation and RGB preprocessing
  -> FibonacciNet prediction
  -> calibration and feature-space similarity
  -> image quality and technical reliability
  -> Grad-CAM visualization
  -> web response or DOCX report
```

## Model

FibonacciNet uses channel widths 21, 34, 55, 89, 144, 233, and 377. The implementation contains BatchNorm, ReLU, PCB auxiliary paths, Avg2MaxPooling, depthwise-separable convolutions, global average pooling, and a sigmoid binary classifier. It does **not** contain squeeze-and-excitation blocks.

Input shape is `(224, 224, 3)` with pixels scaled to `[0, 1]`. Class ID `0` is benign and class ID `1` is malignant according to the existing model contract.

## Explainability

Grad-CAM highlights spatial regions contributing to the model output. It is an approximate model-attention visualization, not a causal explanation and not evidence that a highlighted region is medically diagnostic.

## Trustworthy AI

### Image Quality

Quality combines resolution, brightness, contrast, and Laplacian sharpness checks. Warnings such as excessive blur, low resolution, darkness, brightness, or low contrast lower the technical reliability signal. These checks are engineering heuristics and are not clinically validated.

### Calibration

The persisted artifact uses Platt scaling. It was fit on 623 samples from a deduplicated partition and records Brier score `0.0762` and ECE `0.0364` for that sample. The historical model split was not persisted, so independence from model training cannot be verified. A calibrated probability is not a clinically validated probability.

### Input Similarity / OOD

The application extracts the pre-classifier global-average-pooled embedding, computes a regularized Mahalanobis distance against a persisted reference distribution, compares the distance with the stored threshold, and maps it to `exp(-distance / threshold)`. An OOD result means the input differs from the reference feature distribution; it does not establish medical validity.

### AI Reliability

```
reliability = 0.30 * model_certainty
            + 0.20 * image_quality
            + 0.30 * input_similarity
            + 0.20 * calibration_quality
```

Weights and thresholds are configurable in `utils/config.py`. Current engineering levels are HIGH >= 80, MODERATE >= 60, and LOW < 60. A strong classifier score cannot by itself produce high technical reliability; poor quality, poor calibration, or an OOD signal lowers the result.

## Evaluation

Use the evaluation module with an actual held-out CSV containing `label,score` columns:

```bash
python -m evaluation.evaluate_model path\to\held_out_predictions.csv --output-dir results
```

The module calculates accuracy, precision, sensitivity/recall, specificity, F1, ROC-AUC, NPV, FPR, FNR, confusion matrix, Brier score, and ECE. It writes JSON and plot artifacts. No verified independent performance metrics are committed because the historical split manifest is unavailable.

## Historical Methodology and Limitations

The notebook downloads the Kaggle dataset, uses class folders `0` and `1`, resizes to 224x224 RGB, rescales pixels, upsamples the minority class with replacement, and then makes an 80/10/10 train/validation/test split. Upsampling before splitting creates a potential duplicate-leakage risk. The original split manifest and exact dataset version were not persisted. There is no external validation or clinical validation.

For future experiments, use: original data -> exact-byte and perceptual deduplication -> stratified split -> training-only balancing and augmentation.

See [MODEL_CARD.md](MODEL_CARD.md) for the full model card.

## Project Structure

```
app.py                         FastAPI entry point
backend/routes.py              Upload, analysis, and report endpoints
frontend/                      HTML, CSS, and browser JavaScript
utils/                         Model, preprocessing, quality, reliability, and reports
evaluation/                    Reusable metrics and plotting helpers
experiments/                   Historical notebook
model artifacts/               Model-adjacent OOD and calibration artifacts
test_*.py                      Focused regression and live checks
MODEL_CARD.md                 Model and safety documentation
```

## Installation

```bash
python -m venv cenv
cenv\Scripts\activate
pip install -r requirements.txt
```

The model is downloaded from the configured Hugging Face repository on first use. Update `utils/config.py` only when intentionally changing model or artifact configuration.

## Usage

Start FastAPI:

```bash
python app.py
```

Open `http://localhost:8000`. Upload a JPEG or PNG image under 10 MB. The interface presents the raw scan, prediction, raw model score, calibrated probability when available, network class ID, technical reliability factors, recommendation, and Grad-CAM output.

The optional Streamlit interface can be started with:

```bash
streamlit run streamlit_app.py
```

## API

`POST /analyze` accepts multipart field `file` and returns `label`, `score`, `model_score_percent`, `calibrated_probability`, `class_id`, `is_malignant`, `original_image`, `gradcam_image`, and a structured `reliability` object. Component failures are reported as `NOT_AVAILABLE` while prediction remains available where possible.

`POST /report` accepts the same upload and returns a DOCX AI decision-support report that distinguishes raw model score, calibrated probability, and AI technical reliability.

## Screenshots

No screenshots are committed yet. Suggested paths for future evidence are `docs/screenshots/upload.png`, `analysis.png`, `reliability.png`, `gradcam.png`, and `dashboard.png`. Do not add synthetic screenshots or fabricated evaluation charts.

## Safety Notice

This project is an AI decision-support prototype for research and education. It is not a standalone diagnosis, is not clinically validated, and must not replace qualified clinical review. Do not upload identifying patient information.

## Future Work

- Persist a versioned, deduplicated dataset manifest and independent test split.
- Run external and prospective validation with appropriate governance.
- Add a research dashboard only when real evaluation artifacts are available.
- Evaluate calibration and OOD behavior across acquisition devices and sites.

## Authors

Add project authors and dataset attribution here.
