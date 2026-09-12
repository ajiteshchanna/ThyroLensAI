# ThyroLens AI Model Card

## 1. Model Overview

ThyroLens AI is a research and educational decision-support prototype for binary thyroid image classification. It returns a raw model score, optional calibrated probability, technical reliability signals, and Grad-CAM visualization.

## 2. Intended Use

The system is intended for software evaluation, portfolio demonstration, and research-oriented exploration of trustworthy AI patterns around thyroid ultrasound image classification.

## 3. Out-of-Scope Use

It must not be used as a standalone diagnosis, a cancer-risk estimate, a patient-risk score, or a replacement for qualified clinical review. It does not establish medical validity of an uploaded image.

## 4. Architecture

FibonacciNet uses Fibonacci-based channel widths (21, 34, 55, 89, 144, 233, 377), convolutional blocks with BatchNorm and ReLU, PCB auxiliary paths, Avg2MaxPooling, depthwise-separable convolutions, global average pooling, and a sigmoid binary classifier. The implementation does not contain squeeze-and-excitation blocks.

## 5. Training Data

The historical notebook downloads the Kaggle thyroid ultrasound dataset `diveshzz/thyroid-cancer-classification-ultrasound-dataset`, reads class folders `0` and `1`, resizes images to 224x224 RGB, and rescales pixels to [0, 1]. The exact dataset version and original split manifest were not persisted in this repository.

## 6. Dataset Limitations

The historical pipeline upsampled the minority class before splitting into train, validation, and test partitions. Because upsampling sampled with replacement, duplicate files or repeated source images could cross partition boundaries. This creates a potential leakage risk. There is no external validation set recorded here.

## 7. Evaluation Methodology

`evaluation/evaluate_model.py` evaluates a caller-provided held-out CSV with `label,score` columns. It computes accuracy, precision, sensitivity/recall, specificity, F1, ROC-AUC, NPV, FPR, FNR, and a confusion matrix. It also computes Brier score and equal-width ECE and can generate a reliability diagram. No metrics are reported unless produced from actual supplied predictions.

## 8. Performance Metrics

Not available in this repository as a verified independent test result. The historical notebook title includes an accuracy claim, but this model card does not treat that title as a reproducible evaluation result.

## 9. Calibration

Persisted calibration uses Platt scaling on 623 samples from a deduplicated dataset partition. The artifact records Brier score 0.0762 and ECE 0.0364 for that calibration sample. Independence from historical model training cannot be verified because the original split manifest was not persisted. A calibrated probability is not a clinically validated probability.

## 10. OOD Detection

The system extracts the pre-classifier global-average-pooled embedding, computes a regularized Mahalanobis distance against a persisted reference distribution, compares it with the persisted threshold, and maps distance to a similarity score using `exp(-distance / threshold)`. This detects feature-space difference from the reference data; it does not guarantee that an image is medically valid.

## 11. Explainability

Grad-CAM targets the final depthwise-separable convolution layer and overlays the resulting heatmap on the input image. The heatmap is an approximate visualization of model attention, not a clinical explanation or proof of causality.

## 12. Reliability Assessment

AI technical reliability combines model certainty, image quality, input similarity, and calibration quality using configurable engineering weights. It is not cancer probability, diagnostic probability, probability of correctness, patient risk, or clinical risk. HIGH, MODERATE, and LOW thresholds are engineering heuristics.

## 13. Known Limitations

- Potential duplicate leakage in the historical split methodology.
- No persisted original split manifest.
- No external validation or prospective clinical evaluation.
- No clinical validation of image-quality, OOD, calibration, or reliability thresholds.
- OOD detection is feature-space screening, not medical validity checking.
- Grad-CAM is not a causal or clinical explanation.
- Supported uploads are readable JPEG and PNG files, not DICOM.

## 14. Ethical and Safety Considerations

Outputs can be wrong, overconfident, or sensitive to image acquisition and dataset shift. Results should be reviewed by an appropriately qualified professional. Avoid uploading identifying patient information to this prototype.

## 15. Human Oversight

The application always presents a recommendation to use the output only as supporting information. Low reliability, poor image quality, or an OOD signal requires clinical review and must not be treated as an automated conclusion.
