# HuggingFace Model Configuration
REPO_ID = "Diveshj/thyroid_models"
MODEL_FILENAME = "thyroid_cancer_model.keras"

# These are engineering heuristics for demonstrating AI reliability assessment
# and are not clinically validated decision thresholds.
RELIABILITY_CONFIG = {
	"weights": {
		"model_certainty": 0.3,
		"image_quality": 0.2,
		"input_similarity": 0.3,
		"calibration": 0.2,
	},
	"levels": {"high": 80, "moderate": 60},
	"confidence": {"high": 0.8, "moderate": 0.5},
}

RELIABILITY_ARTIFACT_DIR = "model artifacts"
OOD_REFERENCE_FILENAME = "ood_reference.npz"
CALIBRATION_FILENAME = "calibration.json"

UPLOAD_CONFIG = {
	"max_bytes": 10 * 1024 * 1024,
	"allowed_extensions": {".jpg", ".jpeg", ".png"},
}

IMAGE_QUALITY_CONFIG = {
	"minimum_width": 224,
	"minimum_height": 224,
	"brightness_low": 0.08,
	"brightness_high": 0.92,
	"contrast_low": 0.08,
	"contrast_reference": 0.25,
	"sharpness_low": 20.0,
	"sharpness_reference": 100.0,
}