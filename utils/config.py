# HuggingFace Model Configuration
REPO_ID = "Diveshj/thyroid_models"
MODEL_FILENAME = "thyroid_cancer_model.keras"

# These are engineering heuristics for demonstrating AI reliability assessment
# and are not clinically validated decision thresholds.
RELIABILITY_CONFIG = {
	"weights": {"model_certainty": 0.6, "image_quality": 0.4},
	"levels": {"high": 80, "moderate": 60},
	"confidence": {"high": 0.8, "moderate": 0.5},
}

IMAGE_QUALITY_CONFIG = {
	"minimum_width": 224,
	"minimum_height": 224,
	"brightness_low": 0.08,
	"brightness_high": 0.92,
	"contrast_low": 0.08,
	"sharpness_low": 20.0,
}