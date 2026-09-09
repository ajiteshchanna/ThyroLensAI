from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import io
import base64
import traceback
import numpy as np
from PIL import Image
import tensorflow as tf
from huggingface_hub import hf_hub_download

# Import shared utils
from utils.config import REPO_ID, MODEL_FILENAME
from utils.logger import logger
from utils.model_architecture import Avg2MaxPooling, DepthwiseSeparableConv
from utils.processing import preprocess_image
from utils.gradcam import make_gradcam_heatmap, save_and_display_gradcam
from utils.report_generator import generate_docx_report
from utils.image_quality import assess_image_quality
from utils.reliability import calculate_reliability_with_embedding, extract_feature_embedding

# Create Router
router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# Global Model Variable
MODEL = None

def load_model():
    """Loads model from Hugging Face Hub"""
    global MODEL
    if MODEL is None:
        try:
            logger.info("Loading model from Hugging Face...")
            model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
            custom_objects = {
                "Avg2MaxPooling": Avg2MaxPooling, 
                "DepthwiseSeparableConv": DepthwiseSeparableConv
            }
            MODEL = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

# Helper
def get_image_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _calculate_reliability(image, score, processed_image, model):
    """Keep reliability failures isolated from prediction and Grad-CAM."""
    try:
        image_quality = assess_image_quality(image)
        logger.info(
            "Image quality: %s (%.2f)",
            image_quality["quality_level"],
            image_quality["quality_score"],
        )
        embedding = extract_feature_embedding(model, processed_image)
        reliability = calculate_reliability_with_embedding(score, image_quality, embedding)
        logger.info(
            "Model certainty: %s (%.2f)",
            reliability["model_certainty"]["level"],
            reliability["model_certainty"]["score"],
        )
        logger.info("AI reliability: %d/100 (%s)", reliability["score"], reliability["level"])
        return reliability
    except Exception as exc:
        logger.error("Reliability assessment failed: %s\n%s", exc, traceback.format_exc())
        return {
            "score": None,
            "level": "NOT_AVAILABLE",
            "model_certainty": {"score": None, "percent": None, "level": "NOT_AVAILABLE"},
            "image_quality": {"score": None, "percent": None, "level": "NOT_AVAILABLE", "warnings": []},
            "input_similarity": {"status": "NOT_AVAILABLE"},
            "ood": {"status": "NOT_AVAILABLE"},
            "calibration": {"status": "NOT_AVAILABLE"},
            "components_used": [],
            "recommendation": "Reliability assessment unavailable. Clinical review is required.",
        }

# --- Routes ---

@router.on_event("startup")
async def startup_event():
    load_model()

@router.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"request": request}
    )

@router.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    logger.info(f"Analyze request received for file: {file.filename}")
    if MODEL is None:
        # Try loading if not loaded (fallback)
        load_model()
        if MODEL is None:
            return JSONResponse(status_code=503, content={"error": "Model not loaded"})
    
    # Read Image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    
    # Process
    processed_img = preprocess_image(image)
    
    # Predict
    preds = MODEL.predict(processed_img)
    score = float(preds[0][0])
    is_malignant = score > 0.5
    
    # Grad-CAM
    gradcam_b64 = None
    try:
        # Search from the end of the model for the last DepthwiseSeparableConv layer.
        # This is the final spatial feature map (7x7x377) before GlobalAveragePooling2D.
        last_conv = next(
            (l.name for l in MODEL.layers[::-1] if "depthwise_separable_conv" in l.name.lower()),
            None
        )
        if last_conv is None:
            logger.error("Grad-CAM: Could not find a 'depthwise_separable_conv' layer in model.")
        else:
            heatmap = make_gradcam_heatmap(processed_img, MODEL, last_conv)
            if heatmap is not None:
                gradcam_img = save_and_display_gradcam(image, heatmap)
                gradcam_b64 = get_image_base64(gradcam_img)
                logger.info(f"Grad-CAM generated successfully using layer: {last_conv}")
            else:
                logger.error("Grad-CAM: make_gradcam_heatmap returned None.")
    except Exception as e:
        logger.error(f"Grad-CAM generation failed: {e}\n{traceback.format_exc()}")

    reliability = _calculate_reliability(image, score, processed_img, MODEL)
    return {
        "label": "Malignant" if is_malignant else "Benign",
        "score": score,
        "percent": score * 100 if is_malignant else (1 - score) * 100,
        "model_score_percent": score * 100,
        "confidence_percent": score * 100 if is_malignant else (1 - score) * 100,
        "class_id": 1 if is_malignant else 0,
        "is_malignant": is_malignant,
        "original_image": get_image_base64(image),
        "gradcam_image": gradcam_b64,
        "reliability": reliability,
    }

@router.post("/report")
async def get_report(file: UploadFile = File(...)):
    logger.info(f"Report request received for file: {file.filename}")
    try:
        if MODEL is None:
            load_model()
            if MODEL is None:
                return JSONResponse(status_code=503, content={"error": "Model not loaded"})

        # Read Image (again)
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Re-Run Prediction
        processed_img = preprocess_image(image)
        preds = MODEL.predict(processed_img)
        score = float(preds[0][0])
        is_malignant = score > 0.5
        label = "Malignant" if is_malignant else "Benign"
        conf_percent = score * 100 if is_malignant else (1 - score) * 100
        reliability = _calculate_reliability(image, score, processed_img, MODEL)
        
        # Re-Run Grad-CAM for report
        gradcam_bytes = None
        try:
            last_conv = next(
                (l.name for l in MODEL.layers[::-1] if "depthwise_separable_conv" in l.name.lower()),
                None
            )
            if last_conv:
                heatmap = make_gradcam_heatmap(processed_img, MODEL, last_conv)
                if heatmap is not None:
                    gradcam_img = save_and_display_gradcam(image, heatmap)
                    gradcam_bytes = io.BytesIO()
                    gradcam_img.save(gradcam_bytes, format='PNG')
                    gradcam_bytes.seek(0)
                    logger.info(f"Grad-CAM for report generated using layer: {last_conv}")
        except Exception as e:
            logger.error(f"Grad-CAM for report failed: {e}\n{traceback.format_exc()}")

        # Prepare Original Image Bytes
        img_bytes = io.BytesIO()
        image.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Generate Report
        report_buffer = generate_docx_report(
            image_buffer=img_bytes,
            prediction_label=label,
            confidence_score=score,
            confidence_percent=conf_percent,
            gradcam_buffer=gradcam_bytes,
            reliability=reliability,
        )
        report_buffer.seek(0)
        
        # Return File
        headers = {'Content-Disposition': 'attachment; filename="thyroid_analysis_report.docx"'}
        return StreamingResponse(
            report_buffer,
            headers=headers, 
            media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
