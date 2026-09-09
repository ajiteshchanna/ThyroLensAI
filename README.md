# Thyroid Cancer Detection System

An AI-powered web application for detecting thyroid cancer from medical images using deep learning and Grad-CAM interpretability.

## 🌟 Features

- **AI-Powered Analysis**: Uses a custom FibonacciNet deep learning model for accurate thyroid cancer detection
- **Grad-CAM Visualization**: Provides interpretable heatmaps showing which regions the AI focused on
- **Dual Interface**: 
  - Modern web app (FastAPI + HTML/CSS)
  - Streamlit dashboard
- **Report Generation**: Download detailed DOCX reports with analysis results
- **Professional UI**: Clean, medical-themed interface with teal/white color scheme

## 🏗️ Project Structure

```
Thyroid new/
├── app.py                      # FastAPI entry point
├── streamlit_app.py            # Streamlit application
├── model_architecture.py       # Custom neural network layers
├── requirements.txt            # Python dependencies
├── backend/
│   └── routes.py              # API endpoints
├── frontend/
│   ├── static/
│   │   ├── style.css          # Styling
│   │   └── app.js             # Frontend logic
│   └── templates/
│       └── index.html         # Main page
├── utils/
│   ├── config.py              # Configuration
│   ├── processing.py          # Image preprocessing
│   ├── gradcam.py             # Grad-CAM implementation
│   ├── report_generator.py   # DOCX report generation
│   ├── image_quality.py      # Deterministic input quality checks
│   ├── reliability.py        # AI reliability assessment
│   └── logger.py              # Logging configuration
└── logs/
    └── app.log                # Application logs
```

## 🚀 Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Thyroid new"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Hugging Face** (if model is private)
   ```bash
   huggingface-cli login
   ```

## 💻 Usage

### Option 1: Web Application (FastAPI)

1. **Start the server**
   ```bash
   python app.py
   ```

2. **Open browser**
   Navigate to: `http://localhost:8000`

3. **Upload & Analyze**
   - Click "Upload Image"
   - Select a thyroid ultrasound/pathology image
   - View AI analysis and Grad-CAM heatmap
   - Download DOCX report

### Option 2: Streamlit Dashboard

1. **Run Streamlit**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Access dashboard**
   Opens automatically in browser (usually `http://localhost:8501`)

## 🧠 Model Architecture

**FibonacciNet** - Custom CNN with:
- SE (Squeeze-and-Excitation) blocks
- Depthwise separable convolutions
- Avg2Max pooling layers
- Progressive channel expansion following Fibonacci sequence

**Input**: 224x224 RGB images  
**Output**: Binary classification (Benign/Malignant)

## Trustworthy AI Reliability Assessment

The **AI Reliability Score** is a technical, engineering-level assessment of how much reliance
should be placed on the current model output given the available signals. It is **not** a cancer
probability, patient risk score, diagnostic risk, or clinically validated measure.

The current score combines:

```
reliability = 0.30 * model certainty + 0.20 * image quality +
               0.30 * input similarity + 0.20 * calibration quality
```

Model certainty is `abs(model_score - 0.5) * 2`. Image quality combines resolution,
brightness, contrast, and sharpness checks. Reliability thresholds are `HIGH >= 80`,
`MODERATE >= 60`, and `LOW < 60`; these are engineering heuristics, not clinical thresholds.

Input similarity uses a global-average-pooled FibonacciNet feature vector and a regularized
Mahalanobis distance against 2,492 deduplicated reference images. The threshold is the 99th
percentile reference distance (`25.8113`). Similarity is `exp(-distance / threshold)`.

Calibration uses Platt scaling fitted on 623 held-out images from the deduplicated original
dataset. The persisted calibration artifact reports Brier score `0.0762` and ECE `0.0364`.
The historical notebook did not persist its original split and performed upsampling before
splitting, so independence from historical model training cannot be verified. These are
engineering-level Trustworthy AI signals, not clinically validated measures.

## 📊 API Endpoints

### `POST /analyze`
Analyzes uploaded image and returns prediction with Grad-CAM.

**Request**: Multipart form-data with image file

**Response**:
```json
{
  "label": "Malignant (Cancerous)",
  "score": 0.9876,
   "percent": 98.76,
   "model_score_percent": 98.76,
  "class_id": 1,
  "is_malignant": true,
  "original_image": "base64...",
   "gradcam_image": "base64...",
   "reliability": {
      "score": 86,
      "level": "HIGH",
      "model_certainty": {"score": 0.98, "percent": 98, "level": "HIGH"},
      "image_quality": {"score": 0.91, "percent": 91, "level": "GOOD", "warnings": []},
      "input_similarity": {"score": 0.89, "percent": 89, "status": "NORMAL", "ood": false},
      "calibration": {"status": "CALIBRATED", "method": "Platt scaling", "calibrated_probability": 0.98, "ece": 0.0364, "brier_score": 0.0762}
   }
}
```

### `POST /report`
Generates and downloads DOCX report.

**Request**: Multipart form-data with image file

**Response**: DOCX file download

## 🛠️ Technologies

- **Backend**: FastAPI, Python
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML Framework**: TensorFlow/Keras
- **Model Hosting**: Hugging Face Hub
- **Visualization**: Grad-CAM, Matplotlib
- **Reporting**: python-docx
- **UI Framework**: Streamlit (alternative interface)

## 📝 Configuration

Edit `utils/config.py` to change:
- Hugging Face repository ID
- Model filename
- Other settings

## 🔍 Logging

Logs are stored in `logs/app.log` and include:
- Model loading events
- Prediction requests
- Errors and warnings
- Grad-CAM generation status

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

[Add your license here]

## 👥 Authors

[Add author information]

## 🙏 Acknowledgments

- FibonacciNet architecture design
- Grad-CAM implementation
- Medical imaging community
