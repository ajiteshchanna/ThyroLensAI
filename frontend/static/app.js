const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const loader = document.getElementById('loader');
const resultsSection = document.getElementById('resultsSection');
const predBadge = document.getElementById('predBadge');
const downloadBtn = document.getElementById('downloadBtn');
const customAlert = document.getElementById('customAlert');

// --- Helper: Show Alert ---
function showAlert(message, type = 'info') {
    customAlert.textContent = message;
    customAlert.style.display = 'block';
    customAlert.style.color = type === 'error' ? '#ff5252' : '#00c0a3';
    setTimeout(() => {
        customAlert.style.display = 'none';
    }, 5000);
}

// --- Drag & Drop Handlers ---
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, e => {
        e.preventDefault();
        e.stopPropagation();
    }, false);
});

['dragenter', 'dragover'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.add('active'), false);
});

['dragleave', 'drop'].forEach(eventName => {
    dropZone.addEventListener(eventName, () => dropZone.classList.remove('active'), false);
});

dropZone.addEventListener('drop', e => {
    const dt = e.dataTransfer;
    const files = dt.files;
    if (files.length) fileInput.files = files;
    handleFiles(files);
});

fileInput.addEventListener('change', e => {
    handleFiles(e.target.files);
});

async function handleFiles(files) {
    const file = files[0];
    if (!file) return;

    if (!file.type.startsWith('image/')) {
        showAlert("Please upload a valid medical image file.", "error");
        return;
    }

    // Reset UI
    resultsSection.classList.remove('show');
    loader.style.display = 'block';

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        if (!response.ok) throw new Error("Diagnostic analysis failed. Please try again.");

        const result = await response.json();

        // Update Images
        document.getElementById('originalImg').src = `data:image/png;base64,${result.original_image}`;
        const gradcamImg = document.getElementById('gradcamImg');
        gradcamImg.src = result.gradcam_image ? `data:image/png;base64,${result.gradcam_image}` : '';
        gradcamImg.alt = result.gradcam_image ? 'Attention Heatmap' : 'Grad-CAM unavailable';

        // Update Metrics
        predBadge.textContent = result.label;
        predBadge.className = `prediction-badge ${result.is_malignant ? 'malignant' : 'benign'}`;

        document.getElementById('modelScore').textContent = (result.model_score_percent ?? result.score * 100).toFixed(2) + "%";
        document.getElementById('classId').textContent = result.class_id;
        document.getElementById('calibratedProbability').textContent = result.calibrated_probability == null
            ? 'Not available'
            : `${(result.calibrated_probability * 100).toFixed(2)}%`;

        const reliability = result.reliability || {};
        const level = reliability.level || 'NOT_AVAILABLE';
        const score = reliability.score;
        document.getElementById('reliabilityScore').textContent = score == null ? 'Not available' : score;
        const levelElement = document.getElementById('reliabilityLevel');
        levelElement.textContent = level;
        levelElement.className = `reliability-level ${level.toLowerCase()}`;
        const certainty = reliability.model_certainty?.score;
        const quality = reliability.image_quality?.score;
        const certaintyPercent = reliability.model_certainty?.percent;
        const qualityPercent = reliability.image_quality?.percent;
        document.getElementById('certaintyValue').textContent = certaintyPercent == null ? 'Not evaluated' : `${certaintyPercent}%`;
        document.getElementById('qualityValue').textContent = qualityPercent == null ? 'Not evaluated' : `${qualityPercent}% (${reliability.image_quality.level})`;
        document.getElementById('certaintyBar').style.width = certainty == null ? '0%' : `${certainty * 100}%`;
        document.getElementById('qualityBar').style.width = quality == null ? '0%' : `${quality * 100}%`;
        const similarity = reliability.input_similarity || {};
        const similarityPercent = similarity.percent;
        const similarityStatus = similarity.status || 'NOT_AVAILABLE';
        document.getElementById('inputSimilarityValue').textContent = similarityPercent == null ? similarityStatus : `${similarityPercent}% - ${similarityStatus}`;
        document.getElementById('similarityBar').style.width = similarityPercent == null ? '0%' : `${similarityPercent}%`;
        document.getElementById('similarityDetail').textContent = similarity.distance == null ? 'Feature-space comparison' : `Distance ${similarity.distance.toFixed(2)} / threshold ${similarity.threshold.toFixed(2)}`;
        const calibration = reliability.calibration || {};
        document.getElementById('calibrationValue').textContent = calibration.status || 'NOT_AVAILABLE';
        document.getElementById('eceValue').textContent = calibration.ece == null ? '-' : `${(calibration.ece * 100).toFixed(2)}%`;
        document.getElementById('brierValue').textContent = calibration.brier_score == null ? '-' : calibration.brier_score.toFixed(3);
        document.getElementById('recommendation').textContent = reliability.recommendation || 'Clinical review is required.';

        // Show Results with Animation
        loader.style.display = 'none';
        resultsSection.classList.add('show');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

    } catch (err) {
        loader.style.display = 'none';
        showAlert(err.message, "error");
    }
}

// --- Report Generation ---
downloadBtn.addEventListener('click', async () => {
    const file = fileInput.files[0] || null;
    if (!file) {
        showAlert("No file available for report generation.", "error");
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const originalText = downloadBtn.innerHTML;
        downloadBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating Report...';
        downloadBtn.disabled = true;

        const response = await fetch('/report', { method: 'POST', body: formData });
        if (!response.ok) throw new Error("Report generation failed.");

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `Thyroid_Report_${new Date().getTime()}.docx`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);

        downloadBtn.innerHTML = originalText;
        downloadBtn.disabled = false;
        showAlert("Clinical report downloaded successfully.");
    } catch (err) {
        showAlert(err.message, "error");
        downloadBtn.disabled = false;
    }
});
