// ============================================================
// DermaAgent - Frontend API Integration (Connected to Real ResNet18 + Agents)
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const cameraInput = document.getElementById('cameraInput');
    const computerUploadBtn = document.getElementById('computerUploadBtn');
    const cameraUploadBtn = document.getElementById('cameraUploadBtn');

    const previewContainer = document.getElementById('previewContainer');
    const imagePreview = document.getElementById('imagePreview');
    const removeImageBtn = document.getElementById('removeImage');
    const analyzeBtn = document.getElementById('analyzeBtn') || document.getElementById('analyzeButton');
    const resultsSection = document.getElementById('resultsSection');

    const predictionDisease = document.getElementById('predictionDisease');
    const predictionConfidence = document.getElementById('predictionConfidence');
    const uncertaintyLevel = document.getElementById('uncertaintyLevel');
    const reliabilityAdvisory = document.getElementById('reliabilityAdvisory') || document.getElementById('medicalAdvisory');
    const top3Container = document.getElementById('top3Container') || document.getElementById('top3Predictions');
    const gradcamImage = document.getElementById('gradcamImage');

    let selectedFile = null;

    // Disease Taxonomy Mapping
    const DISEASE_NAMES = {
        "akiec": "Actinic Keratosis / Bowen's Disease",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi (Mole)",
        "vasc": "Vascular Lesion"
    };

    // Computer Upload Button Click
    if (computerUploadBtn && imageInput) {
        computerUploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            imageInput.click();
        });
    }

    // Camera Upload Button Click
    if (cameraUploadBtn && cameraInput) {
        cameraUploadBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            cameraInput.click();
        });
    }

    // Fallback: Click on upload area
    if (uploadArea && imageInput) {
        uploadArea.addEventListener('click', (e) => {
            if (e.target.closest('#computerUploadBtn') || e.target.closest('#cameraUploadBtn')) return;
            imageInput.click();
        });
    }

    // Drag and Drop handlers
    if (uploadArea) {
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadArea.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, (e) => {
                e.preventDefault();
                uploadArea.classList.remove('drag-over');
            });
        });

        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                handleFileSelect(files[0]);
            }
        });
    }

    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    if (cameraInput) {
        cameraInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        if (!file.type.match('image.*')) {
            alert('Please select a valid image file (JPG, JPEG, PNG).');
            return;
        }

        selectedFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            if (imagePreview) imagePreview.src = e.target.result;
            if (uploadArea) uploadArea.style.display = 'none';
            if (previewContainer) previewContainer.classList.add('active');
        };
        reader.readAsDataURL(file);
    }

    if (removeImageBtn) {
        removeImageBtn.addEventListener('click', () => {
            selectedFile = null;
            if (imageInput) imageInput.value = '';
            if (cameraInput) cameraInput.value = '';
            if (imagePreview) imagePreview.src = '';
            if (previewContainer) previewContainer.classList.remove('active');
            if (uploadArea) uploadArea.style.display = 'flex';
            if (resultsSection) resultsSection.classList.remove('active');
        });
    }

    // Analyze Button Click Handler
    if (analyzeBtn) {
        analyzeBtn.addEventListener('click', async () => {

            // Get selected file either from variable or file input
            const fileToUpload = selectedFile || (imageInput && imageInput.files[0]) || (cameraInput && cameraInput.files[0]);

            if (!fileToUpload) {
                alert('Please upload a skin image before running analysis.');
                return;
            }

            // Collect checked symptoms
            const checkedSymptoms = Array.from(document.querySelectorAll('.symptoms-grid input[type="checkbox"]:checked'))
                .map(cb => cb.value);

            // Prepare FormData for Flask API
            const formData = new FormData();
            formData.append('image', fileToUpload);
            checkedSymptoms.forEach(sym => formData.append('symptoms', sym));

            // Loading state
            const originalBtnHTML = analyzeBtn.innerHTML;
            analyzeBtn.innerHTML = `<span>Analyzing Skin Image & Symptoms...</span><span class="spinner">⏳</span>`;
            analyzeBtn.disabled = true;

            try {
                const API_URL = '/api/predict';
                const response = await fetch(API_URL, {
                    method: 'POST',
                    body: formData
                });

                const resData = await response.json();

                if (!response.ok || resData.status !== 'success') {
                    throw new Error(resData.error || 'Server error occurred during prediction.');
                }

                const data = resData.data;

                // Render Results
                renderResults(data);

            } catch (error) {
                console.error("Prediction Error:", error);
                alert(`Analysis Error: ${error.message}`);
            } finally {
                analyzeBtn.innerHTML = originalBtnHTML;
                analyzeBtn.disabled = false;
            }
        });
    }

    function renderResults(data) {
        const pred = data.prediction;
        const rel = data.reliability;
        const exp = data.explainability;

        // Top Prediction
        if (predictionDisease && pred) {
            const fullDiseaseName = DISEASE_NAMES[pred.disease] || pred.disease.toUpperCase();
            predictionDisease.textContent = `${fullDiseaseName} (${pred.disease.toUpperCase()})`;
        }

        if (predictionConfidence && pred) {
            predictionConfidence.textContent = `Confidence Score: ${pred.confidence.toFixed(2)}%`;
        }

        // Reliability & Uncertainty
        if (uncertaintyLevel && rel) {
            uncertaintyLevel.textContent = `${rel.uncertainty_level} Uncertainty`;

            // Color coding
            if (rel.uncertainty_level === 'Low') {
                uncertaintyLevel.style.color = 'var(--success)';
            } else if (rel.uncertainty_level === 'Moderate') {
                uncertaintyLevel.style.color = 'var(--warning)';
            } else {
                uncertaintyLevel.style.color = 'var(--danger)';
            }
        }

        if (reliabilityAdvisory && rel) {
            reliabilityAdvisory.textContent = rel.advisory;
        }

        // Top-3 Predictions Breakdown
        if (top3Container && pred && pred.top3) {
            top3Container.innerHTML = '';
            pred.top3.forEach((item, index) => {
                const dName = DISEASE_NAMES[item.class] || item.class.toUpperCase();
                const row = document.createElement('div');
                row.className = 'prediction-row';
                row.style.cssText = "display:flex; justify-content:space-between; align-items:center; padding:14px; margin-bottom:10px; border:1px solid rgba(255,255,255,0.12); border-radius:12px; background:rgba(255,255,255,0.03);";
                row.innerHTML = `
                    <span style="font-weight:600;">${index + 1}. ${dName} (${item.class.toUpperCase()})</span>
                    <span style="color:var(--accent-light); font-weight:700;">${item.confidence.toFixed(2)}%</span>
                `;
                top3Container.appendChild(row);
            });
        }

        // Grad-CAM Image
        if (gradcamImage && exp) {
            if (exp.gradcam_available && exp.gradcam_url) {
                gradcamImage.src = exp.gradcam_url;
                gradcamImage.style.display = 'block';
            } else {
                gradcamImage.style.display = 'none';
            }
        }

        // Show Results Section
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.classList.add('active');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
});
