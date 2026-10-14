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
    const changeImageBtn = document.getElementById('changeImageBtn');
    const previewSourceTag = document.getElementById('previewSourceTag');
    const previewFileInfo = document.getElementById('previewFileInfo');

    // Live Camera Modal Elements
    const cameraModal = document.getElementById('cameraModal');
    const closeCameraBtn = document.getElementById('closeCameraBtn');
    const cancelCameraBtn = document.getElementById('cancelCameraBtn');
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const cameraSnapshotPreview = document.getElementById('cameraSnapshotPreview');
    const cameraReticle = document.getElementById('cameraReticle');
    const cameraGuidance = document.getElementById('cameraGuidance');
    const cameraFlash = document.getElementById('cameraFlash');
    const cameraStreamControls = document.getElementById('cameraStreamControls');
    const cameraReviewControls = document.getElementById('cameraReviewControls');
    const flipCameraBtn = document.getElementById('flipCameraBtn');
    const shutterBtn = document.getElementById('shutterBtn');
    const retakePhotoBtn = document.getElementById('retakePhotoBtn');
    const usePhotoBtn = document.getElementById('usePhotoBtn');

    let activeStream = null;
    let currentFacingMode = 'environment';

    const analyzeBtn = document.getElementById('analyzeBtn') || document.getElementById('analyzeButton');
    const resultsSection = document.getElementById('resultsSection');

    const predictionDisease = document.getElementById('predictionDisease');
    const predictionConfidence = document.getElementById('predictionConfidence');
    const uncertaintyLevel = document.getElementById('uncertaintyLevel');
    const reliabilityAdvisory = document.getElementById('reliabilityAdvisory') || document.getElementById('medicalAdvisory');
    const top3Container = document.getElementById('top3Container') || document.getElementById('top3Predictions');
    const gradcamImage = document.getElementById('gradcamImage');

    let selectedFile = null;

    const modelSelect = document.getElementById('modelSelect');
    const activeModelBadge = document.getElementById('activeModelBadge');

    // Disease Taxonomy Mapping (Supports both ISIC 2019 8-class and HAM10000 7-class)
    const DISEASE_NAMES = {
        // ISIC 2019 (8 Target Classes)
        "MEL": "Melanoma (Malignant)",
        "NV": "Melanocytic Nevi (Benign Mole)",
        "BCC": "Basal Cell Carcinoma (Malignant)",
        "AK": "Actinic Keratosis (Pre-cancerous)",
        "BKL": "Benign Keratosis (Solar Lentigo / Seborrheic)",
        "DF": "Dermatofibroma (Benign)",
        "VASC": "Vascular Lesion (Benign)",
        "SCC": "Squamous Cell Carcinoma (Malignant)",

        // HAM10000 (7 Classes Fallback)
        "akiec": "Actinic Keratosis / Bowen's Disease",
        "bcc": "Basal Cell Carcinoma",
        "bkl": "Benign Keratosis",
        "df": "Dermatofibroma",
        "mel": "Melanoma",
        "nv": "Melanocytic Nevi (Mole)",
        "vasc": "Vascular Lesion"
    };

    const DISEASE_CATEGORIES = {
        "MEL": { name: "Malignant", badgeClass: "badge-malignant" },
        "BCC": { name: "Malignant", badgeClass: "badge-malignant" },
        "SCC": { name: "Malignant", badgeClass: "badge-malignant" },
        "AK": { name: "Pre-cancerous", badgeClass: "badge-precancerous" },
        "akiec": { name: "Pre-cancerous", badgeClass: "badge-precancerous" },
        "NV": { name: "Benign", badgeClass: "badge-benign" },
        "nv": { name: "Benign", badgeClass: "badge-benign" },
        "BKL": { name: "Benign", badgeClass: "badge-benign" },
        "bkl": { name: "Benign", badgeClass: "badge-benign" },
        "DF": { name: "Benign", badgeClass: "badge-benign" },
        "df": { name: "Benign", badgeClass: "badge-benign" },
        "VASC": { name: "Benign", badgeClass: "badge-benign" },
        "vasc": { name: "Benign", badgeClass: "badge-benign" }
    };

    // ============================================================
    // Device File Upload Handler (Isolated from camera intent)
    // ============================================================
    if (computerUploadBtn && imageInput) {
        computerUploadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            imageInput.click();
        });
    }

    // ============================================================
    // Live Camera Scanner Implementation (WebRTC + Fallback)
    // ============================================================
    if (cameraUploadBtn) {
        cameraUploadBtn.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            openLiveCamera('environment');
        });
    }

    async function openLiveCamera(facingMode = 'environment') {
        // Fallback if browser/context does not support getUserMedia (e.g. non-secure origin)
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            console.warn('getUserMedia not supported in current environment; falling back to native camera input.');
            if (cameraInput) cameraInput.click();
            return;
        }

        try {
            // Stop any existing stream
            if (activeStream) {
                activeStream.getTracks().forEach(track => {
                    try { track.stop(); } catch(err) {}
                });
                activeStream = null;
            }

            currentFacingMode = facingMode;

            // Reset UI states to live stream mode
            if (cameraVideo) {
                cameraVideo.style.display = 'block';
                if (currentFacingMode === 'user') {
                    cameraVideo.classList.add('mirrored');
                } else {
                    cameraVideo.classList.remove('mirrored');
                }
            }
            if (cameraSnapshotPreview) {
                cameraSnapshotPreview.style.display = 'none';
                cameraSnapshotPreview.src = '';
            }
            if (cameraReticle) cameraReticle.style.display = 'block';
            if (cameraGuidance) cameraGuidance.style.display = 'flex';
            if (cameraStreamControls) cameraStreamControls.style.display = 'flex';
            if (cameraReviewControls) cameraReviewControls.style.display = 'none';

            // Show camera modal
            if (cameraModal) cameraModal.style.display = 'flex';

            const constraints = {
                video: {
                    facingMode: { ideal: facingMode },
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                },
                audio: false
            };

            activeStream = await navigator.mediaDevices.getUserMedia(constraints);

            if (cameraVideo) {
                cameraVideo.srcObject = activeStream;
                await cameraVideo.play().catch(e => console.log("Camera play notice:", e));
            }
        } catch (error) {
            console.warn('Live Camera access issue:', error);
            stopLiveCamera();

            // Graceful fallback to native device camera
            alert('Live camera streaming not directly available on this browser/permission. Opening system camera...');
            if (cameraInput) {
                cameraInput.click();
            }
        }
    }

    function stopLiveCamera() {
        if (activeStream) {
            activeStream.getTracks().forEach(track => {
                try { track.stop(); } catch(e) {}
            });
            activeStream = null;
        }
        if (cameraVideo) {
            cameraVideo.srcObject = null;
        }
        if (cameraModal) {
            cameraModal.style.display = 'none';
        }
    }

    function capturePhoto() {
        if (!cameraVideo || !cameraCanvas) return;

        // Visual flash feedback
        if (cameraFlash) {
            cameraFlash.classList.add('flash-active');
            setTimeout(() => {
                cameraFlash.classList.remove('flash-active');
            }, 100);
        }

        const videoW = cameraVideo.videoWidth || 1280;
        const videoH = cameraVideo.videoHeight || 720;

        cameraCanvas.width = videoW;
        cameraCanvas.height = videoH;
        const ctx = cameraCanvas.getContext('2d');

        // Mirror canvas if front-facing selfie camera
        if (currentFacingMode === 'user') {
            ctx.translate(videoW, 0);
            ctx.scale(-1, 1);
        }

        ctx.drawImage(cameraVideo, 0, 0, videoW, videoH);

        // Display captured snapshot in review mode
        const dataUrl = cameraCanvas.toDataURL('image/jpeg', 0.95);
        if (cameraSnapshotPreview) {
            cameraSnapshotPreview.src = dataUrl;
            cameraSnapshotPreview.style.display = 'block';
        }

        if (cameraVideo) cameraVideo.style.display = 'none';
        if (cameraReticle) cameraReticle.style.display = 'none';
        if (cameraGuidance) cameraGuidance.style.display = 'none';

        if (cameraStreamControls) cameraStreamControls.style.display = 'none';
        if (cameraReviewControls) cameraReviewControls.style.display = 'flex';
    }

    function retakePhoto() {
        if (cameraSnapshotPreview) {
            cameraSnapshotPreview.style.display = 'none';
            cameraSnapshotPreview.src = '';
        }
        if (cameraVideo) cameraVideo.style.display = 'block';
        if (cameraReticle) cameraReticle.style.display = 'block';
        if (cameraGuidance) cameraGuidance.style.display = 'flex';

        if (cameraStreamControls) cameraStreamControls.style.display = 'flex';
        if (cameraReviewControls) cameraReviewControls.style.display = 'none';
    }

    function useCapturedPhoto() {
        if (!cameraCanvas) return;

        cameraCanvas.toBlob((blob) => {
            if (!blob) {
                alert('Could not process camera image.');
                return;
            }
            const timestamp = Date.now();
            const capturedFile = new File([blob], `live_skin_lesion_${timestamp}.jpg`, { type: 'image/jpeg' });
            stopLiveCamera();
            handleFileSelect(capturedFile, '📸 Live Camera Photo');
        }, 'image/jpeg', 0.95);
    }

    function flipCamera() {
        const nextFacing = (currentFacingMode === 'environment') ? 'user' : 'environment';
        openLiveCamera(nextFacing);
    }

    // Camera Modal Event Listeners
    if (closeCameraBtn) closeCameraBtn.addEventListener('click', stopLiveCamera);
    if (cancelCameraBtn) cancelCameraBtn.addEventListener('click', stopLiveCamera);
    if (shutterBtn) shutterBtn.addEventListener('click', capturePhoto);
    if (retakePhotoBtn) retakePhotoBtn.addEventListener('click', retakePhoto);
    if (usePhotoBtn) usePhotoBtn.addEventListener('click', useCapturedPhoto);
    if (flipCameraBtn) flipCameraBtn.addEventListener('click', flipCamera);

    // Close on backdrop click outside dialog
    if (cameraModal) {
        cameraModal.addEventListener('click', (e) => {
            if (e.target === cameraModal) {
                stopLiveCamera();
            }
        });
    }

    // Close on Escape key press
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && cameraModal && cameraModal.style.display === 'flex') {
            stopLiveCamera();
        }
    });

    // ============================================================
    // Drag & Drop handlers (Dropzone)
    // ============================================================
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
                handleFileSelect(files[0], '📁 Dropped Image');
            }
        });
    }

    // File Input changes
    if (imageInput) {
        imageInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0], '📁 Device File: ' + e.target.files[0].name);
            }
        });
    }

    // Native Camera Fallback Input changes
    if (cameraInput) {
        cameraInput.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                handleFileSelect(e.target.files[0], '📷 Camera Photo: ' + e.target.files[0].name);
            }
        });
    }

    // ============================================================
    // Image Selection & Preview Handler
    // ============================================================
    function handleFileSelect(file, sourceLabel = null) {
        if (!file || !file.type.match('image.*')) {
            alert('Please select a valid image file (JPG, JPEG, PNG, WEBP).');
            return;
        }

        selectedFile = file;

        // Update preview source tag & file info
        if (previewSourceTag) {
            previewSourceTag.textContent = sourceLabel || ('📁 File: ' + (file.name || 'skin_image.jpg'));
            if (sourceLabel && sourceLabel.includes('Live Camera')) {
                previewSourceTag.style.background = 'rgba(85, 224, 193, 0.15)';
                previewSourceTag.style.borderColor = 'rgba(85, 224, 193, 0.4)';
                previewSourceTag.style.color = '#55e0c1';
            } else {
                previewSourceTag.style.background = 'rgba(108, 124, 255, 0.15)';
                previewSourceTag.style.borderColor = 'rgba(141, 153, 255, 0.3)';
                previewSourceTag.style.color = 'var(--accent-light)';
            }
        }

        if (previewFileInfo && file.size) {
            const sizeKB = (file.size / 1024).toFixed(1);
            previewFileInfo.textContent = `${sizeKB} KB`;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            if (imagePreview) imagePreview.src = e.target.result;
            if (uploadArea) uploadArea.style.display = 'none';
            if (previewContainer) previewContainer.classList.add('active');
        };
        reader.readAsDataURL(file);
    }

    // Remove Image button
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

    // Change Image button
    if (changeImageBtn) {
        changeImageBtn.addEventListener('click', () => {
            if (previewContainer) previewContainer.classList.remove('active');
            if (uploadArea) uploadArea.style.display = 'flex';
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

            const selectedModel = (modelSelect && modelSelect.value) ? modelSelect.value : 'isic2019';
            formData.append('model_type', selectedModel);

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

        // Active Model Badge
        if (activeModelBadge) {
            const mLabel = data.model_type === 'isic2019' 
                ? 'ISIC 2019 ResNet18 (8 Target Classes)' 
                : 'HAM10000 ResNet18 (7 Classes Baseline)';
            activeModelBadge.textContent = mLabel;
        }

        // Top Prediction
        if (predictionDisease && pred) {
            const diseaseKey = pred.disease;
            const fullDiseaseName = DISEASE_NAMES[diseaseKey] || DISEASE_NAMES[diseaseKey.toLowerCase()] || DISEASE_NAMES[diseaseKey.toUpperCase()] || diseaseKey;
            const categoryInfo = DISEASE_CATEGORIES[diseaseKey] || DISEASE_CATEGORIES[diseaseKey.toLowerCase()] || DISEASE_CATEGORIES[diseaseKey.toUpperCase()];
            const badgeHTML = categoryInfo ? `<span class="badge-tag ${categoryInfo.badgeClass}">${categoryInfo.name}</span>` : '';
            predictionDisease.innerHTML = `${fullDiseaseName} ${badgeHTML}`;
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

        // Top-3 Predictions Breakdown with visual progress bars
        if (top3Container && pred && pred.top3) {
            top3Container.innerHTML = '';
            pred.top3.forEach((item, index) => {
                const diseaseKey = item.class;
                const dName = DISEASE_NAMES[diseaseKey] || DISEASE_NAMES[diseaseKey.toLowerCase()] || DISEASE_NAMES[diseaseKey.toUpperCase()] || diseaseKey;
                const categoryInfo = DISEASE_CATEGORIES[diseaseKey] || DISEASE_CATEGORIES[diseaseKey.toLowerCase()] || DISEASE_CATEGORIES[diseaseKey.toUpperCase()];
                const badgeHTML = categoryInfo ? `<span class="badge-tag ${categoryInfo.badgeClass}" style="margin-left:6px; font-size:10px;">${categoryInfo.name}</span>` : '';
                
                const card = document.createElement('div');
                card.className = 'prediction-row';
                card.style.cssText = "padding:14px; margin-bottom:12px; border:1px solid rgba(255,255,255,0.12); border-radius:14px; background:rgba(255,255,255,0.03);";
                card.innerHTML = `
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-weight:600; display:flex; align-items:center;">
                            ${index + 1}. ${dName} ${badgeHTML}
                        </span>
                        <span style="color:var(--accent-light); font-weight:700; font-size:15px;">${item.confidence.toFixed(2)}%</span>
                    </div>
                    <div class="confidence-bar-bg">
                        <div class="confidence-bar-fill" style="width: ${Math.min(100, Math.max(3, item.confidence))}%;"></div>
                    </div>
                `;
                top3Container.appendChild(card);
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

        // Image Quality Agent Output
        const qualityContainer = document.getElementById('qualityContainer');
        if (qualityContainer && data.image_quality) {
            const iq = data.image_quality;
            qualityContainer.style.display = 'flex';
            const statusColor = iq.is_acceptable ? 'var(--success)' : 'var(--danger)';
            const statusIcon = iq.is_acceptable ? '✓' : '⚠️';
            const reuploadNote = iq.reupload_recommended 
                ? `<div style="width:100%; color:var(--danger); font-size:13px; margin-top:6px; font-weight:600;">⚠️ ${iq.advisory}</div>`
                : `<div style="width:100%; color:var(--text-secondary); font-size:13px; margin-top:4px;">${iq.advisory}</div>`;

            qualityContainer.innerHTML = `
                <div style="display:flex; align-items:center; gap:10px;">
                    <span style="display:grid; place-items:center; width:30px; height:30px; border-radius:50%; background:${statusColor}22; color:${statusColor}; font-weight:bold;">${statusIcon}</span>
                    <div>
                        <strong style="color:var(--text-primary); font-size:14px;">Image Quality: ${iq.sharpness_level || 'Evaluated'} (${iq.quality_score}/100)</strong>
                        <small style="display:block; color:var(--text-secondary);">Lighting: ${iq.lighting_level || 'Normal'} | Res: ${iq.metrics?.dimensions || 'Standard'}</small>
                    </div>
                </div>
                <div class="quality-metrics">
                    <span class="quality-pill">Sharpness: <strong>${iq.metrics?.sharpness ?? '—'}</strong></span>
                    <span class="quality-pill">Contrast: <strong>${iq.metrics?.contrast ?? '—'}</strong></span>
                </div>
                ${reuploadNote}
            `;
        }

        // Risk Assessment Agent Output (Screening Priority)
        const riskContainer = document.getElementById('riskContainer');
        if (riskContainer && data.risk_assessment) {
            const ra = data.risk_assessment;
            const pClass = ra.priority_level === 'High' ? 'risk-priority-high' : (ra.priority_level === 'Moderate' ? 'risk-priority-moderate' : 'risk-priority-low');
            const pBadgeClass = ra.priority_level === 'High' ? 'badge-malignant' : (ra.priority_level === 'Moderate' ? 'badge-precancerous' : 'badge-benign');

            const reasonsHTML = (ra.reasoning || []).map(r => `<li>${r}</li>`).join('');

            riskContainer.innerHTML = `
                <div class="risk-priority-banner ${pClass}">
                    <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
                        <span style="font-size:16px; font-weight:700;">
                            Screening Urgency: <span class="badge-tag ${pBadgeClass}" style="font-size:13px;">${ra.priority_level} Priority</span>
                        </span>
                        <span style="font-size:13px; font-weight:600; color:var(--text-secondary);">${ra.urgency_timeline}</span>
                    </div>
                    <ul class="reasoning-list">
                        ${reasonsHTML}
                    </ul>
                    <div style="margin-top:12px; padding:12px 14px; background:rgba(0,0,0,0.2); border-radius:10px; font-size:13.5px;">
                        <strong style="color:var(--accent-light);">Doctor Evaluation Guidance:</strong>
                        <p style="margin-top:4px; color:var(--text-primary);">${ra.recommendation}</p>
                    </div>
                    <p style="margin-top:10px; font-size:11.5px; color:var(--text-secondary); font-style:italic;">
                        ${ra.disclaimer}
                    </p>
                </div>
            `;
        }

        // Doctor Consultation Questions Agent Output
        const questionsContainer = document.getElementById('questionsContainer');
        const questionsSummaryPrompt = document.getElementById('questionsSummaryPrompt');
        if (questionsContainer && data.clinical_questions) {
            const cq = data.clinical_questions;
            if (questionsSummaryPrompt && cq.summary_prompt) {
                questionsSummaryPrompt.textContent = cq.summary_prompt;
            }
            questionsContainer.innerHTML = '';
            (cq.questions || []).forEach((q, idx) => {
                const impBadge = q.importance === 'High' ? 'badge-malignant' : 'badge-precancerous';
                const qDiv = document.createElement('div');
                qDiv.className = 'question-item';
                qDiv.innerHTML = `
                    <div class="question-header">
                        <span style="color:var(--accent-light); font-weight:600;">#${idx + 1} — ${q.category}</span>
                        <span class="badge-tag ${impBadge}" style="font-size:10px;">${q.importance} Priority</span>
                    </div>
                    <div class="question-text">${q.question}</div>
                    <div class="question-rationale">💡 Clinical purpose: ${q.rationale}</div>
                `;
                questionsContainer.appendChild(qDiv);
            });
        }

        // Show Results Section
        if (resultsSection) {
            resultsSection.style.display = 'block';
            resultsSection.classList.add('active');
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
    }
});
