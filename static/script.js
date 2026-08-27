// =========================================================
// DASHBOARD DOM ELEMENTS & STATE
// =========================================================

const $ = id => document.getElementById(id);

const imageInput = $("imageInput");
const detectButton = $("detectButton");
const previewImage = $("previewImage");
const previewMessage = $("previewMessage");
const resultImage = $("resultImage");
const resultMessage = $("resultMessage");
const detectionSummary = $("detectionSummary");
const detectionsBox = $("detections");

const webcam = $("webcam");
const canvas = $("detectionCanvas");
const ctx = canvas ? canvas.getContext("2d") : null;
const startWebcamButton = $("startWebcamButton");
const stopWebcamButton = $("stopWebcamButton");
const webcamMessage = $("webcamMessage");
const liveStatus = $("liveDetectionStatus");

const confidenceSlider = $("confidenceSlider");
const confidenceValue = $("confidenceValue");

let stream = null;
let timer = null;
let detecting = false;
let isPredicting = false; // Prevents parallel health check collisions
let detectionHistory = [];

// Throttled interval for cloud CPU stability
const INTERVAL = 800;


// =========================================================
// THRESHOLD SLIDER CONTROLLER
// =========================================================

if (confidenceSlider && confidenceValue) {
    confidenceSlider.addEventListener("input", (e) => {
        confidenceValue.textContent = `${e.target.value}%`;
    });
}

function getConfidenceThreshold() {
    return confidenceSlider ? (parseFloat(confidenceSlider.value) / 100) : 0.40;
}


// =========================================================
// IMAGE COMPRESSOR HELPER (PREVENTS 502 TIMEOUTS)
// =========================================================

function compressImage(fileOrCanvas, maxWidth = 416, quality = 0.7) {
    return new Promise((resolve) => {
        const img = new Image();
        img.src = typeof fileOrCanvas === "string" ? fileOrCanvas : URL.createObjectURL(fileOrCanvas);
        img.onload = () => {
            const tempCanvas = document.createElement("canvas");
            let width = img.width;
            let height = img.height;

            if (width > maxWidth) {
                height = Math.round((height * maxWidth) / width);
                width = maxWidth;
            }

            tempCanvas.width = width;
            tempCanvas.height = height;

            const tempCtx = tempCanvas.getContext("2d");
            tempCtx.drawImage(img, 0, 0, width, height);

            tempCanvas.toBlob((blob) => resolve(blob), "image/jpeg", quality);
        };
    });
}


// =========================================================
// UNIFIED HEALTH & SYSTEM STATUS CHECK
// =========================================================

async function checkSystemHealth() {
    if (isPredicting) return; // Pause health check during active model inference

    const pill = $("apiStatusPill");
    const pillText = $("apiStatusText");
    const systemStatus = $("systemStatus");
    const statusDot = $("statusIndicator");

    try {
        const response = await fetch("/health", { method: "GET" });
        const rawText = await response.text();
        
        let data = {};
        try {
            data = JSON.parse(rawText);
        } catch (e) {
            throw new Error(`Server returned status ${response.status}`);
        }

        const isRunning = response.ok && (data.status === "running" || data.status === "ok");

        if (isRunning) {
            if (pill && pillText) {
                pill.className = "api-pill online";
                pillText.textContent = "API Online";
            }
            if (systemStatus) {
                systemStatus.className = "status-running";
                systemStatus.textContent = "System Running — Model Loaded";
            }
            if (statusDot) {
                statusDot.className = "status-indicator running";
            }
        } else {
            throw new Error(data.message || data.error || `HTTP ${response.status}`);
        }
    } catch (err) {
        if (pill && pillText) {
            pill.className = "api-pill offline";
            pillText.textContent = "API Offline";
        }
        if (systemStatus) {
            systemStatus.className = "status-error";
            systemStatus.textContent = err.name === "TypeError" 
                ? "API Offline — Connection Failed" 
                : `System Warning — ${err.message}`;
        }
        if (statusDot) {
            statusDot.className = "status-indicator error";
        }
    }
}


// =========================================================
// DETECTION DETAILS DISPLAY
// =========================================================

function showDetails() {
    if (!detectionSummary || !detectionsBox) return;

    detectionSummary.textContent = `Objects Detected: ${detectionHistory.length}`;

    if (!detectionHistory.length) {
        detectionsBox.innerHTML = '<div class="detection-item">No hazardous objects detected.</div>';
        return;
    }

    // Green (#4ade80) for ShockAbsorber (Index 0), Orange (#f97316) for cylinder (Index 1)
    detectionsBox.innerHTML = detectionHistory.map((d, i) => `
        <div class="detection-item">
            <strong>Detection ${i + 1}</strong><br>
            Class: <span style="color: ${d.class_name === 'ShockAbsorber' ? '#4ade80' : '#f97316'}; font-weight: bold;">${d.class_name || d.label || 'Hazard'}</span><br>
            Confidence: ${(d.confidence * 100).toFixed(2)}%
        </div>
    `).join("");
}

function clearDetails() {
    detectionHistory = [];
    showDetails();
}


// =========================================================
// IMAGE PREVIEW & FILE UPLOAD DETECTION
// =========================================================

if (imageInput) {
    imageInput.addEventListener("change", () => {
        const file = imageInput.files[0];
        if (!file) return;

        clearDetails();
        const objectUrl = URL.createObjectURL(file);
        
        previewImage.src = objectUrl;
        previewImage.style.display = "block";
        previewMessage.style.display = "none";

        resultImage.src = "";
        resultImage.style.display = "none";
        resultMessage.style.display = "block";
        resultMessage.textContent = "Click Detect Objects to analyze the image.";
    });
}

if (detectButton) {
    detectButton.addEventListener("click", async () => {
        const file = imageInput?.files[0];
        if (!file) return alert("Please select an image first.");

        clearDetails();
        detectButton.disabled = true;
        detectButton.textContent = "Detecting...";
        isPredicting = true;

        try {
            // Compress uploaded image down to 416px width to prevent 502 timeouts
            const compressedBlob = await compressImage(file, 416, 0.7);

            const form = new FormData();
            form.append("image", compressedBlob, "upload.jpg");
            form.append("threshold", getConfidenceThreshold());

            const res = await fetch("/predict", { method: "POST", body: form });
            const rawText = await res.text();
            let data;

            try {
                data = JSON.parse(rawText);
            } catch (jsonErr) {
                throw new Error(`Server Error (${res.status}): Non-JSON response received.`);
            }

            if (!res.ok || data.error || data.success === false) {
                throw new Error(data.error || `Server HTTP ${res.status}`);
            }

            detectionHistory = data.detections || [];
            showDetails();

            // Display annotated image returned from Flask server
            if (data.annotated_image) {
                resultImage.src = `data:image/jpeg;base64,${data.annotated_image}`;
                resultImage.style.display = "block";
                resultMessage.style.display = "none";
            } else {
                resultImage.src = previewImage.src;
                resultImage.style.display = "block";
                resultMessage.style.display = "none";
            }

        } catch (e) {
            resultMessage.textContent = `Detection failed: ${e.message}`;
            resultMessage.style.display = "block";
        } finally {
            isPredicting = false;
            detectButton.disabled = false;
            detectButton.textContent = "Detect Objects";
        }
    });
}


// =========================================================
// WEBCAM LIVE STREAM & AUTOMATIC CAPTURE DETECTION
// =========================================================

if (startWebcamButton) {
    startWebcamButton.addEventListener("click", async () => {
        try {
            clearDetails();
            stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: { ideal: 640 }, height: { ideal: 480 } }, 
                audio: false 
            });

            webcam.srcObject = stream;
            await webcam.play();

            canvas.width = webcam.videoWidth;
            canvas.height = webcam.videoHeight;

            webcam.style.display = "block";
            canvas.style.display = "block";
            webcamMessage.style.display = "none";

            startWebcamButton.disabled = true;
            stopWebcamButton.disabled = false;
            if (liveStatus) liveStatus.textContent = "● Live Detection Running";

            timer = setInterval(detectWebcam, INTERVAL);

        } catch (err) {
            alert("Please allow camera permission.");
        }
    });
}

async function detectWebcam() {
    if (!stream || detecting || !webcam.videoWidth) return;
    detecting = true;
    isPredicting = true;

    try {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = 416;
        tempCanvas.height = 416;
        const tempCtx = tempCanvas.getContext("2d");
        
        tempCtx.drawImage(webcam, 0, 0, 416, 416);

        const blob = await new Promise(res => tempCanvas.toBlob(res, "image/jpeg", 0.5));
        
        const form = new FormData();
        form.append("image", blob, "webcam.jpg");
        form.append("threshold", getConfidenceThreshold());

        const res = await fetch("/predict", { method: "POST", body: form });
        const rawText = await res.text();
        
        let data;
        try {
            data = JSON.parse(rawText);
        } catch (jsonErr) {
            return;
        }

        if (!res.ok || data.error) return;

        const detections = data.detections || [];

        if (detections.length > 0) {
            detections.sort((a, b) => b.confidence - a.confidence);

            const scaleX = webcam.videoWidth / 416;
            const scaleY = webcam.videoHeight / 416;

            const rescaledDetections = detections.map(d => {
                if (!d.bbox) return d;
                return {
                    ...d,
                    bbox: {
                        x1: d.bbox.x1 * scaleX,
                        y1: d.bbox.y1 * scaleY,
                        x2: d.bbox.x2 * scaleX,
                        y2: d.bbox.y2 * scaleY
                    }
                };
            });

            drawBoxes(rescaledDetections);
            detectionHistory = detections;
            showDetails();

            if (data.annotated_image) {
                resultImage.src = `data:image/jpeg;base64,${data.annotated_image}`;
                resultImage.style.display = "block";
                resultMessage.style.display = "none";
            }
        } else {
            if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    } catch (e) {
        console.error("Webcam processing error:", e);
    } finally {
        detecting = false;
        isPredicting = false;
    }
}

function drawBoxes(detections) {
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    detections.forEach(d => {
        const b = d.bbox;
        if (!b) return;
        const label = `${d.class_name || d.label} ${(d.confidence * 100).toFixed(1)}%`;
        
        // Green for ShockAbsorber (Index 0), Orange for cylinder (Index 1)
        const color = (d.class_name === 'ShockAbsorber' || d.c_id === 0) ? "#4ade80" : "#f97316";

        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(b.x1, b.y1, b.x2 - b.x1, b.y2 - b.y1);

        ctx.fillStyle = color;
        ctx.font = "bold 15px Inter, sans-serif";
        ctx.fillText(label, b.x1, Math.max(18, b.y1 - 7));
    });
}

if (stopWebcamButton) {
    stopWebcamButton.addEventListener("click", () => {
        clearInterval(timer);
        timer = null;

        if (stream) stream.getTracks().forEach(t => t.stop());
        stream = null;
        webcam.srcObject = null;

        if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height);

        webcam.style.display = "none";
        canvas.style.display = "none";
        webcamMessage.style.display = "block";

        startWebcamButton.disabled = false;
        stopWebcamButton.disabled = true;

        if (liveStatus) liveStatus.textContent = "Webcam stopped.";
    });
}


// =========================================================
// INITIALIZATION
// =========================================================

document.addEventListener("DOMContentLoaded", () => {
    checkSystemHealth();
    setInterval(checkSystemHealth, 10000);
});

window.addEventListener("beforeunload", () => {
    stream?.getTracks().forEach(t => t.stop());
});