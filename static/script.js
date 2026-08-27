// =========================================================
// HAZARDOUS DETECTION DASHBOARD
// =========================================================


// =========================================================
// DOM ELEMENTS
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


// =========================================================
// APPLICATION STATE
// =========================================================

let stream = null;

// Used for the webcam detection loop
let webcamLoopActive = false;

// Prevent multiple prediction requests
let detecting = false;

// Prevent health check while prediction is running
let isPredicting = false;

// Store current detections
let detectionHistory = [];


// =========================================================
// PERFORMANCE SETTINGS
// =========================================================

// YOLO inference size
const MODEL_SIZE = 416;

// Time to wait after a completed detection
//
// 600 ms = approximately 1.6 detections/sec maximum.
//
// This is more suitable for a Render CPU backend than
// trying to continuously send requests every 400 ms.
const DETECTION_INTERVAL = 600;

// Webcam capture quality
const WEBCAM_JPEG_QUALITY = 0.65;

// Upload image compression quality
const UPLOAD_JPEG_QUALITY = 0.80;

// Request timeout
const REQUEST_TIMEOUT = 15000;


// =========================================================
// CONFIDENCE SLIDER
// =========================================================

if (confidenceSlider && confidenceValue) {

    confidenceSlider.addEventListener("input", (event) => {

        confidenceValue.textContent =
            `${event.target.value}%`;

    });
}


function getConfidenceThreshold() {

    if (!confidenceSlider) {
        return 0.40;
    }

    const value =
        parseFloat(confidenceSlider.value) / 100;

    return Math.max(
        0.0,
        Math.min(1.0, value)
    );
}


// =========================================================
// IMAGE COMPRESSION
// =========================================================

function compressImage(
    fileOrCanvas,
    maxWidth = MODEL_SIZE,
    quality = UPLOAD_JPEG_QUALITY
) {

    return new Promise((resolve, reject) => {

        const img = new Image();

        let objectUrl = null;

        if (typeof fileOrCanvas === "string") {

            img.src = fileOrCanvas;

        } else {

            objectUrl =
                URL.createObjectURL(fileOrCanvas);

            img.src = objectUrl;
        }


        img.onload = () => {

            try {

                let width = img.width;
                let height = img.height;


                // Preserve aspect ratio
                if (width > maxWidth) {

                    const ratio =
                        maxWidth / width;

                    width = maxWidth;
                    height =
                        Math.round(height * ratio);
                }


                const tempCanvas =
                    document.createElement("canvas");

                tempCanvas.width = width;
                tempCanvas.height = height;


                const tempCtx =
                    tempCanvas.getContext("2d");

                tempCtx.drawImage(
                    img,
                    0,
                    0,
                    width,
                    height
                );


                tempCanvas.toBlob(
                    blob => {

                        if (objectUrl) {
                            URL.revokeObjectURL(objectUrl);
                        }

                        if (!blob) {
                            reject(
                                new Error(
                                    "Image compression failed."
                                )
                            );
                            return;
                        }

                        resolve(blob);

                    },
                    "image/jpeg",
                    quality
                );

            } catch (error) {

                if (objectUrl) {
                    URL.revokeObjectURL(objectUrl);
                }

                reject(error);
            }
        };


        img.onerror = () => {

            if (objectUrl) {
                URL.revokeObjectURL(objectUrl);
            }

            reject(
                new Error(
                    "Unable to load image."
                )
            );
        };
    });
}


// =========================================================
// FETCH WITH TIMEOUT
// =========================================================

async function fetchWithTimeout(
    url,
    options = {},
    timeout = REQUEST_TIMEOUT
) {

    const controller =
        new AbortController();

    const timeoutId =
        setTimeout(
            () => controller.abort(),
            timeout
        );


    try {

        return await fetch(
            url,
            {
                ...options,
                signal: controller.signal
            }
        );

    } finally {

        clearTimeout(timeoutId);
    }
}


// =========================================================
// API / SYSTEM HEALTH
// =========================================================

async function checkSystemHealth() {

    // Don't run a health request while
    // the prediction request is active.
    if (isPredicting) {
        return;
    }


    const pill =
        $("apiStatusPill");

    const pillText =
        $("apiStatusText");

    const systemStatus =
        $("systemStatus");

    const statusDot =
        $("statusIndicator");


    try {

        const response =
            await fetchWithTimeout(
                "/health",
                {
                    method: "GET"
                },
                8000
            );


        const rawText =
            await response.text();


        let data = {};

        try {

            data =
                JSON.parse(rawText);

        } catch {

            throw new Error(
                `Server returned status ${response.status}`
            );
        }


        const isRunning =
            response.ok &&
            data.status === "running" &&
            data.model_loaded === true;


        if (!isRunning) {

            throw new Error(
                data.message ||
                data.error ||
                "API is not running."
            );
        }


        // ---------------------------------------------
        // ONLINE
        // ---------------------------------------------

        if (pill && pillText) {

            pill.className =
                "api-pill online";

            pillText.textContent =
                "API Online";
        }


        if (systemStatus) {

            systemStatus.className =
                "status-running";

            systemStatus.textContent =
                "System Running — Model Loaded";
        }


        if (statusDot) {

            statusDot.className =
                "status-indicator running";
        }


    } catch (error) {

        // ---------------------------------------------
        // OFFLINE
        // ---------------------------------------------

        if (pill && pillText) {

            pill.className =
                "api-pill offline";

            pillText.textContent =
                "API Offline";
        }


        if (systemStatus) {

            systemStatus.className =
                "status-error";

            if (error.name === "AbortError") {

                systemStatus.textContent =
                    "API Offline — Request Timeout";

            } else {

                systemStatus.textContent =
                    `API Offline — ${error.message}`;
            }
        }


        if (statusDot) {

            statusDot.className =
                "status-indicator error";
        }
    }
}


// =========================================================
// DETECTION DETAILS
// =========================================================

function showDetails() {

    if (!detectionSummary || !detectionsBox) {
        return;
    }


    detectionSummary.textContent =
        `Objects Detected: ${detectionHistory.length}`;


    if (detectionHistory.length === 0) {

        detectionsBox.innerHTML =
            '<div class="detection-item">' +
            'No hazardous objects detected.' +
            '</div>';

        return;
    }


    detectionsBox.innerHTML =
        detectionHistory.map((d, i) => {

            const isShockAbsorber =
                d.class_name === "ShockAbsorber" ||
                d.c_id === 0;


            const classColor =
                isShockAbsorber
                    ? "#4ade80"
                    : "#f97316";


            const confidence =
                (
                    Number(d.confidence || 0) * 100
                ).toFixed(2);


            return `
                <div class="detection-item">
                    <strong>Detection ${i + 1}</strong><br>

                    Class:
                    <span
                        style="
                            color: ${classColor};
                            font-weight: bold;
                        "
                    >
                        ${d.class_name || d.label || "Hazard"}
                    </span>
                    <br>

                    Confidence:
                    ${confidence}%
                </div>
            `;

        }).join("");
}


// =========================================================
// CLEAR DETECTION DETAILS
// =========================================================

function clearDetails() {

    detectionHistory = [];

    showDetails();
}


// =========================================================
// IMAGE PREVIEW
// =========================================================

if (imageInput) {

    imageInput.addEventListener(
        "change",
        () => {

            const file =
                imageInput.files[0];


            if (!file) {
                return;
            }


            clearDetails();


            const objectUrl =
                URL.createObjectURL(file);


            if (previewImage) {

                previewImage.src =
                    objectUrl;

                previewImage.style.display =
                    "block";
            }


            if (previewMessage) {

                previewMessage.style.display =
                    "none";
            }


            if (resultImage) {

                resultImage.src = "";

                resultImage.style.display =
                    "none";
            }


            if (resultMessage) {

                resultMessage.style.display =
                    "block";

                resultMessage.textContent =
                    "Click Detect Objects to analyze the image.";
            }
        }
    );
}


// =========================================================
// IMAGE UPLOAD DETECTION
// =========================================================

if (detectButton) {

    detectButton.addEventListener(
        "click",
        async () => {

            const file =
                imageInput?.files[0];


            if (!file) {

                alert(
                    "Please select an image first."
                );

                return;
            }


            clearDetails();


            detectButton.disabled =
                true;

            detectButton.textContent =
                "Detecting...";


            isPredicting = true;


            try {

                // -----------------------------------------
                // Compress uploaded image
                // -----------------------------------------

                const compressedBlob =
                    await compressImage(
                        file,
                        MODEL_SIZE,
                        UPLOAD_JPEG_QUALITY
                    );


                // -----------------------------------------
                // Prepare request
                // -----------------------------------------

                const form =
                    new FormData();


                form.append(
                    "image",
                    compressedBlob,
                    "upload.jpg"
                );


                form.append(
                    "threshold",
                    getConfidenceThreshold()
                );


                form.append(
                    "is_webcam",
                    "false"
                );


                // -----------------------------------------
                // Send to Flask
                // -----------------------------------------

                const response =
                    await fetchWithTimeout(
                        "/predict",
                        {
                            method: "POST",
                            body: form
                        }
                    );


                const rawText =
                    await response.text();


                let data;


                try {

                    data =
                        JSON.parse(rawText);

                } catch {

                    throw new Error(
                        `Server Error (${response.status}): ` +
                        `Non-JSON response received.`
                    );
                }


                if (
                    !response.ok ||
                    data.error ||
                    data.success === false
                ) {

                    throw new Error(
                        data.error ||
                        `Server HTTP ${response.status}`
                    );
                }


                // -----------------------------------------
                // Store detections
                // -----------------------------------------

                detectionHistory =
                    data.detections || [];


                detectionHistory.sort(
                    (a, b) =>
                        b.confidence -
                        a.confidence
                );


                showDetails();


                // -----------------------------------------
                // Display annotated image
                // -----------------------------------------

                if (
                    data.annotated_image
                ) {

                    resultImage.src =
                        `data:image/jpeg;base64,` +
                        data.annotated_image;

                    resultImage.style.display =
                        "block";


                    if (resultMessage) {

                        resultMessage.style.display =
                            "none";
                    }

                } else {

                    resultImage.src =
                        previewImage.src;

                    resultImage.style.display =
                        "block";


                    if (resultMessage) {

                        resultMessage.style.display =
                            "none";
                    }
                }


            } catch (error) {

                console.error(
                    "Image detection error:",
                    error
                );


                if (resultMessage) {

                    resultMessage.textContent =
                        `Detection failed: ${error.message}`;

                    resultMessage.style.display =
                        "block";
                }


            } finally {

                isPredicting =
                    false;


                detectButton.disabled =
                    false;

                detectButton.textContent =
                    "Detect Objects";
            }
        }
    );
}


// =========================================================
// START WEBCAM
// =========================================================

if (startWebcamButton) {

    startWebcamButton.addEventListener(
        "click",
        async () => {

            try {

                clearDetails();


                // -----------------------------------------
                // Request camera
                // -----------------------------------------

                stream =
                    await navigator.mediaDevices
                        .getUserMedia({
                            video: {
                                width: {
                                    ideal: 640
                                },
                                height: {
                                    ideal: 480
                                },
                                facingMode: "environment"
                            },
                            audio: false
                        });


                webcam.srcObject =
                    stream;


                await webcam.play();


                // -----------------------------------------
                // Canvas matches actual webcam size
                // -----------------------------------------

                canvas.width =
                    webcam.videoWidth || 640;

                canvas.height =
                    webcam.videoHeight || 480;


                webcam.style.display =
                    "block";

                canvas.style.display =
                    "block";


                if (webcamMessage) {

                    webcamMessage.style.display =
                        "none";
                }


                startWebcamButton.disabled =
                    true;

                stopWebcamButton.disabled =
                    false;


                if (liveStatus) {

                    liveStatus.textContent =
                        "● Live Detection Running";
                }


                // -----------------------------------------
                // Start detection loop
                // -----------------------------------------

                webcamLoopActive =
                    true;


                runWebcamDetectionLoop();


            } catch (error) {

                console.error(
                    "Webcam error:",
                    error
                );


                alert(
                    "Please allow camera permission."
                );
            }
        }
    );
}


// =========================================================
// WEBCAM DETECTION LOOP
// =========================================================
//
// IMPORTANT:
// We do NOT use setInterval here.
//
// The next request starts only after the previous
// request has completely finished.
//
// This is much better for Render.
// =========================================================

async function runWebcamDetectionLoop() {

    while (
        webcamLoopActive &&
        stream
    ) {

        // ---------------------------------------------
        // Perform one detection
        // ---------------------------------------------

        await detectWebcam();


        // ---------------------------------------------
        // Wait before next detection
        // ---------------------------------------------

        if (
            webcamLoopActive &&
            stream
        ) {

            await sleep(
                DETECTION_INTERVAL
            );
        }
    }
}


// =========================================================
// SLEEP HELPER
// =========================================================

function sleep(milliseconds) {

    return new Promise(
        resolve =>
            setTimeout(
                resolve,
                milliseconds
            )
    );
}


// =========================================================
// WEBCAM DETECTION
// =========================================================

async function detectWebcam() {

    if (
        !stream ||
        detecting ||
        !webcam.videoWidth ||
        !webcam.videoHeight
    ) {
        return;
    }


    detecting =
        true;

    isPredicting =
        true;


    try {

        // ---------------------------------------------
        // Keep original webcam aspect ratio
        // ---------------------------------------------

        const videoWidth =
            webcam.videoWidth;

        const videoHeight =
            webcam.videoHeight;


        // Use a 416-wide canvas while preserving
        // the original 4:3 camera aspect ratio.
        const scale =
            MODEL_SIZE / videoWidth;


        const captureWidth =
            MODEL_SIZE;

        const captureHeight =
            Math.round(
                videoHeight * scale
            );


        const tempCanvas =
            document.createElement("canvas");


        tempCanvas.width =
            captureWidth;

        tempCanvas.height =
            captureHeight;


        const tempCtx =
            tempCanvas.getContext("2d");


        // ---------------------------------------------
        // Draw webcam frame
        // ---------------------------------------------

        tempCtx.drawImage(
            webcam,
            0,
            0,
            captureWidth,
            captureHeight
        );


        // ---------------------------------------------
        // Convert frame to JPEG
        // ---------------------------------------------

        const blob =
            await new Promise(
                resolve =>
                    tempCanvas.toBlob(
                        resolve,
                        "image/jpeg",
                        WEBCAM_JPEG_QUALITY
                    )
            );


        if (!blob) {
            return;
        }


        // ---------------------------------------------
        // Prepare request
        // ---------------------------------------------

        const form =
            new FormData();


        form.append(
            "image",
            blob,
            "webcam.jpg"
        );


        form.append(
            "threshold",
            getConfidenceThreshold()
        );


        form.append(
            "is_webcam",
            "true"
        );


        // ---------------------------------------------
        // Send to Render
        // ---------------------------------------------

        const response =
            await fetchWithTimeout(
                "/predict",
                {
                    method: "POST",
                    body: form
                }
            );


        const rawText =
            await response.text();


        let data;


        try {

            data =
                JSON.parse(rawText);

        } catch {

            console.warn(
                "Render returned non-JSON response."
            );

            return;
        }


        if (
            !response.ok ||
            data.error ||
            data.success === false
        ) {

            console.warn(
                "Prediction error:",
                data.error
            );

            return;
        }


        // ---------------------------------------------
        // Get detections
        // ---------------------------------------------

        const detections =
            data.detections || [];


        if (detections.length > 0) {

            // Highest confidence first
            detections.sort(
                (a, b) =>
                    b.confidence -
                    a.confidence
            );


            // -----------------------------------------
            // Scale backend coordinates to the
            // actual webcam canvas
            // -----------------------------------------

            const scaleX =
                canvas.width /
                captureWidth;


            const scaleY =
                canvas.height /
                captureHeight;


            const rescaledDetections =
                detections.map(
                    detection => {

                        if (!detection.bbox) {
                            return detection;
                        }


                        return {
                            ...detection,

                            bbox: {

                                x1:
                                    detection.bbox.x1 *
                                    scaleX,

                                y1:
                                    detection.bbox.y1 *
                                    scaleY,

                                x2:
                                    detection.bbox.x2 *
                                    scaleX,

                                y2:
                                    detection.bbox.y2 *
                                    scaleY
                            }
                        };
                    }
                );


            // -----------------------------------------
            // Draw boxes
            // -----------------------------------------

            drawBoxes(
                rescaledDetections
            );


            // -----------------------------------------
            // Update details
            // -----------------------------------------

            detectionHistory =
                detections;


            showDetails();


        } else {

            // No objects
            if (ctx) {

                ctx.clearRect(
                    0,
                    0,
                    canvas.width,
                    canvas.height
                );
            }


            detectionHistory = [];

            showDetails();
        }


    } catch (error) {

        if (
            error.name !==
            "AbortError"
        ) {

            console.error(
                "Webcam processing error:",
                error
            );
        }


    } finally {

        detecting =
            false;

        isPredicting =
            false;
    }
}


// =========================================================
// DRAW DETECTION BOXES
// =========================================================

function drawBoxes(detections) {

    if (!ctx) {
        return;
    }


    // Clear previous boxes
    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );


    detections.forEach(
        detection => {

            const bbox =
                detection.bbox;


            if (!bbox) {
                return;
            }


            const className =
                detection.class_name ||
                detection.label ||
                "Hazard";


            const confidence =
                (
                    Number(
                        detection.confidence || 0
                    ) * 100
                ).toFixed(1);


            const label =
                `${className} ${confidence}%`;


            // -----------------------------------------
            // Class colors
            // -----------------------------------------

            const isShockAbsorber =
                className ===
                "ShockAbsorber" ||
                detection.c_id === 0;


            const color =
                isShockAbsorber
                    ? "#4ade80"
                    : "#f97316";


            // -----------------------------------------
            // Bounding box
            // -----------------------------------------

            ctx.strokeStyle =
                color;

            ctx.lineWidth =
                3;


            ctx.strokeRect(
                bbox.x1,
                bbox.y1,
                bbox.x2 - bbox.x1,
                bbox.y2 - bbox.y1
            );


            // -----------------------------------------
            // Label
            // -----------------------------------------

            ctx.fillStyle =
                color;

            ctx.font =
                "bold 15px Inter, sans-serif";


            ctx.fillText(
                label,
                bbox.x1,
                Math.max(
                    18,
                    bbox.y1 - 7
                )
            );
        }
    );
}


// =========================================================
// STOP WEBCAM
// =========================================================

if (stopWebcamButton) {

    stopWebcamButton.addEventListener(
        "click",
        stopWebcam
    );
}


function stopWebcam() {

    // ---------------------------------------------
    // Stop detection loop
    // ---------------------------------------------

    webcamLoopActive =
        false;


    detecting =
        false;


    isPredicting =
        false;


    // ---------------------------------------------
    // Stop camera tracks
    // ---------------------------------------------

    if (stream) {

        stream
            .getTracks()
            .forEach(
                track =>
                    track.stop()
            );
    }


    stream =
        null;


    // ---------------------------------------------
    // Disconnect video
    // ---------------------------------------------

    if (webcam) {

        webcam.pause();

        webcam.srcObject =
            null;

        webcam.style.display =
            "none";
    }


    // ---------------------------------------------
    // Clear canvas
    // ---------------------------------------------

    if (ctx) {

        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );
    }


    if (canvas) {

        canvas.style.display =
            "none";
    }


    // ---------------------------------------------
    // Show webcam message
    // ---------------------------------------------

    if (webcamMessage) {

        webcamMessage.style.display =
            "block";
    }


    // ---------------------------------------------
    // Update buttons
    // ---------------------------------------------

    if (startWebcamButton) {

        startWebcamButton.disabled =
            false;
    }


    if (stopWebcamButton) {

        stopWebcamButton.disabled =
            true;
    }


    if (liveStatus) {

        liveStatus.textContent =
            "Webcam stopped.";
    }
}


// =========================================================
// INITIALIZATION
// =========================================================

document.addEventListener(
    "DOMContentLoaded",
    () => {

        // Initial API check
        checkSystemHealth();


        // Check API every 10 seconds
        setInterval(
            checkSystemHealth,
            10000
        );
    }
);


// =========================================================
// CLEANUP WHEN PAGE CLOSES
// =========================================================

window.addEventListener(
    "beforeunload",
    () => {

        webcamLoopActive =
            false;


        if (stream) {

            stream
                .getTracks()
                .forEach(
                    track =>
                        track.stop()
                );
        }
    }
);