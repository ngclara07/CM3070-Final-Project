// === web_app/static/script.js ===


// ============================================================
// DOM references
// ============================================================

const textInput =
  document.getElementById("textInput");

const webcam =
  document.getElementById("webcam");

const canvas =
  document.getElementById("frameCanvas");


const startBtn =
  document.getElementById("startBtn");

const stopBtn =
  document.getElementById("stopBtn");

const resetBtn =
  document.getElementById("resetBtn");

const resetTemporalBtn =
  document.getElementById("resetTemporalBtn");


const statusBox =
  document.getElementById("status");

const sessionStatus =
  document.getElementById("sessionStatus");

const audioStatus =
  document.getElementById("audioStatus");

const webcamStatus =
  document.getElementById("webcamStatus");

const modelStatusText =
  document.getElementById("modelStatusText");

const webcamModelStatusText =
  document.getElementById("webcamModelStatusText");


const predictionBox =
  document.getElementById("prediction");

const confidencePercent =
  document.getElementById("confidencePercent");

const confidenceFill =
  document.getElementById("confidenceFill");

const confidenceLevel =
  document.getElementById("confidenceLevel");


const rawPrediction =
  document.getElementById("rawPrediction");

const rawConfidence =
  document.getElementById("rawConfidence");


const temporalSamples =
  document.getElementById("temporalSamples");

const temporalWindow =
  document.getElementById("temporalWindow");

const temporalWindowStatus =
  document.getElementById("temporalWindowStatus");


const secondaryState =
  document.getElementById("secondaryState");

const confidenceGap =
  document.getElementById("confidenceGap");

const featureDimension =
  document.getElementById("featureDimension");

const deviceInfo =
  document.getElementById("deviceInfo");

const probabilitiesBox =
  document.getElementById("probabilities");

const rawProbabilitiesBox =
  document.getElementById("rawProbabilities");


const webcamPrediction =
  document.getElementById("webcamPrediction");

const webcamConfidence =
  document.getElementById("webcamConfidence");

const webcamProbabilityBars =
  document.getElementById("webcamProbabilityBars");


const webcamCalibrationUsed =
  document.getElementById("webcamCalibrationUsed");

const activeModalities =
  document.getElementById("activeModalities");

const technicalRawState =
  document.getElementById("technicalRawState");

const technicalTemporalSamples =
  document.getElementById("technicalTemporalSamples");

const sessionIdDisplay =
  document.getElementById("sessionIdDisplay");


const modelReady =
  document.getElementById("modelReady");

const webcamModelReady =
  document.getElementById("webcamModelReady");

const textReady =
  document.getElementById("textReady");

const keyReady =
  document.getElementById("keyReady");

const audioReady =
  document.getElementById("audioReady");

const imageReady =
  document.getElementById("imageReady");


const textCard =
  document.getElementById("textCard");

const webcamCard =
  document.getElementById("webcamCard");

const audioCard =
  document.getElementById("audioCard");


const charCount =
  document.getElementById("charCount");

const keyCount =
  document.getElementById("keyCount");


// ============================================================
// Configuration
// ============================================================

const MIN_TEXT_CHARS = 20;
const MIN_KEYPRESSES = 20;

// Keep this aligned with the backend prediction cadence.
const LIVE_INTERVAL_MS = 15000;


// ============================================================
// Session state
// ============================================================

let keystrokeEvents = [];
let activeKeys = new Set();


let mediaStream = null;
let mediaRecorder = null;

let audioChunks = [];
let latestAudioBlob = null;


let liveTimer = null;

let sessionActive = false;

let fusionModelLoaded = false;
let webcamModelLoaded = false;

let predictionInFlight = false;


// ============================================================
// Browser session identifier
// ============================================================

function createSessionId() {

  if (
    window.crypto &&
    typeof window.crypto.randomUUID === "function"
  ) {
    return window.crypto.randomUUID();
  }

  return (
    "session-" +
    Date.now().toString(36) +
    "-" +
    Math.random().toString(36).slice(2)
  );
}


let sessionId = createSessionId();


// ============================================================
// Key normalisation
// ============================================================

function normaliseKey(event) {

  if (event.key === "Backspace") {
    return "backspace";
  }

  if (event.key === "Delete") {
    return "delete";
  }

  if (event.key === " ") {
    return "space";
  }

  if (event.key.length === 1) {
    return event.key.toLowerCase();
  }

  return event.key.toLowerCase();
}


// ============================================================
// Readiness helpers
// ============================================================

function setReady(
  element,
  isReady,
  readyText = "Ready",
  missingText = "Missing"
) {

  element.classList.toggle(
    "active",
    isReady
  );

  element.querySelector("b").textContent =
    isReady
      ? readyText
      : missingText;
}


function updateReadiness() {

  const textOk =
    textInput.value.trim().length
    >= MIN_TEXT_CHARS;


  const keydowns =
    keystrokeEvents.filter(
      event =>
        event.type === "down"
    ).length;


  const keyOk =
    keydowns
    >= MIN_KEYPRESSES;


  const audioOk =
    latestAudioBlob !== null;


  const imageOk =
    mediaStream !== null;


  charCount.textContent =
    textInput.value.trim().length;


  keyCount.textContent =
    keydowns;


  setReady(
    modelReady,
    fusionModelLoaded,
    "Loaded",
    "Failed"
  );


  setReady(
    webcamModelReady,
    webcamModelLoaded,
    "Loaded",
    "Unavailable"
  );


  setReady(
    textReady,
    textOk
  );


  setReady(
    keyReady,
    keyOk
  );


  setReady(
    audioReady,
    audioOk,
    "Ready",
    "Optional"
  );


  setReady(
    imageReady,
    imageOk,
    "Ready",
    "Optional"
  );


  textCard.classList.toggle(
    "active",
    textOk && keyOk
  );


  webcamCard.classList.toggle(
    "active",
    imageOk
  );


  audioCard.classList.toggle(
    "active",
    audioOk
  );


  textCard
    .querySelector(".badge")
    .textContent =
      textOk && keyOk
        ? "active"
        : "inactive";


  webcamCard
    .querySelector(".badge")
    .textContent =
      imageOk
        ? "active"
        : "inactive";


  audioCard
    .querySelector(".badge")
    .textContent =
      audioOk
        ? "active"
        : "inactive";
}


// ============================================================
// Model status
// ============================================================

async function checkModelStatus() {

  try {

    const response =
      await fetch(
        "/model-status"
      );


    const data =
      await response.json();


    fusionModelLoaded =
      Boolean(
        data.fusion_model
      );


    webcamModelLoaded =
      Boolean(
        data.webcam_calibrated_image_model
      );


    const configuredWindow =
      Number(
        data.temporal_probability_window
        || 5
      );


    temporalWindow.textContent =
      configuredWindow;


    temporalWindowStatus.textContent =
      `0 / ${configuredWindow}`;


    modelStatusText.textContent =
      fusionModelLoaded
        ? (
            "Fusion model loaded. " +
            `Backend: ${data.inference_backend}`
          )
        : (
            "Fusion model unavailable. " +
            "Fallback active. " +
            `${data.error || ""}`
          );


    webcamModelStatusText.textContent =
      webcamModelLoaded
        ? (
            "Webcam-calibrated image " +
            "classifier loaded."
          )
        : (
            "Webcam-calibrated classifier " +
            "is unavailable."
          );


    updateReadiness();

  } catch (error) {

    fusionModelLoaded = false;
    webcamModelLoaded = false;

    modelStatusText.textContent =
      "Could not query model status.";

    webcamModelStatusText.textContent =
      "Webcam classifier status unavailable.";

    updateReadiness();
  }
}


// ============================================================
// Keystroke capture
// ============================================================

textInput.addEventListener(
  "keydown",
  event => {

    const key =
      normaliseKey(event);


    if (activeKeys.has(key)) {
      return;
    }


    activeKeys.add(key);


    keystrokeEvents.push(
      {
        type: "down",

        key: key,

        timestamp_perf:
          performance.now()
          / 1000,

        timestamp_epoch:
          Date.now()
          / 1000
      }
    );


    updateReadiness();
  }
);


textInput.addEventListener(
  "keyup",
  event => {

    const key =
      normaliseKey(event);


    activeKeys.delete(key);


    keystrokeEvents.push(
      {
        type: "up",

        key: key,

        timestamp_perf:
          performance.now()
          / 1000,

        timestamp_epoch:
          Date.now()
          / 1000
      }
    );


    updateReadiness();
  }
);


textInput.addEventListener(
  "input",
  updateReadiness
);


// ============================================================
// Webcam frame capture
// ============================================================

function captureWebcamFrame() {

  if (
    !mediaStream ||
    webcam.videoWidth === 0 ||
    webcam.videoHeight === 0
  ) {
    return null;
  }


  canvas.width =
    webcam.videoWidth;


  canvas.height =
    webcam.videoHeight;


  const context =
    canvas.getContext("2d");


  context.drawImage(
    webcam,
    0,
    0,
    canvas.width,
    canvas.height
  );


  return canvas.toDataURL(
    "image/jpeg",
    0.90
  );
}


// ============================================================
// Audio recording cycle
// ============================================================

function beginAudioRecordingCycle() {

  if (
    !mediaRecorder ||
    !sessionActive
  ) {
    return;
  }


  if (
    mediaRecorder.state
    !== "inactive"
  ) {
    return;
  }


  audioChunks = [];


  mediaRecorder.start();


  setTimeout(
    () => {

      if (
        mediaRecorder &&
        mediaRecorder.state === "recording"
      ) {

        mediaRecorder.stop();
      }

    },
    LIVE_INTERVAL_MS
  );
}


// ============================================================
// Start session
// ============================================================

async function startSession() {

  if (sessionActive) {
    return;
  }


  // Start a fresh temporal sequence.
  sessionId =
    createSessionId();


  resetPredictionDisplay();


  try {

    mediaStream =
      await navigator.mediaDevices
        .getUserMedia(
          {
            video: true,
            audio: true
          }
        );


    webcam.srcObject =
      mediaStream;


    await webcam.play();


    webcamStatus.textContent =
      "Webcam capturing.";


    audioStatus.textContent =
      "Microphone recording.";


    mediaRecorder =
      new MediaRecorder(
        mediaStream
      );


    mediaRecorder.ondataavailable =
      event => {

        if (event.data.size > 0) {

          audioChunks.push(
            event.data
          );
        }
      };


    mediaRecorder.onstop =
      () => {

        if (audioChunks.length > 0) {

          latestAudioBlob =
            new Blob(
              audioChunks,
              {
                type: "audio/webm"
              }
            );


          audioStatus.textContent =
            "Latest audio chunk ready.";
        }


        audioChunks = [];


        updateReadiness();


        if (sessionActive) {

          beginAudioRecordingCycle();
        }
      };


    sessionActive = true;


    startBtn.disabled = true;
    stopBtn.disabled = false;


    sessionStatus.textContent =
      "Live session running.";


    statusBox.textContent =
      "Capturing live multimodal behaviour...";


    beginAudioRecordingCycle();


    liveTimer =
      setInterval(
        runLivePrediction,
        LIVE_INTERVAL_MS
      );


    updateReadiness();

  } catch (error) {

    // --------------------------------------------------------
    // Limited modality mode
    // --------------------------------------------------------

    sessionActive = true;

    startBtn.disabled = true;
    stopBtn.disabled = false;


    mediaStream = null;
    mediaRecorder = null;


    sessionStatus.textContent =
      "Session running with limited modalities.";


    webcamStatus.textContent =
      "Webcam unavailable.";


    audioStatus.textContent =
      "Microphone unavailable.";


    statusBox.textContent =
      "Camera/microphone unavailable. " +
      "Text and keystroke prediction remain available.";


    liveTimer =
      setInterval(
        runLivePrediction,
        LIVE_INTERVAL_MS
      );


    updateReadiness();
  }
}


// ============================================================
// Stop session
// ============================================================

function stopSession() {

  sessionActive = false;


  if (liveTimer) {

    clearInterval(
      liveTimer
    );

    liveTimer = null;
  }


  if (
    mediaRecorder &&
    mediaRecorder.state === "recording"
  ) {

    mediaRecorder.stop();
  }


  if (mediaStream) {

    mediaStream
      .getTracks()
      .forEach(
        track => track.stop()
      );
  }


  mediaStream = null;
  mediaRecorder = null;


  webcam.srcObject = null;


  startBtn.disabled = false;
  stopBtn.disabled = true;


  sessionStatus.textContent =
    "Session stopped.";


  statusBox.textContent =
    "Live session stopped.";


  webcamStatus.textContent =
    "Webcam inactive.";


  audioStatus.textContent =
    "Microphone inactive.";


  updateReadiness();
}


// ============================================================
// Temporal reset
// ============================================================

async function resetTemporalWindow() {

  try {

    const formData =
      new FormData();


    formData.append(
      "session_id",
      sessionId
    );


    const response =
      await fetch(
        "/reset_temporal",
        {
          method: "POST",
          body: formData
        }
      );


    const data =
      await response.json();


    if (!response.ok) {

      throw new Error(
        data.detail ||
        "Could not reset temporal history."
      );
    }


    temporalSamples.textContent =
      "0";


    temporalWindowStatus.textContent =
      `0 / ${data.temporal_window || 5}`;


    technicalTemporalSamples.textContent =
      "0";


    statusBox.textContent =
      "Temporal probability window reset.";

  } catch (error) {

    statusBox.textContent =
      "Temporal reset failed: " +
      error.message;
  }
}


// ============================================================
// Full reset
// ============================================================

async function resetSession() {

  try {

    await resetTemporalWindow();

  } catch (_) {
    // Continue resetting the interface.
  }


  stopSession();


  textInput.value = "";


  keystrokeEvents = [];

  activeKeys.clear();


  audioChunks = [];

  latestAudioBlob = null;


  // Generate a completely new temporal session.
  sessionId =
    createSessionId();


  resetPredictionDisplay();


  statusBox.textContent =
    "Session reset.";


  sessionStatus.textContent =
    "Session not started.";


  updateReadiness();
}


// ============================================================
// Prediction display reset
// ============================================================

function resetPredictionDisplay() {

  predictionBox.textContent =
    "—";


  confidencePercent.textContent =
    "—";


  confidenceFill.style.width =
    "0%";


  confidenceLevel.textContent =
    "—";


  confidenceLevel.classList.remove(
    "confidence-high",
    "confidence-medium",
    "confidence-low"
  );


  rawPrediction.textContent =
    "—";


  rawConfidence.textContent =
    "—";


  temporalSamples.textContent =
    "0";


  secondaryState.textContent =
    "—";


  confidenceGap.textContent =
    "—";


  featureDimension.textContent =
    "—";


  deviceInfo.textContent =
    "—";


  webcamPrediction.textContent =
    "—";


  webcamConfidence.textContent =
    "—";


  webcamCalibrationUsed.textContent =
    "—";


  activeModalities.textContent =
    "—";


  technicalRawState.textContent =
    "—";


  technicalTemporalSamples.textContent =
    "—";


  sessionIdDisplay.textContent =
    sessionId;


  probabilitiesBox.innerHTML =
    "";


  rawProbabilitiesBox.innerHTML =
    "";


  webcamProbabilityBars.innerHTML =
    "";
}


// ============================================================
// Run live prediction
// ============================================================

async function runLivePrediction() {

  if (predictionInFlight) {
    return;
  }


  try {

    updateReadiness();


    const text =
      textInput.value.trim();


    const keydowns =
      keystrokeEvents.filter(
        event =>
          event.type === "down"
      ).length;


    if (
      text.length < MIN_TEXT_CHARS ||
      keydowns < MIN_KEYPRESSES
    ) {

      statusBox.textContent =
        "Waiting for sufficient text " +
        "and keystroke data...";

      return;
    }


    predictionInFlight = true;


    statusBox.textContent =
      "Running multimodal fusion prediction...";


    const frameData =
      captureWebcamFrame();


    const formData =
      new FormData();


    formData.append(
      "session_id",
      sessionId
    );


    formData.append(
      "text",
      text
    );


    formData.append(
      "keystroke_events",
      JSON.stringify(
        keystrokeEvents
      )
    );


    if (frameData !== null) {

      formData.append(
        "image_frame",
        frameData
      );
    }


    if (latestAudioBlob !== null) {

      formData.append(
        "audio_chunk",
        latestAudioBlob,
        "live_audio.webm"
      );
    }


    const response =
      await fetch(
        "/predict_live",
        {
          method: "POST",
          body: formData
        }
      );


    const data =
      await response.json();


    if (!response.ok) {

      throw new Error(
        data.detail ||
        data.error ||
        "Live prediction failed."
      );
    }


    updatePredictionUI(
      data
    );

  } catch (error) {

    statusBox.textContent =
      "Live prediction failed: " +
      error.message;

  } finally {

    predictionInFlight = false;
  }
}


// ============================================================
// Probability renderer
// ============================================================

function renderProbabilityBars(
  container,
  probabilities,
  fillClass = ""
) {

  container.innerHTML =
    "";


  if (!probabilities) {
    return;
  }


  Object.entries(
    probabilities
  )
    .sort(
      (a, b) =>
        Number(b[1])
        - Number(a[1])
    )
    .forEach(
      ([label, probability]) => {

        const percentage =
          (
            Number(probability)
            * 100
          ).toFixed(2);


        const row =
          document.createElement(
            "div"
          );


        row.className =
          "prob-row";


        row.innerHTML = `
          <div class="prob-label">
            <span>${label}</span>
            <span>${percentage}%</span>
          </div>

          <div class="track">
            <div
              class="fill ${fillClass}"
              data-width="${percentage}%"
            ></div>
          </div>
        `;


        container.appendChild(
          row
        );
      }
    );


  requestAnimationFrame(
    () => {

      container
        .querySelectorAll(".fill")
        .forEach(
          bar => {

            bar.style.width =
              bar.dataset.width;
          }
        );
    }
  );
}


// ============================================================
// Update prediction UI
// ============================================================

function updatePredictionUI(data) {

  // ----------------------------------------------------------
  // Final temporally aggregated result
  // ----------------------------------------------------------

  const state =
    data.current_state ||
    data.prediction ||
    "unknown";


  const confidence =
    Number(
      data.confidence || 0
    );


  const confidencePct =
    Number(
      data.confidence_percent
      ?? confidence * 100
    );


  const gap =
    Number(
      data.confidence_gap || 0
    );


  const level =
    data.confidence_level ||
    "Low";


  predictionBox.textContent =
    state.toUpperCase();


  confidencePercent.textContent =
    `${confidencePct.toFixed(2)}%`;


  confidenceFill.style.width =
    `${
      Math.max(
        0,
        Math.min(
          confidencePct,
          100
        )
      )
    }%`;


  confidenceLevel.textContent =
    level;


  confidenceLevel.classList.remove(
    "confidence-high",
    "confidence-medium",
    "confidence-low"
  );


  if (level === "High") {

    confidenceLevel.classList.add(
      "confidence-high"
    );

  } else if (
    level === "Medium"
  ) {

    confidenceLevel.classList.add(
      "confidence-medium"
    );

  } else {

    confidenceLevel.classList.add(
      "confidence-low"
    );
  }


  // ----------------------------------------------------------
  // Temporal diagnostics
  // ----------------------------------------------------------

  const samples =
    Number(
      data.temporal_samples || 0
    );


  const windowSize =
    Number(
      data.temporal_window || 5
    );


  temporalSamples.textContent =
    samples;


  temporalWindow.textContent =
    windowSize;


  temporalWindowStatus.textContent =
    `${samples} / ${windowSize}`;


  // ----------------------------------------------------------
  // Raw current fusion result
  // ----------------------------------------------------------

  const rawState =
    data.raw_prediction ||
    "—";


  const rawConfidencePct =
    Number(
      data.raw_confidence_percent
      ?? (
        Number(
          data.raw_confidence || 0
        )
        * 100
      )
    );


  rawPrediction.textContent =
    rawState.toUpperCase();


  rawConfidence.textContent =
    `${rawConfidencePct.toFixed(2)}%`;


  // ----------------------------------------------------------
  // Technical details
  // ----------------------------------------------------------

  const details =
    data.technical_details || {};


  secondaryState.textContent =
    (
      details.second_class ||
      "—"
    ).toUpperCase();


  confidenceGap.textContent =
    gap.toFixed(4);


  featureDimension.textContent =
    data.feature_dimension ??
    details.feature_dimension ??
    "—";


  deviceInfo.textContent =
    data.device ||
    details.device ||
    "—";


  webcamCalibrationUsed.textContent =
    data.webcam_calibration_used
      ? "Yes"
      : "No";


  activeModalities.textContent =
    formatModalities(
      data.used_modalities
    );


  technicalRawState.textContent =
    rawState.toUpperCase();


  technicalTemporalSamples.textContent =
    `${samples} / ${windowSize}`;


  sessionIdDisplay.textContent =
    data.session_id ||
    sessionId;


  // ----------------------------------------------------------
  // Final aggregated probabilities
  // ----------------------------------------------------------

  renderProbabilityBars(
    probabilitiesBox,
    data.probabilities
  );


  // ----------------------------------------------------------
  // Raw probabilities
  // ----------------------------------------------------------

  renderProbabilityBars(
    rawProbabilitiesBox,
    data.raw_probabilities,
    "raw-fill"
  );


  // ----------------------------------------------------------
  // Webcam-calibrated result
  // ----------------------------------------------------------

  updateWebcamResult(
    data.webcam_prediction
  );


  // ----------------------------------------------------------
  // Status message
  // ----------------------------------------------------------

  statusBox.textContent =
    (
      "Behavioural-state prediction complete | " +
      `Temporal window: ${samples}/${windowSize} | ` +
      `Active modalities: ${
        formatModalities(
          data.used_modalities
        )
      }`
    );
}


// ============================================================
// Webcam-calibrated result
// ============================================================

function updateWebcamResult(
  webcamResult
) {

  if (!webcamResult) {

    webcamPrediction.textContent =
      "—";


    webcamConfidence.textContent =
      "—";


    webcamProbabilityBars.innerHTML =
      "";


    return;
  }


  const state =
    webcamResult.current_state ||
    webcamResult.prediction ||
    "unknown";


  const confidencePct =
    Number(
      webcamResult.confidence_percent
      ?? (
        Number(
          webcamResult.confidence || 0
        )
        * 100
      )
    );


  webcamPrediction.textContent =
    state.toUpperCase();


  webcamConfidence.textContent =
    `${confidencePct.toFixed(2)}%`;


  renderProbabilityBars(
    webcamProbabilityBars,
    webcamResult.probabilities,
    "webcam-fill"
  );
}


// ============================================================
// Active modality formatter
// ============================================================

function formatModalities(
  modalities
) {

  if (!modalities) {
    return "unknown";
  }


  const active =
    Object.entries(
      modalities
    )
      .filter(
        ([, enabled]) =>
          Boolean(enabled)
      )
      .map(
        ([name]) => name
      );


  return (
    active.length
      ? active.join(", ")
      : "none"
  );
}


// ============================================================
// Event handlers
// ============================================================

startBtn.addEventListener(
  "click",
  startSession
);


stopBtn.addEventListener(
  "click",
  stopSession
);


resetTemporalBtn.addEventListener(
  "click",
  resetTemporalWindow
);


resetBtn.addEventListener(
  "click",
  resetSession
);


// ============================================================
// Initialisation
// ============================================================

resetPredictionDisplay();

checkModelStatus();

updateReadiness();
