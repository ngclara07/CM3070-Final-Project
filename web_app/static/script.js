// static/script.js

const textInput = document.getElementById("textInput");
const webcam = document.getElementById("webcam");
const canvas = document.getElementById("frameCanvas");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");

const statusBox = document.getElementById("status");
const sessionStatus = document.getElementById("sessionStatus");
const audioStatus = document.getElementById("audioStatus");
const webcamStatus = document.getElementById("webcamStatus");
const modelStatusText = document.getElementById("modelStatusText");

const predictionBox = document.getElementById("prediction");
const confidencePercent = document.getElementById("confidencePercent");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceLevel = document.getElementById("confidenceLevel");

const secondaryState = document.getElementById("secondaryState");
const confidenceGap = document.getElementById("confidenceGap");
const featureDimension = document.getElementById("featureDimension");
const deviceInfo = document.getElementById("deviceInfo");
const probabilitiesBox = document.getElementById("probabilities");

const modelReady = document.getElementById("modelReady");
const textReady = document.getElementById("textReady");
const keyReady = document.getElementById("keyReady");
const audioReady = document.getElementById("audioReady");
const imageReady = document.getElementById("imageReady");

const textCard = document.getElementById("textCard");
const webcamCard = document.getElementById("webcamCard");
const audioCard = document.getElementById("audioCard");

const charCount = document.getElementById("charCount");
const keyCount = document.getElementById("keyCount");

let keystrokeEvents = [];
let activeKeys = new Set();

let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let latestAudioBlob = null;

let liveTimer = null;
let sessionActive = false;
let fusionModelLoaded = false;

const LIVE_INTERVAL_MS = 15000;

function normaliseKey(event) {
  if (event.key === "Backspace") return "backspace";
  if (event.key === "Delete") return "delete";
  if (event.key === " ") return "space";
  if (event.key.length === 1) return event.key.toLowerCase();
  return event.key.toLowerCase();
}

function setReady(element, isReady, readyText = "Ready", missingText = "Missing") {
  element.classList.toggle("active", isReady);
  element.querySelector("b").textContent = isReady ? readyText : missingText;
}

function updateReadiness() {
  const textOk = textInput.value.trim().length >= 20;
  const keydowns = keystrokeEvents.filter(e => e.type === "down").length;
  const keyOk = keydowns >= 20;
  const audioOk = latestAudioBlob !== null;
  const imageOk = mediaStream !== null;

  charCount.textContent = textInput.value.trim().length;
  keyCount.textContent = keydowns;

  setReady(modelReady, fusionModelLoaded, "Loaded", "Failed");
  setReady(textReady, textOk);
  setReady(keyReady, keyOk);
  setReady(audioReady, audioOk, "Ready", "Optional");
  setReady(imageReady, imageOk, "Ready", "Optional");

  textCard.classList.toggle("active", textOk && keyOk);
  webcamCard.classList.toggle("active", imageOk);
  audioCard.classList.toggle("active", audioOk);

  textCard.querySelector(".badge").textContent = textOk && keyOk ? "active" : "inactive";
  webcamCard.querySelector(".badge").textContent = imageOk ? "active" : "inactive";
  audioCard.querySelector(".badge").textContent = audioOk ? "active" : "inactive";
}

async function checkModelStatus() {
  try {
    const response = await fetch("/model-status");
    const data = await response.json();

    fusionModelLoaded = Boolean(data.fusion_model);

    modelStatusText.textContent = fusionModelLoaded
      ? `Fusion model loaded. Backend: ${data.inference_backend}`
      : `Fusion model unavailable. Backend fallback active. ${data.error || ""}`;

    updateReadiness();
  } catch (error) {
    fusionModelLoaded = false;
    modelStatusText.textContent = "Could not query model status.";
    updateReadiness();
  }
}

textInput.addEventListener("keydown", (event) => {
  const key = normaliseKey(event);
  if (activeKeys.has(key)) return;

  activeKeys.add(key);

  keystrokeEvents.push({
    type: "down",
    key: key,
    timestamp_perf: performance.now() / 1000,
    timestamp_epoch: Date.now() / 1000
  });

  updateReadiness();
});

textInput.addEventListener("keyup", (event) => {
  const key = normaliseKey(event);
  activeKeys.delete(key);

  keystrokeEvents.push({
    type: "up",
    key: key,
    timestamp_perf: performance.now() / 1000,
    timestamp_epoch: Date.now() / 1000
  });

  updateReadiness();
});

textInput.addEventListener("input", updateReadiness);

function captureWebcamFrame() {
  if (!mediaStream || webcam.videoWidth === 0 || webcam.videoHeight === 0) {
    return null;
  }

  canvas.width = webcam.videoWidth;
  canvas.height = webcam.videoHeight;

  const ctx = canvas.getContext("2d");
  ctx.drawImage(webcam, 0, 0, canvas.width, canvas.height);

  return canvas.toDataURL("image/jpeg", 0.85);
}

async function startSession() {
  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: true
    });

    webcam.srcObject = mediaStream;
    webcamStatus.textContent = "Webcam capturing.";
    audioStatus.textContent = "Microphone recording.";

    mediaRecorder = new MediaRecorder(mediaStream);

    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) audioChunks.push(event.data);
    };

    mediaRecorder.onstop = () => {
      latestAudioBlob = new Blob(audioChunks, { type: "audio/webm" });
      audioChunks = [];
      audioStatus.textContent = "Audio chunk ready.";
      updateReadiness();

      if (sessionActive) {
        mediaRecorder.start();
        setTimeout(() => {
          if (mediaRecorder && mediaRecorder.state === "recording") {
            mediaRecorder.stop();
          }
        }, LIVE_INTERVAL_MS);
      }
    };

    sessionActive = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;

    sessionStatus.textContent = "Live session running.";
    statusBox.textContent = "Capturing live multimodal behaviour...";

    mediaRecorder.start();

    setTimeout(() => {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
    }, LIVE_INTERVAL_MS);

    liveTimer = setInterval(runLivePrediction, LIVE_INTERVAL_MS);

    updateReadiness();

  } catch (error) {
    sessionActive = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;

    mediaStream = null;
    mediaRecorder = null;

    sessionStatus.textContent = "Session running with limited modalities.";
    webcamStatus.textContent = "Webcam unavailable.";
    audioStatus.textContent = "Microphone unavailable.";
    statusBox.textContent = "Camera/microphone unavailable. Text and keystroke prediction still enabled.";

    liveTimer = setInterval(runLivePrediction, LIVE_INTERVAL_MS);
    updateReadiness();
  }
}

function stopSession() {
  sessionActive = false;

  if (liveTimer) clearInterval(liveTimer);

  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach(track => track.stop());
  }

  mediaStream = null;
  mediaRecorder = null;

  startBtn.disabled = false;
  stopBtn.disabled = true;

  sessionStatus.textContent = "Session stopped.";
  statusBox.textContent = "Live session stopped.";
  webcamStatus.textContent = "Webcam inactive.";
  audioStatus.textContent = "Microphone inactive.";

  updateReadiness();
}

function resetSession() {
  stopSession();

  textInput.value = "";
  keystrokeEvents = [];
  activeKeys.clear();
  audioChunks = [];
  latestAudioBlob = null;

  predictionBox.textContent = "—";
  confidencePercent.textContent = "—";
  confidenceFill.style.width = "0%";
  confidenceLevel.textContent = "—";
  confidenceLevel.classList.remove("confidence-high", "confidence-medium", "confidence-low");

  secondaryState.textContent = "—";
  confidenceGap.textContent = "—";
  featureDimension.textContent = "—";
  deviceInfo.textContent = "—";
  probabilitiesBox.innerHTML = "";

  statusBox.textContent = "Session reset.";
  sessionStatus.textContent = "Session not started.";

  updateReadiness();
}

async function runLivePrediction() {
  try {
    updateReadiness();

    const text = textInput.value.trim();
    const keydowns = keystrokeEvents.filter(e => e.type === "down").length;

    if (text.length < 20 || keydowns < 20) {
      statusBox.textContent = "Waiting for sufficient text and keystroke data...";
      return;
    }

    statusBox.textContent = "Running behavioural-state prediction...";

    const frameData = captureWebcamFrame();

    const formData = new FormData();
    formData.append("text", text);
    formData.append("keystroke_events", JSON.stringify(keystrokeEvents));

    if (frameData !== null) {
      formData.append("image_frame", frameData);
    }

    if (latestAudioBlob !== null) {
      formData.append("audio_chunk", latestAudioBlob, "live_audio.webm");
    }

    const response = await fetch("/predict_live", {
      method: "POST",
      body: formData
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || data.error || "Live prediction failed.");
    }

    updatePredictionUI(data);

  } catch (error) {
    statusBox.textContent = "Live prediction failed: " + error.message;
  }
}

function updatePredictionUI(data) {
  const state = data.current_state || data.prediction;
  const confidence = Number(data.confidence || 0);
  const confidencePct = Number(data.confidence_percent || confidence * 100);
  const gap = Number(data.confidence_gap || 0);
  const level = data.confidence_level || "Low";

  predictionBox.textContent = state.toUpperCase();
  confidencePercent.textContent = `${confidencePct.toFixed(2)}%`;
  confidenceFill.style.width = `${Math.max(0, Math.min(confidencePct, 100))}%`;

  confidenceLevel.textContent = level;
  confidenceLevel.classList.remove("confidence-high", "confidence-medium", "confidence-low");

  if (level === "High") {
    confidenceLevel.classList.add("confidence-high");
  } else if (level === "Medium") {
    confidenceLevel.classList.add("confidence-medium");
  } else {
    confidenceLevel.classList.add("confidence-low");
  }

  const details = data.technical_details || {};

  secondaryState.textContent = (details.second_class || "—").toUpperCase();
  confidenceGap.textContent = gap.toFixed(4);
  featureDimension.textContent = data.feature_dimension || details.feature_dimension || "—";
  deviceInfo.textContent = data.device || details.device || "—";

  statusBox.textContent =
    `Behavioural-state prediction complete | Active modalities: ${formatModalities(data.used_modalities)}`;

  probabilitiesBox.innerHTML = "";

  Object.entries(data.probabilities)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, probability]) => {
      const percentage = (probability * 100).toFixed(2);

      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <div class="prob-label">
          <span>${label}</span>
          <span>${percentage}%</span>
        </div>
        <div class="track">
          <div class="fill" data-width="${percentage}%"></div>
        </div>
      `;

      probabilitiesBox.appendChild(row);
    });

  requestAnimationFrame(() => {
    document.querySelectorAll(".fill").forEach(bar => {
      bar.style.width = bar.dataset.width;
    });
  });
}

function formatModalities(modalities) {
  if (!modalities) return "unknown";

  return Object.entries(modalities)
    .filter(([, active]) => Boolean(active))
    .map(([name]) => name)
    .join(", ");
}

startBtn.addEventListener("click", startSession);
stopBtn.addEventListener("click", stopSession);
resetBtn.addEventListener("click", resetSession);

checkModelStatus();
updateReadiness();
