"use strict";

/* ============================================================
   SenseFuzeAI browser client
   ============================================================ */


/* ============================================================
   HELPERS
   ============================================================ */

const el = id => document.getElementById(id);

function setText(element, value) {
  if (element) {
    element.textContent = String(value);
  }
}

function finiteNumber(value, fallback = 0) {
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function positiveInteger(value, fallback) {
  const n = Number(value);
  return (
    Number.isInteger(n) && n > 0
  ) ? n : fallback;
}

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

function formatServerError(data) {
  if (!data) {
    return "Unknown server error.";
  }

  const detail =
    data.detail ??
    data.error ??
    data.exception ??
    data.message;

  if (typeof detail === "string") {
    return detail;
  }

  try {
    return JSON.stringify(detail);
  } catch (_) {
    return String(detail);
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(
    url,
    options
  );

  let data = {};

  try {
    data = await response.json();
  } catch (_) {
    data = {};
  }

  if (!response.ok) {
    const error = new Error(
      formatServerError(data)
    );

    error.status = response.status;
    error.data = data;

    throw error;
  }

  return data;
}

async function postForm(url, values) {
  const form = new FormData();

  Object.entries(values).forEach(
    ([key, value]) => {
      if (
        value !== undefined &&
        value !== null
      ) {
        form.append(key, value);
      }
    }
  );

  return fetchJson(
    url,
    {
      method: "POST",
      body: form
    }
  );
}


/* ============================================================
   DOM
   ============================================================ */

const textInput = el("textInput");

const webcam = el("webcam");
const canvas = el("frameCanvas");
const staticImagePreview =
  el("staticImagePreview");

const startBtn = el("startBtn");
const stopBtn = el("stopBtn");
const resetBtn = el("resetBtn");
const resetTemporalBtn =
  el("resetTemporalBtn");

const startMicBtn = el("startMicBtn");
const stopMicBtn = el("stopMicBtn");

const chooseAudioBtn =
  el("chooseAudioBtn");

const audioFileInput =
  el("audioFileInput");

const chooseImageBtn =
  el("chooseImageBtn");

const chooseVideoBtn =
  el("chooseVideoBtn");

const imageFileInput =
  el("imageFileInput");

const videoFileInput =
  el("videoFileInput");

const statusBox = el("status");
const sessionStatus =
  el("sessionStatus");

const audioStatus =
  el("audioStatus");

const audioDiagnostic =
  el("audioDiagnostic");

const webcamStatus =
  el("webcamStatus");

const modelStatusText =
  el("modelStatusText");

const webcamModelStatusText =
  el("webcamModelStatusText");

const predictionBox =
  el("prediction");

const confidencePercent =
  el("confidencePercent");

const confidenceFill =
  el("confidenceFill");

const confidenceLevel =
  el("confidenceLevel");

const rawPrediction =
  el("rawPrediction");

const rawConfidence =
  el("rawConfidence");

const temporalSamples =
  el("temporalSamples");

const temporalWindow =
  el("temporalWindow");

const temporalWindowStatus =
  el("temporalWindowStatus");

const secondaryState =
  el("secondaryState");

const confidenceGap =
  el("confidenceGap");

const featureDimension =
  el("featureDimension");

const deviceInfo =
  el("deviceInfo");

const probabilitiesBox =
  el("probabilities");

const rawProbabilitiesBox =
  el("rawProbabilities");

const webcamPrediction =
  el("webcamPrediction");

const webcamConfidence =
  el("webcamConfidence");

const webcamProbabilityBars =
  el("webcamProbabilityBars");

const webcamCalibrationUsed =
  el("webcamCalibrationUsed");

const activeModalities =
  el("activeModalities");

const technicalRawState =
  el("technicalRawState");

const technicalTemporalSamples =
  el("technicalTemporalSamples");

const sessionIdDisplay =
  el("sessionIdDisplay");

const validationStatus =
  el("validationStatus");

const modelReady = el("modelReady");
const webcamModelReady =
  el("webcamModelReady");

const textReady = el("textReady");
const keyReady = el("keyReady");
const audioReady = el("audioReady");
const imageReady = el("imageReady");

const textCard = el("textCard");
const webcamCard = el("webcamCard");
const audioCard = el("audioCard");

const charCount = el("charCount");
const keyCount = el("keyCount");

const audioStreamState =
  el("audioStreamState");

const audioBufferedSeconds =
  el("audioBufferedSeconds");

const audioLiveLevel =
  el("audioLiveLevel");

const audioPacketCount =
  el("audioPacketCount");


/* ============================================================
   CONFIGURATION
   ============================================================ */

let MIN_TEXT_CHARS = 20;
let MIN_KEYPRESSES = 20;
let LIVE_INTERVAL_MS = 2500;
let TEMPORAL_WINDOW = 5;

let TARGET_AUDIO_SAMPLE_RATE =
  16000;

let AUDIO_STREAM_WINDOW_SECONDS =
  10;

let AUDIO_STREAM_MIN_SECONDS =
  2;

let behaviouralLabels = [];


/* ============================================================
   MODEL STATE
   ============================================================ */

let modelState = "not_loaded";

let fusionModelLoaded = false;
let modelInitialisationInFlight =
  false;

let webcamModelLoaded = false;

let temporalFusionBackend = null;


/* ============================================================
   CONCURRENCY
   ============================================================ */

let serverGeneration = 0;
let clientEpoch = 0;

let stateChangeInProgress = false;
let predictionInFlight = false;

let liveTimer = null;


/* ============================================================
   KEYSTROKES
   ============================================================ */

let keystrokeEvents = [];

const activeKeys = new Set();


/* ============================================================
   AUDIO
   ============================================================ */

let audioSourceReady = false;
let audioSourceName = null;
let audioSourceKind = null;

let microphoneStreaming = false;

let microphoneStream = null;
let microphoneContext = null;
let microphoneSourceNode = null;
let microphoneProcessor = null;
let microphoneSilentGain = null;

let audioSocket = null;
let audioStreamToken = null;

let microphoneExpectedClose = false;

let audioBufferedSec = 0;
let audioPackets = 0;
let audioCurrentDbfs = null;


/* ============================================================
   VISUAL
   ============================================================ */

let visualMode = "none";
let visualSourceReady = false;
let visualSourceName = null;

let webcamStream = null;
let visualObjectUrl = null;


/* ============================================================
   SESSION
   ============================================================ */

const sessionId = createSessionId();

setText(
  sessionIdDisplay,
  sessionId
);


/* ============================================================
   STATE CHANGE HELPERS
   ============================================================ */

function beginStateChange(message) {
  clientEpoch += 1;
  stateChangeInProgress = true;

  if (message) {
    setText(statusBox, message);
  }

  updateReadiness();

  return clientEpoch;
}

function finishStateChange(epoch) {
  if (epoch === clientEpoch) {
    stateChangeInProgress = false;
  }

  updateReadiness();
}

function operationStillCurrent(epoch) {
  return epoch === clientEpoch;
}


/* ============================================================
   SERVER CONFLICTS
   ============================================================ */

function handleConflictResponse(data) {
  const detail =
    data &&
    typeof data.detail === "object" &&
    data.detail !== null
      ? data.detail
      : null;

  if (!detail) {
    return false;
  }

  if (detail.generation !== undefined) {
    const generation =
      Number(detail.generation);

    if (Number.isFinite(generation)) {
      serverGeneration = generation;
    }
  }

  const type =
    String(detail.type || "");

  if (
    type === "stale_generation" ||
    type === "stale_result" ||
    type === "stale_session"
  ) {
    setText(
      statusBox,
      "Stale prediction rejected after a source/reset change."
    );

    return true;
  }

  if (type === "visual_mode_mismatch") {
    setText(
      statusBox,
      (
        "Visual-source state changed on the server "
        + `(${detail.visual_mode || "none"}).`
      )
    );

    return true;
  }

  return false;
}


/* ============================================================
   KEY NORMALISATION
   ============================================================ */

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

  return String(
    event.key
  ).toLowerCase();
}


/* ============================================================
   READINESS UI
   ============================================================ */

function setReady(
  element,
  ready,
  readyText = "Ready",
  missingText = "Missing"
) {
  if (!element) {
    return;
  }

  element.classList.toggle(
    "active",
    ready
  );

  const bold =
    element.querySelector("b");

  if (bold) {
    bold.textContent =
      ready
        ? readyText
        : missingText;
  }
}

function setModelBadge(text, active) {
  if (!modelReady) {
    return;
  }

  modelReady.classList.toggle(
    "active",
    active
  );

  modelReady.classList.toggle(
    "warning",
    modelState === "loading"
  );

  const bold =
    modelReady.querySelector("b");

  if (bold) {
    bold.textContent = text;
  }
}

function currentTextLength() {
  return textInput
    ? textInput.value.trim().length
    : 0;
}

function currentKeydownCount() {
  return keystrokeEvents.filter(
    event => event.type === "down"
  ).length;
}

function audioIsReady() {
  return Boolean(
    audioSourceReady
  );
}

function visualIsReady() {
  if (
    visualMode === "image" ||
    visualMode === "video"
  ) {
    return Boolean(
      visualSourceReady
    );
  }

  if (visualMode === "webcam") {
    return Boolean(
      visualSourceReady &&
      webcamStream &&
      webcam &&
      webcam.videoWidth > 0 &&
      webcam.videoHeight > 0
    );
  }

  return false;
}

/*
 * IMPORTANT FIX:
 *
 * This intentionally does NOT require fusionModelLoaded.
 *
 * Requiring fusionModelLoaded here created the lazy-loading
 * deadlock in the previous script.
 */
function inputModalitiesReady() {
  return (
    !stateChangeInProgress &&
    currentTextLength()
      >= MIN_TEXT_CHARS &&
    currentKeydownCount()
      >= MIN_KEYPRESSES &&
    audioIsReady() &&
    visualIsReady()
  );
}

function updateAudioMetrics() {
  setText(
    audioStreamState,
    microphoneStreaming
      ? (
          audioSourceReady
            ? "Live"
            : "Buffering"
        )
      : (
          audioSourceKind === "file"
            ? "Fixed file"
            : "Stopped"
        )
  );

  setText(
    audioBufferedSeconds,
    `${audioBufferedSec.toFixed(1)} s`
  );

  setText(
    audioPacketCount,
    audioPackets
  );

  setText(
    audioLiveLevel,
    Number.isFinite(
      audioCurrentDbfs
    )
      ? `${audioCurrentDbfs.toFixed(1)} dBFS`
      : "—"
  );
}

function updateReadiness() {
  const textCount =
    currentTextLength();

  const keydowns =
    currentKeydownCount();

  const textOk =
    textCount >= MIN_TEXT_CHARS;

  const keyOk =
    keydowns >= MIN_KEYPRESSES;

  const audioOk =
    audioIsReady();

  const visualOk =
    visualIsReady();

  setText(charCount, textCount);
  setText(keyCount, keydowns);

  switch (modelState) {
    case "ready":
      setModelBadge(
        "Ready",
        true
      );
      break;

    case "loading":
      setModelBadge(
        "Loading",
        false
      );
      break;

    case "failed":
      setModelBadge(
        "Failed",
        false
      );
      break;

    default:
      setModelBadge(
        "Standby",
        false
      );
  }

  setReady(
    webcamModelReady,
    webcamModelLoaded,
    "Loaded",
    "Not required"
  );

  setReady(
    textReady,
    textOk,
    "Ready",
    "Missing"
  );

  setReady(
    keyReady,
    keyOk,
    "Ready",
    "Missing"
  );

  setReady(
    audioReady,
    audioOk,
    microphoneStreaming
      ? "Streaming"
      : "Ready",
    microphoneStreaming
      ? "Buffering"
      : "Required"
  );

  setReady(
    imageReady,
    visualOk,
    "Ready",
    "Required"
  );

  if (textCard) {
    textCard.classList.toggle(
      "active",
      textOk && keyOk
    );
  }

  if (audioCard) {
    audioCard.classList.toggle(
      "active",
      audioOk
    );

    audioCard.classList.toggle(
      "streaming",
      microphoneStreaming
    );
  }

  if (webcamCard) {
    webcamCard.classList.toggle(
      "active",
      visualOk
    );
  }

  if (startMicBtn) {
    startMicBtn.disabled =
      microphoneStreaming;
  }

  if (stopMicBtn) {
    stopMicBtn.disabled =
      !microphoneStreaming;
  }

  updateAudioMetrics();
}


/* ============================================================
   MODEL STATUS
   ============================================================ */

function applyModelStatus(data) {
  modelState =
    String(
      data.state || "not_loaded"
    );

  fusionModelLoaded =
    Boolean(
      data.fusion_model ||
      data.predictor_loaded
    );

  webcamModelLoaded =
    Boolean(
      data.webcam_calibrated_image_model
    );

  temporalFusionBackend =
    data.temporal_fusion_backend ||
    null;

  MIN_TEXT_CHARS =
    positiveInteger(
      data.min_text_chars,
      20
    );

  MIN_KEYPRESSES =
    positiveInteger(
      data.min_keypresses,
      20
    );

  LIVE_INTERVAL_MS =
    positiveInteger(
      data.live_interval_ms,
      2500
    );

  TEMPORAL_WINDOW =
    positiveInteger(
      data.temporal_probability_window,
      5
    );

  TARGET_AUDIO_SAMPLE_RATE =
    positiveInteger(
      data.target_audio_sample_rate,
      16000
    );

  AUDIO_STREAM_WINDOW_SECONDS =
    finiteNumber(
      data.audio_stream_window_seconds,
      10
    );

  AUDIO_STREAM_MIN_SECONDS =
    finiteNumber(
      data.audio_stream_min_seconds,
      2
    );

  if (Array.isArray(data.labels)) {
    behaviouralLabels =
      data.labels
        .map(
          value =>
            String(value).trim()
        )
        .filter(Boolean);
  }

  setText(
    temporalWindow,
    TEMPORAL_WINDOW
  );

  if (modelState === "ready") {
    setText(
      modelStatusText,
      (
        "Fusion backend ready"
        + (
            temporalFusionBackend
              ? (
                  " | Temporal backend: "
                  + temporalFusionBackend
                )
              : ""
          )
      )
    );

  } else if (modelState === "loading") {
    setText(
      modelStatusText,
      "Fusion backend is loading..."
    );

  } else if (modelState === "failed") {
    setText(
      modelStatusText,
      (
        "Fusion backend failed: "
        + String(
            data.error ||
            "unknown error"
          )
      )
    );

  } else {
    setText(
      modelStatusText,
      (
        "Fusion backend in standby. "
        + "It will initialise when "
        + "prediction becomes ready."
      )
    );
  }

  setText(
    webcamModelStatusText,
    webcamModelLoaded
      ? (
          "Webcam-calibrated image "
          + "augmentation loaded."
        )
      : (
          "Webcam calibration not "
          + "currently loaded/required."
        )
  );

  updateReadiness();
}

async function checkModelStatus() {
  try {
    const data =
      await fetchJson(
        "/model-status",
        {
          cache: "no-store"
        }
      );

    applyModelStatus(data);

  } catch (error) {
    setText(
      modelStatusText,
      (
        "Model-status query failed: "
        + String(
            error.message || error
          )
      )
    );
  }
}

/*
 * IMPORTANT FIX:
 *
 * Explicitly asks the backend to initialise the predictor.
 */
async function ensureModelsReady() {
  if (
    modelState === "ready" &&
    fusionModelLoaded
  ) {
    return true;
  }

  if (modelInitialisationInFlight) {
    return false;
  }

  modelInitialisationInFlight = true;
  modelState = "loading";

  setText(
    statusBox,
    (
      "Initialising multimodal "
      + "fusion backend..."
    )
  );

  updateReadiness();

  try {
    const data =
      await fetchJson(
        "/initialize-models",
        {
          method: "POST"
        }
      );

    if (data.model_status) {
      applyModelStatus(
        data.model_status
      );
    } else {
      await checkModelStatus();
    }

    return (
      modelState === "ready" &&
      fusionModelLoaded
    );

  } catch (error) {
    modelState = "failed";
    fusionModelLoaded = false;

    const serverStatus =
      error.data?.model_status;

    if (serverStatus) {
      applyModelStatus(
        serverStatus
      );
    }

    setText(
      statusBox,
      (
        "Fusion model initialisation failed: "
        + String(
            error.message || error
          )
      )
    );

    return false;

  } finally {
    modelInitialisationInFlight =
      false;

    updateReadiness();
  }
}


/* ============================================================
   KEYSTROKE ACQUISITION
   ============================================================ */

if (textInput) {
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
          key,
          timestamp_perf:
            performance.now() / 1000,
          timestamp_epoch:
            Date.now() / 1000
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
          key,
          timestamp_perf:
            performance.now() / 1000,
          timestamp_epoch:
            Date.now() / 1000
        }
      );

      updateReadiness();
    }
  );

  textInput.addEventListener(
    "input",
    updateReadiness
  );
}


/* ============================================================
   AUDIO DSP
   ============================================================ */

function resampleLinear(
  input,
  inputRate,
  outputRate
) {
  if (
    !input ||
    input.length === 0
  ) {
    return new Float32Array(0);
  }

  if (inputRate === outputRate) {
    return new Float32Array(input);
  }

  const outputLength =
    Math.max(
      1,
      Math.round(
        input.length *
        outputRate /
        inputRate
      )
    );

  const output =
    new Float32Array(
      outputLength
    );

  const ratio =
    inputRate / outputRate;

  for (
    let i = 0;
    i < outputLength;
    i += 1
  ) {
    const position =
      i * ratio;

    const left =
      Math.floor(position);

    const right =
      Math.min(
        left + 1,
        input.length - 1
      );

    const fraction =
      position - left;

    output[i] =
      input[left] +
      (
        input[right] -
        input[left]
      ) *
      fraction;
  }

  return output;
}

function float32ToPCM16Buffer(samples) {
  const buffer =
    new ArrayBuffer(
      samples.length * 2
    );

  const view =
    new DataView(buffer);

  for (
    let i = 0;
    i < samples.length;
    i += 1
  ) {
    const value =
      Math.max(
        -1,
        Math.min(
          1,
          samples[i]
        )
      );

    const pcm =
      value < 0
        ? value * 32768
        : value * 32767;

    view.setInt16(
      i * 2,
      Math.round(pcm),
      true
    );
  }

  return buffer;
}

function calculateDbfs(samples) {
  if (
    !samples ||
    samples.length === 0
  ) {
    return -120;
  }

  let squareSum = 0;

  for (
    let i = 0;
    i < samples.length;
    i += 1
  ) {
    squareSum +=
      samples[i] * samples[i];
  }

  const rms =
    Math.sqrt(
      squareSum /
      samples.length
    );

  return (
    20 *
    Math.log10(
      Math.max(
        rms,
        1e-12
      )
    )
  );
}

function updateAudioDiagnostic(audio) {
  const value = audio || {};

  const dbfs =
    Number(value.dbfs);

  const duration =
    Number(
      value.duration_sec ??
      value.analysed_duration_sec
    );

  let text =
    (
      "Audio condition: "
      + String(
          value.condition ||
          "unknown"
        )
    );

  if (Number.isFinite(duration)) {
    text +=
      (
        " | Window: "
        + duration.toFixed(2)
        + "s"
      );
  }

  if (Number.isFinite(dbfs)) {
    text +=
      (
        " | Level: "
        + dbfs.toFixed(1)
        + " dBFS"
      );

    audioCurrentDbfs = dbfs;
  }

  if (value.note) {
    text +=
      " | " + String(value.note);
  }

  setText(
    audioDiagnostic,
    text
  );

  updateAudioMetrics();
}


/* ============================================================
   MICROPHONE WEBSOCKET
   ============================================================ */

function websocketUrl(token) {
  const scheme =
    window.location.protocol === "https:"
      ? "wss"
      : "ws";

  return (
    `${scheme}://${window.location.host}`
    + `/ws/audio/${encodeURIComponent(sessionId)}`
    + `?token=${encodeURIComponent(token)}`
  );
}

function waitForSocketOpen(socket) {
  return new Promise(
    (resolve, reject) => {
      const timer =
        window.setTimeout(
          () => reject(
            new Error(
              "Audio WebSocket connection timed out."
            )
          ),
          10000
        );

      socket.addEventListener(
        "open",
        () => {
          clearTimeout(timer);
          resolve();
        },
        { once: true }
      );

      socket.addEventListener(
        "error",
        () => {
          clearTimeout(timer);

          reject(
            new Error(
              "Audio WebSocket connection failed."
            )
          );
        },
        { once: true }
      );
    }
  );
}

function cleanupMicrophoneLocal() {
  microphoneStreaming = false;

  if (microphoneProcessor) {
    microphoneProcessor.onaudioprocess =
      null;

    try {
      microphoneProcessor.disconnect();
    } catch (_) {}
  }

  try {
    microphoneSourceNode?.disconnect();
  } catch (_) {}

  try {
    microphoneSilentGain?.disconnect();
  } catch (_) {}

  microphoneStream
    ?.getTracks()
    .forEach(
      track => track.stop()
    );

  microphoneStream = null;
  microphoneProcessor = null;
  microphoneSourceNode = null;
  microphoneSilentGain = null;

  if (microphoneContext) {
    void microphoneContext
      .close()
      .catch(() => {});

    microphoneContext = null;
  }

  if (audioSocket) {
    const socket = audioSocket;

    audioSocket = null;

    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;

    try {
      socket.close();
    } catch (_) {}
  }

  audioStreamToken = null;

  updateReadiness();
}

function installAudioSocketHandlers(
  socket
) {
  socket.onmessage =
    event => {
      let data;

      try {
        data =
          JSON.parse(event.data);
      } catch (_) {
        return;
      }

      if (
        data.type !== "audio_status"
      ) {
        return;
      }

      audioBufferedSec =
        finiteNumber(
          data.buffered_seconds,
          0
        );

      audioPackets =
        finiteNumber(
          data.packets_received,
          0
        );

      audioSourceReady =
        Boolean(
          data.audio_ready
        );

      if (data.audio_diagnostics) {
        updateAudioDiagnostic(
          data.audio_diagnostics
        );
      }

      setText(
        audioStatus,
        audioSourceReady
          ? (
              "Live microphone streaming "
              + `| ${audioBufferedSec.toFixed(1)}s`
            )
          : (
              "Live microphone buffering "
              + `| ${audioBufferedSec.toFixed(1)}`
              + `/${AUDIO_STREAM_MIN_SECONDS.toFixed(1)}s`
            )
      );

      updateReadiness();
    };

  socket.onerror =
    () => {
      setText(
        audioStatus,
        "Microphone transport error."
      );
    };

  socket.onclose =
    () => {
      if (!microphoneExpectedClose) {
        audioSourceReady = false;

        cleanupMicrophoneLocal();

        setText(
          statusBox,
          "Microphone stream disconnected."
        );
      }
    };
}

async function startMicrophoneStream() {
  if (microphoneStreaming) {
    return;
  }

  if (
    !navigator.mediaDevices ||
    !navigator.mediaDevices.getUserMedia
  ) {
    setText(
      statusBox,
      "Microphone API unavailable."
    );
    return;
  }

  const AudioContextClass =
    window.AudioContext ||
    window.webkitAudioContext;

  if (!AudioContextClass) {
    setText(
      statusBox,
      "Web Audio API unavailable."
    );
    return;
  }

  const epoch =
    beginStateChange(
      "Starting microphone..."
    );

  microphoneExpectedClose = false;

  try {
    const stream =
      await navigator.mediaDevices
        .getUserMedia(
          {
            audio: {
              channelCount: 1,
              echoCancellation: false,
              noiseSuppression: false,
              autoGainControl: false
            },
            video: false
          }
        );

    let context;

    try {
      context =
        new AudioContextClass(
          {
            sampleRate:
              TARGET_AUDIO_SAMPLE_RATE
          }
        );
    } catch (_) {
      context =
        new AudioContextClass();
    }

    await context.resume();

    const startData =
      await postForm(
        "/audio_stream/start",
        {
          session_id:
            sessionId
        }
      );

    if (
      !operationStillCurrent(epoch)
    ) {
      stream
        .getTracks()
        .forEach(
          track => track.stop()
        );
      return;
    }

    serverGeneration =
      finiteNumber(
        startData.generation,
        serverGeneration
      );

    audioStreamToken =
      String(
        startData.stream_token
      );

    const socket =
      new WebSocket(
        websocketUrl(
          audioStreamToken
        )
      );

    audioSocket = socket;

    await waitForSocketOpen(
      socket
    );

    installAudioSocketHandlers(
      socket
    );

    microphoneStream = stream;
    microphoneContext = context;

    microphoneSourceNode =
      context.createMediaStreamSource(
        stream
      );

    microphoneProcessor =
      context.createScriptProcessor(
        4096,
        1,
        1
      );

    microphoneSilentGain =
      context.createGain();

    microphoneSilentGain.gain.value =
      0;

    microphoneStreaming = true;

    microphoneProcessor
      .connect(
        microphoneSilentGain
      );

    microphoneSilentGain
      .connect(
        context.destination
      );

    microphoneSourceNode
      .connect(
        microphoneProcessor
      );

    microphoneProcessor
      .onaudioprocess =
        event => {
          if (
            !audioSocket ||
            audioSocket.readyState
              !== WebSocket.OPEN
          ) {
            return;
          }

          const input =
            event.inputBuffer
              .getChannelData(0);

          audioCurrentDbfs =
            calculateDbfs(input);

          const resampled =
            resampleLinear(
              input,
              context.sampleRate,
              TARGET_AUDIO_SAMPLE_RATE
            );

          audioSocket.send(
            float32ToPCM16Buffer(
              resampled
            )
          );

          updateAudioMetrics();
        };

    audioSourceKind =
      "microphone_stream";

    audioSourceName =
      "Live microphone";

    setText(
      audioStatus,
      "Live microphone buffering..."
    );

  } catch (error) {
    cleanupMicrophoneLocal();

    setText(
      statusBox,
      (
        "Microphone start failed: "
        + String(
            error.message || error
          )
      )
    );

  } finally {
    finishStateChange(epoch);
  }
}

async function stopMicrophoneStream() {
  const epoch =
    beginStateChange(
      "Stopping microphone..."
    );

  microphoneExpectedClose = true;

  try {
    const data =
      await postForm(
        "/audio_stream/stop",
        {
          session_id:
            sessionId
        }
      );

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    cleanupMicrophoneLocal();

    audioSourceReady = false;
    audioSourceName = null;
    audioSourceKind = null;

    audioBufferedSec = 0;
    audioPackets = 0;
    audioCurrentDbfs = null;

    setText(
      audioStatus,
      "Microphone stream stopped."
    );

  } catch (error) {
    setText(
      statusBox,
      (
        "Microphone stop failed: "
        + String(
            error.message || error
          )
      )
    );

  } finally {
    microphoneExpectedClose = false;
    finishStateChange(epoch);
  }
}


/* ============================================================
   FIXED AUDIO FILE
   ============================================================ */

async function setAudioFile(file) {
  if (!file) {
    return;
  }

  if (microphoneStreaming) {
    await stopMicrophoneStream();
  }

  const epoch =
    beginStateChange(
      "Loading audio file..."
    );

  const form = new FormData();

  form.append(
    "session_id",
    sessionId
  );

  form.append(
    "source_kind",
    "file"
  );

  form.append(
    "audio_file",
    file,
    file.name
  );

  try {
    const data =
      await fetchJson(
        "/set_audio_source",
        {
          method: "POST",
          body: form
        }
      );

    if (
      !operationStillCurrent(epoch)
    ) {
      return;
    }

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    audioSourceReady = true;
    audioSourceName =
      data.audio_name || file.name;

    audioSourceKind = "file";

    audioBufferedSec =
      finiteNumber(
        data.audio_diagnostics
          ?.duration_sec,
        0
      );

    updateAudioDiagnostic(
      data.audio_diagnostics
    );

    setText(
      audioStatus,
      "Fixed audio file: "
      + audioSourceName
    );

  } catch (error) {
    setText(
      statusBox,
      "Audio file failed: "
      + String(
          error.message || error
        )
    );

  } finally {
    finishStateChange(epoch);
  }
}


/* ============================================================
   VISUAL UTILITIES
   ============================================================ */

function revokeVisualObjectUrl() {
  if (visualObjectUrl) {
    URL.revokeObjectURL(
      visualObjectUrl
    );

    visualObjectUrl = null;
  }
}

function hideStaticImagePreview() {
  if (!staticImagePreview) {
    return;
  }

  staticImagePreview.classList.add(
    "hidden"
  );

  staticImagePreview.removeAttribute(
    "src"
  );
}

function stopWebcamStreamLocally() {
  webcamStream
    ?.getTracks()
    .forEach(
      track => track.stop()
    );

  webcamStream = null;

  if (webcam) {
    webcam.srcObject = null;
  }
}

function stopVideoPreview() {
  if (!webcam) {
    return;
  }

  try {
    webcam.pause();
  } catch (_) {}

  webcam.removeAttribute("src");

  try {
    webcam.load();
  } catch (_) {}
}


/* ============================================================
   IMAGE
   ============================================================ */

async function setVisualImage(file) {
  if (!file) {
    return;
  }

  const epoch =
    beginStateChange(
      "Loading image source..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();
  revokeVisualObjectUrl();

  visualMode = "none";
  visualSourceReady = false;

  const form = new FormData();

  form.append(
    "session_id",
    sessionId
  );

  form.append(
    "image_file",
    file,
    file.name
  );

  try {
    const data =
      await fetchJson(
        "/set_visual_image",
        {
          method: "POST",
          body: form
        }
      );

    if (
      !operationStillCurrent(epoch)
    ) {
      return;
    }

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    visualMode = "image";
    visualSourceReady = true;

    visualSourceName =
      data.visual_name || file.name;

    visualObjectUrl =
      URL.createObjectURL(file);

    if (staticImagePreview) {
      staticImagePreview.src =
        visualObjectUrl;

      staticImagePreview
        .classList
        .remove("hidden");
    }

    if (webcam) {
      webcam.classList.add(
        "hidden"
      );
    }

    setText(
      webcamStatus,
      "Image source: "
      + visualSourceName
    );

  } catch (error) {
    visualMode = "none";
    visualSourceReady = false;

    setText(
      statusBox,
      "Image source failed: "
      + String(
          error.message || error
        )
    );

  } finally {
    finishStateChange(epoch);
  }
}


/* ============================================================
   VIDEO
   ============================================================ */

async function setVisualVideo(file) {
  if (!file) {
    return;
  }

  const epoch =
    beginStateChange(
      "Loading video source..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();
  hideStaticImagePreview();
  revokeVisualObjectUrl();

  const form = new FormData();

  form.append(
    "session_id",
    sessionId
  );

  form.append(
    "video_file",
    file,
    file.name
  );

  try {
    const data =
      await fetchJson(
        "/set_visual_video",
        {
          method: "POST",
          body: form
        }
      );

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    visualMode = "video";
    visualSourceReady = true;
    visualSourceName =
      data.visual_name || file.name;

    visualObjectUrl =
      URL.createObjectURL(file);

    if (webcam) {
      webcam.classList.remove(
        "hidden"
      );

      webcam.srcObject = null;
      webcam.src = visualObjectUrl;
      webcam.loop = true;
      webcam.muted = true;

      try {
        await webcam.play();
      } catch (_) {}
    }

    setText(
      webcamStatus,
      "Video source: "
      + visualSourceName
    );

  } catch (error) {
    visualMode = "none";
    visualSourceReady = false;

    setText(
      statusBox,
      "Video source failed: "
      + String(
          error.message || error
        )
    );

  } finally {
    finishStateChange(epoch);
  }
}


/* ============================================================
   WEBCAM
   ============================================================ */

async function startWebcamMode() {
  if (
    !navigator.mediaDevices ||
    !navigator.mediaDevices.getUserMedia
  ) {
    setText(
      statusBox,
      "Webcam API unavailable."
    );
    return;
  }

  const epoch =
    beginStateChange(
      "Starting webcam..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();
  hideStaticImagePreview();

  try {
    const stream =
      await navigator.mediaDevices
        .getUserMedia(
          {
            video: true,
            audio: false
          }
        );

    if (!webcam) {
      throw new Error(
        "Webcam video element missing."
      );
    }

    webcamStream = stream;

    webcam.classList.remove(
      "hidden"
    );

    webcam.removeAttribute("src");
    webcam.srcObject = stream;
    webcam.muted = true;

    await webcam.play();

    const data =
      await postForm(
        "/set_visual_webcam",
        {
          session_id:
            sessionId
        }
      );

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    visualMode = "webcam";
    visualSourceReady = true;
    visualSourceName = "Webcam";

    setText(
      webcamStatus,
      "Webcam active."
    );

    if (startBtn) {
      startBtn.disabled = true;
    }

    if (stopBtn) {
      stopBtn.disabled = false;
    }

  } catch (error) {
    stopWebcamStreamLocally();

    visualMode = "none";
    visualSourceReady = false;

    setText(
      statusBox,
      "Webcam start failed: "
      + String(
          error.message || error
        )
    );

  } finally {
    finishStateChange(epoch);
  }
}

async function stopVisualMode() {
  const epoch =
    beginStateChange(
      "Stopping visual stream..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();

  try {
    const data =
      await postForm(
        "/stop_visual",
        {
          session_id:
            sessionId
        }
      );

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    visualMode =
      String(
        data.visual_mode || "none"
      );

    visualSourceReady =
      Boolean(
        data.visual_ready
      );

    setText(
      webcamStatus,
      "Visual stream stopped."
    );

  } catch (error) {
    setText(
      statusBox,
      "Stop visual failed: "
      + String(
          error.message || error
        )
    );

  } finally {
    finishStateChange(epoch);
  }
}

function captureWebcamFrame() {
  if (
    visualMode !== "webcam" ||
    !webcamStream ||
    !webcam ||
    !canvas ||
    webcam.videoWidth <= 0 ||
    webcam.videoHeight <= 0
  ) {
    return null;
  }

  canvas.width =
    webcam.videoWidth;

  canvas.height =
    webcam.videoHeight;

  const context =
    canvas.getContext("2d");

  if (!context) {
    return null;
  }

  context.drawImage(
    webcam,
    0,
    0,
    canvas.width,
    canvas.height
  );

  return canvas.toDataURL(
    "image/jpeg",
    0.85
  );
}


/* ============================================================
   PROBABILITY UI
   ============================================================ */

function resolveRenderLabels(
  probabilities
) {
  if (behaviouralLabels.length) {
    return behaviouralLabels;
  }

  return (
    probabilities &&
    typeof probabilities === "object"
  )
    ? Object.keys(probabilities)
    : [];
}

function renderProbabilityBars(
  container,
  probabilities,
  kind,
  probabilitySum = null
) {
  if (!container) {
    return;
  }

  container.innerHTML = "";

  if (
    !probabilities ||
    typeof probabilities !== "object"
  ) {
    return;
  }

  resolveRenderLabels(
    probabilities
  ).forEach(
    label => {
      const probability =
        finiteNumber(
          probabilities[label],
          0
        );

      const percent =
        Math.max(
          0,
          Math.min(
            100,
            probability * 100
          )
        );

      const row =
        document.createElement("div");

      row.className =
        "sf-prob-row";

      const labelRow =
        document.createElement("div");

      labelRow.className =
        "sf-prob-label";

      const name =
        document.createElement("span");

      name.textContent = label;

      const value =
        document.createElement("strong");

      value.textContent =
        `${percent.toFixed(2)}%`;

      labelRow.append(
        name,
        value
      );

      const track =
        document.createElement("div");

      track.className =
        "sf-prob-track";

      const fill =
        document.createElement("div");

      fill.className =
        `sf-prob-fill ${kind}`;

      fill.style.width =
        `${percent}%`;

      track.appendChild(fill);

      row.append(
        labelRow,
        track
      );

      container.appendChild(row);
    }
  );

  if (
    probabilitySum !== null &&
    Number.isFinite(
      Number(probabilitySum)
    )
  ) {
    const row =
      document.createElement("div");

    row.className =
      "sf-prob-sum";

    row.textContent =
      (
        "Probability sum: "
        + Number(
            probabilitySum
          ).toFixed(6)
      );

    container.appendChild(row);
  }
}


/* ============================================================
   RESULT DISPLAY
   ============================================================ */

function updatePredictionUI(data) {
  const state =
    String(
      data.current_state ||
      data.prediction ||
      "unknown"
    );

  const confidencePct =
    finiteNumber(
      data.confidence_percent,
      0
    );

  const gap =
    finiteNumber(
      data.confidence_gap,
      0
    );

  const level =
    String(
      data.confidence_level ||
      "Low"
    );

  const validation =
    data.runtime_validation || {};

  setText(
    predictionBox,
    state.toUpperCase()
  );

  setText(
    confidencePercent,
    `${confidencePct.toFixed(2)}%`
  );

  if (confidenceFill) {
    confidenceFill.style.width =
      `${Math.min(
        100,
        Math.max(
          0,
          confidencePct
        )
      )}%`;
  }

  setText(
    confidenceLevel,
    level
  );

  setText(
    rawPrediction,
    data.raw_top_class || "—"
  );

  const rawPct =
    Number(
      data.raw_confidence_percent
    );

  setText(
    rawConfidence,
    Number.isFinite(rawPct)
      ? `${rawPct.toFixed(2)}%`
      : "—"
  );

  setText(
    temporalSamples,
    data.temporal_samples ?? 0
  );

  setText(
    temporalWindow,
    data.temporal_window ??
    TEMPORAL_WINDOW
  );

  setText(
    temporalWindowStatus,
    (
      `${data.temporal_samples ?? 0}`
      + " / "
      + `${data.temporal_window ?? TEMPORAL_WINDOW}`
    )
  );

  setText(
    secondaryState,
    data.second_class || "—"
  );

  setText(
    confidenceGap,
    gap.toFixed(4)
  );

  setText(
    featureDimension,
    data.feature_dimension ?? "—"
  );

  setText(
    deviceInfo,
    data.device || "—"
  );

  renderProbabilityBars(
    probabilitiesBox,
    data.probabilities,
    "temporal",
    validation.temporal_probability_sum
  );

  renderProbabilityBars(
    rawProbabilitiesBox,
    data.raw_probabilities,
    "raw",
    validation.raw_probability_sum
  );

  const modalities =
    data.used_modalities || {};

  const active =
    Array.isArray(modalities)
      ? modalities.map(String)
      : Object.entries(modalities)
          .filter(
            ([, enabled]) =>
              Boolean(enabled)
          )
          .map(
            ([name]) => name
          );

  setText(
    activeModalities,
    active.length
      ? active.join(", ")
      : "—"
  );

  setText(
    technicalRawState,
    data.raw_top_class || "—"
  );

  setText(
    technicalTemporalSamples,
    (
      `${data.temporal_samples ?? 0}`
      + "/"
      + `${data.temporal_window ?? TEMPORAL_WINDOW}`
    )
  );

  const webcamResult =
    data.webcam_prediction;

  if (webcamResult) {
    setText(
      webcamPrediction,
      webcamResult.current_state ||
      "—"
    );

    const pct =
      Number(
        webcamResult.confidence_percent
      );

    setText(
      webcamConfidence,
      Number.isFinite(pct)
        ? `${pct.toFixed(2)}%`
        : "—"
    );

    setText(
      webcamCalibrationUsed,
      "Yes"
    );

    renderProbabilityBars(
      webcamProbabilityBars,
      webcamResult.probabilities,
      "webcam"
    );

  } else {
    setText(
      webcamPrediction,
      "Not used"
    );

    setText(
      webcamConfidence,
      "—"
    );

    setText(
      webcamCalibrationUsed,
      "No"
    );
  }

  if (data.audio_diagnostics) {
    updateAudioDiagnostic(
      data.audio_diagnostics
    );
  }

  const validationText =
    (
      "Runtime validation: "
      + (
          validation.pass
            ? "PASS"
            : "CHECK"
        )
      + " | Raw sum: "
      + finiteNumber(
          validation.raw_probability_sum
        ).toFixed(6)
      + " | Temporal sum: "
      + finiteNumber(
          validation.temporal_probability_sum
        ).toFixed(6)
    );

  setText(
    validationStatus,
    validationText
  );

  setText(
    statusBox,
    validationText
  );
}

function resetPredictionDisplay() {
  setText(predictionBox, "—");
  setText(confidencePercent, "—");
  setText(confidenceLevel, "—");

  if (confidenceFill) {
    confidenceFill.style.width =
      "0%";
  }

  setText(rawPrediction, "—");
  setText(rawConfidence, "—");

  setText(temporalSamples, "0");

  setText(
    temporalWindow,
    TEMPORAL_WINDOW
  );

  setText(
    temporalWindowStatus,
    `0 / ${TEMPORAL_WINDOW}`
  );

  setText(secondaryState, "—");
  setText(confidenceGap, "—");
  setText(featureDimension, "—");
  setText(deviceInfo, "—");
  setText(webcamPrediction, "—");
  setText(webcamConfidence, "—");
  setText(webcamCalibrationUsed, "—");
  setText(activeModalities, "—");
  setText(technicalRawState, "—");
  setText(technicalTemporalSamples, "0");
  setText(validationStatus, "");

  probabilitiesBox &&
    (probabilitiesBox.innerHTML = "");

  rawProbabilitiesBox &&
    (rawProbabilitiesBox.innerHTML = "");

  webcamProbabilityBars &&
    (webcamProbabilityBars.innerHTML = "");
}


/* ============================================================
   LIVE PREDICTION
   ============================================================ */

async function runLivePrediction() {
  if (
    predictionInFlight ||
    stateChangeInProgress ||
    modelInitialisationInFlight
  ) {
    return;
  }

  updateReadiness();

  if (!inputModalitiesReady()) {
    return;
  }

  /*
   * CRITICAL FIX:
   *
   * The model is initialised AFTER input modalities become
   * ready. We no longer require the model to already be loaded
   * before reaching this point.
   */
  if (
    modelState !== "ready" ||
    !fusionModelLoaded
  ) {
    const ready =
      await ensureModelsReady();

    if (!ready) {
      return;
    }
  }

  let webcamFrame = null;

  if (visualMode === "webcam") {
    webcamFrame =
      captureWebcamFrame();

    if (!webcamFrame) {
      return;
    }
  }

  predictionInFlight = true;

  const requestEpoch =
    clientEpoch;

  const requestGeneration =
    serverGeneration;

  try {
    const form = new FormData();

    form.append(
      "session_id",
      sessionId
    );

    form.append(
      "generation",
      String(requestGeneration)
    );

    form.append(
      "text",
      textInput
        ? textInput.value.trim()
        : ""
    );

    form.append(
      "keystroke_events",
      JSON.stringify(
        keystrokeEvents
      )
    );

    form.append(
      "visual_mode",
      visualMode
    );

    if (webcamFrame) {
      form.append(
        "webcam_frame",
        webcamFrame
      );
    }

    setText(
      statusBox,
      (
        "Running multimodal "
        + "fusion inference..."
      )
    );

    const response =
      await fetch(
        "/predict_live",
        {
          method: "POST",
          body: form
        }
      );

    let data = {};

    try {
      data =
        await response.json();
    } catch (_) {}

    if (!response.ok) {
      if (
        response.status === 409 &&
        handleConflictResponse(data)
      ) {
        return;
      }

      /*
       * If inference causes a backend model failure,
       * refresh the status before displaying it.
       */
      await checkModelStatus();

      throw new Error(
        formatServerError(data)
      );
    }

    if (
      requestEpoch !== clientEpoch
    ) {
      return;
    }

    const returnedGeneration =
      Number(data.generation);

    if (
      !Number.isFinite(
        returnedGeneration
      ) ||
      returnedGeneration
        !== requestGeneration
    ) {
      return;
    }

    serverGeneration =
      returnedGeneration;

    updatePredictionUI(data);

  } catch (error) {
    setText(
      statusBox,
      (
        "Live prediction failed: "
        + String(
            error.message || error
          )
      )
    );

  } finally {
    predictionInFlight = false;
  }
}


/* ============================================================
   RESETS
   ============================================================ */

async function resetTemporalWindow() {
  const epoch =
    beginStateChange(
      "Resetting temporal history..."
    );

  try {
    const data =
      await postForm(
        "/reset_temporal",
        {
          session_id:
            sessionId
        }
      );

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    resetPredictionDisplay();

  } catch (error) {
    setText(
      statusBox,
      "Temporal reset failed: "
      + String(
          error.message || error
        )
    );

  } finally {
    finishStateChange(epoch);
  }
}

async function resetSession() {
  const epoch =
    beginStateChange(
      "Resetting session..."
    );

  microphoneExpectedClose = true;

  try {
    await postForm(
      "/full_reset",
      {
        session_id:
          sessionId
      }
    );

    cleanupMicrophoneLocal();

    stopWebcamStreamLocally();
    stopVideoPreview();

    if (textInput) {
      textInput.value = "";
    }

    keystrokeEvents = [];
    activeKeys.clear();

    audioSourceReady = false;
    audioSourceName = null;
    audioSourceKind = null;

    visualMode = "none";
    visualSourceReady = false;
    visualSourceName = null;

    resetPredictionDisplay();

    setText(
      statusBox,
      "Session reset."
    );

  } catch (error) {
    setText(
      statusBox,
      "Full reset failed: "
      + String(
          error.message || error
        )
    );

  } finally {
    microphoneExpectedClose = false;
    finishStateChange(epoch);
  }
}


/* ============================================================
   BUTTON BINDINGS
   ============================================================ */

startMicBtn?.addEventListener(
  "click",
  () => void startMicrophoneStream()
);

stopMicBtn?.addEventListener(
  "click",
  () => void stopMicrophoneStream()
);

chooseAudioBtn?.addEventListener(
  "click",
  () => audioFileInput?.click()
);

chooseImageBtn?.addEventListener(
  "click",
  () => imageFileInput?.click()
);

chooseVideoBtn?.addEventListener(
  "click",
  () => videoFileInput?.click()
);

audioFileInput?.addEventListener(
  "change",
  () => {
    const file =
      audioFileInput.files?.[0];

    if (file) {
      void setAudioFile(file);
    }

    audioFileInput.value = "";
  }
);

imageFileInput?.addEventListener(
  "change",
  () => {
    const file =
      imageFileInput.files?.[0];

    if (file) {
      void setVisualImage(file);
    }

    imageFileInput.value = "";
  }
);

videoFileInput?.addEventListener(
  "change",
  () => {
    const file =
      videoFileInput.files?.[0];

    if (file) {
      void setVisualVideo(file);
    }

    videoFileInput.value = "";
  }
);

startBtn?.addEventListener(
  "click",
  () => void startWebcamMode()
);

stopBtn?.addEventListener(
  "click",
  () => void stopVisualMode()
);

resetTemporalBtn?.addEventListener(
  "click",
  () => void resetTemporalWindow()
);

resetBtn?.addEventListener(
  "click",
  () => void resetSession()
);


/* ============================================================
   SHUTDOWN
   ============================================================ */

window.addEventListener(
  "beforeunload",
  () => {
    microphoneExpectedClose = true;

    if (liveTimer !== null) {
      clearInterval(liveTimer);
    }

    cleanupMicrophoneLocal();
    stopWebcamStreamLocally();
    revokeVisualObjectUrl();
  }
);


/* ============================================================
   INITIALISATION
   ============================================================ */

async function initialise() {
  await checkModelStatus();

  resetPredictionDisplay();
  updateReadiness();

  setText(
    sessionStatus,
    "Session ready."
  );

  setText(
    statusBox,
    (
      "Ready. Provide text/keystrokes, "
      + "audio, and a visual source. "
      + "The fusion backend will load "
      + "when inference becomes ready."
    )
  );

  if (liveTimer !== null) {
    clearInterval(liveTimer);
  }

  liveTimer =
    window.setInterval(
      runLivePrediction,
      LIVE_INTERVAL_MS
    );
}

void initialise();
