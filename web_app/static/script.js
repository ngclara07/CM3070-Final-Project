"use strict";

/* ============================================================
   SenseFuzeAI continuous live-fusion browser client
   ============================================================ */


/* ============================================================
   HELPERS
   ============================================================ */

function getElement(id) {
  return document.getElementById(id);
}


function setText(element, value) {
  if (element) {
    element.textContent = String(value);
  }
}


function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}


function positiveInteger(value, fallback) {
  const number = Number(value);

  return (
    Number.isInteger(number)
    && number > 0
  )
    ? number
    : fallback;
}


function createSessionId() {
  if (
    window.crypto
    && typeof window.crypto.randomUUID === "function"
  ) {
    return window.crypto.randomUUID();
  }

  return (
    "session-"
    + Date.now().toString(36)
    + "-"
    + Math.random().toString(36).slice(2)
  );
}


function formatServerError(data) {
  if (!data) {
    return "Unknown server error.";
  }

  const detail =
    data.detail
    ?? data.error
    ?? data.message;

  if (typeof detail === "string") {
    return detail;
  }

  try {
    return JSON.stringify(detail);
  } catch (_) {
    return String(detail);
  }
}


async function postForm(url, values) {
  const formData = new FormData();

  Object.entries(values).forEach(
    ([key, value]) => {
      if (value !== undefined && value !== null) {
        formData.append(key, value);
      }
    }
  );

  const response = await fetch(
    url,
    {
      method: "POST",
      body: formData
    }
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


/* ============================================================
   DOM
   ============================================================ */

const textInput = getElement("textInput");

const webcam = getElement("webcam");
const canvas = getElement("frameCanvas");
const staticImagePreview =
  getElement("staticImagePreview");

const startBtn = getElement("startBtn");
const stopBtn = getElement("stopBtn");
const resetBtn = getElement("resetBtn");
const resetTemporalBtn =
  getElement("resetTemporalBtn");

const startMicBtn =
  getElement("startMicBtn");

const stopMicBtn =
  getElement("stopMicBtn");

const chooseAudioBtn =
  getElement("chooseAudioBtn");

const audioFileInput =
  getElement("audioFileInput");

const chooseImageBtn =
  getElement("chooseImageBtn");

const chooseVideoBtn =
  getElement("chooseVideoBtn");

const imageFileInput =
  getElement("imageFileInput");

const videoFileInput =
  getElement("videoFileInput");

const statusBox = getElement("status");
const sessionStatus =
  getElement("sessionStatus");

const audioStatus =
  getElement("audioStatus");

const audioDiagnostic =
  getElement("audioDiagnostic");

const webcamStatus =
  getElement("webcamStatus");

const modelStatusText =
  getElement("modelStatusText");

const webcamModelStatusText =
  getElement("webcamModelStatusText");

const predictionBox =
  getElement("prediction");

const confidencePercent =
  getElement("confidencePercent");

const confidenceFill =
  getElement("confidenceFill");

const confidenceLevel =
  getElement("confidenceLevel");

const rawPrediction =
  getElement("rawPrediction");

const rawConfidence =
  getElement("rawConfidence");

const temporalSamples =
  getElement("temporalSamples");

const temporalWindow =
  getElement("temporalWindow");

const temporalWindowStatus =
  getElement("temporalWindowStatus");

const secondaryState =
  getElement("secondaryState");

const confidenceGap =
  getElement("confidenceGap");

const featureDimension =
  getElement("featureDimension");

const deviceInfo =
  getElement("deviceInfo");

const probabilitiesBox =
  getElement("probabilities");

const rawProbabilitiesBox =
  getElement("rawProbabilities");

const webcamPrediction =
  getElement("webcamPrediction");

const webcamConfidence =
  getElement("webcamConfidence");

const webcamProbabilityBars =
  getElement("webcamProbabilityBars");

const webcamCalibrationUsed =
  getElement("webcamCalibrationUsed");

const activeModalities =
  getElement("activeModalities");

const technicalRawState =
  getElement("technicalRawState");

const technicalTemporalSamples =
  getElement("technicalTemporalSamples");

const sessionIdDisplay =
  getElement("sessionIdDisplay");

const validationStatus =
  getElement("validationStatus");

const modelReady =
  getElement("modelReady");

const webcamModelReady =
  getElement("webcamModelReady");

const textReady =
  getElement("textReady");

const keyReady =
  getElement("keyReady");

const audioReady =
  getElement("audioReady");

const imageReady =
  getElement("imageReady");

const textCard =
  getElement("textCard");

const webcamCard =
  getElement("webcamCard");

const audioCard =
  getElement("audioCard");

const charCount =
  getElement("charCount");

const keyCount =
  getElement("keyCount");

const audioStreamState =
  getElement("audioStreamState");

const audioBufferedSeconds =
  getElement("audioBufferedSeconds");

const audioLiveLevel =
  getElement("audioLiveLevel");

const audioPacketCount =
  getElement("audioPacketCount");


/* ============================================================
   CONFIGURATION
   ============================================================ */

let MIN_TEXT_CHARS = 20;
let MIN_KEYPRESSES = 20;

let LIVE_INTERVAL_MS = 2500;

let TEMPORAL_WINDOW = 5;

let TARGET_AUDIO_SAMPLE_RATE = 16000;

let AUDIO_STREAM_WINDOW_SECONDS = 10;
let AUDIO_STREAM_MIN_SECONDS = 2;

let behaviouralLabels = [];


/* ============================================================
   MODEL STATE
   ============================================================ */

let fusionModelLoaded = false;
let webcamModelLoaded = false;

let temporalFusionBackend = null;


/* ============================================================
   GENERATION / CONCURRENCY
   ============================================================ */

let serverGeneration = 0;

let clientEpoch = 0;

let stateChangeInProgress = false;

let predictionInFlight = false;

let liveTimer = null;


/* ============================================================
   KEYSTROKE STATE
   ============================================================ */

let keystrokeEvents = [];

const activeKeys = new Set();


/* ============================================================
   AUDIO STATE
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
   VISUAL STATE
   ============================================================ */

let visualMode = "none";

let visualSourceReady = false;
let visualSourceName = null;

let webcamStream = null;
let visualObjectUrl = null;


/* ============================================================
   SESSION
   ============================================================ */

const sessionId =
  createSessionId();

setText(
  sessionIdDisplay,
  sessionId
);


/* ============================================================
   EPOCH HELPERS
   ============================================================ */

function beginStateChange(message) {
  clientEpoch += 1;

  stateChangeInProgress = true;

  if (message) {
    setText(
      statusBox,
      message
    );
  }

  updateReadiness();

  return clientEpoch;
}


function finishStateChange(operationEpoch) {
  if (operationEpoch === clientEpoch) {
    stateChangeInProgress = false;
  }

  updateReadiness();
}


function operationStillCurrent(operationEpoch) {
  return operationEpoch === clientEpoch;
}


/* ============================================================
   SERVER CONFLICT HANDLING
   ============================================================ */

function handleConflictResponse(data) {
  const detail =
    data
    && typeof data.detail === "object"
    && data.detail !== null
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
    type === "stale_generation"
    || type === "stale_result"
    || type === "stale_session"
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
   READINESS
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
  return Boolean(audioSourceReady);
}


function visualIsReady() {
  if (
    visualMode === "image"
    || visualMode === "video"
  ) {
    return Boolean(visualSourceReady);
  }

  if (visualMode === "webcam") {
    return Boolean(
      visualSourceReady
      && webcamStream
      && webcam
      && webcam.videoWidth > 0
      && webcam.videoHeight > 0
    );
  }

  return false;
}


function allModalitiesReady() {
  return (
    !stateChangeInProgress
    && fusionModelLoaded
    && currentTextLength() >= MIN_TEXT_CHARS
    && currentKeydownCount() >= MIN_KEYPRESSES
    && audioIsReady()
    && visualIsReady()
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
    Number.isFinite(audioCurrentDbfs)
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

  setText(
    charCount,
    textCount
  );

  setText(
    keyCount,
    keydowns
  );

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

  if (audioReady) {
    audioReady.classList.toggle(
      "warning",
      microphoneStreaming
      && !audioOk
    );
  }

  setReady(
    imageReady,
    visualOk,
    "Ready",
    "Required"
  );

  if (textCard) {
    const active =
      textOk && keyOk;

    textCard.classList.toggle(
      "active",
      active
    );

    const badge =
      textCard.querySelector(".badge");

    if (badge) {
      badge.textContent =
        active
          ? "active"
          : "inactive";
    }
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

    const badge =
      audioCard.querySelector(".badge");

    if (badge) {
      badge.textContent =
        microphoneStreaming
          ? (
              audioOk
                ? "live"
                : "buffering"
            )
          : (
              audioOk
                ? "ready"
                : "inactive"
            );
    }
  }

  if (webcamCard) {
    webcamCard.classList.toggle(
      "active",
      visualOk
    );

    const badge =
      webcamCard.querySelector(".badge");

    if (badge) {
      badge.textContent =
        visualOk
          ? "active"
          : "inactive";
    }
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

async function checkModelStatus() {
  try {
    const response =
      await fetch(
        "/model-status",
        {
          cache: "no-store"
        }
      );

    const data =
      await response.json();

    if (!response.ok) {
      throw new Error(
        formatServerError(data)
      );
    }

    fusionModelLoaded =
      Boolean(data.fusion_model);

    webcamModelLoaded =
      Boolean(
        data.webcam_calibrated_image_model
      );

    temporalFusionBackend =
      data.temporal_fusion_backend
      || null;

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
          .filter(
            value =>
              value.length > 0
          );
    }

    setText(
      temporalWindow,
      TEMPORAL_WINDOW
    );

    setText(
      temporalWindowStatus,
      `0 / ${TEMPORAL_WINDOW}`
    );

    if (fusionModelLoaded) {
      setText(
        modelStatusText,
        (
          "FinalMultimodalInference loaded"
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
    } else {
      setText(
        modelStatusText,
        (
          "Fusion backend unavailable: "
          + String(
              data.error
              || "unknown error"
            )
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
            "Webcam calibration not required "
            + "by the current fusion schema."
          )
    );

  } catch (error) {
    fusionModelLoaded = false;

    setText(
      modelStatusText,
      (
        "Model-status query failed: "
        + String(
            error.message
            || error
          )
      )
    );
  }

  updateReadiness();
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
  if (!input || input.length === 0) {
    return new Float32Array(0);
  }

  if (inputRate === outputRate) {
    return new Float32Array(input);
  }

  const outputLength =
    Math.max(
      1,
      Math.round(
        input.length
        * outputRate
        / inputRate
      )
    );

  const output =
    new Float32Array(
      outputLength
    );

  const ratio =
    inputRate / outputRate;

  for (
    let index = 0;
    index < outputLength;
    index += 1
  ) {
    const position =
      index * ratio;

    const left =
      Math.floor(position);

    const right =
      Math.min(
        left + 1,
        input.length - 1
      );

    const fraction =
      position - left;

    output[index] =
      input[left]
      + (
          input[right]
          - input[left]
        )
      * fraction;
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

  let offset = 0;

  for (
    let index = 0;
    index < samples.length;
    index += 1
  ) {
    const value =
      Math.max(
        -1,
        Math.min(
          1,
          samples[index]
        )
      );

    const pcm =
      value < 0
        ? value * 32768
        : value * 32767;

    view.setInt16(
      offset,
      Math.round(pcm),
      true
    );

    offset += 2;
  }

  return buffer;
}


function calculateDbfs(samples) {
  if (!samples || samples.length === 0) {
    return -120;
  }

  let squareSum = 0;

  for (
    let index = 0;
    index < samples.length;
    index += 1
  ) {
    const value =
      samples[index];

    squareSum +=
      value * value;
  }

  const rms =
    Math.sqrt(
      squareSum
      / samples.length
    );

  return (
    20
    * Math.log10(
        Math.max(
          rms,
          1e-12
        )
      )
  );
}


/* ============================================================
   AUDIO DIAGNOSTICS
   ============================================================ */

function updateAudioDiagnostic(audio) {
  const value =
    audio || {};

  const dbfs =
    Number(value.dbfs);

  const duration =
    Number(
      value.duration_sec
      ?? value.analysed_duration_sec
    );

  let text =
    (
      "Audio condition: "
      + String(
          value.condition
          || "unknown"
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
  }

  if (value.note) {
    text +=
      (
        " | "
        + String(value.note)
      );
  }

  setText(
    audioDiagnostic,
    text
  );

  if (Number.isFinite(dbfs)) {
    audioCurrentDbfs = dbfs;
  }

  updateAudioMetrics();
}


/* ============================================================
   CONTINUOUS MICROPHONE - WEBSOCKET
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
      const timeout =
        window.setTimeout(
          () => {
            reject(
              new Error(
                "Audio WebSocket connection timed out."
              )
            );
          },
          5000
        );

      socket.addEventListener(
        "open",
        () => {
          window.clearTimeout(timeout);
          resolve();
        },
        {
          once: true
        }
      );

      socket.addEventListener(
        "error",
        () => {
          window.clearTimeout(timeout);

          reject(
            new Error(
              "Audio WebSocket connection failed."
            )
          );
        },
        {
          once: true
        }
      );
    }
  );
}


async function synchroniseAfterUnexpectedAudioClose() {
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

  } catch (_) {
    // Future server interactions will repair generation
    // through the normal stale-generation mechanism.
  }
}


function cleanupMicrophoneLocal() {
  microphoneStreaming = false;

  if (microphoneProcessor) {
    microphoneProcessor.onaudioprocess =
      null;

    try {
      microphoneProcessor.disconnect();
    } catch (_) {
      // Ignore.
    }
  }

  if (microphoneSourceNode) {
    try {
      microphoneSourceNode.disconnect();
    } catch (_) {
      // Ignore.
    }
  }

  if (microphoneSilentGain) {
    try {
      microphoneSilentGain.disconnect();
    } catch (_) {
      // Ignore.
    }
  }

  if (microphoneStream) {
    microphoneStream
      .getTracks()
      .forEach(
        track => track.stop()
      );
  }

  microphoneStream = null;
  microphoneProcessor = null;
  microphoneSourceNode = null;
  microphoneSilentGain = null;

  if (microphoneContext) {
    const context =
      microphoneContext;

    microphoneContext = null;

    void context.close().catch(
      () => {}
    );
  }

  if (audioSocket) {
    const socket =
      audioSocket;

    audioSocket = null;

    socket.onmessage = null;
    socket.onerror = null;
    socket.onclose = null;

    try {
      socket.close();
    } catch (_) {
      // Ignore.
    }
  }

  audioStreamToken = null;

  updateReadiness();
}


async function handleUnexpectedAudioDisconnect() {
  if (microphoneExpectedClose) {
    return;
  }

  cleanupMicrophoneLocal();

  audioSourceReady = false;
  audioSourceName = null;
  audioSourceKind = null;

  audioBufferedSec = 0;

  setText(
    audioStatus,
    "Microphone stream disconnected."
  );

  setText(
    statusBox,
    "Continuous microphone stream disconnected."
  );

  await synchroniseAfterUnexpectedAudioClose();

  resetPredictionDisplay();
  updateReadiness();
}


function installAudioSocketHandlers(socket) {
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
        data.type
        !== "audio_status"
      ) {
        return;
      }

      audioBufferedSec =
        finiteNumber(
          data.buffered_seconds,
          0
        );

      audioPackets =
        positiveInteger(
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
        (
          audioSourceReady
            ? (
                "Live microphone streaming"
                + ` | rolling ${audioBufferedSec.toFixed(1)}s buffer`
              )
            : (
                "Live microphone buffering"
                + ` | ${audioBufferedSec.toFixed(1)}`
                + `/${AUDIO_STREAM_MIN_SECONDS.toFixed(1)}s`
              )
        )
      );

      updateReadiness();
    };


  socket.onerror =
    () => {
      if (!microphoneExpectedClose) {
        setText(
          audioStatus,
          "Microphone transport error."
        );
      }
    };


  socket.onclose =
    () => {
      if (!microphoneExpectedClose) {
        void handleUnexpectedAudioDisconnect();
      }
    };
}


async function startMicrophoneStream() {
  if (microphoneStreaming) {
    return;
  }

  if (
    !navigator.mediaDevices
    || !navigator.mediaDevices.getUserMedia
  ) {
    setText(
      statusBox,
      "Microphone API is unavailable."
    );

    return;
  }

  const AudioContextClass =
    window.AudioContext
    || window.webkitAudioContext;

  if (!AudioContextClass) {
    setText(
      statusBox,
      "Web Audio API is unavailable."
    );

    return;
  }

  const operationEpoch =
    beginStateChange(
      "Starting continuous microphone stream..."
    );

  microphoneExpectedClose = false;

  let stream = null;
  let context = null;

  try {
    stream =
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

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      stream
        .getTracks()
        .forEach(
          track => track.stop()
        );

      return;
    }

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
      !operationStillCurrent(
        operationEpoch
      )
    ) {
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

    resetPredictionDisplay();

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

    audioSourceReady = false;
    audioSourceName = "Live microphone";
    audioSourceKind =
      "microphone_stream";

    audioBufferedSec = 0;
    audioPackets = 0;
    audioCurrentDbfs = null;

    microphoneProcessor.onaudioprocess =
      event => {
        if (
          !microphoneStreaming
          || !audioSocket
          || audioSocket.readyState
            !== WebSocket.OPEN
        ) {
          return;
        }

        const input =
          new Float32Array(
            event.inputBuffer
              .getChannelData(0)
          );

        audioCurrentDbfs =
          calculateDbfs(input);

        const resampled =
          resampleLinear(
            input,
            context.sampleRate,
            TARGET_AUDIO_SAMPLE_RATE
          );

        if (resampled.length === 0) {
          return;
        }

        const pcm =
          float32ToPCM16Buffer(
            resampled
          );

        try {
          audioSocket.send(pcm);
        } catch (_) {
          // Socket close handler manages disconnection.
        }

        updateAudioMetrics();
      };

    microphoneSourceNode.connect(
      microphoneProcessor
    );

    microphoneProcessor.connect(
      microphoneSilentGain
    );

    microphoneSilentGain.connect(
      context.destination
    );

    setText(
      audioStatus,
      (
        "Microphone live; buffering "
        + `first ${AUDIO_STREAM_MIN_SECONDS.toFixed(1)}s...`
      )
    );

    setText(
      statusBox,
      (
        "Continuous microphone stream started"
        + ` | Generation=${serverGeneration}.`
      )
    );

  } catch (error) {
    if (stream) {
      stream
        .getTracks()
        .forEach(
          track => track.stop()
        );
    }

    if (context) {
      try {
        await context.close();
      } catch (_) {
        // Ignore.
      }
    }

    cleanupMicrophoneLocal();

    try {
      const stopData =
        await postForm(
          "/audio_stream/stop",
          {
            session_id:
              sessionId
          }
        );

      serverGeneration =
        finiteNumber(
          stopData.generation,
          serverGeneration
        );

    } catch (_) {
      // Ignore secondary cleanup failure.
    }

    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {
      setText(
        statusBox,
        (
          "Microphone stream failed: "
          + String(
              error.message
              || error
            )
        )
      );
    }

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


async function stopMicrophoneStream(
  resetDisplay = true
) {
  if (
    !microphoneStreaming
    && audioSourceKind
      !== "microphone_stream"
  ) {
    return;
  }

  const operationEpoch =
    beginStateChange(
      "Stopping continuous microphone stream..."
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

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      return;
    }

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

    if (resetDisplay) {
      resetPredictionDisplay();
    }

    setText(
      audioStatus,
      "Microphone stream stopped."
    );

    setText(
      audioDiagnostic,
      "Audio condition: —"
    );

    setText(
      statusBox,
      (
        "Continuous microphone stopped"
        + ` | Generation=${serverGeneration}.`
      )
    );

  } catch (error) {
    cleanupMicrophoneLocal();

    audioSourceReady = false;

    setText(
      statusBox,
      (
        "Microphone stop failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    microphoneExpectedClose = false;

    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   FIXED AUDIO FILE FALLBACK
   ============================================================ */

async function setAudioFile(file) {
  if (!file) {
    return;
  }

  if (microphoneStreaming) {
    await stopMicrophoneStream(
      false
    );
  }

  const operationEpoch =
    beginStateChange(
      "Loading fixed audio file..."
    );

  const formData =
    new FormData();

  formData.append(
    "session_id",
    sessionId
  );

  formData.append(
    "source_kind",
    "file"
  );

  formData.append(
    "audio_file",
    file,
    file.name
  );

  try {
    const response =
      await fetch(
        "/set_audio_source",
        {
          method: "POST",
          body: formData
        }
      );

    const data =
      await response.json();

    if (!response.ok) {
      throw new Error(
        formatServerError(data)
      );
    }

    if (
      !operationStillCurrent(
        operationEpoch
      )
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
      data.audio_name
      || file.name;

    audioSourceKind = "file";

    audioBufferedSec =
      finiteNumber(
        data.audio_diagnostics
          ?.duration_sec,
        0
      );

    audioPackets = 0;

    resetPredictionDisplay();

    updateAudioDiagnostic(
      data.audio_diagnostics
    );

    setText(
      audioStatus,
      (
        "Fixed audio file: "
        + audioSourceName
      )
    );

    setText(
      statusBox,
      (
        "Fixed audio source ready"
        + ` | Generation=${serverGeneration}.`
      )
    );

  } catch (error) {
    setText(
      statusBox,
      (
        "Audio file failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   VISUAL UTILITY FUNCTIONS
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
  if (webcamStream) {
    webcamStream
      .getTracks()
      .forEach(
        track => track.stop()
      );
  }

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
  } catch (_) {
    // Ignore.
  }

  webcam.removeAttribute(
    "src"
  );

  try {
    webcam.load();
  } catch (_) {
    // Ignore.
  }
}


/* ============================================================
   IMAGE SOURCE
   ============================================================ */

async function setVisualImage(file) {
  if (!file) {
    return;
  }

  const operationEpoch =
    beginStateChange(
      "Loading image source..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();
  revokeVisualObjectUrl();

  visualMode = "none";
  visualSourceReady = false;

  const formData =
    new FormData();

  formData.append(
    "session_id",
    sessionId
  );

  formData.append(
    "image_file",
    file,
    file.name
  );

  try {
    const response =
      await fetch(
        "/set_visual_image",
        {
          method: "POST",
          body: formData
        }
      );

    const data =
      await response.json();

    if (!response.ok) {
      throw new Error(
        formatServerError(data)
      );
    }

    if (
      !operationStillCurrent(
        operationEpoch
      )
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
      data.visual_name
      || file.name;

    visualObjectUrl =
      URL.createObjectURL(file);

    if (staticImagePreview) {
      staticImagePreview.src =
        visualObjectUrl;

      staticImagePreview.classList.remove(
        "hidden"
      );
    }

    if (webcam) {
      webcam.classList.add(
        "hidden"
      );
    }

    resetPredictionDisplay();

    setText(
      webcamStatus,
      (
        "Image source: "
        + visualSourceName
      )
    );

    setText(
      sessionStatus,
      "Static image source active."
    );

    if (startBtn) {
      startBtn.disabled = false;
    }

    if (stopBtn) {
      stopBtn.disabled = true;
    }

  } catch (error) {
    visualMode = "none";
    visualSourceReady = false;
    visualSourceName = null;

    setText(
      statusBox,
      (
        "Image source failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   VIDEO SOURCE
   ============================================================ */

async function setVisualVideo(file) {
  if (!file) {
    return;
  }

  const operationEpoch =
    beginStateChange(
      "Loading video source..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();
  hideStaticImagePreview();
  revokeVisualObjectUrl();

  visualMode = "none";
  visualSourceReady = false;

  const formData =
    new FormData();

  formData.append(
    "session_id",
    sessionId
  );

  formData.append(
    "video_file",
    file,
    file.name
  );

  try {
    const response =
      await fetch(
        "/set_visual_video",
        {
          method: "POST",
          body: formData
        }
      );

    const data =
      await response.json();

    if (!response.ok) {
      throw new Error(
        formatServerError(data)
      );
    }

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      return;
    }

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    visualMode = "video";
    visualSourceReady = true;

    visualSourceName =
      data.visual_name
      || file.name;

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
      } catch (_) {
        // Browser playback is preview only.
      }
    }

    resetPredictionDisplay();

    setText(
      webcamStatus,
      (
        "Video source: "
        + visualSourceName
      )
    );

    setText(
      sessionStatus,
      "Video source running."
    );

    if (startBtn) {
      startBtn.disabled = true;
    }

    if (stopBtn) {
      stopBtn.disabled = false;
    }

  } catch (error) {
    visualMode = "none";
    visualSourceReady = false;
    visualSourceName = null;

    setText(
      statusBox,
      (
        "Video source failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   WEBCAM SOURCE
   ============================================================ */

async function startWebcamMode() {
  if (
    !navigator.mediaDevices
    || !navigator.mediaDevices.getUserMedia
  ) {
    setText(
      statusBox,
      "Webcam API is unavailable."
    );

    return;
  }

  const operationEpoch =
    beginStateChange(
      "Starting webcam..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();
  hideStaticImagePreview();
  revokeVisualObjectUrl();

  visualMode = "none";
  visualSourceReady = false;

  let stream = null;

  try {
    stream =
      await navigator.mediaDevices
        .getUserMedia(
          {
            video: true,
            audio: false
          }
        );

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      stream
        .getTracks()
        .forEach(
          track => track.stop()
        );

      return;
    }

    if (!webcam) {
      throw new Error(
        "Webcam video element is missing."
      );
    }

    webcamStream = stream;

    webcam.classList.remove(
      "hidden"
    );

    webcam.removeAttribute(
      "src"
    );

    webcam.srcObject =
      webcamStream;

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

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      stopWebcamStreamLocally();

      return;
    }

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    visualMode = "webcam";
    visualSourceReady = true;
    visualSourceName = "Webcam";

    resetPredictionDisplay();

    setText(
      webcamStatus,
      "Webcam active."
    );

    setText(
      sessionStatus,
      "Webcam stream running."
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
    visualSourceName = null;

    setText(
      statusBox,
      (
        "Webcam start failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   STOP VISUAL STREAM
   ============================================================ */

async function stopVisualMode() {
  if (
    visualMode !== "video"
    && visualMode !== "webcam"
  ) {
    setText(
      statusBox,
      "No live visual stream is currently running."
    );

    return;
  }

  const operationEpoch =
    beginStateChange(
      "Stopping visual stream..."
    );

  stopWebcamStreamLocally();
  stopVideoPreview();
  revokeVisualObjectUrl();

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
        data.visual_mode
        || "none"
      );

    visualSourceReady =
      Boolean(
        data.visual_ready
      );

    if (visualMode === "none") {
      visualSourceName = null;
    }

    setText(
      webcamStatus,
      "Visual stream stopped."
    );

    setText(
      sessionStatus,
      "Visual stream stopped."
    );

    if (startBtn) {
      startBtn.disabled = false;
    }

    if (stopBtn) {
      stopBtn.disabled = true;
    }

  } catch (error) {
    setText(
      statusBox,
      (
        "Stop visual failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   WEBCAM FRAME CAPTURE
   ============================================================ */

function captureWebcamFrame() {
  if (
    visualMode !== "webcam"
    || !webcamStream
    || !webcam
    || !canvas
    || webcam.videoWidth <= 0
    || webcam.videoHeight <= 0
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
    "image/png"
  );
}


/* ============================================================
   PROBABILITY RENDERING
   ============================================================ */

function resolveRenderLabels(probabilities) {
  if (behaviouralLabels.length > 0) {
    return behaviouralLabels;
  }

  if (
    probabilities
    && typeof probabilities === "object"
  ) {
    return Object.keys(probabilities);
  }

  return [];
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
    !probabilities
    || typeof probabilities !== "object"
  ) {
    return;
  }

  const labels =
    resolveRenderLabels(
      probabilities
    );

  labels.forEach(
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
        document.createElement(
          "div"
        );

      row.className =
        "sf-prob-row";

      const labelRow =
        document.createElement(
          "div"
        );

      labelRow.className =
        "sf-prob-label";

      const name =
        document.createElement(
          "span"
        );

      name.textContent =
        label;

      const value =
        document.createElement(
          "strong"
        );

      value.textContent =
        `${percent.toFixed(2)}%`;

      labelRow.appendChild(name);
      labelRow.appendChild(value);

      const track =
        document.createElement(
          "div"
        );

      track.className =
        "sf-prob-track";

      const fill =
        document.createElement(
          "div"
        );

      fill.className =
        (
          "sf-prob-fill "
          + kind
        );

      fill.style.width =
        `${percent}%`;

      track.appendChild(fill);

      row.appendChild(labelRow);
      row.appendChild(track);

      container.appendChild(row);
    }
  );

  if (
    probabilitySum !== null
    && Number.isFinite(
      Number(probabilitySum)
    )
  ) {
    const sumRow =
      document.createElement(
        "div"
      );

    sumRow.className =
      "sf-prob-sum";

    const label =
      document.createElement(
        "span"
      );

    label.textContent =
      "Probability sum";

    const value =
      document.createElement(
        "strong"
      );

    value.textContent =
      Number(
        probabilitySum
      ).toFixed(6);

    sumRow.appendChild(label);
    sumRow.appendChild(value);

    container.appendChild(sumRow);
  }
}


/* ============================================================
   RESULT DISPLAY
   ============================================================ */

function updatePredictionUI(data) {
  if (
    !data
    || typeof data !== "object"
  ) {
    return;
  }

  const generation =
    Number(data.generation);

  if (Number.isFinite(generation)) {
    serverGeneration = generation;
  }

  const state =
    String(
      data.current_state
      || data.prediction
      || "unknown"
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
      data.confidence_level
      || "Low"
    );

  const validation =
    data.runtime_validation
    || {};

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
      (
        Math.max(
          0,
          Math.min(
            100,
            confidencePct
          )
        )
        + "%"
      );
  }

  setText(
    confidenceLevel,
    level
  );

  if (confidenceLevel) {
    confidenceLevel.classList.remove(
      "confidence-high",
      "confidence-medium",
      "confidence-low"
    );

    const className =
      (
        "confidence-"
        + level
          .toLowerCase()
          .replace(
            /[^a-z]+/g,
            "-"
          )
      );

    confidenceLevel.classList.add(
      className
    );
  }

  setText(
    rawPrediction,
    data.raw_top_class
    || "—"
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
    data.temporal_samples
    ?? 0
  );

  setText(
    temporalWindow,
    data.temporal_window
    ?? TEMPORAL_WINDOW
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
    data.second_class
    || "—"
  );

  setText(
    confidenceGap,
    gap.toFixed(4)
  );

  setText(
    featureDimension,
    data.feature_dimension
    ?? "—"
  );

  setText(
    deviceInfo,
    data.device
    || "—"
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
    data.used_modalities
    || {};

  let active = [];

  if (Array.isArray(modalities)) {
    active =
      modalities.map(String);

  } else if (
    modalities
    && typeof modalities === "object"
  ) {
    active =
      Object.entries(modalities)
        .filter(
          ([, enabled]) =>
            Boolean(enabled)
        )
        .map(
          ([name]) => name
        );
  }

  setText(
    activeModalities,
    active.length > 0
      ? active.join(", ")
      : "—"
  );

  setText(
    technicalRawState,
    data.raw_top_class
    || "—"
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

  if (
    webcamResult
    && typeof webcamResult === "object"
  ) {
    setText(
      webcamPrediction,
      webcamResult.current_state
      || "—"
    );

    const webcamPct =
      Number(
        webcamResult.confidence_percent
      );

    setText(
      webcamConfidence,
      Number.isFinite(webcamPct)
        ? `${webcamPct.toFixed(2)}%`
        : "—"
    );

    setText(
      webcamCalibrationUsed,
      "Yes"
    );

    renderProbabilityBars(
      webcamProbabilityBars,
      webcamResult.probabilities,
      "webcam",
      null
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

    if (webcamProbabilityBars) {
      webcamProbabilityBars.innerHTML =
        "";
    }
  }

  if (data.audio_diagnostics) {
    updateAudioDiagnostic(
      data.audio_diagnostics
    );
  }

  const rawSum =
    finiteNumber(
      validation.raw_probability_sum,
      0
    );

  const temporalSum =
    finiteNumber(
      validation.temporal_probability_sum,
      0
    );

  const windowFull =
    Boolean(
      data.temporal_window_full
      ?? validation.temporal_window_full
    );

  const validationText =
    (
      "Runtime validation: "
      + (
          validation.pass
            ? "PASS"
            : "CHECK"
        )
      + " | Raw sum: "
      + rawSum.toFixed(6)
      + " | Temporal sum: "
      + temporalSum.toFixed(6)
      + " | Window: "
      + (
          windowFull
            ? "FULL"
            : "WARMING UP"
        )
    );

  setText(
    validationStatus,
    validationText
  );

  setText(
    statusBox,
    (
      validationText
      + ` | Generation=${serverGeneration}`
      + ` | Audio=${data.audio_source_kind || audioSourceKind || "—"}`
      + ` | Visual=${data.visual_source_type || visualMode}`
    )
  );
}


/* ============================================================
   RESET RESULT DISPLAY
   ============================================================ */

function resetPredictionDisplay() {
  setText(
    predictionBox,
    "—"
  );

  setText(
    confidencePercent,
    "—"
  );

  if (confidenceFill) {
    confidenceFill.style.width =
      "0%";
  }

  setText(
    confidenceLevel,
    "—"
  );

  if (confidenceLevel) {
    confidenceLevel.classList.remove(
      "confidence-high",
      "confidence-medium",
      "confidence-low"
    );
  }

  setText(
    rawPrediction,
    "—"
  );

  setText(
    rawConfidence,
    "—"
  );

  setText(
    temporalSamples,
    "0"
  );

  setText(
    temporalWindow,
    TEMPORAL_WINDOW
  );

  setText(
    temporalWindowStatus,
    `0 / ${TEMPORAL_WINDOW}`
  );

  setText(
    secondaryState,
    "—"
  );

  setText(
    confidenceGap,
    "—"
  );

  setText(
    featureDimension,
    "—"
  );

  setText(
    deviceInfo,
    "—"
  );

  setText(
    webcamPrediction,
    "—"
  );

  setText(
    webcamConfidence,
    "—"
  );

  setText(
    webcamCalibrationUsed,
    "—"
  );

  setText(
    activeModalities,
    "—"
  );

  setText(
    technicalRawState,
    "—"
  );

  setText(
    technicalTemporalSamples,
    "0"
  );

  setText(
    sessionIdDisplay,
    sessionId
  );

  if (probabilitiesBox) {
    probabilitiesBox.innerHTML = "";
  }

  if (rawProbabilitiesBox) {
    rawProbabilitiesBox.innerHTML = "";
  }

  if (webcamProbabilityBars) {
    webcamProbabilityBars.innerHTML = "";
  }

  setText(
    validationStatus,
    ""
  );
}


/* ============================================================
   LIVE PREDICTION
   ============================================================ */

async function runLivePrediction() {
  if (
    predictionInFlight
    || stateChangeInProgress
  ) {
    return;
  }

  updateReadiness();

  if (!allModalitiesReady()) {
    return;
  }

  let webcamFrame = null;

  if (visualMode === "webcam") {
    webcamFrame =
      captureWebcamFrame();

    if (!webcamFrame) {
      setText(
        statusBox,
        "Current webcam frame is unavailable."
      );

      return;
    }
  }

  predictionInFlight = true;

  const requestEpoch =
    clientEpoch;

  const requestGeneration =
    serverGeneration;

  try {
    const formData =
      new FormData();

    formData.append(
      "session_id",
      sessionId
    );

    formData.append(
      "generation",
      String(
        requestGeneration
      )
    );

    formData.append(
      "text",
      textInput
        ? textInput.value.trim()
        : ""
    );

    formData.append(
      "keystroke_events",
      JSON.stringify(
        keystrokeEvents
      )
    );

    formData.append(
      "visual_mode",
      visualMode
    );

    if (webcamFrame) {
      formData.append(
        "webcam_frame",
        webcamFrame
      );
    }

    setText(
      statusBox,
      "Running canonical multimodal fusion inference..."
    );

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
      if (
        response.status === 409
        && handleConflictResponse(data)
      ) {
        return;
      }

      throw new Error(
        formatServerError(data)
      );
    }

    if (
      requestEpoch
      !== clientEpoch
    ) {
      return;
    }

    const returnedGeneration =
      Number(
        data.generation
      );

    if (
      !Number.isFinite(
        returnedGeneration
      )
      || returnedGeneration
        !== requestGeneration
    ) {
      return;
    }

    serverGeneration =
      returnedGeneration;

    updatePredictionUI(data);

  } catch (error) {
    if (
      requestEpoch
      === clientEpoch
    ) {
      setText(
        statusBox,
        (
          "Live prediction failed: "
          + String(
              error.message
              || error
            )
        )
      );
    }

  } finally {
    predictionInFlight = false;
  }
}


/* ============================================================
   TEMPORAL RESET
   ============================================================ */

async function resetTemporalWindow() {
  const operationEpoch =
    beginStateChange(
      "Resetting temporal probability history..."
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

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      return;
    }

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    TEMPORAL_WINDOW =
      positiveInteger(
        data.temporal_window,
        TEMPORAL_WINDOW
      );

    resetPredictionDisplay();

    setText(
      statusBox,
      (
        "Temporal probability history reset"
        + ` | Generation=${serverGeneration}.`
        + (
            microphoneStreaming
              ? " Live microphone remains streaming."
              : ""
          )
      )
    );

  } catch (error) {
    setText(
      statusBox,
      (
        "Temporal reset failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   FULL RESET
   ============================================================ */

async function resetSession() {
  const operationEpoch =
    beginStateChange(
      "Performing full session reset..."
    );

  microphoneExpectedClose = true;

  stopWebcamStreamLocally();
  stopVideoPreview();
  hideStaticImagePreview();
  revokeVisualObjectUrl();

  try {
    const data =
      await postForm(
        "/full_reset",
        {
          session_id:
            sessionId
        }
      );

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      return;
    }

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    cleanupMicrophoneLocal();

    if (textInput) {
      textInput.value = "";
    }

    keystrokeEvents = [];
    activeKeys.clear();

    audioSourceReady = false;
    audioSourceName = null;
    audioSourceKind = null;

    audioBufferedSec = 0;
    audioPackets = 0;
    audioCurrentDbfs = null;

    visualMode = "none";
    visualSourceReady = false;
    visualSourceName = null;

    if (audioFileInput) {
      audioFileInput.value = "";
    }

    if (imageFileInput) {
      imageFileInput.value = "";
    }

    if (videoFileInput) {
      videoFileInput.value = "";
    }

    resetPredictionDisplay();

    setText(
      audioStatus,
      "Microphone stream inactive."
    );

    setText(
      audioDiagnostic,
      "Audio condition: —"
    );

    setText(
      webcamStatus,
      "Visual input inactive."
    );

    setText(
      sessionStatus,
      "Session reset."
    );

    setText(
      statusBox,
      (
        "Full session reset"
        + ` | Generation=${serverGeneration}.`
      )
    );

    if (startBtn) {
      startBtn.disabled = false;
    }

    if (stopBtn) {
      stopBtn.disabled = true;
    }

  } catch (error) {
    cleanupMicrophoneLocal();

    setText(
      statusBox,
      (
        "Full reset failed: "
        + String(
            error.message
            || error
          )
      )
    );

  } finally {
    microphoneExpectedClose = false;

    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   BUTTON / FILE BINDINGS
   ============================================================ */

if (startMicBtn) {
  startMicBtn.addEventListener(
    "click",
    () => {
      void startMicrophoneStream();
    }
  );
}


if (stopMicBtn) {
  stopMicBtn.addEventListener(
    "click",
    () => {
      void stopMicrophoneStream();
    }
  );
}


if (chooseAudioBtn) {
  chooseAudioBtn.addEventListener(
    "click",
    () => {
      if (audioFileInput) {
        audioFileInput.click();
      }
    }
  );
}


if (audioFileInput) {
  audioFileInput.addEventListener(
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
}


if (chooseImageBtn) {
  chooseImageBtn.addEventListener(
    "click",
    () => {
      imageFileInput?.click();
    }
  );
}


if (chooseVideoBtn) {
  chooseVideoBtn.addEventListener(
    "click",
    () => {
      videoFileInput?.click();
    }
  );
}


if (imageFileInput) {
  imageFileInput.addEventListener(
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
}


if (videoFileInput) {
  videoFileInput.addEventListener(
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
}


if (startBtn) {
  startBtn.addEventListener(
    "click",
    () => {
      void startWebcamMode();
    }
  );
}


if (stopBtn) {
  stopBtn.addEventListener(
    "click",
    () => {
      void stopVisualMode();
    }
  );
}


if (resetTemporalBtn) {
  resetTemporalBtn.addEventListener(
    "click",
    () => {
      void resetTemporalWindow();
    }
  );
}


if (resetBtn) {
  resetBtn.addEventListener(
    "click",
    () => {
      void resetSession();
    }
  );
}


/* ============================================================
   SHUTDOWN
   ============================================================ */

window.addEventListener(
  "beforeunload",
  () => {
    microphoneExpectedClose = true;

    if (liveTimer !== null) {
      window.clearInterval(
        liveTimer
      );

      liveTimer = null;
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

  if (startBtn) {
    startBtn.disabled = false;
  }

  if (stopBtn) {
    stopBtn.disabled = true;
  }

  setText(
    sessionStatus,
    "Session not started."
  );

  setText(
    statusBox,
    (
      "Ready. Provide text and keystrokes, "
      + "start the continuous microphone or select "
      + "an audio file, and provide a visual source."
    )
  );

  if (liveTimer !== null) {
    window.clearInterval(
      liveTimer
    );
  }

  liveTimer =
    window.setInterval(
      runLivePrediction,
      LIVE_INTERVAL_MS
    );
}


void initialise();
