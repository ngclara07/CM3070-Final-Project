"use strict";

/* ============================================================
   SenseFuzeAI
   Recoverable continuous multimodal browser client
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

  return Number.isFinite(number)
    ? number
    : fallback;
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


function sleep(milliseconds) {
  return new Promise(
    resolve => {
      window.setTimeout(
        resolve,
        milliseconds
      );
    }
  );
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


function formatServerError(
  data,
  fallback = "Unknown server error."
) {
  if (!data) {
    return fallback;
  }

  const detail =
    data.detail
    ?? data.error
    ?? data.exception
    ?? data.message;

  if (
    detail === undefined
    || detail === null
  ) {
    return fallback;
  }

  if (typeof detail === "string") {
    return detail;
  }

  if (
    typeof detail === "object"
    && detail.message
  ) {
    return String(
      detail.message
    );
  }

  try {
    return JSON.stringify(detail);
  } catch (_) {
    return String(detail);
  }
}


async function fetchJson(
  url,
  options = {}
) {
  let response;

  try {
    response = await fetch(
      url,
      options
    );

  } catch (networkError) {
    const error = new Error(
      (
        "Network request failed: "
        + (
            networkError?.message
            || String(networkError)
          )
      )
    );

    error.cause =
      networkError;

    throw error;
  }

  let data = {};

  try {
    data =
      await response.json();

  } catch (_) {
    try {
      const text =
        await response.text();

      if (text) {
        data = {
          detail: text
        };
      }

    } catch (_) {
      data = {};
    }
  }

  if (!response.ok) {
    const error = new Error(
      formatServerError(
        data,
        `HTTP ${response.status}`
      )
    );

    error.status =
      response.status;

    error.data =
      data;

    throw error;
  }

  return data;
}


async function postForm(
  url,
  values
) {
  const formData =
    new FormData();

  Object.entries(
    values
  ).forEach(
    ([key, value]) => {
      if (
        value !== undefined
        && value !== null
      ) {
        formData.append(
          key,
          value
        );
      }
    }
  );

  return fetchJson(
    url,
    {
      method: "POST",
      body: formData
    }
  );
}


/* ============================================================
   DOM
   ============================================================ */

const textInput =
  getElement("textInput");

const webcam =
  getElement("webcam");

const canvas =
  getElement("frameCanvas");

const staticImagePreview =
  getElement("staticImagePreview");


const startBtn =
  getElement("startBtn");

const stopBtn =
  getElement("stopBtn");

const resetBtn =
  getElement("resetBtn");

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


const statusBox =
  getElement("status");

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
  getElement(
    "webcamModelStatusText"
  );


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
  getElement(
    "temporalWindowStatus"
  );

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
  getElement(
    "webcamProbabilityBars"
  );

const webcamCalibrationUsed =
  getElement(
    "webcamCalibrationUsed"
  );


const activeModalities =
  getElement("activeModalities");

const technicalRawState =
  getElement("technicalRawState");

const technicalTemporalSamples =
  getElement(
    "technicalTemporalSamples"
  );

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
  getElement(
    "audioBufferedSeconds"
  );

const audioLiveLevel =
  getElement("audioLiveLevel");

const audioPacketCount =
  getElement("audioPacketCount");


/* ============================================================
   CONFIGURATION
   ============================================================ */

let MIN_TEXT_CHARS = 20;
let MIN_KEYPRESSES = 20;

let LIVE_INTERVAL_MS = 15000;
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

let modelState =
  "not_loaded";

let fusionModelLoaded =
  false;

let webcamModelLoaded =
  false;

let temporalFusionBackend =
  null;

let modelInitialisationInFlight =
  false;


/* ============================================================
   CLIENT / GENERATION STATE
   ============================================================ */

let serverGeneration = 0;

let clientEpoch = 0;

let stateChangeInProgress =
  false;

let predictionInFlight =
  false;

let recoveryInFlight =
  false;

let liveTimer =
  null;


/* ============================================================
   KEYSTROKES
   ============================================================ */

let keystrokeEvents = [];

const activeKeys =
  new Set();


/* ============================================================
   AUDIO
   ============================================================ */

let audioSourceReady =
  false;

let audioSourceName =
  null;

let audioSourceKind =
  null;

let microphoneStreaming =
  false;

let microphoneStream =
  null;

let microphoneContext =
  null;

let microphoneSourceNode =
  null;

let microphoneProcessor =
  null;

let microphoneSilentGain =
  null;

let audioSocket =
  null;

let audioStreamToken =
  null;

let microphoneExpectedClose =
  false;

let microphoneReconnectInFlight =
  false;

let microphoneReconnectAttempts =
  0;

const MAX_MICROPHONE_RECONNECT_ATTEMPTS =
  5;

let audioBufferedSec =
  0;

let audioPackets =
  0;

let audioCurrentDbfs =
  null;

/*
 * Fixed audio File is retained while the page remains open.
 * This permits automatic reconstruction after server restart.
 */
let fixedAudioFile =
  null;


/* ============================================================
   VISUAL
   ============================================================ */

let visualMode =
  "none";

let visualSourceReady =
  false;

let visualSourceName =
  null;

let visualSourceFile =
  null;

let webcamStream =
  null;

let visualObjectUrl =
  null;


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
   STATE CHANGE
   ============================================================ */

function beginStateChange(message) {
  clientEpoch += 1;

  stateChangeInProgress =
    true;

  if (message) {
    setText(
      statusBox,
      message
    );
  }

  updateReadiness();

  return clientEpoch;
}


function finishStateChange(
  operationEpoch
) {
  if (
    operationEpoch === clientEpoch
  ) {
    stateChangeInProgress =
      false;
  }

  updateReadiness();
}


function operationStillCurrent(
  operationEpoch
) {
  return (
    operationEpoch === clientEpoch
  );
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
    Boolean(ready)
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


function setModelReadinessBadge() {
  if (!modelReady) {
    return;
  }

  const bold =
    modelReady.querySelector("b");

  modelReady.classList.remove(
    "active"
  );

  modelReady.classList.remove(
    "warning"
  );

  if (modelState === "ready") {
    modelReady.classList.add(
      "active"
    );

    if (bold) {
      bold.textContent = "Ready";
    }

  } else if (
    modelState === "loading"
  ) {
    modelReady.classList.add(
      "warning"
    );

    if (bold) {
      bold.textContent = "Loading";
    }

  } else if (
    modelState === "failed"
  ) {
    if (bold) {
      bold.textContent = "Failed";
    }

  } else {
    if (bold) {
      bold.textContent = "Standby";
    }
  }
}


function currentTextLength() {
  return textInput
    ? textInput.value.trim().length
    : 0;
}


function currentKeydownCount() {
  return keystrokeEvents.filter(
    event =>
      event.type === "down"
  ).length;
}


function audioIsReady() {
  return Boolean(
    audioSourceReady
  );
}


function visualIsReady() {
  if (
    visualMode === "image"
    || visualMode === "video"
  ) {
    return Boolean(
      visualSourceReady
    );
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


function inputModalitiesReady() {
  return (
    !stateChangeInProgress
    && !recoveryInFlight
    && currentTextLength()
      >= MIN_TEXT_CHARS
    && currentKeydownCount()
      >= MIN_KEYPRESSES
    && audioIsReady()
    && visualIsReady()
  );
}


function updateAudioMetrics() {
  setText(
    audioStreamState,
    (
      microphoneStreaming
        ? (
            audioSocket
            && audioSocket.readyState
              === WebSocket.OPEN
              ? (
                  audioSourceReady
                    ? "Live"
                    : "Buffering"
                )
              : "Reconnecting"
          )
        : (
            audioSourceKind === "file"
              ? "Fixed file"
              : "Stopped"
          )
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
    (
      Number.isFinite(
        audioCurrentDbfs
      )
        ? (
            `${audioCurrentDbfs.toFixed(1)} dBFS`
          )
        : "—"
    )
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

  setModelReadinessBadge();

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
    (
      microphoneStreaming
        ? "Streaming"
        : "Ready"
    ),
    (
      microphoneStreaming
        ? (
            microphoneReconnectInFlight
              ? "Reconnecting"
              : "Buffering"
          )
        : "Required"
    )
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
      microphoneStreaming
      || microphoneReconnectInFlight;
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

function deriveModelState(data) {
  const explicit =
    String(
      data?.state || ""
    )
      .trim()
      .toLowerCase();

  if (
    [
      "not_loaded",
      "loading",
      "ready",
      "failed"
    ].includes(explicit)
  ) {
    return explicit;
  }

  if (
    data?.fusion_model
    || data?.predictor_loaded
    || data?.initialised
  ) {
    return "ready";
  }

  if (data?.error) {
    return "failed";
  }

  return "not_loaded";
}


function applyModelStatus(data) {
  modelState =
    deriveModelState(data);

  fusionModelLoaded =
    Boolean(
      data.fusion_model
      || data.predictor_loaded
      || data.initialised
      || modelState === "ready"
    );

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
      MIN_TEXT_CHARS
    );

  MIN_KEYPRESSES =
    positiveInteger(
      data.min_keypresses,
      MIN_KEYPRESSES
    );

  LIVE_INTERVAL_MS =
    positiveInteger(
      data.live_interval_ms,
      LIVE_INTERVAL_MS
    );

  TEMPORAL_WINDOW =
    positiveInteger(
      data.temporal_probability_window,
      TEMPORAL_WINDOW
    );

  TARGET_AUDIO_SAMPLE_RATE =
    positiveInteger(
      data.target_audio_sample_rate,
      TARGET_AUDIO_SAMPLE_RATE
    );

  AUDIO_STREAM_WINDOW_SECONDS =
    finiteNumber(
      data.audio_stream_window_seconds,
      AUDIO_STREAM_WINDOW_SECONDS
    );

  AUDIO_STREAM_MIN_SECONDS =
    finiteNumber(
      data.audio_stream_min_seconds,
      AUDIO_STREAM_MIN_SECONDS
    );

  if (
    Array.isArray(data.labels)
  ) {
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

  } else if (
    modelState === "loading"
  ) {
    setText(
      modelStatusText,
      (
        "Fusion backend is loading. "
        + "First inference may take longer."
      )
    );

  } else if (
    modelState === "failed"
  ) {
    setText(
      modelStatusText,
      (
        "Fusion backend failed: "
        + String(
            data.error
            || "unknown error"
          )
      )
    );

  } else {
    setText(
      modelStatusText,
      (
        "Fusion backend in standby. "
        + "It will initialise when "
        + "inference becomes ready."
      )
    );
  }

  setText(
    webcamModelStatusText,
    (
      webcamModelLoaded
        ? (
            "Webcam-calibrated image "
            + "augmentation loaded."
          )
        : (
            "Webcam calibration not required "
            + "or not yet loaded."
          )
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

    return data;

  } catch (error) {
    console.error(
      "[SenseFuzeAI] "
      + "Model-status query failed:",
      error
    );

    setText(
      modelStatusText,
      (
        "Model-status query failed: "
        + (
            error?.message
            || String(error)
          )
      )
    );

    return null;
  }
}


async function ensureModelsReady() {
  if (
    modelState === "ready"
    && fusionModelLoaded
  ) {
    return true;
  }

  if (modelInitialisationInFlight) {
    return false;
  }

  modelInitialisationInFlight =
    true;

  modelState =
    "loading";

  updateReadiness();

  setText(
    statusBox,
    (
      "Initialising multimodal "
      + "fusion backend..."
    )
  );

  try {
    const data =
      await fetchJson(
        "/initialize-models",
        {
          method: "POST",
          cache: "no-store"
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
      modelState === "ready"
      && fusionModelLoaded
    );

  } catch (error) {
    modelState =
      "failed";

    fusionModelLoaded =
      false;

    if (
      error.data?.model_status
    ) {
      applyModelStatus(
        error.data.model_status
      );
    }

    setText(
      statusBox,
      (
        "Fusion model initialisation failed: "
        + (
            error?.message
            || String(error)
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
   KEYSTROKES
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

  textInput.addEventListener(
    "blur",
    () => {
      activeKeys.clear();
    }
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
    !input
    || input.length === 0
  ) {
    return new Float32Array(0);
  }

  if (inputRate === outputRate) {
    return new Float32Array(
      input
    );
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
      index * 2,
      Math.round(pcm),
      true
    );
  }

  return buffer;
}


function calculateDbfs(samples) {
  if (
    !samples
    || samples.length === 0
  ) {
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

    audioCurrentDbfs =
      dbfs;
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

  updateAudioMetrics();
}


/* ============================================================
   MICROPHONE TRANSPORT
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
          10000
        );

      socket.addEventListener(
        "open",
        () => {
          window.clearTimeout(
            timeout
          );

          resolve();
        },
        {
          once: true
        }
      );

      socket.addEventListener(
        "error",
        () => {
          window.clearTimeout(
            timeout
          );

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


function closeAudioSocketOnly() {
  if (!audioSocket) {
    return;
  }

  const socket =
    audioSocket;

  audioSocket =
    null;

  socket.onmessage =
    null;

  socket.onerror =
    null;

  socket.onclose =
    null;

  try {
    socket.close();
  } catch (_) {}
}


function cleanupMicrophoneCapture() {
  microphoneStreaming =
    false;

  closeAudioSocketOnly();

  if (microphoneProcessor) {
    microphoneProcessor
      .onaudioprocess =
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

  microphoneStream =
    null;

  microphoneProcessor =
    null;

  microphoneSourceNode =
    null;

  microphoneSilentGain =
    null;

  if (microphoneContext) {
    const context =
      microphoneContext;

    microphoneContext =
      null;

    void context
      .close()
      .catch(() => {});
  }

  audioStreamToken =
    null;

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
          JSON.parse(
            event.data
          );

      } catch (_) {
        return;
      }

      if (data.type === "error") {
        console.warn(
          "[SenseFuzeAI] "
          + "Audio socket server error:",
          data
        );

        return;
      }

      if (
        data.type !==
        "audio_status"
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

      if (
        data.audio_diagnostics
      ) {
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
                + ` | ${audioBufferedSec.toFixed(1)}s buffer`
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
      if (
        !microphoneExpectedClose
      ) {
        setText(
          audioStatus,
          "Microphone transport interrupted."
        );
      }
    };


  socket.onclose =
    () => {
      if (
        microphoneExpectedClose
      ) {
        return;
      }

      if (
        socket !== audioSocket
      ) {
        return;
      }

      audioSocket =
        null;

      audioSourceReady =
        false;

      updateReadiness();

      void recoverMicrophoneConnection();
    };
}


async function attachAudioSocket(
  token
) {
  closeAudioSocketOnly();

  const socket =
    new WebSocket(
      websocketUrl(token)
    );

  audioSocket =
    socket;

  await waitForSocketOpen(
    socket
  );

  installAudioSocketHandlers(
    socket
  );
}


/* ============================================================
   MICROPHONE RECOVERY
   ============================================================ */

async function recoverMicrophoneConnection() {
  if (
    microphoneExpectedClose
    || !microphoneStreaming
    || !microphoneStream
    || microphoneReconnectInFlight
  ) {
    return false;
  }

  microphoneReconnectInFlight =
    true;

  audioSourceReady =
    false;

  updateReadiness();

  try {
    for (
      let attempt = 1;
      attempt <=
        MAX_MICROPHONE_RECONNECT_ATTEMPTS;
      attempt += 1
    ) {
      if (
        microphoneExpectedClose
        || !microphoneStreaming
      ) {
        return false;
      }

      microphoneReconnectAttempts =
        attempt;

      setText(
        audioStatus,
        (
          "Microphone connection interrupted. "
          + `Recovering ${attempt}/`
          + `${MAX_MICROPHONE_RECONNECT_ATTEMPTS}...`
        )
      );

      await sleep(
        Math.min(
          attempt * 1000,
          5000
        )
      );

      try {
        const data =
          await postForm(
            "/audio_stream/reconnect",
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

        audioStreamToken =
          String(
            data.stream_token
          );

        await attachAudioSocket(
          audioStreamToken
        );

        audioSourceKind =
          "microphone_stream";

        audioSourceName =
          "Live microphone";

        audioBufferedSec =
          0;

        audioPackets =
          0;

        setText(
          audioStatus,
          (
            "Microphone transport recovered; "
            + "rebuilding audio buffer..."
          )
        );

        setText(
          statusBox,
          (
            "Microphone connection recovered"
            + ` | Generation=${serverGeneration}.`
          )
        );

        microphoneReconnectAttempts =
          0;

        return true;

      } catch (error) {
        console.warn(
          "[SenseFuzeAI] "
          + `Microphone recovery attempt ${attempt} failed:`,
          error
        );
      }
    }

    setText(
      audioStatus,
      (
        "Automatic microphone recovery failed. "
        + "Press Start Microphone to retry."
      )
    );

    return false;

  } finally {
    microphoneReconnectInFlight =
      false;

    updateReadiness();
  }
}


/* ============================================================
   MICROPHONE START
   ============================================================ */

async function startMicrophoneStream() {
  if (
    microphoneStreaming
    && microphoneStream
  ) {
    if (
      !audioSocket
      || audioSocket.readyState
        !== WebSocket.OPEN
    ) {
      await recoverMicrophoneConnection();
    }

    return;
  }

  if (
    !navigator.mediaDevices
    || !navigator.mediaDevices
      .getUserMedia
  ) {
    setText(
      statusBox,
      "Microphone API unavailable."
    );

    return;
  }

  const AudioContextClass =
    window.AudioContext
    || window.webkitAudioContext;

  if (!AudioContextClass) {
    setText(
      statusBox,
      "Web Audio API unavailable."
    );

    return;
  }

  const operationEpoch =
    beginStateChange(
      "Starting microphone..."
    );

  microphoneExpectedClose =
    false;

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
      stream
        .getTracks()
        .forEach(
          track => track.stop()
        );

      await context.close();

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

    await attachAudioSocket(
      audioStreamToken
    );

    microphoneStream =
      stream;

    microphoneContext =
      context;

    microphoneSourceNode =
      context
        .createMediaStreamSource(
          stream
        );

    microphoneProcessor =
      context
        .createScriptProcessor(
          4096,
          1,
          1
        );

    microphoneSilentGain =
      context
        .createGain();

    microphoneSilentGain
      .gain
      .value =
        0;

    microphoneStreaming =
      true;

    audioSourceReady =
      false;

    audioSourceKind =
      "microphone_stream";

    audioSourceName =
      "Live microphone";

    fixedAudioFile =
      null;

    audioBufferedSec =
      0;

    audioPackets =
      0;

    microphoneProcessor
      .onaudioprocess =
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
            event.inputBuffer
              .getChannelData(0);

          audioCurrentDbfs =
            calculateDbfs(
              input
            );

          const resampled =
            resampleLinear(
              input,
              context.sampleRate,
              TARGET_AUDIO_SAMPLE_RATE
            );

          if (
            resampled.length === 0
          ) {
            return;
          }

          try {
            audioSocket.send(
              float32ToPCM16Buffer(
                resampled
              )
            );

          } catch (_) {}
        };

    microphoneSourceNode
      .connect(
        microphoneProcessor
      );

    microphoneProcessor
      .connect(
        microphoneSilentGain
      );

    microphoneSilentGain
      .connect(
        context.destination
      );

    setText(
      audioStatus,
      (
        "Microphone connected; "
        + "building audio buffer..."
      )
    );

    setText(
      statusBox,
      (
        "Continuous microphone started"
        + ` | Generation=${serverGeneration}.`
      )
    );

  } catch (error) {
    console.error(
      "[SenseFuzeAI] "
      + "Microphone start failed:",
      error
    );

    try {
      stream
        ?.getTracks()
        .forEach(
          track => track.stop()
        );
    } catch (_) {}

    try {
      await context?.close();
    } catch (_) {}

    cleanupMicrophoneCapture();

    audioSourceReady =
      false;

    audioSourceKind =
      null;

    audioSourceName =
      null;

    setText(
      statusBox,
      (
        "Microphone start failed: "
        + (
            error?.message
            || String(error)
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
   EXPLICIT MICROPHONE STOP
   ============================================================ */

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
      "Stopping microphone..."
    );

  microphoneExpectedClose =
    true;

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

  } catch (error) {
    console.warn(
      "[SenseFuzeAI] "
      + "Server microphone stop failed:",
      error
    );

  } finally {
    cleanupMicrophoneCapture();

    audioSourceReady =
      false;

    audioSourceKind =
      null;

    audioSourceName =
      null;

    audioBufferedSec =
      0;

    audioPackets =
      0;

    audioCurrentDbfs =
      null;

    if (resetDisplay) {
      resetPredictionDisplay();
    }

    setText(
      audioStatus,
      "Microphone stopped."
    );

    microphoneExpectedClose =
      false;

    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   FIXED AUDIO
   ============================================================ */

async function uploadFixedAudio(
  file,
  {
    recovery = false
  } = {}
) {
  const form =
    new FormData();

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

  const data =
    await fetchJson(
      "/set_audio_source",
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

  audioSourceReady =
    true;

  audioSourceName =
    data.audio_name
    || file.name;

  audioSourceKind =
    "file";

  fixedAudioFile =
    file;

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
    (
      recovery
        ? (
            "Fixed audio source "
            + "restored after server-state loss."
          )
        : (
            "Fixed audio file: "
            + audioSourceName
          )
    )
  );

  return data;
}


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
      "Loading audio file..."
    );

  try {
    await uploadFixedAudio(
      file
    );

    resetPredictionDisplay();

  } catch (error) {
    audioSourceReady =
      false;

    setText(
      statusBox,
      (
        "Audio file failed: "
        + (
            error?.message
            || String(error)
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
   VISUAL UTILITIES
   ============================================================ */

function revokeVisualObjectUrl() {
  if (visualObjectUrl) {
    URL.revokeObjectURL(
      visualObjectUrl
    );

    visualObjectUrl =
      null;
  }
}


function hideStaticImagePreview() {
  if (!staticImagePreview) {
    return;
  }

  staticImagePreview
    .classList
    .add("hidden");

  staticImagePreview
    .removeAttribute("src");
}


function stopWebcamStreamLocally() {
  webcamStream
    ?.getTracks()
    .forEach(
      track => track.stop()
    );

  webcamStream =
    null;

  if (webcam) {
    webcam.srcObject =
      null;
  }
}


function stopVideoPreview() {
  if (!webcam) {
    return;
  }

  try {
    webcam.pause();
  } catch (_) {}

  webcam.removeAttribute(
    "src"
  );

  try {
    webcam.load();
  } catch (_) {}
}


/* ============================================================
   VISUAL REGISTRATION HELPERS
   ============================================================ */

async function registerImageWithServer(
  file
) {
  const form =
    new FormData();

  form.append(
    "session_id",
    sessionId
  );

  form.append(
    "image_file",
    file,
    file.name
  );

  const data =
    await fetchJson(
      "/set_visual_image",
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

  return data;
}


async function registerVideoWithServer(
  file
) {
  const form =
    new FormData();

  form.append(
    "session_id",
    sessionId
  );

  form.append(
    "video_file",
    file,
    file.name
  );

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

  return data;
}


async function registerWebcamWithServer() {
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

  return data;
}


/* ============================================================
   IMAGE
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

  visualMode =
    "none";

  visualSourceReady =
    false;

  try {
    const data =
      await registerImageWithServer(
        file
      );

    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {
      return;
    }

    visualMode =
      "image";

    visualSourceReady =
      true;

    visualSourceName =
      data.visual_name
      || file.name;

    visualSourceFile =
      file;

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
      webcam
        .classList
        .add("hidden");
    }

    resetPredictionDisplay();

    setText(
      webcamStatus,
      (
        "Image source: "
        + visualSourceName
      )
    );

  } catch (error) {
    visualMode =
      "none";

    visualSourceReady =
      false;

    visualSourceFile =
      null;

    setText(
      statusBox,
      (
        "Image source failed: "
        + (
            error?.message
            || String(error)
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
   VIDEO
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

  try {
    const data =
      await registerVideoWithServer(
        file
      );

    visualMode =
      "video";

    visualSourceReady =
      true;

    visualSourceName =
      data.visual_name
      || file.name;

    visualSourceFile =
      file;

    visualObjectUrl =
      URL.createObjectURL(file);

    if (webcam) {
      webcam
        .classList
        .remove("hidden");

      webcam.srcObject =
        null;

      webcam.src =
        visualObjectUrl;

      webcam.loop =
        true;

      webcam.muted =
        true;

      try {
        await webcam.play();
      } catch (_) {}
    }

    resetPredictionDisplay();

    setText(
      webcamStatus,
      (
        "Video source: "
        + visualSourceName
      )
    );

  } catch (error) {
    visualMode =
      "none";

    visualSourceReady =
      false;

    visualSourceFile =
      null;

    setText(
      statusBox,
      (
        "Video source failed: "
        + (
            error?.message
            || String(error)
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
   WEBCAM
   ============================================================ */

async function startWebcamMode() {
  if (
    !navigator.mediaDevices
    || !navigator.mediaDevices
      .getUserMedia
  ) {
    setText(
      statusBox,
      "Webcam API unavailable."
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

    if (!webcam) {
      throw new Error(
        "Webcam video element missing."
      );
    }

    webcamStream =
      stream;

    webcam
      .classList
      .remove("hidden");

    webcam.removeAttribute(
      "src"
    );

    webcam.srcObject =
      stream;

    webcam.muted =
      true;

    await webcam.play();

    await registerWebcamWithServer();

    visualMode =
      "webcam";

    visualSourceReady =
      true;

    visualSourceName =
      "Webcam";

    visualSourceFile =
      null;

    resetPredictionDisplay();

    setText(
      webcamStatus,
      "Webcam active."
    );

    if (startBtn) {
      startBtn.disabled =
        true;
    }

    if (stopBtn) {
      stopBtn.disabled =
        false;
    }

  } catch (error) {
    stream
      ?.getTracks()
      .forEach(
        track => track.stop()
      );

    stopWebcamStreamLocally();

    visualMode =
      "none";

    visualSourceReady =
      false;

    setText(
      statusBox,
      (
        "Webcam start failed: "
        + (
            error?.message
            || String(error)
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


async function stopVisualMode() {
  const operationEpoch =
    beginStateChange(
      "Stopping visual input..."
    );

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

    stopWebcamStreamLocally();
    stopVideoPreview();
    hideStaticImagePreview();
    revokeVisualObjectUrl();

    visualMode =
      "none";

    visualSourceReady =
      false;

    visualSourceName =
      null;

    visualSourceFile =
      null;

    setText(
      webcamStatus,
      "Visual input inactive."
    );

    resetPredictionDisplay();

  } catch (error) {
    setText(
      statusBox,
      (
        "Stop visual failed: "
        + (
            error?.message
            || String(error)
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


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
    canvas.getContext(
      "2d"
    );

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
   SERVER SESSION RECOVERY
   ============================================================ */

async function recoverVisualRegistration() {
  if (
    visualMode === "none"
  ) {
    return false;
  }

  try {
    if (
      visualMode === "webcam"
      && webcamStream
    ) {
      await registerWebcamWithServer();

      setText(
        statusBox,
        (
          "Server visual state recovered: "
          + "webcam re-registered."
        )
      );

      return true;
    }

    if (
      visualMode === "image"
      && visualSourceFile
    ) {
      await registerImageWithServer(
        visualSourceFile
      );

      setText(
        statusBox,
        (
          "Server visual state recovered: "
          + "image re-uploaded."
        )
      );

      return true;
    }

    if (
      visualMode === "video"
      && visualSourceFile
    ) {
      await registerVideoWithServer(
        visualSourceFile
      );

      setText(
        statusBox,
        (
          "Server visual state recovered: "
          + "video re-uploaded."
        )
      );

      return true;
    }

  } catch (error) {
    console.error(
      "[SenseFuzeAI] "
      + "Visual recovery failed:",
      error
    );
  }

  return false;
}


async function recoverAudioRegistration() {
  try {
    if (
      microphoneStreaming
      && microphoneStream
    ) {
      return await recoverMicrophoneConnection();
    }

    if (
      audioSourceKind === "file"
      && fixedAudioFile
    ) {
      await uploadFixedAudio(
        fixedAudioFile,
        {
          recovery: true
        }
      );

      return true;
    }

  } catch (error) {
    console.error(
      "[SenseFuzeAI] "
      + "Audio recovery failed:",
      error
    );
  }

  return false;
}


async function synchroniseSessionFromServer() {
  try {
    const data =
      await fetchJson(
        (
          "/session-status/"
          + encodeURIComponent(
              sessionId
            )
        ),
        {
          cache: "no-store"
        }
      );

    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );

    return data;

  } catch (error) {
    console.warn(
      "[SenseFuzeAI] "
      + "Session status unavailable:",
      error
    );

    return null;
  }
}


async function recoverServerSession(
  detail = {}
) {
  if (recoveryInFlight) {
    return true;
  }

  recoveryInFlight =
    true;

  predictionInFlight =
    false;

  updateReadiness();

  try {
    if (
      detail.generation
      !== undefined
    ) {
      serverGeneration =
        finiteNumber(
          detail.generation,
          serverGeneration
        );
    }

    const server =
      await synchroniseSessionFromServer();

    let visualRecovered =
      true;

    let audioRecovered =
      true;

    /*
     * If server state differs from browser state,
     * reconstruct the browser-selected source.
     */
    if (
      visualMode !== "none"
      && server?.visual_mode
        !== visualMode
    ) {
      visualRecovered =
        await recoverVisualRegistration();
    }

    if (
      audioSourceKind
      === "microphone_stream"
      && (
        server?.audio_source_kind
        !== "microphone_stream"
        || !server
          ?.audio_stream_connected
      )
    ) {
      audioRecovered =
        await recoverAudioRegistration();

    } else if (
      audioSourceKind === "file"
      && server?.audio_source_kind
        !== "file"
    ) {
      audioRecovered =
        await recoverAudioRegistration();
    }

    await synchroniseSessionFromServer();

    return (
      visualRecovered
      && audioRecovered
    );

  } finally {
    recoveryInFlight =
      false;

    updateReadiness();
  }
}


/* ============================================================
   CONFLICT HANDLING
   ============================================================ */

async function handleConflictResponse(
  data
) {
  const detail =
    data
    && typeof data.detail
      === "object"
    && data.detail !== null
      ? data.detail
      : null;

  if (!detail) {
    return false;
  }

  if (
    detail.generation
    !== undefined
  ) {
    serverGeneration =
      finiteNumber(
        detail.generation,
        serverGeneration
      );
  }

  const type =
    String(
      detail.type || ""
    );

  if (
    type === "visual_mode_mismatch"
    || type === "audio_state_mismatch"
    || type === "stale_session"
  ) {
    setText(
      statusBox,
      (
        "Server session state changed. "
        + "Recovering active inputs..."
      )
    );

    await recoverServerSession(
      detail
    );

    return true;
  }

  if (
    type === "stale_generation"
    || type === "stale_result"
  ) {
    /*
     * Generation mismatch alone does not require destroying
     * any active browser source. Adopt the server generation.
     */
    setText(
      statusBox,
      (
        "Prediction generation resynchronised "
        + `to ${serverGeneration}.`
      )
    );

    return true;
  }

  return false;
}


/* ============================================================
   PROBABILITY UI
   ============================================================ */

function resolveRenderLabels(
  probabilities
) {
  if (
    behaviouralLabels.length
  ) {
    return behaviouralLabels;
  }

  if (
    probabilities
    && typeof probabilities
      === "object"
  ) {
    return Object.keys(
      probabilities
    );
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

  container.innerHTML =
    "";

  if (
    !probabilities
    || typeof probabilities
      !== "object"
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

      labelRow.append(
        name,
        value
      );

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
        `sf-prob-fill ${kind}`;

      fill.style.width =
        `${percent}%`;

      track.appendChild(
        fill
      );

      row.append(
        labelRow,
        track
      );

      container.appendChild(
        row
      );
    }
  );

  if (
    probabilitySum !== null
    && Number.isFinite(
      Number(probabilitySum)
    )
  ) {
    const row =
      document.createElement(
        "div"
      );

    row.className =
      "sf-prob-sum";

    row.textContent =
      (
        "Probability sum: "
        + Number(
            probabilitySum
          ).toFixed(6)
      );

    container.appendChild(
      row
    );
  }
}


/* ============================================================
   RESULT UI
   ============================================================ */

function updatePredictionUI(data) {
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
    data.confidence_level || "—"
  );

  setText(
    rawPrediction,
    data.raw_top_class
    || data.raw_prediction
    || "—"
  );

  const rawPct =
    Number(
      data.raw_confidence_percent
    );

  setText(
    rawConfidence,
    (
      Number.isFinite(rawPct)
        ? `${rawPct.toFixed(2)}%`
        : "—"
    )
  );

  const samples =
    finiteNumber(
      data.temporal_samples,
      0
    );

  const windowSize =
    positiveInteger(
      data.temporal_window,
      TEMPORAL_WINDOW
    );

  setText(
    temporalSamples,
    samples
  );

  setText(
    temporalWindow,
    windowSize
  );

  setText(
    temporalWindowStatus,
    `${samples} / ${windowSize}`
  );

  setText(
    secondaryState,
    data.second_class || "—"
  );

  setText(
    confidenceGap,
    finiteNumber(
      data.confidence_gap,
      0
    ).toFixed(4)
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
    validation
      .temporal_probability_sum
  );

  renderProbabilityBars(
    rawProbabilitiesBox,
    data.raw_probabilities,
    "raw",
    validation
      .raw_probability_sum
  );

  const modalities =
    data.used_modalities
    || {};

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
    (
      active.length
        ? active.join(", ")
        : "—"
    )
  );

  setText(
    technicalRawState,
    data.raw_top_class
    || data.raw_prediction
    || "—"
  );

  setText(
    technicalTemporalSamples,
    `${samples}/${windowSize}`
  );

  const webcamResult =
    data.webcam_prediction;

  if (webcamResult) {
    setText(
      webcamPrediction,
      webcamResult.current_state
      || "—"
    );

    const pct =
      Number(
        webcamResult.confidence_percent
      );

    setText(
      webcamConfidence,
      (
        Number.isFinite(pct)
          ? `${pct.toFixed(2)}%`
          : "—"
      )
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

    if (
      webcamProbabilityBars
    ) {
      webcamProbabilityBars.innerHTML =
        "";
    }
  }

  if (
    data.audio_diagnostics
  ) {
    updateAudioDiagnostic(
      data.audio_diagnostics
    );
  }

  const rawSum =
    Number(
      validation.raw_probability_sum
    );

  const temporalSum =
    Number(
      validation.temporal_probability_sum
    );

  setText(
    validationStatus,
    (
      "Runtime validation: "
      + (
          validation.pass
            ? "PASS"
            : "CHECK"
        )
      + (
          Number.isFinite(rawSum)
            ? (
                " | Raw sum: "
                + rawSum.toFixed(6)
              )
            : ""
        )
      + (
          Number.isFinite(temporalSum)
            ? (
                " | Temporal sum: "
                + temporalSum.toFixed(6)
              )
            : ""
        )
    )
  );

  const inferenceSeconds =
    Number(
      data.inference_seconds
    );

  setText(
    statusBox,
    (
      "Live prediction successful"
      + ` | ${state}`
      + ` | ${confidencePct.toFixed(2)}%`
      + (
          Number.isFinite(
            inferenceSeconds
          )
            ? (
                ` | inference ${inferenceSeconds.toFixed(1)}s`
              )
            : ""
        )
    )
  );
}


function resetPredictionDisplay() {
  setText(
    predictionBox,
    "—"
  );

  setText(
    confidencePercent,
    "—"
  );

  setText(
    confidenceLevel,
    "—"
  );

  if (confidenceFill) {
    confidenceFill.style.width =
      "0%";
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
    validationStatus,
    ""
  );

  if (probabilitiesBox) {
    probabilitiesBox.innerHTML =
      "";
  }

  if (rawProbabilitiesBox) {
    rawProbabilitiesBox.innerHTML =
      "";
  }

  if (
    webcamProbabilityBars
  ) {
    webcamProbabilityBars.innerHTML =
      "";
  }
}


/* ============================================================
   LIVE PREDICTION
   ============================================================ */

async function runLivePrediction() {
  if (
    predictionInFlight
    || stateChangeInProgress
    || modelInitialisationInFlight
    || recoveryInFlight
  ) {
    return;
  }

  updateReadiness();

  if (
    !inputModalitiesReady()
  ) {
    return;
  }

  if (
    modelState !== "ready"
    || !fusionModelLoaded
  ) {
    const ready =
      await ensureModelsReady();

    if (!ready) {
      return;
    }
  }

  let webcamFrame =
    null;

  if (
    visualMode === "webcam"
  ) {
    webcamFrame =
      captureWebcamFrame();

    if (!webcamFrame) {
      return;
    }
  }

  predictionInFlight =
    true;

  const requestEpoch =
    clientEpoch;

  const requestGeneration =
    serverGeneration;

  try {
    const form =
      new FormData();

    form.append(
      "session_id",
      sessionId
    );

    form.append(
      "generation",
      String(
        requestGeneration
      )
    );

    form.append(
      "text",
      (
        textInput
          ? textInput.value.trim()
          : ""
      )
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
        "Running multimodal fusion inference..."
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
      console.error(
        "[SenseFuzeAI] "
        + "Prediction HTTP failure:",
        response.status,
        data
      );

      if (
        response.status === 409
        && await handleConflictResponse(
          data
        )
      ) {
        return;
      }

      if (
        response.status === 503
      ) {
        modelState =
          "failed";

        fusionModelLoaded =
          false;

        void checkModelStatus();
      }

      const error =
        new Error(
          formatServerError(
            data,
            `HTTP ${response.status}`
          )
        );

      error.status =
        response.status;

      error.data =
        data;

      throw error;
    }

    if (
      requestEpoch !== clientEpoch
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
    ) {
      throw new Error(
        (
          "Prediction response did not "
          + "contain a valid generation."
        )
      );
    }

    if (
      returnedGeneration
      !== requestGeneration
    ) {
      serverGeneration =
        returnedGeneration;

      setText(
        statusBox,
        (
          "Prediction completed after "
          + "generation change; result discarded."
        )
      );

      return;
    }

    serverGeneration =
      returnedGeneration;

    modelState =
      "ready";

    fusionModelLoaded =
      true;

    updatePredictionUI(
      data
    );

  } catch (error) {
    console.error(
      "[SenseFuzeAI] "
      + "Live prediction failed:",
      error
    );

    /*
     * A network failure may indicate worker restart.
     * Attempt recovery without deleting browser sources.
     */
    if (
      !error.status
      && requestEpoch === clientEpoch
    ) {
      await recoverServerSession();
    }

    if (
      requestEpoch === clientEpoch
    ) {
      setText(
        statusBox,
        (
          "Live prediction failed: "
          + (
              error?.message
              || String(error)
            )
        )
      );
    }

  } finally {
    predictionInFlight =
      false;
  }
}


/* ============================================================
   RESETS
   ============================================================ */

async function resetTemporalWindow() {
  const operationEpoch =
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

    TEMPORAL_WINDOW =
      positiveInteger(
        data.temporal_window,
        TEMPORAL_WINDOW
      );

    resetPredictionDisplay();

    setText(
      statusBox,
      (
        "Temporal history reset"
        + ` | Generation=${serverGeneration}.`
      )
    );

  } catch (error) {
    setText(
      statusBox,
      (
        "Temporal reset failed: "
        + (
            error?.message
            || String(error)
          )
      )
    );

  } finally {
    finishStateChange(
      operationEpoch
    );
  }
}


async function resetSession() {
  const operationEpoch =
    beginStateChange(
      "Resetting session..."
    );

  microphoneExpectedClose =
    true;

  try {
    const data =
      await postForm(
        "/full_reset",
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

  } catch (error) {
    console.warn(
      "[SenseFuzeAI] "
      + "Server full reset failed:",
      error
    );

  } finally {
    cleanupMicrophoneCapture();

    stopWebcamStreamLocally();
    stopVideoPreview();
    hideStaticImagePreview();
    revokeVisualObjectUrl();

    if (textInput) {
      textInput.value =
        "";
    }

    keystrokeEvents =
      [];

    activeKeys.clear();

    audioSourceReady =
      false;

    audioSourceName =
      null;

    audioSourceKind =
      null;

    fixedAudioFile =
      null;

    audioBufferedSec =
      0;

    audioPackets =
      0;

    audioCurrentDbfs =
      null;

    visualMode =
      "none";

    visualSourceReady =
      false;

    visualSourceName =
      null;

    visualSourceFile =
      null;

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
      startBtn.disabled =
        false;
    }

    if (stopBtn) {
      stopBtn.disabled =
        true;
    }

    microphoneExpectedClose =
      false;

    finishStateChange(
      operationEpoch
    );
  }
}


/* ============================================================
   BUTTON BINDINGS
   ============================================================ */

startMicBtn?.addEventListener(
  "click",
  () => {
    void startMicrophoneStream();
  }
);


stopMicBtn?.addEventListener(
  "click",
  () => {
    void stopMicrophoneStream();
  }
);


chooseAudioBtn?.addEventListener(
  "click",
  () => {
    audioFileInput?.click();
  }
);


audioFileInput?.addEventListener(
  "change",
  () => {
    const file =
      audioFileInput.files?.[0];

    if (file) {
      void setAudioFile(file);
    }

    audioFileInput.value =
      "";
  }
);


chooseImageBtn?.addEventListener(
  "click",
  () => {
    imageFileInput?.click();
  }
);


chooseVideoBtn?.addEventListener(
  "click",
  () => {
    videoFileInput?.click();
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

    imageFileInput.value =
      "";
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

    videoFileInput.value =
      "";
  }
);


startBtn?.addEventListener(
  "click",
  () => {
    void startWebcamMode();
  }
);


stopBtn?.addEventListener(
  "click",
  () => {
    void stopVisualMode();
  }
);


resetTemporalBtn?.addEventListener(
  "click",
  () => {
    void resetTemporalWindow();
  }
);


resetBtn?.addEventListener(
  "click",
  () => {
    void resetSession();
  }
);


/* ============================================================
   PAGE SHUTDOWN
   ============================================================ */

window.addEventListener(
  "beforeunload",
  () => {
    microphoneExpectedClose =
      true;

    if (liveTimer !== null) {
      window.clearInterval(
        liveTimer
      );
    }

    cleanupMicrophoneCapture();

    stopWebcamStreamLocally();

    revokeVisualObjectUrl();
  }
);


/* ============================================================
   INITIALISATION
   ============================================================ */

async function initialiseBrowserClient() {
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
      + "audio and a visual source. "
      + "Live inference will start automatically."
    )
  );

  if (liveTimer !== null) {
    window.clearInterval(
      liveTimer
    );
  }

  liveTimer =
    window.setInterval(
      () => {
        void runLivePrediction();
      },
      LIVE_INTERVAL_MS
    );
}


void initialiseBrowserClient();
