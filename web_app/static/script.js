"use strict";


// ============================================================
// SenseFuzeAI canonical web live-fusion client
//
// File:
//     web_app/static/script.js
//
// Architectural responsibility
// --------------------------------
//
// This file is responsible for:
//
//     - browser UI state
//     - text acquisition
//     - keystroke acquisition
//     - audio acquisition
//     - image/video/webcam acquisition
//     - HTTP communication with app.py
//     - stale-client-response protection
//     - rendering server results
//
// This file DOES NOT implement:
//
//     - probability normalisation
//     - temporal probability history
//     - rolling probability averaging
//     - temporal confidence calculation
//     - confidence-gap thresholds
//     - server reset generation
//
// Those operations are canonical in:
//
//     temporal_fusion.py
//
// Processing architecture:
//
//     Browser inputs
//          |
//          v
//     script.js
//          |
//          v
//     web_app/app.py
//          |
//          v
//     FinalMultimodalInference.predict(...)
//          |
//          | raw probability vector
//          v
//     TemporalFusionEngine
//          |
//          v
//     app.py JSON result
//          |
//          v
//     script.js DISPLAY ONLY
//
// Behaviour aligned with live_fusion_gui.py:
//
//     - minimum 20 text characters
//     - minimum 20 key-down events
//     - 2500 ms scheduler
//     - all four modalities required
//     - fixed audio source until replaced/reset
//     - image / video / webcam modes
//     - source changes reset temporal history server-side
//     - temporal window = 5
//     - generation-safe stale-result rejection
//
// ============================================================


// ============================================================
// DOM helpers
// ============================================================

function getElement(id) {

  return document.getElementById(
    id
  );
}


function setText(
  element,
  value
) {

  if (element) {

    element.textContent =
      String(value);
  }
}


function sleep(ms) {

  return new Promise(
    resolve => {

      window.setTimeout(
        resolve,
        ms
      );
    }
  );
}


function finiteNumber(
  value,
  fallback = 0
) {

  const numeric =
    Number(value);


  return (
    Number.isFinite(numeric)
      ? numeric
      : fallback
  );
}


function positiveInteger(
  value,
  fallback
) {

  const numeric =
    Number(value);


  if (
    Number.isInteger(numeric)
    &&
    numeric > 0
  ) {

    return numeric;
  }


  return fallback;
}


// ============================================================
// Existing DOM references
// ============================================================

const textInput =
  getElement(
    "textInput"
  );


const webcam =
  getElement(
    "webcam"
  );


const canvas =
  getElement(
    "frameCanvas"
  );


const startBtn =
  getElement(
    "startBtn"
  );


const stopBtn =
  getElement(
    "stopBtn"
  );


const resetBtn =
  getElement(
    "resetBtn"
  );


const resetTemporalBtn =
  getElement(
    "resetTemporalBtn"
  );


const statusBox =
  getElement(
    "status"
  );


const sessionStatus =
  getElement(
    "sessionStatus"
  );


const audioStatus =
  getElement(
    "audioStatus"
  );


const webcamStatus =
  getElement(
    "webcamStatus"
  );


const modelStatusText =
  getElement(
    "modelStatusText"
  );


const webcamModelStatusText =
  getElement(
    "webcamModelStatusText"
  );


const predictionBox =
  getElement(
    "prediction"
  );


const confidencePercent =
  getElement(
    "confidencePercent"
  );


const confidenceFill =
  getElement(
    "confidenceFill"
  );


const confidenceLevel =
  getElement(
    "confidenceLevel"
  );


const rawPrediction =
  getElement(
    "rawPrediction"
  );


const rawConfidence =
  getElement(
    "rawConfidence"
  );


const temporalSamples =
  getElement(
    "temporalSamples"
  );


const temporalWindow =
  getElement(
    "temporalWindow"
  );


const temporalWindowStatus =
  getElement(
    "temporalWindowStatus"
  );


const secondaryState =
  getElement(
    "secondaryState"
  );


const confidenceGap =
  getElement(
    "confidenceGap"
  );


const featureDimension =
  getElement(
    "featureDimension"
  );


const deviceInfo =
  getElement(
    "deviceInfo"
  );


const probabilitiesBox =
  getElement(
    "probabilities"
  );


const rawProbabilitiesBox =
  getElement(
    "rawProbabilities"
  );


const webcamPrediction =
  getElement(
    "webcamPrediction"
  );


const webcamConfidence =
  getElement(
    "webcamConfidence"
  );


const webcamProbabilityBars =
  getElement(
    "webcamProbabilityBars"
  );


const webcamCalibrationUsed =
  getElement(
    "webcamCalibrationUsed"
  );


const activeModalities =
  getElement(
    "activeModalities"
  );


const technicalRawState =
  getElement(
    "technicalRawState"
  );


const technicalTemporalSamples =
  getElement(
    "technicalTemporalSamples"
  );


const sessionIdDisplay =
  getElement(
    "sessionIdDisplay"
  );


const modelReady =
  getElement(
    "modelReady"
  );


const webcamModelReady =
  getElement(
    "webcamModelReady"
  );


const textReady =
  getElement(
    "textReady"
  );


const keyReady =
  getElement(
    "keyReady"
  );


const audioReady =
  getElement(
    "audioReady"
  );


const imageReady =
  getElement(
    "imageReady"
  );


const textCard =
  getElement(
    "textCard"
  );


const webcamCard =
  getElement(
    "webcamCard"
  );


const audioCard =
  getElement(
    "audioCard"
  );


const charCount =
  getElement(
    "charCount"
  );


const keyCount =
  getElement(
    "keyCount"
  );


const validationStatus =
  getElement(
    "validationStatus"
  );


const audioDiagnostic =
  getElement(
    "audioDiagnostic"
  );


// ============================================================
// Configuration
//
// These values are bootstrap defaults only.
//
// /model-status supplied by app.py is authoritative.
// ============================================================

let MIN_TEXT_CHARS = 20;

let MIN_KEYPRESSES = 20;

let LIVE_INTERVAL_MS = 2500;

let TEMPORAL_WINDOW = 5;

let AUDIO_CAPTURE_SECONDS = 10;

let TARGET_AUDIO_SAMPLE_RATE = 16000;


// ============================================================
// Canonical labels
//
// Deliberately NOT hard-coded.
//
// app.py should return:
//
//     "labels": list(LABELS)
//
// from /model-status.
//
// If an older backend does not return labels, rendering falls
// back to the keys contained in the probability response.
// ============================================================

let behaviouralLabels = [];


// ============================================================
// Model state
// ============================================================

let fusionModelLoaded =
  false;


let webcamModelLoaded =
  false;


let temporalFusionBackend =
  null;


// ============================================================
// Generation / concurrency state
// ============================================================

// Server-side generation returned by TemporalFusionEngine.
let serverGeneration = 0;


// Independent browser epoch.
//
// Whenever the browser begins a reset or modality-source
// replacement, this value changes immediately.
//
// A response produced for an older browser epoch is therefore
// prevented from repainting the interface.
let clientEpoch = 0;


// True while a modality/reset operation is modifying the
// current experiment condition.
//
// Automatic prediction pauses during such changes.
let stateChangeInProgress =
  false;


// Prevent overlapping fusion requests.
let predictionInFlight =
  false;


// Automatic scheduler handle.
let liveTimer =
  null;


// ============================================================
// Keystroke state
// ============================================================

let keystrokeEvents = [];

let activeKeys =
  new Set();


// ============================================================
// Audio state
//
// Audio remains FIXED after selection/recording until explicitly
// replaced or Full Reset is performed.
// ============================================================

let audioSourceReady =
  false;


let audioSourceName =
  null;


let audioSourceKind =
  null;


let microphoneRecording =
  false;


// ============================================================
// Visual state
// ============================================================

let visualMode =
  "none";


let visualSourceReady =
  false;


let visualSourceName =
  null;


let webcamStream =
  null;


let visualObjectUrl =
  null;


let staticImagePreview =
  null;


// ============================================================
// Dynamic file inputs
// ============================================================

let audioFileInput =
  null;


let imageFileInput =
  null;


let videoFileInput =
  null;


// ============================================================
// Session ID
// ============================================================

function createSessionId() {

  if (
    window.crypto
    &&
    typeof window.crypto.randomUUID
      === "function"
  ) {

    return (
      window.crypto.randomUUID()
    );
  }


  return (
    "session-"
    + Date.now()
      .toString(36)
    + "-"
    + Math.random()
      .toString(36)
      .slice(2)
  );
}


const sessionId =
  createSessionId();


setText(
  sessionIdDisplay,
  sessionId
);


// ============================================================
// Client operation / epoch helpers
// ============================================================

function beginStateChange(
  message
) {

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
    operationEpoch
    === clientEpoch
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
    operationEpoch
    === clientEpoch
  );
}


// ============================================================
// Error helpers
// ============================================================

function formatServerError(
  data
) {

  if (!data) {

    return (
      "Unknown server error."
    );
  }


  const detail =
    (
      data.detail
      ??
      data.error
      ??
      data.message
    );


  if (
    typeof detail
    === "string"
  ) {

    return detail;
  }


  try {

    return JSON.stringify(
      detail
    );

  } catch (_) {

    return String(
      detail
    );
  }
}


function handleConflictResponse(
  data
) {

  const detail =
    data
    &&
    typeof data.detail
      === "object"
    &&
    data.detail !== null

      ? data.detail

      : null;


  if (!detail) {

    return false;
  }


  if (
    detail.generation
    !== undefined
  ) {

    const generation =
      Number(
        detail.generation
      );


    if (
      Number.isFinite(
        generation
      )
    ) {

      serverGeneration =
        generation;
    }
  }


  const type =
    String(
      detail.type
      || ""
    );


  if (
    type
    === "stale_generation"
    ||
    type
    === "stale_result"
    ||
    type
    === "stale_session"
  ) {

    setText(
      statusBox,
      (
        "Stale prediction rejected "
        + "after reset/source change."
      )
    );


    return true;
  }


  if (
    type
    === "visual_mode_mismatch"
  ) {

    const serverVisualMode =
      String(
        detail.visual_mode
        || "none"
      );


    setText(
      statusBox,
      (
        "Visual-source state changed "
        + `on server (${serverVisualMode}).`
      )
    );


    return true;
  }


  return false;
}


// ============================================================
// Styles for dynamically generated controls/results
// ============================================================

function installStyles() {

  if (
    getElement(
      "sensefuzeDynamicStyles"
    )
  ) {

    return;
  }


  const style =
    document.createElement(
      "style"
    );


  style.id =
    "sensefuzeDynamicStyles";


  style.textContent = `
    #probabilities,
    #rawProbabilities,
    #webcamProbabilityBars {
      display: block !important;
      width: 100%;
      min-height: 130px;
      max-height: none !important;
      overflow: visible !important;
    }

    .sf-prob-row {
      margin: 5px 0;
      width: 100%;
    }

    .sf-prob-label {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      font-size: .9rem;
      margin-bottom: 3px;
    }

    .sf-prob-track {
      width: 100%;
      height: 9px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(148, 163, 184, .20);
    }

    .sf-prob-fill {
      height: 100%;
      border-radius: 999px;
      background: #49e8cf;
      transition: width .2s ease;
    }

    .sf-prob-fill.raw {
      background: #f5b942;
    }

    .sf-prob-fill.temporal {
      background: #49e8cf;
    }

    .sf-prob-fill.webcam {
      background: #8b7bff;
    }

    .sf-prob-sum {
      display: flex;
      justify-content: space-between;
      margin-top: 8px;
      padding-top: 5px;
      border-top: 1px solid rgba(148, 163, 184, .25);
      color: #b9c8da;
      font-size: .82rem;
    }

    .sf-source-controls {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 8px;
      margin: 10px 0;
    }

    .sf-source-controls button {
      cursor: pointer;
      padding: 7px 11px;
    }

    #sfStaticImagePreview {
      display: none;
      width: 100%;
      max-height: 320px;
      margin-top: 8px;
      object-fit: contain;
    }
  `;


  document.head.appendChild(
    style
  );
}


// ============================================================
// Dynamic control creation
// ============================================================

function makeButton(
  text,
  handler
) {

  const button =
    document.createElement(
      "button"
    );


  button.type =
    "button";


  button.textContent =
    text;


  button.addEventListener(
    "click",
    handler
  );


  return button;
}


function installAudioControls() {

  const host =
    (
      audioCard
      ||
      (
        audioStatus
          ? audioStatus.parentElement
          : null
      )
    );


  if (!host) {

    return;
  }


  if (
    getElement(
      "sfAudioSourceControls"
    )
  ) {

    return;
  }


  const row =
    document.createElement(
      "div"
    );


  row.id =
    "sfAudioSourceControls";


  row.className =
    "sf-source-controls";


  audioFileInput =
    document.createElement(
      "input"
    );


  audioFileInput.type =
    "file";


  audioFileInput.accept =
    (
      "audio/*,"
      + ".wav,.mp3,.flac,.ogg,"
      + ".m4a,.webm"
    );


  audioFileInput.style.display =
    "none";


  audioFileInput.addEventListener(
    "change",

    async () => {

      const file =
        (
          audioFileInput.files
          &&
          audioFileInput.files.length > 0

            ? audioFileInput.files[0]

            : null
        );


      if (file) {

        await setAudioFile(
          file,
          "file"
        );
      }


      audioFileInput.value =
        "";
    }
  );


  const chooseButton =
    makeButton(
      "Choose Audio File",

      () => {

        audioFileInput.click();
      }
    );


  const recordButton =
    makeButton(
      (
        "Record Microphone "
        + `(${AUDIO_CAPTURE_SECONDS}s)`
      ),

      recordMicrophoneOnce
    );


  row.appendChild(
    chooseButton
  );


  row.appendChild(
    recordButton
  );


  row.appendChild(
    audioFileInput
  );


  host.appendChild(
    row
  );
}


function installVisualControls() {

  const host =
    (
      webcamCard
      ||
      (
        webcamStatus
          ? webcamStatus.parentElement
          : null
      )
    );


  if (!host) {

    return;
  }


  if (
    getElement(
      "sfVisualSourceControls"
    )
  ) {

    return;
  }


  const row =
    document.createElement(
      "div"
    );


  row.id =
    "sfVisualSourceControls";


  row.className =
    "sf-source-controls";


  imageFileInput =
    document.createElement(
      "input"
    );


  imageFileInput.type =
    "file";


  imageFileInput.accept =
    "image/*";


  imageFileInput.style.display =
    "none";


  imageFileInput.addEventListener(
    "change",

    async () => {

      const file =
        (
          imageFileInput.files
          &&
          imageFileInput.files.length > 0

            ? imageFileInput.files[0]

            : null
        );


      if (file) {

        await setVisualImage(
          file
        );
      }


      imageFileInput.value =
        "";
    }
  );


  videoFileInput =
    document.createElement(
      "input"
    );


  videoFileInput.type =
    "file";


  videoFileInput.accept =
    "video/*";


  videoFileInput.style.display =
    "none";


  videoFileInput.addEventListener(
    "change",

    async () => {

      const file =
        (
          videoFileInput.files
          &&
          videoFileInput.files.length > 0

            ? videoFileInput.files[0]

            : null
        );


      if (file) {

        await setVisualVideo(
          file
        );
      }


      videoFileInput.value =
        "";
    }
  );


  row.appendChild(
    makeButton(
      "Choose Image",

      () => {

        imageFileInput.click();
      }
    )
  );


  row.appendChild(
    makeButton(
      "Choose Video",

      () => {

        videoFileInput.click();
      }
    )
  );


  row.appendChild(
    imageFileInput
  );


  row.appendChild(
    videoFileInput
  );


  host.appendChild(
    row
  );


  staticImagePreview =
    document.createElement(
      "img"
    );


  staticImagePreview.id =
    "sfStaticImagePreview";


  staticImagePreview.alt =
    "Selected visual source";


  host.appendChild(
    staticImagePreview
  );
}


// ============================================================
// Key normalisation
// ============================================================

function normaliseKey(
  event
) {

  if (
    event.key
    === "Backspace"
  ) {

    return "backspace";
  }


  if (
    event.key
    === "Delete"
  ) {

    return "delete";
  }


  if (
    event.key
    === " "
  ) {

    return "space";
  }


  if (
    typeof event.key
      === "string"
    &&
    event.key.length
      === 1
  ) {

    return (
      event.key
        .toLowerCase()
    );
  }


  return (
    String(
      event.key
    )
    .toLowerCase()
  );
}


// ============================================================
// Readiness
// ============================================================

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
    element.querySelector(
      "b"
    );


  if (bold) {

    bold.textContent =
      (
        ready
          ? readyText
          : missingText
      );
  }
}


function currentTextLength() {

  if (!textInput) {

    return 0;
  }


  return (
    textInput.value
      .trim()
      .length
  );
}


function currentKeydownCount() {

  return (
    keystrokeEvents.filter(
      event =>
        event.type === "down"
    ).length
  );
}


function audioIsReady() {

  return Boolean(
    audioSourceReady
  );
}


function visualIsReady() {

  if (
    visualMode
    === "image"
  ) {

    return Boolean(
      visualSourceReady
    );
  }


  if (
    visualMode
    === "video"
  ) {

    return Boolean(
      visualSourceReady
    );
  }


  if (
    visualMode
    === "webcam"
  ) {

    return Boolean(
      visualSourceReady
      &&
      webcamStream
      &&
      webcam
      &&
      webcam.videoWidth > 0
      &&
      webcam.videoHeight > 0
    );
  }


  return false;
}


function allModalitiesReady() {

  return (
    !stateChangeInProgress
    &&
    fusionModelLoaded
    &&
    currentTextLength()
      >= MIN_TEXT_CHARS
    &&
    currentKeydownCount()
      >= MIN_KEYPRESSES
    &&
    audioIsReady()
    &&
    visualIsReady()
  );
}


function updateReadiness() {

  const textCount =
    currentTextLength();


  const keydowns =
    currentKeydownCount();


  const textOk =
    (
      textCount
      >= MIN_TEXT_CHARS
    );


  const keyOk =
    (
      keydowns
      >= MIN_KEYPRESSES
    );


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
    "Required"
  );


  setReady(
    imageReady,
    visualOk,
    "Ready",
    "Required"
  );


  if (textCard) {

    const active =
      (
        textOk
        &&
        keyOk
      );


    textCard.classList.toggle(
      "active",
      active
    );


    const badge =
      textCard.querySelector(
        ".badge"
      );


    if (badge) {

      badge.textContent =
        (
          active
            ? "active"
            : "inactive"
        );
    }
  }


  if (audioCard) {

    audioCard.classList.toggle(
      "active",
      audioOk
    );


    const badge =
      audioCard.querySelector(
        ".badge"
      );


    if (badge) {

      badge.textContent =
        (
          audioOk
            ? "active"
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
      webcamCard.querySelector(
        ".badge"
      );


    if (badge) {

      badge.textContent =
        (
          visualOk
            ? "active"
            : "inactive"
        );
    }
  }
}


// ============================================================
// Model / canonical configuration status
// ============================================================

async function checkModelStatus() {

  try {

    const response =
      await fetch(
        "/model-status",
        {
          cache:
            "no-store"
        }
      );


    const data =
      await response.json();


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
    }


    fusionModelLoaded =
      Boolean(
        data.fusion_model
      );


    webcamModelLoaded =
      Boolean(
        data.webcam_calibrated_image_model
      );


    temporalFusionBackend =
      (
        data.temporal_fusion_backend
        || null
      );


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


    AUDIO_CAPTURE_SECONDS =
      positiveInteger(
        data.audio_capture_seconds,
        10
      );


    TARGET_AUDIO_SAMPLE_RATE =
      positiveInteger(
        data.target_audio_sample_rate,
        16000
      );


    if (
      Array.isArray(
        data.labels
      )
    ) {

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
            "Webcam calibration not "
            + "required by current fusion schema."
          )
    );


  } catch (error) {

    fusionModelLoaded =
      false;


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


// ============================================================
// Keystroke capture
// ============================================================

if (textInput) {

  textInput.addEventListener(
    "keydown",

    event => {

      const key =
        normaliseKey(
          event
        );


      // Match desktop behaviour:
      // ignore operating-system/browser key-repeat events.
      if (
        activeKeys.has(
          key
        )
      ) {

        return;
      }


      activeKeys.add(
        key
      );


      keystrokeEvents.push(
        {
          type:
            "down",

          key:
            key,

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
        normaliseKey(
          event
        );


      activeKeys.delete(
        key
      );


      keystrokeEvents.push(
        {
          type:
            "up",

          key:
            key,

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
}


// ============================================================
// Audio source: file
// ============================================================

async function setAudioFile(
  file,
  sourceKind,
  existingOperationEpoch = null
) {

  if (!file) {

    return;
  }


  const operationEpoch =
    (
      existingOperationEpoch
      !== null

        ? existingOperationEpoch

        : beginStateChange(
            "Loading new audio source..."
          )
    );


  if (
    !operationStillCurrent(
      operationEpoch
    )
  ) {

    return;
  }


  const formData =
    new FormData();


  formData.append(
    "session_id",
    sessionId
  );


  formData.append(
    "source_kind",
    sourceKind
  );


  formData.append(
    "audio_file",
    file,
    file.name
  );


  try {

    setText(
      statusBox,
      "Uploading audio source..."
    );


    const response =
      await fetch(
        "/set_audio_source",
        {
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      return;
    }


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
    }


    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );


    audioSourceReady =
      true;


    audioSourceName =
      (
        data.audio_name
        || file.name
      );


    audioSourceKind =
      (
        data.audio_source_kind
        || sourceKind
      );


    // Source replacement performs a canonical temporal reset
    // in app.py / TemporalFusionEngine.
    resetPredictionDisplay();


    setText(
      audioStatus,
      (
        "Audio: "
        + audioSourceName
        + " | fixed source"
      )
    );


    updateAudioDiagnostic(
      data.audio_diagnostics
    );


    setText(
      statusBox,
      (
        "Audio source ready"
        + ` | Generation=${serverGeneration}`
        + " | Temporal history reset."
      )
    );


  } catch (error) {

    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {

      setText(
        statusBox,
        (
          "Audio source failed: "
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


// ============================================================
// Browser microphone acquisition
//
// Behaviour:
//     - exactly ONE recording operation
//     - 10 seconds by default
//     - mono
//     - resampled to 16 kHz
//     - PCM16 WAV
//     - reused for subsequent predictions
//
// No automatic recurring audio replacement occurs.
// ============================================================

function concatenateFloat32(
  chunks
) {

  const totalLength =
    chunks.reduce(
      (
        total,
        chunk
      ) => {

        return (
          total
          + chunk.length
        );
      },
      0
    );


  const output =
    new Float32Array(
      totalLength
    );


  let offset = 0;


  for (
    const chunk
    of chunks
  ) {

    output.set(
      chunk,
      offset
    );


    offset +=
      chunk.length;
  }


  return output;
}


function resampleLinear(
  input,
  inputRate,
  outputRate
) {

  if (
    input.length === 0
  ) {

    return (
      new Float32Array(0)
    );
  }


  if (
    inputRate
    === outputRate
  ) {

    return input;
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
    inputRate
    / outputRate;


  for (
    let index = 0;
    index < outputLength;
    index += 1
  ) {

    const sourcePosition =
      index
      * ratio;


    const left =
      Math.floor(
        sourcePosition
      );


    const right =
      Math.min(
        left + 1,
        input.length - 1
      );


    const fraction =
      sourcePosition
      - left;


    output[index] =
      (
        input[left]
        * (
            1
            - fraction
          )
        +
        input[right]
        * fraction
      );
  }


  return output;
}


function forceExactDuration(
  samples,
  sampleRate,
  seconds
) {

  const requiredLength =
    Math.round(
      sampleRate
      * seconds
    );


  const output =
    new Float32Array(
      requiredLength
    );


  const copyLength =
    Math.min(
      samples.length,
      requiredLength
    );


  output.set(
    samples.subarray(
      0,
      copyLength
    ),
    0
  );


  return output;
}


function encodePcm16Wav(
  samples,
  sampleRate
) {

  const bytesPerSample =
    2;


  const buffer =
    new ArrayBuffer(
      44
      + samples.length
      * bytesPerSample
    );


  const view =
    new DataView(
      buffer
    );


  function writeString(
    offset,
    value
  ) {

    for (
      let index = 0;
      index < value.length;
      index += 1
    ) {

      view.setUint8(
        offset + index,
        value.charCodeAt(
          index
        )
      );
    }
  }


  writeString(
    0,
    "RIFF"
  );


  view.setUint32(
    4,
    36
    + samples.length
    * bytesPerSample,
    true
  );


  writeString(
    8,
    "WAVE"
  );


  writeString(
    12,
    "fmt "
  );


  view.setUint32(
    16,
    16,
    true
  );


  // PCM
  view.setUint16(
    20,
    1,
    true
  );


  // Mono
  view.setUint16(
    22,
    1,
    true
  );


  view.setUint32(
    24,
    sampleRate,
    true
  );


  view.setUint32(
    28,
    sampleRate
    * bytesPerSample,
    true
  );


  view.setUint16(
    32,
    bytesPerSample,
    true
  );


  view.setUint16(
    34,
    16,
    true
  );


  writeString(
    36,
    "data"
  );


  view.setUint32(
    40,
    samples.length
    * bytesPerSample,
    true
  );


  let offset = 44;


  for (
    let index = 0;
    index < samples.length;
    index += 1
  ) {

    const sample =
      Math.max(
        -1,
        Math.min(
          1,
          samples[index]
        )
      );


    const pcm =
      (
        sample < 0

          ? sample * 32768

          : sample * 32767
      );


    view.setInt16(
      offset,
      Math.round(
        pcm
      ),
      true
    );


    offset +=
      bytesPerSample;
  }


  return new Blob(
    [buffer],
    {
      type:
        "audio/wav"
    }
  );
}


async function recordMicrophoneOnce() {

  if (microphoneRecording) {

    return;
  }


  if (
    !navigator.mediaDevices
    ||
    !navigator.mediaDevices
      .getUserMedia
  ) {

    setText(
      statusBox,
      "Microphone API is unavailable."
    );


    return;
  }


  microphoneRecording =
    true;


  const operationEpoch =
    beginStateChange(
      (
        "Recording fixed "
        + `${AUDIO_CAPTURE_SECONDS}s `
        + "microphone sample..."
      )
    );


  let stream = null;

  let context = null;


  try {

    stream =
      await navigator.mediaDevices
        .getUserMedia(
          {
            audio: {
              channelCount:
                1,

              echoCancellation:
                false,

              noiseSuppression:
                false,

              autoGainControl:
                false
            },

            video:
              false
          }
        );


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      return;
    }


    const AudioContextClass =
      (
        window.AudioContext
        ||
        window.webkitAudioContext
      );


    if (!AudioContextClass) {

      throw new Error(
        "Web Audio API is unavailable."
      );
    }


    context =
      new AudioContextClass();


    await context.resume();


    const source =
      context.createMediaStreamSource(
        stream
      );


    const processor =
      context.createScriptProcessor(
        4096,
        1,
        1
      );


    const silentGain =
      context.createGain();


    silentGain.gain.value =
      0;


    const chunks = [];


    processor.onaudioprocess =
      event => {

        chunks.push(
          new Float32Array(
            event.inputBuffer
              .getChannelData(0)
          )
        );
      };


    source.connect(
      processor
    );


    processor.connect(
      silentGain
    );


    silentGain.connect(
      context.destination
    );


    setText(
      audioStatus,
      (
        "Recording microphone "
        + `(${AUDIO_CAPTURE_SECONDS}s)...`
      )
    );


    await sleep(
      AUDIO_CAPTURE_SECONDS
      * 1000
    );


    processor.disconnect();

    source.disconnect();

    silentGain.disconnect();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      return;
    }


    const combined =
      concatenateFloat32(
        chunks
      );


    if (
      combined.length === 0
    ) {

      throw new Error(
        "Microphone produced no audio samples."
      );
    }


    const resampled =
      resampleLinear(
        combined,
        context.sampleRate,
        TARGET_AUDIO_SAMPLE_RATE
      );


    const exact =
      forceExactDuration(
        resampled,
        TARGET_AUDIO_SAMPLE_RATE,
        AUDIO_CAPTURE_SECONDS
      );


    const wavBlob =
      encodePcm16Wav(
        exact,
        TARGET_AUDIO_SAMPLE_RATE
      );


    const file =
      new File(
        [wavBlob],
        "microphone.wav",
        {
          type:
            "audio/wav"
        }
      );


    // Use the SAME operation epoch.
    // Do not create a second source-change epoch.
    await setAudioFile(
      file,
      "microphone",
      operationEpoch
    );


  } catch (error) {

    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {

      setText(
        statusBox,
        (
          "Microphone recording failed: "
          + String(
              error.message
              || error
            )
        )
      );
    }


  } finally {

    if (context) {

      try {

        await context.close();

      } catch (_) {

        // Ignore shutdown errors.
      }
    }


    if (stream) {

      stream
        .getTracks()
        .forEach(
          track => {

            track.stop();
          }
        );
    }


    microphoneRecording =
      false;


    finishStateChange(
      operationEpoch
    );
  }
}


// ============================================================
// Audio diagnostic display
//
// Diagnostic only.
// No model probability is modified in JavaScript.
// ============================================================

function updateAudioDiagnostic(
  audio
) {

  const value =
    (
      audio
      || {}
    );


  const dbfs =
    Number(
      value.dbfs
    );


  const duration =
    Number(
      (
        value.duration_sec
        ??
        value.analysed_duration_sec
      )
    );


  let diagnosticText =
    (
      "Audio condition: "
      + String(
          value.condition
          || "unknown"
        )
    );


  if (
    Number.isFinite(
      duration
    )
  ) {

    diagnosticText +=
      (
        " | Analysed: "
        + duration.toFixed(2)
        + "s"
      );
  }


  if (
    Number.isFinite(
      dbfs
    )
  ) {

    diagnosticText +=
      (
        " | Level: "
        + dbfs.toFixed(1)
        + " dBFS"
      );
  }


  if (value.note) {

    diagnosticText +=
      (
        " | "
        + String(
            value.note
          )
      );
  }


  if (audioDiagnostic) {

    setText(
      audioDiagnostic,
      diagnosticText
    );

  } else {

    setText(
      audioStatus,
      diagnosticText
    );
  }
}


// ============================================================
// Visual preview utilities
// ============================================================

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


  staticImagePreview.style.display =
    "none";


  staticImagePreview.removeAttribute(
    "src"
  );
}


function stopWebcamStreamLocally() {

  if (webcamStream) {

    webcamStream
      .getTracks()
      .forEach(
        track => {

          track.stop();
        }
      );
  }


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


// ============================================================
// Static image source
// ============================================================

async function setVisualImage(
  file
) {

  if (!file) {

    return;
  }


  const operationEpoch =
    beginStateChange(
      "Loading new image source..."
    );


  stopWebcamStreamLocally();

  stopVideoPreview();

  hideStaticImagePreview();

  revokeVisualObjectUrl();


  visualMode =
    "none";


  visualSourceReady =
    false;


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
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      return;
    }


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
    }


    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );


    visualMode =
      "image";


    visualSourceReady =
      true;


    visualSourceName =
      (
        data.visual_name
        || file.name
      );


    visualObjectUrl =
      URL.createObjectURL(
        file
      );


    if (staticImagePreview) {

      staticImagePreview.src =
        visualObjectUrl;


      staticImagePreview.style.display =
        "block";
    }


    if (webcam) {

      webcam.style.display =
        "none";
    }


    resetPredictionDisplay();


    setText(
      webcamStatus,
      (
        "Image: "
        + visualSourceName
      )
    );


    setText(
      statusBox,
      (
        "Image source ready"
        + ` | Generation=${serverGeneration}`
        + " | Temporal history reset."
      )
    );


  } catch (error) {

    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {

      visualMode =
        "none";


      visualSourceReady =
        false;


      visualSourceName =
        null;


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
    }


  } finally {

    finishStateChange(
      operationEpoch
    );
  }
}


// ============================================================
// Video source
//
// Server controls canonical OpenCV frame extraction.
//
// Browser playback is preview only.
// ============================================================

async function setVisualVideo(
  file
) {

  if (!file) {

    return;
  }


  const operationEpoch =
    beginStateChange(
      "Loading new video source..."
    );


  stopWebcamStreamLocally();

  stopVideoPreview();

  hideStaticImagePreview();

  revokeVisualObjectUrl();


  visualMode =
    "none";


  visualSourceReady =
    false;


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
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      return;
    }


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
    }


    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );


    visualMode =
      "video";


    visualSourceReady =
      true;


    visualSourceName =
      (
        data.visual_name
        || file.name
      );


    visualObjectUrl =
      URL.createObjectURL(
        file
      );


    if (webcam) {

      webcam.srcObject =
        null;


      webcam.src =
        visualObjectUrl;


      webcam.loop =
        true;


      webcam.muted =
        true;


      webcam.style.display =
        "";


      try {

        await webcam.play();

      } catch (_) {

        // Browser autoplay restrictions do not affect
        // server-side canonical video inference.
      }
    }


    resetPredictionDisplay();


    setText(
      webcamStatus,
      (
        "Video: "
        + visualSourceName
      )
    );


    setText(
      statusBox,
      (
        "Video source ready"
        + ` | Generation=${serverGeneration}`
        + " | Temporal history reset."
      )
    );


  } catch (error) {

    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {

      visualMode =
        "none";


      visualSourceReady =
        false;


      visualSourceName =
        null;


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
    }


  } finally {

    finishStateChange(
      operationEpoch
    );
  }
}


// ============================================================
// Webcam source
// ============================================================

async function startWebcamMode() {

  if (
    !navigator.mediaDevices
    ||
    !navigator.mediaDevices
      .getUserMedia
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


  visualMode =
    "none";


  visualSourceReady =
    false;


  let newStream =
    null;


  try {

    newStream =
      await navigator.mediaDevices
        .getUserMedia(
          {
            video:
              true,

            audio:
              false
          }
        );


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      newStream
        .getTracks()
        .forEach(
          track => {

            track.stop();
          }
        );


      return;
    }


    if (!webcam) {

      throw new Error(
        "Webcam video element is missing."
      );
    }


    webcamStream =
      newStream;


    webcam.removeAttribute(
      "src"
    );


    webcam.srcObject =
      webcamStream;


    webcam.muted =
      true;


    webcam.style.display =
      "";


    await webcam.play();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      stopWebcamStreamLocally();

      return;
    }


    const formData =
      new FormData();


    formData.append(
      "session_id",
      sessionId
    );


    const response =
      await fetch(
        "/set_visual_webcam",
        {
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      stopWebcamStreamLocally();

      return;
    }


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
    }


    serverGeneration =
      finiteNumber(
        data.generation,
        serverGeneration
      );


    visualMode =
      "webcam";


    visualSourceReady =
      true;


    visualSourceName =
      (
        data.visual_name
        || "Webcam"
      );


    resetPredictionDisplay();


    setText(
      webcamStatus,
      "Webcam active."
    );


    setText(
      sessionStatus,
      "Webcam session running."
    );


    setText(
      statusBox,
      (
        "Webcam ready"
        + ` | Generation=${serverGeneration}`
        + " | Temporal history reset."
      )
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

    stopWebcamStreamLocally();


    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {

      visualMode =
        "none";


      visualSourceReady =
        false;


      visualSourceName =
        null;


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
    }


  } finally {

    finishStateChange(
      operationEpoch
    );
  }
}


// ============================================================
// Stop visual
//
// Mirrors current desktop semantics:
//
// Static image:
//     remains selected.
//
// Video / webcam:
//     stream stops.
//
// Stop Visual itself DOES NOT reset temporal history.
// ============================================================

async function stopVisualMode() {

  if (
    visualMode
    === "image"
  ) {

    setText(
      statusBox,
      (
        "Static image remains selected. "
        + "Stop Visual applies to "
        + "video/webcam streams."
      )
    );


    return;
  }


  if (
    visualMode !== "video"
    &&
    visualMode !== "webcam"
  ) {

    return;
  }


  stopWebcamStreamLocally();

  stopVideoPreview();

  revokeVisualObjectUrl();


  visualSourceReady =
    false;


  try {

    const formData =
      new FormData();


    formData.append(
      "session_id",
      sessionId
    );


    const response =
      await fetch(
        "/stop_visual",
        {
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
    }


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


    if (
      visualMode
      === "none"
    ) {

      visualSourceName =
        null;
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

      startBtn.disabled =
        false;
    }


    if (stopBtn) {

      stopBtn.disabled =
        true;
    }


  } catch (error) {

    setText(
      statusBox,
      (
        "Stop Visual failed: "
        + String(
            error.message
            || error
          )
      )
    );


  } finally {

    updateReadiness();
  }
}


// ============================================================
// Webcam frame capture
//
// Browser sends PNG so an unnecessary lossy JPEG encode is
// avoided.
//
// app.py decodes the frame and creates the canonical OpenCV
// JPEG snapshot before FinalMultimodalInference.
// ============================================================

function captureWebcamFrame() {

  if (
    visualMode
      !== "webcam"
    ||
    !webcamStream
    ||
    !webcam
    ||
    !canvas
    ||
    webcam.videoWidth
      <= 0
    ||
    webcam.videoHeight
      <= 0
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


  return (
    canvas.toDataURL(
      "image/png"
    )
  );
}


// ============================================================
// Probability rendering
//
// IMPORTANT:
//
// This function DOES NOT normalise probabilities,
// aggregate them, rank them, calculate confidence,
// or calculate the confidence gap.
//
// It only converts server-provided decimal values into
// percentages for display.
// ============================================================

function resolveRenderLabels(
  probabilities
) {

  if (
    behaviouralLabels.length > 0
  ) {

    return (
      behaviouralLabels
    );
  }


  if (
    probabilities
    &&
    typeof probabilities
      === "object"
  ) {

    return (
      Object.keys(
        probabilities
      )
    );
  }


  return [];
}


function renderProbabilityBars(
  container,
  probabilities,
  type = "temporal",
  probabilitySum = null
) {

  if (!container) {

    return;
  }


  container.innerHTML =
    "";


  if (
    !probabilities
    ||
    typeof probabilities
      !== "object"
  ) {

    return;
  }


  const labels =
    resolveRenderLabels(
      probabilities
    );


  for (
    const label
    of labels
  ) {

    const probability =
      finiteNumber(
        probabilities[label],
        0
      );


    const percentage =
      probability
      * 100;


    const boundedPercentage =
      Math.max(
        0,
        Math.min(
          100,
          percentage
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
      (
        percentage.toFixed(2)
        + "%"
      );


    labelRow.appendChild(
      name
    );


    labelRow.appendChild(
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
      (
        "sf-prob-fill "
        + type
      );


    fill.style.width =
      (
        boundedPercentage
        + "%"
      );


    track.appendChild(
      fill
    );


    row.appendChild(
      labelRow
    );


    row.appendChild(
      track
    );


    container.appendChild(
      row
    );
  }


  if (
    probabilitySum !== null
    &&
    Number.isFinite(
      Number(
        probabilitySum
      )
    )
  ) {

    const sumRow =
      document.createElement(
        "div"
      );


    sumRow.className =
      "sf-prob-sum";


    const sumName =
      document.createElement(
        "span"
      );


    sumName.textContent =
      "Probability sum";


    const sumValue =
      document.createElement(
        "strong"
      );


    sumValue.textContent =
      Number(
        probabilitySum
      ).toFixed(6);


    sumRow.appendChild(
      sumName
    );


    sumRow.appendChild(
      sumValue
    );


    container.appendChild(
      sumRow
    );
  }
}


// ============================================================
// Prediction-result rendering
//
// Every substantive prediction value here comes from app.py /
// TemporalFusionEngine.
//
// No temporal inference is reconstructed in JavaScript.
// ============================================================

function updatePredictionUI(
  data
) {

  if (
    !data
    ||
    typeof data
      !== "object"
  ) {

    return;
  }


  const returnedGeneration =
    Number(
      data.generation
    );


  if (
    Number.isFinite(
      returnedGeneration
    )
  ) {

    serverGeneration =
      returnedGeneration;
  }


  const state =
    String(
      (
        data.current_state
        ||
        data.prediction
        ||
        "unknown"
      )
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
    (
      data.runtime_validation
      || {}
    );


  setText(
    predictionBox,
    state.toUpperCase()
  );


  setText(
    confidencePercent,
    (
      confidencePct.toFixed(2)
      + "%"
    )
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


    confidenceLevel.classList.add(
      (
        "confidence-"
        + level.toLowerCase()
      )
    );
  }


  setText(
    rawPrediction,
    (
      data.raw_top_class
      || "—"
    )
  );


  const rawConfidencePct =
    Number(
      data.raw_confidence_percent
    );


  setText(
    rawConfidence,

    Number.isFinite(
      rawConfidencePct
    )

      ? (
          rawConfidencePct
            .toFixed(2)
          + "%"
        )

      : "—"
  );


  setText(
    temporalSamples,
    (
      data.temporal_samples
      ?? 0
    )
  );


  setText(
    temporalWindow,
    (
      data.temporal_window
      ?? TEMPORAL_WINDOW
    )
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
    (
      data.second_class
      || "—"
    )
  );


  setText(
    confidenceGap,
    gap.toFixed(4)
  );


  setText(
    featureDimension,
    (
      data.feature_dimension
      ?? "—"
    )
  );


  setText(
    deviceInfo,
    (
      data.device
      || "—"
    )
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
    (
      data.used_modalities
      || {}
    );


  const active =
    Object.entries(
      modalities
    )
      .filter(
        ([, enabled]) => {

          return Boolean(
            enabled
          );
        }
      )
      .map(
        ([name]) => {

          return name;
        }
      );


  setText(
    activeModalities,
    (
      active.length > 0

        ? active.join(
            ", "
          )

        : "—"
    )
  );


  setText(
    technicalRawState,
    (
      data.raw_top_class
      || "—"
    )
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
    &&
    typeof webcamResult
      === "object"
  ) {

    setText(
      webcamPrediction,
      (
        webcamResult.current_state
        || "—"
      )
    );


    const webcamConfidencePct =
      Number(
        webcamResult.confidence_percent
      );


    setText(
      webcamConfidence,

      Number.isFinite(
        webcamConfidencePct
      )

        ? (
            webcamConfidencePct
              .toFixed(2)
            + "%"
          )

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


  updateAudioDiagnostic(
    data.audio_diagnostics
  );


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
      (
        data.temporal_window_full
        ??
        validation.temporal_window_full
      )
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


  if (validationStatus) {

    setText(
      validationStatus,
      validationText
    );
  }


  setText(
    statusBox,
    (
      validationText
      + ` | Generation=${serverGeneration}`
      + ` | Audio=${audioSourceName || "—"}`
      + ` | Visual=${data.visual_source_type || visualMode}`
    )
  );
}


// ============================================================
// Automatic live prediction
//
// Equivalent scheduler policy:
//
//     every LIVE_INTERVAL_MS:
//         if inference already running -> skip
//         if modalities not ready      -> skip
//         otherwise                    -> predict
//
// The backend owns temporal history.
// ============================================================

async function runLivePrediction() {

  if (
    predictionInFlight
    ||
    stateChangeInProgress
  ) {

    return;
  }


  updateReadiness();


  if (
    !allModalitiesReady()
  ) {

    return;
  }


  let webcamFrame =
    null;


  if (
    visualMode
    === "webcam"
  ) {

    webcamFrame =
      captureWebcamFrame();


    if (!webcamFrame) {

      setText(
        statusBox,
        (
          "Current webcam frame "
          + "is unavailable."
        )
      );


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
      (
        textInput
          ? textInput.value.trim()
          : ""
      )
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
      (
        "Running canonical multimodal "
        + "fusion inference..."
      )
    );


    const response =
      await fetch(
        "/predict_live",
        {
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (!response.ok) {

      if (
        response.status === 409
        &&
        handleConflictResponse(
          data
        )
      ) {

        return;
      }


      throw new Error(
        formatServerError(
          data
        )
      );
    }


    // Browser reset/source replacement occurred while request
    // was running.
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


    // The backend response must belong to exactly the same
    // TemporalFusionEngine generation used by this request.
    if (
      !Number.isFinite(
        returnedGeneration
      )
      ||
      returnedGeneration
      !== requestGeneration
    ) {

      return;
    }


    serverGeneration =
      returnedGeneration;


    updatePredictionUI(
      data
    );


  } catch (error) {

    // Do not replace UI with an old error following a reset.
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

    predictionInFlight =
      false;
  }
}


// ============================================================
// Prediction-display reset
//
// UI-only operation.
//
// It does NOT manipulate temporal history.
// Temporal history exists only in TemporalFusionEngine.
// ============================================================

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

    probabilitiesBox.innerHTML =
      "";
  }


  if (rawProbabilitiesBox) {

    rawProbabilitiesBox.innerHTML =
      "";
  }


  if (webcamProbabilityBars) {

    webcamProbabilityBars.innerHTML =
      "";
  }


  if (validationStatus) {

    validationStatus.textContent =
      "";
  }
}


// ============================================================
// Temporal reset
//
// Audio and visual sources remain selected.
//
// Canonical generation increment + temporal-history clear occur
// only in:
//
//     app.py
//         ->
//     TemporalFusionEngine.reset()
// ============================================================

async function resetTemporalWindow() {

  const operationEpoch =
    beginStateChange(
      "Resetting temporal history..."
    );


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
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      return;
    }


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
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
      )
    );


  } catch (error) {

    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {

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
    }


  } finally {

    finishStateChange(
      operationEpoch
    );
  }
}


// ============================================================
// Full reset
//
// Full browser session reset + canonical backend reset.
// ============================================================

async function resetSession() {

  const operationEpoch =
    beginStateChange(
      "Performing full reset..."
    );


  stopWebcamStreamLocally();

  stopVideoPreview();

  hideStaticImagePreview();

  revokeVisualObjectUrl();


  try {

    const formData =
      new FormData();


    formData.append(
      "session_id",
      sessionId
    );


    const response =
      await fetch(
        "/full_reset",
        {
          method:
            "POST",

          body:
            formData
        }
      );


    const data =
      await response.json();


    if (
      !operationStillCurrent(
        operationEpoch
      )
    ) {

      return;
    }


    if (!response.ok) {

      throw new Error(
        formatServerError(
          data
        )
      );
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


    if (textInput) {

      textInput.value =
        "";
    }


    keystrokeEvents = [];

    activeKeys.clear();


    audioSourceReady =
      false;


    audioSourceName =
      null;


    audioSourceKind =
      null;


    visualMode =
      "none";


    visualSourceReady =
      false;


    visualSourceName =
      null;


    resetPredictionDisplay();


    setText(
      audioStatus,
      "Audio not loaded."
    );


    if (audioDiagnostic) {

      setText(
        audioDiagnostic,
        "Audio condition: —"
      );
    }


    setText(
      webcamStatus,
      "Visual input not loaded."
    );


    setText(
      sessionStatus,
      "Session not started."
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


  } catch (error) {

    if (
      operationStillCurrent(
        operationEpoch
      )
    ) {

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
    }


  } finally {

    finishStateChange(
      operationEpoch
    );
  }
}


// ============================================================
// Existing button bindings
//
// Existing Start button = Start Webcam.
//
// Automatic multimodal prediction itself remains continuously
// scheduled in the background once the page is initialised.
// ============================================================

if (startBtn) {

  startBtn.addEventListener(
    "click",
    startWebcamMode
  );
}


if (stopBtn) {

  stopBtn.addEventListener(
    "click",
    stopVisualMode
  );
}


if (resetTemporalBtn) {

  resetTemporalBtn.addEventListener(
    "click",
    resetTemporalWindow
  );
}


if (resetBtn) {

  resetBtn.addEventListener(
    "click",
    resetSession
  );
}


// ============================================================
// Shutdown
// ============================================================

window.addEventListener(
  "beforeunload",

  () => {

    if (
      liveTimer
      !== null
    ) {

      window.clearInterval(
        liveTimer
      );


      liveTimer =
        null;
    }


    stopWebcamStreamLocally();


    revokeVisualObjectUrl();
  }
);


// ============================================================
// Initialisation
// ============================================================

async function initialise() {

  installStyles();


  // Load authoritative backend configuration FIRST.
  await checkModelStatus();


  // Dynamic button labels now use backend-supplied values,
  // e.g. AUDIO_CAPTURE_SECONDS.
  installAudioControls();

  installVisualControls();


  resetPredictionDisplay();

  updateReadiness();


  if (startBtn) {

    startBtn.disabled =
      false;
  }


  if (stopBtn) {

    stopBtn.disabled =
      true;
  }


  setText(
    sessionStatus,
    "Session not started."
  );


  setText(
    statusBox,
    (
      "Ready. Provide text, keystrokes, "
      + "one fixed audio source and "
      + "one visual source."
    )
  );


  if (
    liveTimer
    !== null
  ) {

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


// ============================================================
// Start
// ============================================================

void initialise();
