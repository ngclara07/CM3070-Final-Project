// === web_app/static/script.js ===


// =============================================================================
// DOM REFERENCES
// =============================================================================

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


// Final fusion result

const predictionBox =
  document.getElementById("prediction");

const confidencePercent =
  document.getElementById("confidencePercent");

const confidenceFill =
  document.getElementById("confidenceFill");

const confidenceLevel =
  document.getElementById("confidenceLevel");


// Webcam modality result

const webcamPrediction =
  document.getElementById("webcamPrediction");

const webcamConfidence =
  document.getElementById("webcamConfidence");

const webcamProbabilityBars =
  document.getElementById("webcamProbabilityBars");


// Technical information

const secondaryState =
  document.getElementById("secondaryState");

const confidenceGap =
  document.getElementById("confidenceGap");

const featureDimension =
  document.getElementById("featureDimension");

const deviceInfo =
  document.getElementById("deviceInfo");

const webcamCalibrationUsed =
  document.getElementById("webcamCalibrationUsed");

const activeModalities =
  document.getElementById("activeModalities");

const probabilitiesBox =
  document.getElementById("probabilities");


// Readiness

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


// Cards

const textCard =
  document.getElementById("textCard");

const webcamCard =
  document.getElementById("webcamCard");

const audioCard =
  document.getElementById("audioCard");


// Metrics

const charCount =
  document.getElementById("charCount");

const keyCount =
  document.getElementById("keyCount");


// =============================================================================
// RUNTIME STATE
// =============================================================================

let keystrokeEvents = [];

let activeKeys =
  new Set();


let mediaStream =
  null;

let mediaRecorder =
  null;

let audioChunks =
  [];

let latestAudioBlob =
  null;


let liveTimer =
  null;

let sessionActive =
  false;


let fusionModelLoaded =
  false;

let webcamCalibratedModelLoaded =
  false;


// 15-second multimodal windows.
// This can be reduced later if runtime performance permits.

const LIVE_INTERVAL_MS =
  15000;


// =============================================================================
// KEY NORMALISATION
// =============================================================================

function normaliseKey(event) {

  if (
    event.key === "Backspace"
  ) {
    return "backspace";
  }

  if (
    event.key === "Delete"
  ) {
    return "delete";
  }

  if (
    event.key === " "
  ) {
    return "space";
  }

  if (
    event.key.length === 1
  ) {
    return event.key.toLowerCase();
  }

  return event.key.toLowerCase();
}


// =============================================================================
// READINESS UI
// =============================================================================

function setReady(
  element,
  isReady,
  readyText = "Ready",
  missingText = "Missing"
) {

  if (!element) {
    return;
  }

  element.classList.toggle(
    "active",
    isReady
  );

  const valueElement =
    element.querySelector("b");

  if (valueElement) {

    valueElement.textContent =
      isReady
        ? readyText
        : missingText;
  }
}


function updateReadiness() {

  const textLength =
    textInput.value.trim().length;

  const keydowns =
    keystrokeEvents.filter(
      event =>
        event.type === "down"
    ).length;


  const textOk =
    textLength >= 20;

  const keyOk =
    keydowns >= 20;

  const audioOk =
    latestAudioBlob !== null;

  const imageOk =
    mediaStream !== null;


  charCount.textContent =
    textLength;

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
    webcamCalibratedModelLoaded,
    "Loaded",
    "Missing"
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


  const textBadge =
    textCard.querySelector(".badge");

  const webcamBadge =
    webcamCard.querySelector(".badge");

  const audioBadge =
    audioCard.querySelector(".badge");


  if (textBadge) {

    textBadge.textContent =
      textOk && keyOk
        ? "active"
        : "inactive";
  }


  if (webcamBadge) {

    webcamBadge.textContent =
      imageOk
        ? "active"
        : "inactive";
  }


  if (audioBadge) {

    audioBadge.textContent =
      audioOk
        ? "active"
        : "inactive";
  }
}


// =============================================================================
// MODEL STATUS
// =============================================================================

async function checkModelStatus() {

  try {

    const response =
      await fetch(
        "/model-status"
      );

    if (!response.ok) {

      throw new Error(
        "Model status endpoint failed."
      );
    }


    const data =
      await response.json();


    fusionModelLoaded =
      Boolean(
        data.fusion_model
      );


    webcamCalibratedModelLoaded =
      Boolean(
        data.webcam_calibrated_image_model
      );


    if (fusionModelLoaded) {

      modelStatusText.textContent =
        `Fusion backend loaded: ${data.inference_backend}`;

    } else {

      modelStatusText.textContent =
        `Fusion backend unavailable. ${
          data.error || ""
        }`;
    }


    if (
      webcamCalibratedModelLoaded
    ) {

      webcamModelStatusText.textContent =
        "Webcam-calibrated image classifier loaded successfully.";

    } else {

      webcamModelStatusText.textContent =
        `Webcam calibration model unavailable. ${
          data.webcam_error || ""
        }`;
    }


    updateReadiness();

  } catch (error) {

    fusionModelLoaded =
      false;

    webcamCalibratedModelLoaded =
      false;


    modelStatusText.textContent =
      "Could not query model status.";

    webcamModelStatusText.textContent =
      "Could not query webcam model status.";


    updateReadiness();
  }
}


// =============================================================================
// KEYSTROKE COLLECTION
// =============================================================================

textInput.addEventListener(
  "keydown",
  event => {

    const key =
      normaliseKey(event);


    if (
      activeKeys.has(key)
    ) {
      return;
    }


    activeKeys.add(key);


    keystrokeEvents.push({
      type:
        "down",

      key:
        key,

      timestamp_perf:
        performance.now() / 1000,

      timestamp_epoch:
        Date.now() / 1000
    });


    updateReadiness();
  }
);


textInput.addEventListener(
  "keyup",
  event => {

    const key =
      normaliseKey(event);


    activeKeys.delete(key);


    keystrokeEvents.push({
      type:
        "up",

      key:
        key,

      timestamp_perf:
        performance.now() / 1000,

      timestamp_epoch:
        Date.now() / 1000
    });


    updateReadiness();
  }
);


textInput.addEventListener(
  "input",
  updateReadiness
);


// =============================================================================
// WEBCAM FRAME
// =============================================================================

function captureWebcamFrame() {

  if (
    !mediaStream
    || webcam.videoWidth === 0
    || webcam.videoHeight === 0
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


// =============================================================================
// AUDIO RECORDING
// =============================================================================

function startAudioWindow() {

  if (
    !mediaRecorder
    || !sessionActive
  ) {
    return;
  }


  audioChunks = [];


  if (
    mediaRecorder.state === "inactive"
  ) {

    mediaRecorder.start();
  }


  window.setTimeout(
    () => {

      if (
        mediaRecorder
        && mediaRecorder.state === "recording"
      ) {

        mediaRecorder.stop();
      }

    },
    LIVE_INTERVAL_MS
  );
}


// =============================================================================
// START SESSION
// =============================================================================

async function startSession() {

  if (sessionActive) {
    return;
  }


  try {

    mediaStream =
      await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });


    webcam.srcObject =
      mediaStream;


    await webcam.play();


    webcamStatus.textContent =
      webcamCalibratedModelLoaded
        ? "Webcam capturing with calibrated image classifier."
        : "Webcam capturing; calibrated classifier unavailable.";


    audioStatus.textContent =
      "Microphone recording.";


    const supportedTypes = [
      "audio/webm;codecs=opus",
      "audio/webm"
    ];


    let selectedType =
      "";


    for (
      const candidate
      of supportedTypes
    ) {

      if (
        MediaRecorder.isTypeSupported(candidate)
      ) {

        selectedType =
          candidate;

        break;
      }
    }


    mediaRecorder =
      selectedType
        ? new MediaRecorder(
            mediaStream,
            {
              mimeType:
                selectedType
            }
          )
        : new MediaRecorder(
            mediaStream
          );


    mediaRecorder.ondataavailable =
      event => {

        if (
          event.data
          && event.data.size > 0
        ) {

          audioChunks.push(
            event.data
          );
        }
      };


    mediaRecorder.onstop =
      () => {

        if (
          audioChunks.length > 0
        ) {

          latestAudioBlob =
            new Blob(
              audioChunks,
              {
                type:
                  mediaRecorder.mimeType
                  || "audio/webm"
              }
            );


          audioStatus.textContent =
            "Latest audio analysis window ready.";
        }


        audioChunks = [];


        updateReadiness();


        if (sessionActive) {

          startAudioWindow();
        }
      };


    sessionActive =
      true;


    startBtn.disabled =
      true;

    stopBtn.disabled =
      false;


    sessionStatus.textContent =
      "Live multimodal session running.";


    statusBox.textContent =
      "Capturing live multimodal behavioural signals...";


    startAudioWindow();


    liveTimer =
      window.setInterval(
        runLivePrediction,
        LIVE_INTERVAL_MS
      );


    updateReadiness();

  } catch (error) {

    // ------------------------------------------------------------
    // Limited session:
    // text + keystrokes remain available
    // ------------------------------------------------------------

    sessionActive =
      true;


    startBtn.disabled =
      true;

    stopBtn.disabled =
      false;


    mediaStream =
      null;

    mediaRecorder =
      null;


    sessionStatus.textContent =
      "Session running with limited modalities.";


    webcamStatus.textContent =
      "Webcam unavailable.";


    audioStatus.textContent =
      "Microphone unavailable.";


    statusBox.textContent =
      "Camera or microphone unavailable. Text and keystroke inference remains enabled.";


    liveTimer =
      window.setInterval(
        runLivePrediction,
        LIVE_INTERVAL_MS
      );


    updateReadiness();
  }
}


// =============================================================================
// STOP SESSION
// =============================================================================

function stopSession() {

  sessionActive =
    false;


  if (liveTimer !== null) {

    window.clearInterval(
      liveTimer
    );

    liveTimer =
      null;
  }


  if (
    mediaRecorder
    && mediaRecorder.state === "recording"
  ) {

    mediaRecorder.stop();
  }


  if (mediaStream) {

    mediaStream
      .getTracks()
      .forEach(
        track =>
          track.stop()
      );
  }


  if (webcam) {

    webcam.srcObject =
      null;
  }


  mediaStream =
    null;

  mediaRecorder =
    null;


  startBtn.disabled =
    false;

  stopBtn.disabled =
    true;


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


// =============================================================================
// RESET
// =============================================================================

function resetSession() {

  stopSession();


  textInput.value =
    "";


  keystrokeEvents =
    [];


  activeKeys.clear();


  audioChunks =
    [];


  latestAudioBlob =
    null;


  // Final prediction

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


  // Webcam modality

  webcamPrediction.textContent =
    "—";


  webcamConfidence.textContent =
    "—";


  webcamProbabilityBars.innerHTML =
    "";


  // Technical

  secondaryState.textContent =
    "—";


  confidenceGap.textContent =
    "—";


  featureDimension.textContent =
    "—";


  deviceInfo.textContent =
    "—";


  webcamCalibrationUsed.textContent =
    "—";


  activeModalities.textContent =
    "—";


  probabilitiesBox.innerHTML =
    "";


  statusBox.textContent =
    "Session reset.";


  sessionStatus.textContent =
    "Session not started.";


  updateReadiness();
}


// =============================================================================
// LIVE PREDICTION
// =============================================================================

async function runLivePrediction() {

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
      text.length < 20
      || keydowns < 20
    ) {

      statusBox.textContent =
        `Waiting for sufficient data: ${text.length}/20 characters, ${keydowns}/20 keypresses.`;

      return;
    }


    statusBox.textContent =
      "Running multimodal behavioural-state inference...";


    const frameData =
      captureWebcamFrame();


    const formData =
      new FormData();


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


    if (
      frameData !== null
    ) {

      formData.append(
        "image_frame",
        frameData
      );
    }


    if (
      latestAudioBlob !== null
    ) {

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
          method:
            "POST",

          body:
            formData
        }
      );


    let data;


    try {

      data =
        await response.json();

    } catch {

      throw new Error(
        `Server returned HTTP ${response.status}.`
      );
    }


    if (
      !response.ok
    ) {

      throw new Error(
        data.detail
        || data.error
        || "Live prediction failed."
      );
    }


    updatePredictionUI(
      data
    );

  } catch (error) {

    statusBox.textContent =
      "Live prediction failed: "
      + error.message;
  }
}


// =============================================================================
// FINAL FUSION UI
// =============================================================================

function updatePredictionUI(
  data
) {

  const state =
    data.current_state
    || data.prediction
    || "unknown";


  const confidence =
    Number(
      data.confidence
      || 0
    );


  const confidencePct =
    Number.isFinite(
      Number(
        data.confidence_percent
      )
    )
      ? Number(
          data.confidence_percent
        )
      : confidence * 100;


  const gap =
    Number(
      data.confidence_gap
      || 0
    );


  const level =
    data.confidence_level
    || "Low";


  // ---------------------------------------------------------------------------
  // Final behavioural state
  // ---------------------------------------------------------------------------

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


  if (
    level === "High"
  ) {

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


  // ---------------------------------------------------------------------------
  // Technical information
  // ---------------------------------------------------------------------------

  const details =
    data.technical_details
    || {};


  secondaryState.textContent =
    (
      details.second_class
      || "—"
    ).toUpperCase();


  confidenceGap.textContent =
    Number.isFinite(gap)
      ? gap.toFixed(4)
      : "—";


  featureDimension.textContent =
    data.feature_dimension
    ?? details.feature_dimension
    ?? "—";


  deviceInfo.textContent =
    data.device
    || details.device
    || "—";


  webcamCalibrationUsed.textContent =
    data.webcam_calibration_used
      ? "YES"
      : "NO";


  activeModalities.textContent =
    formatModalities(
      data.used_modalities
    );


  // ---------------------------------------------------------------------------
  // Final fusion probabilities
  // ---------------------------------------------------------------------------

  renderProbabilityBars(
    probabilitiesBox,
    data.probabilities
    || {},
    false
  );


  // ---------------------------------------------------------------------------
  // Webcam calibrated image result
  // ---------------------------------------------------------------------------

  updateWebcamModalityUI(
    data.image_modality
  );


  // ---------------------------------------------------------------------------
  // Overall status
  // ---------------------------------------------------------------------------

  const calibrationText =
    data.webcam_calibration_used
      ? "Webcam calibration active"
      : "Webcam calibration unavailable/not used";


  statusBox.textContent =
    `Final multimodal prediction complete | ${calibrationText} | Active modalities: ${formatModalities(data.used_modalities)}`;
}


// =============================================================================
// WEBCAM MODALITY UI
// =============================================================================

function updateWebcamModalityUI(
  imageResult
) {

  if (!imageResult) {

    webcamPrediction.textContent =
      "NOT AVAILABLE";


    webcamConfidence.textContent =
      "—";


    webcamProbabilityBars.innerHTML =
      "<p class='sub-status'>No calibrated webcam prediction was available for this inference window.</p>";


    return;
  }


  const state =
    imageResult.current_state
    || imageResult.prediction
    || "unknown";


  const confidence =
    Number(
      imageResult.confidence
      || 0
    );


  const confidencePct =
    Number.isFinite(
      Number(
        imageResult.confidence_percent
      )
    )
      ? Number(
          imageResult.confidence_percent
        )
      : confidence * 100;


  webcamPrediction.textContent =
    state.toUpperCase();


  webcamConfidence.textContent =
    `${confidencePct.toFixed(2)}%`;


  renderProbabilityBars(
    webcamProbabilityBars,
    imageResult.probabilities
    || {},
    true
  );
}


// =============================================================================
// PROBABILITY BAR RENDERER
// =============================================================================

function renderProbabilityBars(
  container,
  probabilities,
  webcamStyle = false
) {

  container.innerHTML =
    "";


  const entries =
    Object.entries(
      probabilities
      || {}
    )
      .sort(
        (a, b) =>
          Number(b[1])
          - Number(a[1])
      );


  if (
    entries.length === 0
  ) {

    container.innerHTML =
      "<p>No probability information available.</p>";

    return;
  }


  entries.forEach(
    ([label, probability]) => {

      const numericProbability =
        Number(probability);


      const percentage =
        (
          numericProbability
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
          <span>${escapeHtml(label)}</span>
          <span>${percentage}%</span>
        </div>

        <div class="track">
          <div
            class="fill ${webcamStyle ? "webcam-fill" : ""}"
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


// =============================================================================
// MODALITY FORMATTER
// =============================================================================

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
        ([name]) =>
          name
      );


  return (
    active.length > 0
      ? active.join(", ")
      : "none"
  );
}


// =============================================================================
// BASIC HTML ESCAPING
// =============================================================================

function escapeHtml(
  value
) {

  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}


// =============================================================================
// BUTTON EVENTS
// =============================================================================

startBtn.addEventListener(
  "click",
  startSession
);


stopBtn.addEventListener(
  "click",
  stopSession
);


resetBtn.addEventListener(
  "click",
  resetSession
);


// =============================================================================
// INITIAL STATE
// =============================================================================

checkModelStatus();
updateReadiness();
