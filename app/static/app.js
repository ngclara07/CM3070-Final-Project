// static/app.js
// SenseFuzeAI - Frontend Runtime Logic
//
// Supports:
//   - live keystroke capture
//   - text, audio, and image submission
//   - audio preview
//   - image preview
//   - calibrated fusion result rendering
//   - low-confidence text evidence handling
//   - clearer audio evidence explanation
//   - base/effective/calibrated fusion diagnostics
//   - JSON / CSV / TXT export

"use strict";

// ============================================================
// DOM references
// ============================================================

const textarea = document.getElementById("typed_text");
const form = document.getElementById("analyze-form");
const statusBox = document.getElementById("status");
const resultBox = document.getElementById("results");
const resultJson = document.getElementById("result-json");

const finalPrediction = document.getElementById("final-prediction");
const finalInterpretation = document.getElementById("final-interpretation");
const predictionBadge = document.getElementById("prediction-badge");
const scoreBars = document.getElementById("score-bars");

const confidenceValue = document.getElementById("confidence-value");
const confidenceFill = document.getElementById("confidence-fill");
const sidebarConfidence = document.getElementById("sidebar-confidence");
const heroPrediction = document.getElementById("hero-current-prediction");
const heroConfidence = document.getElementById("hero-confidence");

const keystrokeResult = document.getElementById("keystroke-result");
const keystrokeDetail = document.getElementById("keystroke-detail");
const textResult = document.getElementById("text-result");
const textSentimentDetail = document.getElementById("text-sentiment-detail");
const audioResult = document.getElementById("audio-result");
const audioDetail = document.getElementById("audio-detail");
const imageResult = document.getElementById("image-result");
const imageDetail = document.getElementById("image-detail");

const keystrokeScoreBars = document.getElementById("keystroke-score-bars");
const textScoreBars = document.getElementById("text-score-bars");
const audioScoreBars = document.getElementById("audio-score-bars");
const imageScoreBars = document.getElementById("image-score-bars");

const behaviourSummary = document.getElementById("behaviour-summary");
const imageCaptionSummary = document.getElementById("image-caption-summary");
const audioEvidenceSummary = document.getElementById("audio-evidence-summary");
const keystrokeEvidenceSummary = document.getElementById("keystroke-evidence-summary");

const fusionMethod = document.getElementById("fusion-method");
const evidenceSource = document.getElementById("evidence-source");
const capturedKeys = document.getElementById("captured-keys");
const uploadedModalities = document.getElementById("uploaded-modalities");
const fusionWeights = document.getElementById("fusion-weights");
const textModelInfo = document.getElementById("text-model-info");
const trainedFusionUsed = document.getElementById("trained-fusion-used");
const visualReliability = document.getElementById("visual-reliability");

const toggleJsonBtn = document.getElementById("toggle-json-btn");
const exportJsonBtn = document.getElementById("export-json-btn");
const exportCsvBtn = document.getElementById("export-csv-btn");
const exportTxtBtn = document.getElementById("export-txt-btn");

const resetBtn = document.getElementById("reset-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const workspaceCard = document.getElementById("workspace");

const audioInput = document.getElementById("audio_file");
const imageInput = document.getElementById("image_file");
const audioFileName = document.getElementById("audio-file-name");
const imageFileName = document.getElementById("image-file-name");

const audioPreviewPanel = document.getElementById("audio-preview-panel");
const audioPreview = document.getElementById("audio-preview");
const audioPreviewMeta = document.getElementById("audio-preview-meta");

const imagePreviewPanel = document.getElementById("image-preview-panel");
const imagePreview = document.getElementById("image-preview");
const imagePreviewMeta = document.getElementById("image-preview-meta");

const typingCounter = document.getElementById("typing-counter");
const keystrokeChip = document.getElementById("keystroke-chip");

// ============================================================
// Runtime state
// ============================================================

const state = {
  keystrokeEvents: [],
  activeKeys: new Set(),
  latestResult: null,
  audioObjectUrl: null,
  imageObjectUrl: null
};

const CLASS_ORDER = ["focused", "distracted", "fatigued", "overloaded"];

const LOW_CONFIDENCE_THRESHOLD = 0.40;
const LOW_MARGIN_THRESHOLD = 0.08;

const CLASS_DISPLAY = {
  focused: "Focused",
  distracted: "Distracted",
  fatigued: "Fatigued",
  overloaded: "Overloaded",
  uncertain: "Uncertain",
  unavailable: "Unavailable",
  not_provided: "Not provided"
};

// ============================================================
// Basic utilities
// ============================================================

function nowMs() {
  return performance.now();
}

function normaliseKey(event) {
  if (event.key === "Backspace") return "backspace";
  if (event.key === "Delete") return "delete";
  if (event.key === " ") return "space";
  if (event.key === "Enter") return "enter";
  return event.key || "unknown";
}

function normaliseLabel(label) {
  return String(label || "-").trim().toLowerCase();
}

function formatEvidenceLabel(label) {
  const clean = normaliseLabel(label);

  if (CLASS_DISPLAY[clean]) {
    return CLASS_DISPLAY[clean];
  }

  if (!clean || clean === "-") {
    return "Unavailable";
  }

  return String(label)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function safeNumber(value, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clamp01(value) {
  return Math.max(0, Math.min(1, safeNumber(value, 0)));
}

function formatPct(value) {
  return `${(clamp01(value) * 100).toFixed(1)}%`;
}

function formatScore(value, digits = 3) {
  return safeNumber(value, 0).toFixed(digits);
}

function formatBytes(bytes) {
  const value = safeNumber(bytes, 0);

  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;

  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function countKeydowns() {
  return state.keystrokeEvents.filter((event) => event.type === "down").length;
}

function getTopLabel(scores) {
  const ranked = getRankedScores(scores);
  return ranked.length ? ranked[0].label : "-";
}

function getTopScore(scores) {
  const ranked = getRankedScores(scores);
  return ranked.length ? ranked[0].score : 0;
}

function getRankedScores(scores) {
  if (!scores || typeof scores !== "object") return [];

  return CLASS_ORDER
    .map((label) => ({
      label,
      score: safeNumber(scores[label], 0)
    }))
    .filter((item) => Number.isFinite(item.score))
    .sort((a, b) => b.score - a.score);
}

function getScoreMargin(scores) {
  const ranked = getRankedScores(scores);

  if (ranked.length < 2) return 0;

  return ranked[0].score - ranked[1].score;
}

function isLowConfidenceEvidence(scores) {
  const ranked = getRankedScores(scores);

  if (!ranked.length) return true;

  const topScore = ranked[0].score;
  const margin = getScoreMargin(scores);

  return topScore < LOW_CONFIDENCE_THRESHOLD || margin < LOW_MARGIN_THRESHOLD;
}

function getSecondBest(scores) {
  const ranked = getRankedScores(scores);
  return ranked.length >= 2 ? ranked[1] : null;
}

function truncateText(text, maxLength = 220) {
  const value = String(text || "").trim();

  if (value.length <= maxLength) return value;

  return `${value.slice(0, maxLength - 3)}...`;
}

function timestampForFilename() {
  const now = new Date();

  const pad = (value) => String(value).padStart(2, "0");

  return [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
    "_",
    pad(now.getHours()),
    pad(now.getMinutes()),
    pad(now.getSeconds())
  ].join("");
}

function downloadBlob(content, filename, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");

  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();

  URL.revokeObjectURL(url);
}

function setText(element, value) {
  if (element) {
    element.textContent = value;
  }
}

// ============================================================
// Keystroke capture
// ============================================================

function updateTypingCounter() {
  const characters = textarea ? textarea.value.length : 0;
  const keys = countKeydowns();

  if (typingCounter) {
    typingCounter.textContent = `${characters} characters · ${keys} captured keys`;
  }

  if (keystrokeChip) {
    keystrokeChip.textContent = `Live event stream enabled · ${keys} valid key presses`;
  }
}

if (textarea) {
  textarea.addEventListener("keydown", (event) => {
    const key = normaliseKey(event);

    if (state.activeKeys.has(key)) return;

    state.activeKeys.add(key);

    state.keystrokeEvents.push({
      type: "down",
      key,
      ts: nowMs()
    });

    updateTypingCounter();
  });

  textarea.addEventListener("keyup", (event) => {
    const key = normaliseKey(event);

    state.activeKeys.delete(key);

    state.keystrokeEvents.push({
      type: "up",
      key,
      ts: nowMs()
    });

    updateTypingCounter();
  });

  textarea.addEventListener("input", updateTypingCounter);
}

// ============================================================
// Status and UI state
// ============================================================

function setStatus(message, stateName = "idle") {
  if (!statusBox) return;

  statusBox.textContent = message;
  statusBox.className = `status-pill ${stateName}`;
}

function setProcessingState(isProcessing) {
  document.body.classList.toggle("is-processing", isProcessing);

  if (workspaceCard) {
    workspaceCard.classList.toggle("processing", isProcessing);
  }

  if (analyzeBtn) {
    analyzeBtn.classList.toggle("is-loading", isProcessing);
    analyzeBtn.disabled = isProcessing;
    analyzeBtn.textContent = isProcessing
      ? "Processing Signals..."
      : "Run Behaviour Analysis";
  }

  if (resetBtn) {
    resetBtn.disabled = isProcessing;
  }
}

function setPredictionBadge(label) {
  if (!predictionBadge) return;

  const cleanLabel = normaliseLabel(label);

  predictionBadge.textContent = formatEvidenceLabel(cleanLabel);
  predictionBadge.className = "prediction-badge";

  if (CLASS_ORDER.includes(cleanLabel)) {
    predictionBadge.classList.add(cleanLabel);
  } else {
    predictionBadge.classList.add("neutral");
  }
}

function animateResultCards() {
  document.querySelectorAll(".animated-card").forEach((card, index) => {
    card.style.animation = "none";
    card.offsetHeight;
    card.style.animation = `riseIn 0.55s ease ${index * 0.06}s both`;
  });
}

// ============================================================
// Upload previews
// ============================================================

function clearAudioPreview() {
  if (state.audioObjectUrl) {
    URL.revokeObjectURL(state.audioObjectUrl);
    state.audioObjectUrl = null;
  }

  if (audioPreview) {
    audioPreview.removeAttribute("src");
    audioPreview.load();
  }

  if (audioPreviewMeta) {
    audioPreviewMeta.textContent = "-";
  }

  if (audioPreviewPanel) {
    audioPreviewPanel.classList.add("hidden");
  }
}

function clearImagePreview() {
  if (state.imageObjectUrl) {
    URL.revokeObjectURL(state.imageObjectUrl);
    state.imageObjectUrl = null;
  }

  if (imagePreview) {
    imagePreview.removeAttribute("src");
  }

  if (imagePreviewMeta) {
    imagePreviewMeta.textContent = "-";
  }

  if (imagePreviewPanel) {
    imagePreviewPanel.classList.add("hidden");
  }
}

function handleAudioSelection() {
  clearAudioPreview();

  const file = audioInput?.files?.[0];

  if (!file) {
    setText(audioFileName, "No file selected");
    return;
  }

  setText(audioFileName, file.name);

  state.audioObjectUrl = URL.createObjectURL(file);

  if (audioPreview) {
    audioPreview.src = state.audioObjectUrl;
    audioPreview.load();
  }

  if (audioPreviewMeta) {
    audioPreviewMeta.textContent = `${file.type || "audio file"} · ${formatBytes(file.size)}`;
  }

  if (audioPreviewPanel) {
    audioPreviewPanel.classList.remove("hidden");
  }
}

function handleImageSelection() {
  clearImagePreview();

  const file = imageInput?.files?.[0];

  if (!file) {
    setText(imageFileName, "No file selected");
    return;
  }

  setText(imageFileName, file.name);

  state.imageObjectUrl = URL.createObjectURL(file);

  if (imagePreview) {
    imagePreview.src = state.imageObjectUrl;
  }

  if (imagePreviewMeta) {
    const baseMeta = `${file.type || "image file"} · ${formatBytes(file.size)}`;

    if (imagePreview) {
      imagePreview.onload = () => {
        imagePreviewMeta.textContent = `${baseMeta} · ${imagePreview.naturalWidth}×${imagePreview.naturalHeight}`;
      };
    } else {
      imagePreviewMeta.textContent = baseMeta;
    }
  }

  if (imagePreviewPanel) {
    imagePreviewPanel.classList.remove("hidden");
  }
}

if (audioInput) {
  audioInput.addEventListener("change", handleAudioSelection);
}

if (imageInput) {
  imageInput.addEventListener("change", handleImageSelection);
}

// ============================================================
// Score bars
// ============================================================

function buildScoreBars(scores, container = scoreBars, compact = false) {
  if (!container) return;

  container.innerHTML = "";

  const ranked = getRankedScores(scores);

  const rows = ranked.length
    ? ranked
    : CLASS_ORDER.map((label) => ({ label, score: 0 }));

  rows.forEach(({ label, score }) => {
    const pct = clamp01(score) * 100;

    const row = document.createElement("div");
    row.className = compact ? "score-row compact-score-row" : "score-row";

    row.innerHTML = `
      <div class="score-label-row">
        <span>${formatEvidenceLabel(label)}</span>
        <strong>${pct.toFixed(1)}%</strong>
      </div>
      <div class="score-track ${compact ? "mini-track" : ""}">
        <div class="score-fill ${normaliseLabel(label)}" style="width:${pct}%"></div>
      </div>
    `;

    container.appendChild(row);
  });
}

// ============================================================
// Result summarisation
// ============================================================

function buildFinalInterpretation(data) {
  const prediction = data.final_prediction || "-";
  const scores = data.final_scores || {};
  const confidence = getTopScore(scores);
  const second = getSecondBest(scores);

  const predictionText = formatEvidenceLabel(prediction);
  const secondText = second
    ? ` The next closest state is ${formatEvidenceLabel(second.label)} at ${formatPct(second.score)}.`
    : "";

  const cleanPrediction = normaliseLabel(prediction);

  let explanation = "This result should be considered alongside the individual modality evidence rather than treated as a standalone diagnosis.";

  if (cleanPrediction === "focused") {
    explanation =
      "This usually indicates stable task engagement, consistent working behaviour, and lower signs of interruption. " +
      explanation;
  } else if (cleanPrediction === "distracted") {
    explanation =
      "This usually indicates possible attention shifts, off-task context, interruptions, or inconsistent behavioural evidence. " +
      explanation;
  } else if (cleanPrediction === "fatigued") {
    explanation =
      "This usually indicates slower rhythm, possible tiredness cues, reduced alertness, or fatigue-related evidence. " +
      explanation;
  } else if (cleanPrediction === "overloaded") {
    explanation =
      "This usually indicates task pressure, stress-related cues, environmental load, or erratic behavioural evidence. " +
      explanation;
  }

  return (
    `The system classifies the current state as ${predictionText} with ${formatPct(confidence)} confidence.` +
    secondText +
    ` ${explanation}`
  );
}

function summariseKeystrokeResult(data) {
  const scores = data.keystroke_scores || {};
  const prediction =
    data.keystroke_prediction ||
    getTopLabel(scores);

  const confidence = getTopScore(scores);
  const keydownCount = countKeydowns();

  if (prediction === "unavailable") {
    return `Keystroke evidence unavailable | ${keydownCount} valid key presses captured`;
  }

  return `${formatEvidenceLabel(prediction)} (${formatPct(confidence)}) | ${keydownCount} valid key presses captured`;
}

function summariseKeystrokeEvidence(data) {
  const features = data.keystroke_features || {};
  const speed = features.typing_speed;
  const delay = features.delay_mean;
  const rhythm = features.rhythm_consistency;

  const parts = [];

  if (speed != null) {
    parts.push(`typing speed ${formatScore(speed, 4)}`);
  }

  if (delay != null) {
    parts.push(`mean delay ${formatScore(delay, 2)}`);
  }

  if (rhythm != null) {
    parts.push(`rhythm consistency ${formatScore(rhythm, 2)}`);
  }

  if (!parts.length) {
    return "No detailed keystroke features were returned.";
  }

  return `Keystroke indicators include ${parts.join(", ")}.`;
}

function summariseTextResult(data) {
  const scores = data.text_scores || data.text_result?.behaviour_scores || {};
  const ranked = getRankedScores(scores);

  const prediction =
    data.text_prediction ||
    data.text_result?.predicted_behaviour ||
    getTopLabel(scores);

  const confidence =
    data.text_result?.behaviour_confidence != null
      ? safeNumber(data.text_result.behaviour_confidence, getTopScore(scores))
      : getTopScore(scores);

  const modelName =
    data.text_result?.model_name ||
    data.text_result?.method ||
    "trained text model";

  if (!ranked.length || isLowConfidenceEvidence(scores)) {
    const top = ranked[0];

    if (!top) {
      return `Low-confidence text evidence | ${modelName}`;
    }

    return (
      `Low-confidence / ambiguous text evidence ` +
      `(${formatEvidenceLabel(top.label)} ${formatPct(top.score)}) | ${modelName}`
    );
  }

  return `${formatEvidenceLabel(prediction)} (${formatPct(confidence)}) | ${modelName}`;
}

function summariseTextSentiment(data) {
  const sentiment =
    data.text_sentiment ||
    {
      sentiment_label: data.text_result?.sentiment_label,
      sentiment_score: data.text_result?.sentiment_score,
      sentiment_method: data.text_result?.sentiment_method
    };

  const label = sentiment?.sentiment_label || "-";
  const score =
    sentiment?.sentiment_score != null
      ? formatPct(sentiment.sentiment_score)
      : "-";

  return `Sentiment: ${formatEvidenceLabel(label)} · ${score}`;
}

function extractAudioLabels(data) {
  return (
    data.audio_result?.yamnet_labels ||
    data.audio_result?.labels ||
    []
  );
}

function formatRawAudioLabels(labels, limit = 3) {
  if (!Array.isArray(labels) || !labels.length) return "";

  return labels
    .slice(0, limit)
    .map((item) => {
      if (Array.isArray(item)) {
        return `${item[0]} (${safeNumber(item[1], 0).toFixed(2)})`;
      }

      if (typeof item === "object" && item !== null) {
        const label = item.label || item.name || item.class || "unknown";
        const score = safeNumber(item.score ?? item.confidence, 0);
        return `${label} (${score.toFixed(2)})`;
      }

      return String(item);
    })
    .join(", ");
}

function summariseAudioResult(data) {
  const scores = data.audio_scores || {};

  const prediction =
    data.audio_prediction ||
    data.audio_result?.predicted_behaviour ||
    data.audio_result?.predicted_label ||
    getTopLabel(scores);

  const confidence = getTopScore(scores);
  const labels = extractAudioLabels(data);
  const rawLabelSummary = formatRawAudioLabels(labels, 3);

  if (!prediction || prediction === "not_provided") {
    return "No audio file analysed.";
  }

  if (prediction === "unavailable") {
    return "Audio analysis unavailable.";
  }

  const rawText = rawLabelSummary
    ? ` Raw acoustic labels: ${rawLabelSummary}.`
    : "";

  return (
    `Audio evidence supports ${formatEvidenceLabel(prediction).toLowerCase()} ` +
    `acoustic context (${formatPct(confidence)}).${rawText}`
  );
}

function summariseImageResult(data) {
  const scores = data.image_scores || {};

  const prediction =
    data.image_prediction ||
    data.image_result?.predicted_behaviour ||
    data.image_result?.predicted_label ||
    getTopLabel(scores);

  const confidence = getTopScore(scores);

  const caption =
    data.image_result?.behaviour_caption ||
    data.image_result?.behaviour_aware_caption ||
    data.image_result?.behaviour_description ||
    data.image_result?.caption ||
    data.image_caption ||
    "";

  if (!prediction || prediction === "not_provided") {
    return "No image file analysed.";
  }

  if (prediction === "unavailable") {
    return "Image analysis unavailable.";
  }

  return `${formatEvidenceLabel(prediction)} (${formatPct(confidence)}) | ${truncateText(caption, 130)}`;
}

function getImageBehaviourCaption(data) {
  return (
    data.image_result?.behaviour_caption ||
    data.image_result?.behaviour_aware_caption ||
    data.image_result?.behaviour_description ||
    data.image_result?.caption ||
    data.image_caption ||
    "No image analysis has been run yet."
  );
}

// ============================================================
// Diagnostics formatting
// ============================================================

function inferEvidenceSource(data) {
  const uploaded =
    data.uploaded_modalities ||
    data.fusion_result?.uploaded_modalities;

  if (Array.isArray(uploaded) && uploaded.length) {
    return uploaded.map(formatEvidenceLabel).join(" + ");
  }

  const sources = ["Keystroke", "Text"];

  const audioProvided =
    audioInput?.files?.length > 0 ||
    data.audio_result?.status === "analyzed";

  const imageProvided =
    imageInput?.files?.length > 0 ||
    data.image_result?.status === "analyzed";

  if (audioProvided) sources.push("Audio");
  if (imageProvided) sources.push("Image");

  return sources.join(" + ");
}

function inferUploadedModalities(data) {
  const uploaded =
    data.uploaded_modalities ||
    data.fusion_result?.uploaded_modalities;

  if (Array.isArray(uploaded) && uploaded.length) {
    return uploaded.map(formatEvidenceLabel).join(", ");
  }

  const modalities = ["Text", "Keystroke"];

  if (audioInput?.files?.length > 0 || data.audio_result?.status === "analyzed") {
    modalities.push("Audio");
  }

  if (imageInput?.files?.length > 0 || data.image_result?.status === "analyzed") {
    modalities.push("Image");
  }

  return modalities.join(", ");
}

function formatFusionWeights(weights) {
  if (!weights || typeof weights !== "object") return "-";

  return Object.entries(weights)
    .map(([key, value]) => `${key}: ${formatPct(value)}`)
    .join(" | ");
}

function formatFusionDiagnostics(data) {
  const baseWeights =
    data.base_fusion_weights ||
    data.fusion_result?.base_fusion_weights ||
    data.fusion_weights;

  const effectiveWeights =
    data.effective_fusion_weights ||
    data.fusion_result?.effective_fusion_weights ||
    data.fusion_weights;

  const smoothing =
    data.modality_smoothing ||
    data.fusion_result?.modality_smoothing;

  const parts = [];

  if (baseWeights) {
    parts.push(`Base: ${formatFusionWeights(baseWeights)}`);
  }

  if (effectiveWeights) {
    parts.push(`Effective: ${formatFusionWeights(effectiveWeights)}`);
  }

  if (smoothing) {
    const smoothingText = Object.entries(smoothing)
      .map(([key, value]) => `${key}: ${safeNumber(value, 0).toFixed(2)}`)
      .join(" | ");

    parts.push(`Calibration: ${smoothingText}`);
  }

  return parts.length ? parts.join(" || ") : "-";
}

function formatVisualReliability(data) {
  const result = data.image_result || {};

  if (!result || result.status === "not_provided") {
    return "No image provided";
  }

  if (result.status === "error") {
    return "Image unavailable";
  }

  const reliability = result.reliability_score;
  const quality = result.quality_flag;

  if (reliability == null && !quality) {
    return "-";
  }

  return `${formatScore(reliability, 4)} · ${quality || "unknown"}`;
}

// ============================================================
// Rendering
// ============================================================

function renderResults(data) {
  state.latestResult = data;

  const prediction = data.final_prediction || "-";
  const finalScores = data.final_scores || {};
  const confidence = getTopScore(finalScores);
  const confidencePct = clamp01(confidence) * 100;

  setText(finalPrediction, formatEvidenceLabel(prediction));
  setText(heroPrediction, formatEvidenceLabel(prediction));
  setText(heroConfidence, formatPct(confidence));
  setText(sidebarConfidence, formatPct(confidence));
  setText(confidenceValue, formatPct(confidence));

  if (confidenceFill) {
    confidenceFill.style.width = `${confidencePct}%`;
  }

  setPredictionBadge(prediction);
  buildScoreBars(finalScores, scoreBars, false);

  const finalText = buildFinalInterpretation(data);
  setText(finalInterpretation, finalText);
  setText(behaviourSummary, finalText);

  setText(keystrokeResult, summariseKeystrokeResult(data));
  setText(keystrokeDetail, `${countKeydowns()} valid key presses captured from the text area.`);
  setText(keystrokeEvidenceSummary, summariseKeystrokeEvidence(data));

  setText(textResult, summariseTextResult(data));
  setText(textSentimentDetail, summariseTextSentiment(data));

  setText(audioResult, summariseAudioResult(data));
  setText(audioDetail, data.audio_result?.status === "analyzed" ? "Audio file analysed as acoustic context." : "Audio file optional.");
  setText(audioEvidenceSummary, summariseAudioResult(data));

  setText(imageResult, summariseImageResult(data));
  setText(imageDetail, data.image_result?.status === "analyzed" ? "Image analysed as visual behavioural evidence." : "Image file optional.");
  setText(imageCaptionSummary, getImageBehaviourCaption(data));

  buildScoreBars(data.keystroke_scores, keystrokeScoreBars, true);
  buildScoreBars(data.text_scores, textScoreBars, true);
  buildScoreBars(data.audio_scores, audioScoreBars, true);
  buildScoreBars(data.image_scores, imageScoreBars, true);

  if (fusionMethod) {
    fusionMethod.textContent =
      data.fusion_method ||
      data.fusion_result?.fusion_method ||
      "calibrated_dynamic_weighted_late_fusion";
  }

  setText(evidenceSource, inferEvidenceSource(data));
  setText(capturedKeys, String(countKeydowns()));
  setText(uploadedModalities, inferUploadedModalities(data));
  setText(fusionWeights, formatFusionDiagnostics(data));

  if (textModelInfo) {
    const modelName =
      data.text_result?.model_name ||
      data.text_result?.method ||
      "text_model.joblib";

    const macroF1 =
      data.text_result?.training_macro_f1 != null
        ? ` | macro F1: ${formatScore(data.text_result.training_macro_f1, 3)}`
        : "";

    textModelInfo.textContent = `${modelName}${macroF1}`;
  }

  if (trainedFusionUsed) {
    trainedFusionUsed.textContent = data.trained_fusion_model_used
      ? "Yes"
      : "No, calibrated dynamic late fusion";
  }

  setText(visualReliability, formatVisualReliability(data));

  if (resultJson) {
    resultJson.textContent = JSON.stringify(data, null, 2);
  }

  if (resultBox) {
    resultBox.classList.remove("hidden");
    animateResultCards();
    resultBox.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// ============================================================
// Export logic
// ============================================================

function requireLatestResult() {
  if (!state.latestResult) {
    setStatus("No result available to export yet.", "error");
    return null;
  }

  return state.latestResult;
}

function flattenObject(obj, prefix = "", output = {}) {
  if (!obj || typeof obj !== "object") {
    output[prefix || "value"] = obj;
    return output;
  }

  Object.entries(obj).forEach(([key, value]) => {
    const safeKey = prefix ? `${prefix}.${key}` : key;

    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value)
    ) {
      flattenObject(value, safeKey, output);
    } else if (Array.isArray(value)) {
      output[safeKey] = JSON.stringify(value);
    } else {
      output[safeKey] = value;
    }
  });

  return output;
}

function csvEscape(value) {
  const text = String(value ?? "");

  if (/[",\n\r]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`;
  }

  return text;
}

function exportJson() {
  const data = requireLatestResult();
  if (!data) return;

  const filename = `sensefuze_result_${timestampForFilename()}.json`;
  const content = JSON.stringify(data, null, 2);

  downloadBlob(content, filename, "application/json");
  setStatus("JSON export generated.", "success");
}

function exportCsv() {
  const data = requireLatestResult();
  if (!data) return;

  const flat = flattenObject(data);
  const headers = Object.keys(flat);
  const values = headers.map((header) => csvEscape(flat[header]));

  const content = `${headers.map(csvEscape).join(",")}\n${values.join(",")}\n`;
  const filename = `sensefuze_result_${timestampForFilename()}.csv`;

  downloadBlob(content, filename, "text/csv");
  setStatus("CSV export generated.", "success");
}

function buildTextReport(data) {
  const prediction = formatEvidenceLabel(data.final_prediction);
  const confidence = formatPct(getTopScore(data.final_scores));

  return [
    "SenseFuzeAI Behaviour Analysis Report",
    "====================================",
    "",
    `Generated at: ${new Date().toISOString()}`,
    "",
    "Final Decision",
    "--------------",
    `Prediction: ${prediction}`,
    `Confidence: ${confidence}`,
    "",
    "Final Interpretation",
    "--------------------",
    buildFinalInterpretation(data),
    "",
    "Final Scores",
    "------------",
    ...getRankedScores(data.final_scores).map(
      (item) => `${formatEvidenceLabel(item.label)}: ${formatPct(item.score)}`
    ),
    "",
    "Modality Evidence",
    "-----------------",
    `Keystroke: ${summariseKeystrokeResult(data)}`,
    `Text: ${summariseTextResult(data)}`,
    `Audio: ${summariseAudioResult(data)}`,
    `Image: ${summariseImageResult(data)}`,
    "",
    "Fusion Diagnostics",
    "------------------",
    `Fusion method: ${data.fusion_method || "calibrated_dynamic_weighted_late_fusion"}`,
    `Uploaded modalities: ${inferUploadedModalities(data)}`,
    `Fusion diagnostics: ${formatFusionDiagnostics(data)}`,
    `Visual reliability: ${formatVisualReliability(data)}`,
    "",
    "Methodological Note",
    "-------------------",
    "This system provides explainable behavioural-state decision support. It is not a medical or psychological diagnosis."
  ].join("\n");
}

function exportTxt() {
  const data = requireLatestResult();
  if (!data) return;

  const filename = `sensefuze_report_${timestampForFilename()}.txt`;
  const content = buildTextReport(data);

  downloadBlob(content, filename, "text/plain");
  setStatus("Text report export generated.", "success");
}

if (exportJsonBtn) {
  exportJsonBtn.addEventListener("click", exportJson);
}

if (exportCsvBtn) {
  exportCsvBtn.addEventListener("click", exportCsv);
}

if (exportTxtBtn) {
  exportTxtBtn.addEventListener("click", exportTxt);
}

// ============================================================
// Reset
// ============================================================

function resetUi() {
  if (form) form.reset();
  if (textarea) textarea.value = "";

  state.keystrokeEvents.length = 0;
  state.activeKeys.clear();
  state.latestResult = null;

  clearAudioPreview();
  clearImagePreview();

  setText(audioFileName, "No file selected");
  setText(imageFileName, "No file selected");

  if (resultBox) resultBox.classList.add("hidden");

  if (resultJson) {
    resultJson.textContent = "";
    resultJson.classList.add("hidden");
  }

  setText(finalPrediction, "-");
  setText(finalInterpretation, "Run an analysis to generate an explainable behavioural interpretation.");
  setText(heroPrediction, "Ready");
  setText(heroConfidence, "-");
  setText(sidebarConfidence, "Waiting");
  setText(confidenceValue, "-");

  if (confidenceFill) {
    confidenceFill.style.width = "0%";
  }

  setText(keystrokeResult, "-");
  setText(keystrokeDetail, "Waiting for captured timing evidence.");
  setText(textResult, "-");
  setText(textSentimentDetail, "Sentiment: -");
  setText(audioResult, "-");
  setText(audioDetail, "Audio file optional.");
  setText(imageResult, "-");
  setText(imageDetail, "Image file optional.");

  setText(behaviourSummary, "No behavioural summary available yet.");
  setText(imageCaptionSummary, "No image analysis has been run yet.");
  setText(audioEvidenceSummary, "No audio analysis has been run yet.");
  setText(keystrokeEvidenceSummary, "No keystroke evidence has been submitted yet.");

  setText(fusionMethod, "-");
  setText(evidenceSource, "-");
  setText(capturedKeys, "-");
  setText(uploadedModalities, "-");
  setText(fusionWeights, "-");
  setText(textModelInfo, "-");
  setText(trainedFusionUsed, "-");
  setText(visualReliability, "-");

  [
    scoreBars,
    keystrokeScoreBars,
    textScoreBars,
    audioScoreBars,
    imageScoreBars
  ].forEach((container) => {
    if (container) container.innerHTML = "";
  });

  setPredictionBadge(null);
  updateTypingCounter();
  setStatus("Ready", "idle");
  setProcessingState(false);
}

if (resetBtn) {
  resetBtn.addEventListener("click", resetUi);
}

if (toggleJsonBtn && resultJson) {
  toggleJsonBtn.addEventListener("click", () => {
    resultJson.classList.toggle("hidden");
  });
}

// ============================================================
// Form submission
// ============================================================

if (form && textarea) {
  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const typedText = textarea.value.trim();
    const keydownCount = countKeydowns();

    if (!typedText) {
      setStatus("Please enter typed text first.", "error");
      textarea.focus();
      return;
    }

    if (keydownCount < 5) {
      setStatus("Please type naturally in the text box before analysing.", "error");
      textarea.focus();
      return;
    }

    setProcessingState(true);
    setStatus("Processing multimodal signals...", "loading");

    if (resultBox) {
      resultBox.classList.add("hidden");
    }

    const formData = new FormData();
    formData.append("typed_text", typedText);
    formData.append("keystroke_events_json", JSON.stringify(state.keystrokeEvents));

    if (audioInput && audioInput.files && audioInput.files.length > 0) {
      formData.append("audio_file", audioInput.files[0]);
    }

    if (imageInput && imageInput.files && imageInput.files.length > 0) {
      formData.append("image_file", imageInput.files[0]);
    }

    try {
      const response = await fetch("/analyze", {
        method: "POST",
        body: formData
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || data.message || "Analysis request failed");
      }

      setStatus(`Prediction complete: ${formatEvidenceLabel(data.final_prediction)}`, "success");
      renderResults(data);
    } catch (error) {
      console.error("Analyze request failed:", error);
      setStatus(`Error: ${error.message}`, "error");
    } finally {
      setProcessingState(false);
    }
  });
}

// ============================================================
// Initialisation
// ============================================================

updateTypingCounter();
setPredictionBadge(null);
