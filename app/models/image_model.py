# === app/models/image_model.py ===
# SenseFuzeAI - Vision Runtime Model
#
# This module provides image / vision analysis for SenseFuzeAI.
#
# It combines:
#   - CLIP zero-shot behavioural prompt classification
#   - BLIP generic image captioning
#   - MediaPipe face/body/posture detection
#   - behaviour-aware visual cue caption generation
#
# Main public function:
#   analyze_image_file(path: str) -> dict

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import re
import warnings

import numpy as np
from PIL import Image, ImageOps

import torch
from transformers import (
    pipeline,
    BlipProcessor,
    BlipForConditionalGeneration,
)


# ============================================================
# Configuration
# ============================================================

CLASS_LABELS = ["focused", "distracted", "fatigued", "overloaded"]

BASE_DIR = Path(__file__).resolve().parents[2]

CLIP_MODEL = str(BASE_DIR / "models" / "clip-vit-large-patch14")
CAPTION_MODEL = str(BASE_DIR / "models" / "blip-image-captioning-large")

MIN_DETECTION_CONFIDENCE = 0.35
MIN_VISIBILITY = 0.25

DEVICE = 0 if torch.cuda.is_available() else -1
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# CLIP behavioural prompts
# ============================================================

BEHAVIOUR_PROMPTS = {
    "focused": [
        "a focused person working attentively at a desk",
        "a person concentrating on a laptop or computer",
        "a person looking at a screen while working",
        "a person with upright posture focused on work",
        "a calm person engaged in a single task",
        "a productive workspace with a person working steadily",
    ],
    "distracted": [
        "a distracted person looking away from work",
        "a person using a phone while working",
        "a person multitasking with distractions",
        "a person not paying attention to the main task",
        "a person interrupted while trying to work",
        "a workspace with visible distractions",
    ],
    "fatigued": [
        "a tired sleepy person working at a desk",
        "a person yawning or rubbing eyes",
        "a person with slouched tired posture",
        "an exhausted person working late",
        "a person resting their head while working",
        "a dim tired working environment",
    ],
    "overloaded": [
        "a stressed person overwhelmed by work",
        "a person frustrated by workload",
        "a person overloaded by documents and tasks",
        "a person surrounded by many work demands",
        "a cluttered desk with many papers and screens",
        "a chaotic workspace with signs of stress",
    ],
}

FACE_PROMPTS = {
    "focused": [
        "attentive focused face",
        "neutral concentrated facial expression",
        "face looking toward the work or screen",
        "calm focused expression",
    ],
    "distracted": [
        "distracted face looking away",
        "inattentive facial expression",
        "face turned away from the screen",
        "person looking elsewhere instead of working",
    ],
    "fatigued": [
        "tired sleepy face",
        "yawning exhausted face",
        "heavy eyes or fatigued expression",
        "face showing tiredness",
    ],
    "overloaded": [
        "stressed overwhelmed face",
        "frustrated tense facial expression",
        "anxious or pressured expression",
        "face showing stress from workload",
    ],
}

BODY_PROMPTS = {
    "focused": [
        "upright attentive working posture",
        "person sitting straight while working",
        "body posture facing the workstation",
        "stable engaged body posture",
    ],
    "distracted": [
        "person turned away from work",
        "person distracted by phone or surroundings",
        "body posture not oriented toward the task",
        "person physically disengaged from work",
    ],
    "fatigued": [
        "slouched tired posture",
        "head supported by hand while working",
        "body posture showing fatigue",
        "person leaning forward tiredly",
    ],
    "overloaded": [
        "tense stressed body posture",
        "person holding head stressed at desk",
        "body posture showing pressure or frustration",
        "person overwhelmed at a workstation",
    ],
}

SCENE_PROMPTS = {
    "focused": [
        "clean organized workspace",
        "tidy desk with minimal distractions",
        "calm productive workspace",
        "single laptop work setup",
        "simple workstation for focused work",
    ],
    "distracted": [
        "workspace with phone distraction",
        "desk with multiple devices and interruptions",
        "busy distracting workspace",
        "workspace with off task distractions",
        "person distracted by surrounding activity",
    ],
    "fatigued": [
        "dim late night workspace",
        "dark tired working atmosphere",
        "workspace with low light and screen glow",
        "sleepy tired working environment",
    ],
    "overloaded": [
        "messy cluttered desk with many papers",
        "chaotic workspace with documents and screens",
        "overloaded workspace with many tasks",
        "desk covered with workload and clutter",
        "stressful workspace full of tasks",
    ],
}


# ============================================================
# Lazy model loading
# ============================================================

_CLIP_CLASSIFIER = None
_BLIP_PROCESSOR = None
_BLIP_MODEL = None

_MEDIAPIPE_INITIALISED = False
_MEDIAPIPE_AVAILABLE = False
_MP_FACE_DETECTION = None
_MP_POSE = None


def get_clip_classifier():
    global _CLIP_CLASSIFIER

    if _CLIP_CLASSIFIER is None:
        _CLIP_CLASSIFIER = pipeline(
            task="zero-shot-image-classification",
            model=CLIP_MODEL,
            device=DEVICE,
        )

    return _CLIP_CLASSIFIER


def get_blip():
    global _BLIP_PROCESSOR
    global _BLIP_MODEL

    if _BLIP_PROCESSOR is None or _BLIP_MODEL is None:
        _BLIP_PROCESSOR = BlipProcessor.from_pretrained(CAPTION_MODEL)
        _BLIP_MODEL = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
        _BLIP_MODEL.to(TORCH_DEVICE)
        _BLIP_MODEL.eval()

    return _BLIP_PROCESSOR, _BLIP_MODEL


def get_mediapipe():
    global _MEDIAPIPE_INITIALISED
    global _MEDIAPIPE_AVAILABLE
    global _MP_FACE_DETECTION
    global _MP_POSE

    if not _MEDIAPIPE_INITIALISED:
        _MEDIAPIPE_INITIALISED = True

        try:
            import mediapipe as mp

            _MP_FACE_DETECTION = mp.solutions.face_detection
            _MP_POSE = mp.solutions.pose
            _MEDIAPIPE_AVAILABLE = True

        except Exception as error:
            _MEDIAPIPE_AVAILABLE = False
            _MP_FACE_DETECTION = None
            _MP_POSE = None
            warnings.warn(f"MediaPipe unavailable: {error}")

    return _MEDIAPIPE_AVAILABLE, _MP_FACE_DETECTION, _MP_POSE


# ============================================================
# Basic utilities
# ============================================================

def load_image(path: str | Path) -> Image.Image:
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def normalise_scores(scores: Dict[str, float]) -> Dict[str, float]:
    clean = {
        label: max(float(scores.get(label, 0.0)), 0.0)
        for label in CLASS_LABELS
    }

    total = sum(clean.values())

    if total <= 0:
        return {label: 1.0 / len(CLASS_LABELS) for label in CLASS_LABELS}

    return {label: value / total for label, value in clean.items()}


def flatten_prompt_dict(
    prompt_dict: Dict[str, List[str]],
) -> Tuple[List[str], Dict[str, str]]:
    prompts: List[str] = []
    prompt_to_label: Dict[str, str] = {}

    for label, prompt_list in prompt_dict.items():
        for prompt in prompt_list:
            prompts.append(prompt)
            prompt_to_label[prompt] = label

    return prompts, prompt_to_label


def classify_with_prompts(
    image: Image.Image,
    prompt_dict: Dict[str, List[str]],
) -> Dict[str, Any]:
    classifier = get_clip_classifier()
    prompts, prompt_to_label = flatten_prompt_dict(prompt_dict)

    results = classifier(
        image,
        candidate_labels=prompts,
    )

    label_scores = {label: 0.0 for label in prompt_dict.keys()}

    for result in results:
        prompt = result["label"]
        score = float(result["score"])
        label = prompt_to_label[prompt]
        label_scores[label] = max(label_scores[label], score)

    label_scores = normalise_scores(label_scores)

    sorted_scores = sorted(
        label_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_label, best_score = sorted_scores[0]
    second_label, second_score = sorted_scores[1]

    return {
        "best_label": best_label,
        "best_score": float(best_score),
        "second_label": second_label,
        "second_score": float(second_score),
        "margin": float(best_score - second_score),
        "scores": label_scores,
    }


def generate_caption(image: Image.Image) -> str:
    processor, model = get_blip()

    inputs = processor(
        images=image,
        return_tensors="pt",
    ).to(TORCH_DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=60,
            num_beams=5,
            repetition_penalty=1.2,
            length_penalty=1.0,
            early_stopping=True,
        )

    caption = processor.decode(
        output_ids[0],
        skip_special_tokens=True,
    ).strip()

    return caption


def caption_image_file(path: str) -> str:
    image = load_image(path)
    return generate_caption(image)


def crop_with_padding(
    image: Image.Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad_ratio: float,
) -> Optional[Image.Image]:
    width, height = image.size

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return None

    pad_x = int(box_w * pad_ratio)
    pad_y = int(box_h * pad_ratio)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return None

    return image.crop((x1, y1, x2, y2))


# ============================================================
# MediaPipe face detection
# ============================================================

def detect_face(image: Image.Image) -> Dict[str, Any]:
    available, mp_face_detection, _ = get_mediapipe()

    if not available:
        return {
            "face_detected": False,
            "face_confidence": 0.0,
            "face_crop": None,
            "face_box": "",
        }

    width, height = image.size
    image_np = np.array(image)

    best_detection = None
    best_score = 0.0

    for model_selection in [1, 0]:
        with mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        ) as detector:
            results = detector.process(image_np)

            if results.detections:
                for detection in results.detections:
                    score = float(detection.score[0]) if detection.score else 0.0

                    if score > best_score:
                        best_score = score
                        best_detection = detection

    if best_detection is None:
        return {
            "face_detected": False,
            "face_confidence": 0.0,
            "face_crop": None,
            "face_box": "",
        }

    box = best_detection.location_data.relative_bounding_box

    x1 = int(box.xmin * width)
    y1 = int(box.ymin * height)
    x2 = int((box.xmin + box.width) * width)
    y2 = int((box.ymin + box.height) * height)

    face_crop = crop_with_padding(
        image=image,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        pad_ratio=0.65,
    )

    return {
        "face_detected": True,
        "face_confidence": round(best_score, 4),
        "face_crop": face_crop,
        "face_box": f"{x1},{y1},{x2},{y2}",
    }


# ============================================================
# MediaPipe pose/body detection
# ============================================================

def detect_pose_and_body_crop(image: Image.Image) -> Dict[str, Any]:
    available, _, mp_pose = get_mediapipe()

    if not available:
        return {
            "body_detected": False,
            "body_crop": None,
            "posture_cue": "mediapipe_unavailable",
            "head_position_cue": "unknown",
            "shoulder_visibility": 0.0,
            "pose_visibility": 0.0,
            "body_box": "",
        }

    width, height = image.size
    image_np = np.array(image)

    best_landmarks = None
    best_visibility = 0.0

    for complexity in [2, 1, 0]:
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=complexity,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        ) as pose:
            results = pose.process(image_np)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                visibility = float(np.mean([lm.visibility for lm in landmarks]))

                if visibility > best_visibility:
                    best_visibility = visibility
                    best_landmarks = landmarks

    if best_landmarks is None:
        return {
            "body_detected": False,
            "body_crop": None,
            "posture_cue": "not_detected",
            "head_position_cue": "unknown",
            "shoulder_visibility": 0.0,
            "pose_visibility": 0.0,
            "body_box": "",
        }

    visible_points = [
        (int(lm.x * width), int(lm.y * height), lm.visibility)
        for lm in best_landmarks
        if lm.visibility >= MIN_VISIBILITY
    ]

    if not visible_points:
        return {
            "body_detected": False,
            "body_crop": None,
            "posture_cue": "not_detected",
            "head_position_cue": "unknown",
            "shoulder_visibility": 0.0,
            "pose_visibility": round(best_visibility, 4),
            "body_box": "",
        }

    xs = [point[0] for point in visible_points]
    ys = [point[1] for point in visible_points]

    x1, x2 = max(0, min(xs)), min(width, max(xs))
    y1, y2 = max(0, min(ys)), min(height, max(ys))

    body_crop = crop_with_padding(
        image=image,
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        pad_ratio=0.35,
    )

    nose = best_landmarks[mp_pose.PoseLandmark.NOSE.value]
    left_shoulder = best_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = best_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    shoulder_visibility = float(
        (left_shoulder.visibility + right_shoulder.visibility) / 2
    )

    shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
    nose_y = nose.y

    if shoulder_visibility < 0.25 or nose.visibility < 0.25:
        posture_cue = "partial_body_uncertain"
        head_position_cue = "uncertain"
    elif nose_y > shoulder_y - 0.04:
        posture_cue = "possible_slouched_or_head_down"
        head_position_cue = "low_head_position"
    else:
        posture_cue = "upright_or_neutral"
        head_position_cue = "normal_head_position"

    return {
        "body_detected": True,
        "body_crop": body_crop,
        "posture_cue": posture_cue,
        "head_position_cue": head_position_cue,
        "shoulder_visibility": round(shoulder_visibility, 4),
        "pose_visibility": round(best_visibility, 4),
        "body_box": f"{x1},{y1},{x2},{y2}",
    }


# ============================================================
# Visual fusion
# ============================================================

def compute_visual_evidence(
    behaviour_pred: Dict[str, Any],
    scene_pred: Dict[str, Any],
    face_pred: Optional[Dict[str, Any]],
    body_pred: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    weighted_scores = {label: 0.0 for label in CLASS_LABELS}

    weights = {
        "behaviour": 0.35,
        "scene": 0.25,
        "face": 0.20,
        "body": 0.20,
    }

    for label, score in behaviour_pred["scores"].items():
        weighted_scores[label] += float(score) * weights["behaviour"]

    for label, score in scene_pred["scores"].items():
        weighted_scores[label] += float(score) * weights["scene"]

    if face_pred is not None:
        for label, score in face_pred["scores"].items():
            weighted_scores[label] += float(score) * weights["face"]

    if body_pred is not None:
        for label, score in body_pred["scores"].items():
            weighted_scores[label] += float(score) * weights["body"]

    weighted_scores = normalise_scores(weighted_scores)

    sorted_items = sorted(
        weighted_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    best_label, best_score = sorted_items[0]
    second_label, second_score = sorted_items[1]

    return {
        "visual_fused_label": best_label,
        "visual_fused_score": float(best_score),
        "visual_fused_second_label": second_label,
        "visual_fused_second_score": float(second_score),
        "visual_fused_margin": float(best_score - second_score),
        "visual_fused_scores": weighted_scores,
    }


def compute_reliability_score(
    fused_pred: Dict[str, Any],
    face_info: Dict[str, Any],
    pose_info: Dict[str, Any],
) -> float:
    margin = float(fused_pred.get("visual_fused_margin", 0.0))

    margin_component = min(max(margin * 3.0, 0.0), 1.0)
    face_component = 1.0 if face_info.get("face_detected") else 0.0
    body_component = 1.0 if pose_info.get("body_detected") else 0.0

    reliability = (
        0.60 * margin_component
        + 0.20 * face_component
        + 0.20 * body_component
    )

    return round(float(reliability), 4)


def build_quality_flag(reliability_score: float) -> str:
    if reliability_score >= 0.80:
        return "high_reliability"
    if reliability_score >= 0.60:
        return "moderate_reliability"
    if reliability_score >= 0.40:
        return "low_reliability_review"
    return "weak_or_ambiguous_review"


# ============================================================
# Behaviour-aware cue generation
# ============================================================

def _caption_has_any(caption: str, terms: List[str]) -> bool:
    text = str(caption).lower()
    return any(term in text for term in terms)


def _prediction_block(
    prediction: Optional[Dict[str, Any]],
    default_label: str = "not_detected",
) -> Tuple[str, float, float]:
    if not prediction:
        return default_label, 0.0, 0.0

    return (
        str(prediction.get("best_label", default_label)),
        float(prediction.get("best_score", 0.0)),
        float(prediction.get("margin", 0.0)),
    )


def evidence_strength_from_reliability(reliability_score: float) -> str:
    if reliability_score >= 0.80:
        return "strong"
    if reliability_score >= 0.60:
        return "moderate"
    if reliability_score >= 0.40:
        return "limited"
    return "weak"


def build_visual_cue_summary(
    behaviour_pred: Dict[str, Any],
    scene_pred: Dict[str, Any],
    face_pred: Optional[Dict[str, Any]],
    body_pred: Optional[Dict[str, Any]],
    fused_pred: Dict[str, Any],
    face_info: Dict[str, Any],
    pose_info: Dict[str, Any],
    generic_caption: str = "",
    reliability_score: float = 0.0,
    quality_flag: str = "unknown",
) -> Dict[str, Any]:
    final_label = str(fused_pred.get("visual_fused_label", "unknown"))
    final_score = float(fused_pred.get("visual_fused_score", 0.0))
    final_margin = float(fused_pred.get("visual_fused_margin", 0.0))

    behaviour_label, behaviour_score, behaviour_margin = _prediction_block(behaviour_pred)
    scene_label, scene_score, scene_margin = _prediction_block(scene_pred)
    face_label, face_score, face_margin = _prediction_block(face_pred)
    body_label, body_score, body_margin = _prediction_block(body_pred)

    posture_cue = str(pose_info.get("posture_cue", "unknown"))
    head_position_cue = str(pose_info.get("head_position_cue", "unknown"))

    supporting_cues: List[str] = []
    possible_conflicting_cues: List[str] = []
    absent_or_uncertain_cues: List[str] = []

    caption = generic_caption or ""

    if _caption_has_any(caption, ["desk", "laptop", "computer", "screen", "workstation"]):
        supporting_cues.append("visible workstation or laptop context")

    if posture_cue == "upright_or_neutral":
        supporting_cues.append("upright or neutral working posture")

    if head_position_cue == "normal_head_position":
        supporting_cues.append("normal head position consistent with task engagement")

    if posture_cue == "possible_slouched_or_head_down":
        supporting_cues.append("low head position or slouched posture is visible")

    if face_info.get("face_detected"):
        supporting_cues.append("face is detected, allowing facial attention evidence")

    if pose_info.get("body_detected"):
        supporting_cues.append("body pose is detected, allowing posture evidence")

    if behaviour_label == final_label:
        supporting_cues.append(f"full-image behavioural cue supports {final_label}")

    if scene_label == final_label:
        supporting_cues.append(f"scene cue supports {final_label}")

    if face_label == final_label:
        supporting_cues.append(f"face cue supports {final_label}")

    if body_label == final_label:
        supporting_cues.append(f"body cue supports {final_label}")

    for cue_name, cue_label, cue_score, cue_margin in [
        ("full-image cue", behaviour_label, behaviour_score, behaviour_margin),
        ("scene cue", scene_label, scene_score, scene_margin),
        ("face cue", face_label, face_score, face_margin),
        ("body cue", body_label, body_score, body_margin),
    ]:
        if cue_label in CLASS_LABELS and cue_label != final_label:
            if cue_score >= 0.55 and cue_margin >= 0.12:
                possible_conflicting_cues.append(
                    f"{cue_name} leans toward {cue_label}"
                )

    if not face_info.get("face_detected"):
        absent_or_uncertain_cues.append("face was not detected clearly")

    if not pose_info.get("body_detected"):
        absent_or_uncertain_cues.append("body pose was not detected clearly")

    if face_label == "not_detected":
        absent_or_uncertain_cues.append("facial behavioural cue unavailable")

    if body_label == "not_detected":
        absent_or_uncertain_cues.append("body behavioural cue unavailable")

    if final_margin < 0.10:
        absent_or_uncertain_cues.append("final visual margin is small")

    if not _caption_has_any(caption, ["phone", "mobile"]):
        if final_label == "focused":
            supporting_cues.append("no strong phone-distraction cue is visible in the caption")

    if not _caption_has_any(caption, ["messy", "clutter", "papers", "crowded"]):
        if final_label in {"focused", "distracted"}:
            supporting_cues.append("no strong clutter or overload cue is visible in the caption")

    return {
        "final_visual_label": final_label,
        "final_visual_score": round(final_score, 4),
        "final_visual_margin": round(final_margin, 4),
        "evidence_strength": evidence_strength_from_reliability(reliability_score),

        "full_image_cue": behaviour_label,
        "full_image_score": round(behaviour_score, 4),
        "full_image_margin": round(behaviour_margin, 4),

        "scene_cue": scene_label,
        "scene_score": round(scene_score, 4),
        "scene_margin": round(scene_margin, 4),

        "face_cue": face_label,
        "face_score": round(face_score, 4),
        "face_margin": round(face_margin, 4),

        "body_cue": body_label,
        "body_score": round(body_score, 4),
        "body_margin": round(body_margin, 4),

        "face_detected": bool(face_info.get("face_detected", False)),
        "body_detected": bool(pose_info.get("body_detected", False)),
        "posture_cue": posture_cue,
        "head_position_cue": head_position_cue,

        "supporting_cues": supporting_cues,
        "possible_conflicting_cues": possible_conflicting_cues,
        "absent_or_uncertain_cues": absent_or_uncertain_cues,

        "reliability_score": round(float(reliability_score), 4),
        "quality_flag": quality_flag,
    }


def generate_behaviour_caption(
    generic_caption: str,
    visual_cue_summary: Dict[str, Any],
) -> str:
    final_label = str(visual_cue_summary.get("final_visual_label", "unknown"))
    evidence_strength = str(visual_cue_summary.get("evidence_strength", "unknown"))
    reliability_score = float(visual_cue_summary.get("reliability_score", 0.0))
    quality_flag = str(visual_cue_summary.get("quality_flag", "unknown"))

    supporting_cues = visual_cue_summary.get("supporting_cues", []) or []
    conflicting_cues = visual_cue_summary.get("possible_conflicting_cues", []) or []
    uncertain_cues = visual_cue_summary.get("absent_or_uncertain_cues", []) or []

    posture_cue = str(visual_cue_summary.get("posture_cue", "unknown"))
    head_position_cue = str(visual_cue_summary.get("head_position_cue", "unknown"))

    label_explanations = {
        "focused": (
            "The visual evidence is most consistent with a focused state. "
            "This is supported by task-oriented work context, stable body posture, "
            "and cues suggesting attention toward the workstation."
        ),
        "distracted": (
            "The visual evidence is most consistent with a distracted state. "
            "This may be indicated by gaze or body orientation away from the task, "
            "phone or device-related interruptions, or competing visual activity."
        ),
        "fatigued": (
            "The visual evidence is most consistent with a fatigued state. "
            "This may be indicated by tired posture, a low head position, dim working "
            "conditions, or signs of reduced physical alertness."
        ),
        "overloaded": (
            "The visual evidence is most consistent with an overloaded state. "
            "This may be indicated by clutter, many visible work demands, tense posture, "
            "or a workspace suggesting high task pressure."
        ),
    }

    caption_parts: List[str] = []

    if generic_caption:
        caption_parts.append(
            f"The generic image caption is: '{generic_caption}'."
        )

    caption_parts.append(
        label_explanations.get(
            final_label,
            "The visual evidence produces an uncertain behavioural interpretation.",
        )
    )

    if supporting_cues:
        selected_support = supporting_cues[:5]
        caption_parts.append(
            "Key supporting visual cues include "
            + "; ".join(selected_support)
            + "."
        )

    if posture_cue == "upright_or_neutral":
        caption_parts.append(
            "The posture appears upright or neutral, which can support task engagement."
        )
    elif posture_cue == "possible_slouched_or_head_down":
        caption_parts.append(
            "The posture suggests a low head position or possible slouching, which may indicate fatigue or reduced alertness."
        )

    if head_position_cue == "normal_head_position":
        caption_parts.append(
            "The head position appears normal, which can be consistent with attentive work."
        )
    elif head_position_cue == "low_head_position":
        caption_parts.append(
            "The head position appears low, which may support fatigue-related interpretation."
        )

    if conflicting_cues:
        caption_parts.append(
            "Possible conflicting cues are also present: "
            + "; ".join(conflicting_cues[:4])
            + "."
        )

    if uncertain_cues:
        caption_parts.append(
            "Some evidence is unavailable or uncertain: "
            + "; ".join(uncertain_cues[:4])
            + "."
        )

    caption_parts.append(
        f"Overall visual evidence strength is {evidence_strength}, "
        f"with reliability score {reliability_score:.2f} and quality flag '{quality_flag}'."
    )

    caption_parts.append(
        "This interpretation should be treated as visual behavioural evidence and fused with text, keystroke, and audio signals for the final system decision."
    )

    return " ".join(caption_parts)


# ============================================================
# Public inference
# ============================================================

def analyze_image_file(path: str) -> Dict[str, Any]:
    image_path = Path(path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    image = load_image(image_path)
    width, height = image.size

    face_info = detect_face(image)
    pose_info = detect_pose_and_body_crop(image)

    behaviour_pred = classify_with_prompts(image, BEHAVIOUR_PROMPTS)
    scene_pred = classify_with_prompts(image, SCENE_PROMPTS)

    face_pred = (
        classify_with_prompts(face_info["face_crop"], FACE_PROMPTS)
        if face_info.get("face_crop") is not None
        else None
    )

    body_pred = (
        classify_with_prompts(pose_info["body_crop"], BODY_PROMPTS)
        if pose_info.get("body_crop") is not None
        else None
    )

    fused_pred = compute_visual_evidence(
        behaviour_pred=behaviour_pred,
        scene_pred=scene_pred,
        face_pred=face_pred,
        body_pred=body_pred,
    )

    reliability_score = compute_reliability_score(
        fused_pred=fused_pred,
        face_info=face_info,
        pose_info=pose_info,
    )

    quality_flag = build_quality_flag(reliability_score)

    generic_caption = generate_caption(image)

    visual_cue_summary = build_visual_cue_summary(
        behaviour_pred=behaviour_pred,
        scene_pred=scene_pred,
        face_pred=face_pred,
        body_pred=body_pred,
        fused_pred=fused_pred,
        face_info=face_info,
        pose_info=pose_info,
        generic_caption=generic_caption,
        reliability_score=reliability_score,
        quality_flag=quality_flag,
    )

    behaviour_caption = generate_behaviour_caption(
        generic_caption=generic_caption,
        visual_cue_summary=visual_cue_summary,
    )

    return {
        "filepath": str(image_path),
        "filename": image_path.name,
        "width": width,
        "height": height,

        "caption": generic_caption,
        "generic_caption": generic_caption,
        "scene_description": generic_caption,

        "behaviour_caption": behaviour_caption,
        "behaviour_aware_caption": behaviour_caption,
        "behaviour_description": behaviour_caption,

        "visual_cue_summary": visual_cue_summary,
        "supporting_visual_cues": visual_cue_summary.get("supporting_cues", []),
        "possible_conflicting_cues": visual_cue_summary.get("possible_conflicting_cues", []),
        "absent_or_uncertain_cues": visual_cue_summary.get("absent_or_uncertain_cues", []),

        "predicted_label": fused_pred["visual_fused_label"],
        "predicted_behaviour": fused_pred["visual_fused_label"],
        "prediction_score": round(fused_pred["visual_fused_score"], 4),
        "prediction_margin": round(fused_pred["visual_fused_margin"], 4),

        "reliability_score": reliability_score,
        "quality_flag": quality_flag,

        "behaviour_prediction": behaviour_pred,
        "scene_prediction": scene_pred,
        "face_prediction": face_pred,
        "body_prediction": body_pred,
        "fused_prediction": fused_pred,

        "behaviour_scores": fused_pred["visual_fused_scores"],
        "scores": fused_pred["visual_fused_scores"],
        "class_probabilities": fused_pred["visual_fused_scores"],

        "face_detected": face_info["face_detected"],
        "face_confidence": face_info["face_confidence"],
        "face_box": face_info["face_box"],

        "body_detected": pose_info["body_detected"],
        "body_box": pose_info["body_box"],
        "posture_cue": pose_info["posture_cue"],
        "head_position_cue": pose_info["head_position_cue"],
        "pose_visibility": pose_info["pose_visibility"],
        "shoulder_visibility": pose_info["shoulder_visibility"],

        "method": (
            "clip_zero_shot_plus_blip_caption_plus_mediapipe_plus_"
            "behaviour_aware_visual_caption"
        ),
    }
