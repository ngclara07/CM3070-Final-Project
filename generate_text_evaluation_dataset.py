# generate_text_evaluation_dataset.py
# SenseFuzeAI - Text Evaluation Dataset Generator
#
# Generates a balanced labelled text dataset for evaluating the
# live text component of SenseFuzeAI.

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "processed_text_dataset.csv"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


TEXT_SAMPLES = {
    "focused": [
        "I am calm and concentrating well on my work.",
        "I feel clear and steady while completing this task.",
        "I can stay focused on one activity without difficulty.",
        "My thoughts are organised and I know what to do next.",
        "I am working productively and maintaining attention.",
        "I feel mentally ready and able to continue.",
        "I am following the task clearly without getting distracted.",
        "I feel in control of my work and my attention is stable.",
        "I can process the information and respond carefully.",
        "I am focused and comfortable with the current task.",
        "I understand the instructions and can continue confidently.",
        "I am paying attention and making steady progress.",
        "I feel alert and able to complete the work.",
        "My concentration is stable and I am not rushing.",
        "I can think clearly and stay engaged with the task.",
        "I am working at a comfortable pace with good focus.",
        "I know what I need to do and can continue smoothly.",
        "I am attentive and able to follow the task requirements.",
        "I feel balanced and ready to continue working.",
        "My focus is steady and I am completing the task properly.",
    ],
    "distracted": [
        "I keep losing attention and checking other things.",
        "My mind keeps drifting away from the task.",
        "I am finding it hard to stay focused on this work.",
        "I keep switching between unrelated activities.",
        "I am easily interrupted and cannot maintain attention.",
        "I keep forgetting what I was supposed to do next.",
        "My attention keeps moving away from the screen.",
        "I am not concentrating properly on the task.",
        "I feel unfocused and keep looking for distractions.",
        "I cannot stay with one task for very long.",
        "I keep pausing because my attention is elsewhere.",
        "I am distracted by other thoughts while trying to work.",
        "I keep moving away from the task before finishing it.",
        "I am struggling to pay attention to the current activity.",
        "I keep checking unrelated things instead of continuing.",
        "My focus keeps breaking while I am working.",
        "I am having difficulty staying engaged with the task.",
        "I keep losing track of what I am doing.",
        "My attention is scattered across too many unrelated things.",
        "I am not fully present while completing this task.",
    ],
    "fatigued": [
        "I feel tired and mentally drained.",
        "My reactions feel slower than usual.",
        "I have low energy and it is hard to continue working.",
        "I feel sleepy and less alert than normal.",
        "My mind feels exhausted after working for a long time.",
        "I am struggling to keep going because I feel tired.",
        "I feel mentally worn out and need a break.",
        "It takes more effort than usual to think clearly.",
        "I feel slow and my concentration is weakening.",
        "I am too tired to work effectively right now.",
        "I feel physically tired and mentally slow.",
        "My energy level is low and I am losing alertness.",
        "I feel exhausted and my thinking feels slower.",
        "I am finding it difficult to continue because I am tired.",
        "My mind feels heavy and I need rest.",
        "I feel less responsive than usual.",
        "I am becoming tired after focusing for a long time.",
        "I feel drained and my attention is fading.",
        "I am not as alert as I normally am.",
        "I feel worn down and need time to recover.",
    ],
    "overloaded": [
        "There are too many tasks happening at once.",
        "I feel overwhelmed by the amount of information.",
        "I cannot handle all the tasks being given to me.",
        "There is too much pressure and I feel stressed.",
        "I feel overloaded because many things need attention.",
        "The workload feels too large to manage properly.",
        "I am stressed because there is too much to process.",
        "I feel pressured by the number of things happening.",
        "I cannot decide what to focus on because everything feels urgent.",
        "The information and tasks feel overwhelming right now.",
        "I feel overloaded by the amount of work I need to complete.",
        "There are too many instructions to process at once.",
        "I feel stressed because several things require attention.",
        "I am struggling because everything feels urgent.",
        "The task feels too complex and demanding right now.",
        "I feel pressure from having too many responsibilities.",
        "I cannot manage all the information being presented.",
        "I feel mentally overloaded by the current workload.",
        "Too many things are happening and I cannot keep up.",
        "I feel overwhelmed because the work is piling up.",
    ],
}


def validate_samples(samples: dict[str, list[str]]) -> None:
    expected_labels = {"focused", "distracted", "fatigued", "overloaded"}

    actual_labels = set(samples.keys())

    if actual_labels != expected_labels:
        raise ValueError(
            f"Labels must be exactly {expected_labels}, but got {actual_labels}"
        )

    for label, texts in samples.items():
        if len(texts) < 20:
            raise ValueError(
                f"Label '{label}' has only {len(texts)} samples. "
                "At least 20 samples are required."
            )

        if len(set(texts)) != len(texts):
            raise ValueError(f"Duplicate text samples found in label '{label}'.")


def generate_dataset(samples_per_class: int = 20) -> pd.DataFrame:
    validate_samples(TEXT_SAMPLES)

    rows = []

    for label, texts in TEXT_SAMPLES.items():
        selected_texts = texts[:samples_per_class]

        for text in selected_texts:
            rows.append(
                {
                    "text": text,
                    "label": label,
                }
            )

    df = pd.DataFrame(rows)

    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


def main() -> None:
    df = generate_dataset(samples_per_class=20)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print("Text evaluation dataset generated successfully.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(f"Total samples: {len(df)}")
    print()
    print("Class distribution:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
