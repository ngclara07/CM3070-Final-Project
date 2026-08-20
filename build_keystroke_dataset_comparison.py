# === build_keystroke_dataset_comparison.py ===
#
# SenseFuzeAI
# EmoSurv IEEE + SenseFuzeAI Keystroke Dataset Harmonisation
#
# PURPOSE
# =============================================================================
#
# This script builds comparable keystroke datasets for the supervisor-
# recommended dataset-augmentation experiment.
#
# It DOES NOT modify:
#
#   data/EmoSurv_IEEE/
#   data/session_aligned/
#
# It creates derived harmonised datasets under:
#
#   data/processed/keystroke_dataset_comparison/
#
#
# PRIMARY EXPERIMENT
# =============================================================================
#
# Conservative three-class semantic proxy:
#
#   EmoSurv Calm  -> focused
#   EmoSurv Sad   -> fatigued
#   EmoSurv Angry -> overloaded
#
# Happy and Neutral are excluded.
#
#
# EXPLORATORY EXPERIMENT
# =============================================================================
#
# Existing four-class weakly supervised proxy:
#
#   Calm   -> focused
#   Angry  -> overloaded
#   Sad    -> fatigued
#
#   Happy / Neutral:
#       irregular timing -> distracted
#       otherwise        -> focused
#
# IMPORTANT:
#
# These EmoSurv behavioural labels are proxy labels.
# They are NOT original EmoSurv behavioural ground truth.
#
#
# FEATURE HARMONISATION
# =============================================================================
#
# SenseFuzeAI raw key-down/up events are re-extracted using exactly the
# same FEATURE_COLUMNS and extract_live_features() implementation used
# by keystroke_live_gui_emosurv_ieee.py.
#
# Therefore EmoSurv and SenseFuzeAI observations enter the comparison
# in one common 23-feature representation.
#
# =============================================================================


from __future__ import annotations

import argparse
import hashlib
import json
import math

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =============================================================================
# Reuse the EXISTING canonical EmoSurv comparison implementation
# =============================================================================

from keystroke_live_gui_emosurv_ieee import (
    BEHAVIOURAL_CLASSES,
    DD_MAX_MS,
    DD_MIN_MS,
    DWELL_MAX_MS,
    DWELL_MIN_MS,
    FEATURE_COLUMNS,
    MIN_WINDOW_SIZE,
    WINDOW_SIZE,
    WINDOW_STEP,
    assign_behaviour_proxy_labels,
    build_window_dataset,
    extract_live_features,
    load_emosurv_datasets,
    normalise_key,
)


# =============================================================================
# Paths
# =============================================================================

ROOT_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

SESSION_ROOT = (
    ROOT_DIR
    / "data"
    / "session_aligned"
)

KEYSTROKE_DIR = (
    SESSION_ROOT
    / "keystrokes"
)

LEGACY_SESSION_CSV = (
    SESSION_ROOT
    / "retroactive_keystroke_features.csv"
)

TEMPORAL_MANIFEST_PATH = (
    SESSION_ROOT
    / "temporal_session_manifest_v2.csv"
)

DEFAULT_PARTICIPANT_MAP_PATH = (
    SESSION_ROOT
    / "participant_map.csv"
)

OUTPUT_DIR = (
    ROOT_DIR
    / "data"
    / "processed"
    / "keystroke_dataset_comparison"
)


# =============================================================================
# Outputs
# =============================================================================

EMOSURV_3_PATH = (
    OUTPUT_DIR
    / "emosurv_harmonised_3class.csv"
)

SENSEFUZE_3_PATH = (
    OUTPUT_DIR
    / "sensefuzeai_harmonised_3class.csv"
)

COMBINED_3_PATH = (
    OUTPUT_DIR
    / "combined_harmonised_3class.csv"
)

EMOSURV_4_PATH = (
    OUTPUT_DIR
    / "emosurv_harmonised_4class.csv"
)

SENSEFUZE_4_PATH = (
    OUTPUT_DIR
    / "sensefuzeai_harmonised_4class.csv"
)

COMBINED_4_PATH = (
    OUTPUT_DIR
    / "combined_harmonised_4class.csv"
)

STATISTICS_PATH = (
    OUTPUT_DIR
    / "dataset_statistics.csv"
)

BUILD_METADATA_PATH = (
    OUTPUT_DIR
    / "build_metadata.json"
)

FEATURE_SCHEMA_PATH = (
    OUTPUT_DIR
    / "harmonised_feature_columns.json"
)


# =============================================================================
# Labels
# =============================================================================

THREE_CLASS_LABELS = (
    "focused",
    "fatigued",
    "overloaded",
)

FOUR_CLASS_LABELS = tuple(
    BEHAVIOURAL_CLASSES
)

DIRECT_EMOSURV_PROXY = {
    "C": "focused",
    "S": "fatigued",
    "A": "overloaded",
}


# =============================================================================
# Metadata columns
# =============================================================================

METADATA_COLUMNS = [
    "dataset_source",
    "participant_id",
    "session_id",
    "sample_id",
    "source_type",
    "label",
    "label_origin",
    "original_label",
    "window_index",
    "window_keystrokes",
]


# =============================================================================
# General utilities
# =============================================================================

def safe_float(
    value: Any,
    default: float = float("nan"),
) -> float:

    try:
        converted = float(
            value
        )

        if math.isfinite(
            converted
        ):
            return converted

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):
        pass

    return default


def normalise_behaviour_label(
    value: Any,
) -> str:

    return (
        str(
            value
        )
        .strip()
        .lower()
    )


def atomic_write_csv(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    dataframe.to_csv(
        temporary,
        index=False,
    )

    temporary.replace(
        path
    )


def atomic_write_json(
    value: Any,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = (
        path.with_suffix(
            path.suffix
            + ".tmp"
        )
    )

    temporary.write_text(
        json.dumps(
            value,
            indent=4,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def dataframe_signature(
    dataframe: pd.DataFrame,
) -> str:

    if (
        dataframe.empty
        or
        "sample_id"
        not in dataframe.columns
    ):

        return ""

    text = "\n".join(
        sorted(
            dataframe[
                "sample_id"
            ]
            .astype(
                str
            )
            .tolist()
        )
    )

    return (
        hashlib.sha256(
            text.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


# =============================================================================
# Participant metadata
# =============================================================================

def load_explicit_participant_map(
    path: Path | None,
) -> dict[str, str]:

    if path is None:

        return {}

    if not path.exists():

        return {}

    dataframe = pd.read_csv(
        path
    )

    required = {
        "session_id",
        "participant_id",
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            f"Participant map is missing columns: "
            f"{sorted(missing)}"
        )

    mapping: dict[
        str,
        str,
    ] = {}

    for _, row in dataframe.iterrows():

        session_id = (
            str(
                row[
                    "session_id"
                ]
            )
            .strip()
        )

        participant_id = (
            str(
                row[
                    "participant_id"
                ]
            )
            .strip()
        )

        if (
            session_id
            and
            participant_id
        ):

            mapping[
                session_id
            ] = participant_id

    return mapping


def load_temporal_participant_map() -> dict[str, str]:

    if not TEMPORAL_MANIFEST_PATH.exists():

        return {}

    dataframe = pd.read_csv(
        TEMPORAL_MANIFEST_PATH
    )

    if not {
        "session_id",
        "participant_id",
    }.issubset(
        dataframe.columns
    ):

        return {}

    mapping: dict[
        str,
        str,
    ] = {}

    for _, row in dataframe.iterrows():

        session_id = (
            str(
                row.get(
                    "session_id",
                    "",
                )
            )
            .strip()
        )

        participant_id = (
            str(
                row.get(
                    "participant_id",
                    "",
                )
            )
            .strip()
        )

        if participant_id.lower() in {
            "",
            "nan",
            "none",
        }:

            continue

        if (
            session_id
            and
            participant_id
        ):

            mapping[
                session_id
            ] = participant_id

    return mapping


def build_participant_lookup(
    explicit_map_path: Path | None,
) -> dict[str, str]:

    temporal = (
        load_temporal_participant_map()
    )

    explicit = (
        load_explicit_participant_map(
            explicit_map_path
        )
    )

    # Explicit mapping overrides temporal manifest.
    return {
        **temporal,
        **explicit,
    }


# =============================================================================
# SenseFuzeAI legacy metadata
# =============================================================================

def load_legacy_session_metadata() -> dict[
    str,
    dict[str, Any],
]:

    if not LEGACY_SESSION_CSV.exists():

        return {}

    dataframe = pd.read_csv(
        LEGACY_SESSION_CSV
    )

    if "session_id" not in dataframe.columns:

        return {}

    output: dict[
        str,
        dict[str, Any],
    ] = {}

    for _, row in dataframe.iterrows():

        session_id = (
            str(
                row.get(
                    "session_id",
                    "",
                )
            )
            .strip()
        )

        if not session_id:

            continue

        output[
            session_id
        ] = row.to_dict()

    return output


# =============================================================================
# Pair SenseFuzeAI key-down/up events
# =============================================================================

def pair_key_events(
    events: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Convert raw key-down/up event stream into ordered keystroke records.

    Each record contains:
        key
        down_timestamp
        up_timestamp

    Unmatched key-down events are retained with up_timestamp=None.
    """

    records: list[
        dict[str, Any]
    ] = []

    active: dict[
        str,
        deque[int],
    ] = defaultdict(
        deque
    )

    for event in events:

        event_type = (
            str(
                event.get(
                    "type",
                    "",
                )
            )
            .strip()
            .lower()
        )

        key = (
            normalise_key(
                event.get(
                    "key",
                    "",
                )
            )
        )

        timestamp = safe_float(
            event.get(
                "timestamp_perf",
                event.get(
                    "timestamp_epoch",
                    float("nan"),
                ),
            )
        )

        if not math.isfinite(
            timestamp
        ):

            continue

        if event_type == "down":

            record_index = len(
                records
            )

            records.append(
                {
                    "key":
                        key,

                    "down_timestamp":
                        timestamp,

                    "up_timestamp":
                        None,
                }
            )

            active[
                key
            ].append(
                record_index
            )

        elif event_type == "up":

            if (
                key in active
                and
                active[
                    key
                ]
            ):

                record_index = (
                    active[
                        key
                    ]
                    .popleft()
                )

                records[
                    record_index
                ][
                    "up_timestamp"
                ] = timestamp

    records.sort(
        key=lambda record:
            float(
                record[
                    "down_timestamp"
                ]
            )
    )

    return records


def records_to_events(
    records: list[
        dict[str, Any]
    ],
) -> list[
    dict[str, Any]
]:
    """
    Reconstruct a window-local event stream for the canonical EmoSurv
    live feature extractor.
    """

    output: list[
        dict[str, Any]
    ] = []

    for record in records:

        key = (
            str(
                record[
                    "key"
                ]
            )
        )

        down_timestamp = float(
            record[
                "down_timestamp"
            ]
        )

        output.append(
            {
                "type":
                    "down",

                "key":
                    key,

                "timestamp_perf":
                    down_timestamp,
            }
        )

        up_timestamp = (
            record.get(
                "up_timestamp"
            )
        )

        if (
            up_timestamp is not None
            and
            math.isfinite(
                float(
                    up_timestamp
                )
            )
        ):

            output.append(
                {
                    "type":
                        "up",

                    "key":
                        key,

                    "timestamp_perf":
                        float(
                            up_timestamp
                        ),
                }
            )

    output.sort(
        key=lambda event: (
            float(
                event[
                    "timestamp_perf"
                ]
            ),
            0
            if event[
                "type"
            ]
            == "down"
            else 1,
        )
    )

    return output


# =============================================================================
# SenseFuzeAI window validation
# =============================================================================

def count_valid_dwell(
    records: list[
        dict[str, Any]
    ],
) -> int:

    count = 0

    for record in records:

        up_timestamp = (
            record.get(
                "up_timestamp"
            )
        )

        if up_timestamp is None:

            continue

        dwell_ms = (
            float(
                up_timestamp
            )
            -
            float(
                record[
                    "down_timestamp"
                ]
            )
        ) * 1000.0

        if (
            DWELL_MIN_MS
            <= dwell_ms
            <= DWELL_MAX_MS
        ):

            count += 1

    return count


def count_valid_dd(
    records: list[
        dict[str, Any]
    ],
) -> int:

    count = 0

    for index in range(
        len(
            records
        )
        - 1
    ):

        dd_ms = (
            float(
                records[
                    index + 1
                ][
                    "down_timestamp"
                ]
            )
            -
            float(
                records[
                    index
                ][
                    "down_timestamp"
                ]
            )
        ) * 1000.0

        if (
            DD_MIN_MS
            <= dd_ms
            <= DD_MAX_MS
        ):

            count += 1

    return count


# =============================================================================
# SenseFuzeAI raw JSON loading
# =============================================================================

def load_sensefuze_session(
    path: Path,
    legacy_metadata: dict[
        str,
        dict[str, Any],
    ],
) -> tuple[
    str,
    str,
    list[
        dict[str, Any]
    ],
]:

    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )

    if isinstance(
        payload,
        dict,
    ):

        session_id = (
            str(
                payload.get(
                    "session_id",
                    path.stem,
                )
            )
            .strip()
        )

        events = (
            payload.get(
                "events",
                [],
            )
        )

        label = (
            normalise_behaviour_label(
                payload.get(
                    "label",
                    "",
                )
            )
        )

    elif isinstance(
        payload,
        list,
    ):

        session_id = (
            path.stem
        )

        events = payload

        label = ""

    else:

        raise ValueError(
            f"Unsupported keystroke JSON format: "
            f"{path}"
        )

    legacy = (
        legacy_metadata.get(
            session_id,
            {},
        )
    )

    if not label:

        label = (
            normalise_behaviour_label(
                legacy.get(
                    "label",
                    "",
                )
            )
        )

    if label not in FOUR_CLASS_LABELS:

        raise ValueError(
            f"Session {session_id} has unsupported "
            f"behaviour label: {label!r}"
        )

    if not isinstance(
        events,
        list,
    ):

        raise ValueError(
            f"Session {session_id} has invalid events."
        )

    return (
        session_id,
        label,
        events,
    )


# =============================================================================
# Build SenseFuzeAI harmonised windows
# =============================================================================

def build_sensefuze_windows(
    *,
    participant_lookup: dict[
        str,
        str,
    ],
    default_participant_id: str,
    strict_participant_map: bool,
) -> pd.DataFrame:

    if not KEYSTROKE_DIR.exists():

        raise FileNotFoundError(
            f"SenseFuzeAI keystroke directory "
            f"not found:\n{KEYSTROKE_DIR}"
        )

    legacy_metadata = (
        load_legacy_session_metadata()
    )

    rows: list[
        dict[str, Any]
    ] = []

    skipped_too_short = 0
    skipped_low_quality = 0
    missing_participant_sessions: list[str] = []

    json_paths = sorted(
        KEYSTROKE_DIR.glob(
            "*.json"
        )
    )

    if not json_paths:

        raise FileNotFoundError(
            "No SenseFuzeAI keystroke JSON "
            f"files found in:\n{KEYSTROKE_DIR}"
        )

    print()
    print("=" * 88)
    print("BUILDING SENSEFUZEAI HARMONISED WINDOWS")
    print("=" * 88)

    for json_path in json_paths:

        try:

            (
                session_id,
                label,
                events,
            ) = (
                load_sensefuze_session(
                    json_path,
                    legacy_metadata,
                )
            )

        except Exception as exc:

            print(
                "WARNING: skipping "
                f"{json_path.name}: {exc}"
            )

            continue

        participant_id = (
            participant_lookup.get(
                session_id
            )
        )

        if not participant_id:

            missing_participant_sessions.append(
                session_id
            )

            if strict_participant_map:

                continue

            participant_id = (
                default_participant_id
            )

        participant_id = (
            f"sensefuzeai_"
            f"{participant_id}"
        )

        records = (
            pair_key_events(
                events
            )
        )

        if len(
            records
        ) < MIN_WINDOW_SIZE:

            skipped_too_short += 1
            continue

        window_number = 0

        for start in range(
            0,
            len(
                records
            ),
            WINDOW_STEP,
        ):

            window_records = (
                records[
                    start:
                    start
                    + WINDOW_SIZE
                ]
            )

            if len(
                window_records
            ) < MIN_WINDOW_SIZE:

                continue

            valid_dwell = (
                count_valid_dwell(
                    window_records
                )
            )

            valid_dd = (
                count_valid_dd(
                    window_records
                )
            )

            # Match the existing EmoSurv window-quality principle.
            if (
                valid_dwell < 5
                or
                valid_dd < 5
            ):

                skipped_low_quality += 1
                continue

            window_events = (
                records_to_events(
                    window_records
                )
            )

            features = (
                extract_live_features(
                    window_events
                )
            )

            row = {
                "dataset_source":
                    "sensefuzeai",

                "participant_id":
                    participant_id,

                "session_id":
                    session_id,

                "sample_id":
                    (
                        f"sensefuzeai_"
                        f"{session_id}_"
                        f"window_{window_number:03d}"
                    ),

                "source_type":
                    "guided_multimodal_session",

                "label":
                    label,

                "label_origin":
                    (
                        "sensefuzeai_collected_"
                        "behaviour_label"
                    ),

                "original_label":
                    label,

                "window_index":
                    window_number,

                "window_keystrokes":
                    len(
                        window_records
                    ),

                **{
                    feature:
                        safe_float(
                            features.get(
                                feature
                            )
                        )
                    for feature
                    in FEATURE_COLUMNS
                },
            }

            rows.append(
                row
            )

            window_number += 1

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:

        raise ValueError(
            "No valid SenseFuzeAI harmonised "
            "keystroke windows were generated."
        )

    if strict_participant_map:

        missing_unique = sorted(
            set(
                missing_participant_sessions
            )
        )

        if missing_unique:

            raise ValueError(
                "Strict participant-map mode is enabled, "
                "but participant IDs are missing for "
                f"{len(missing_unique)} sessions.\n\n"
                f"Examples:\n{missing_unique[:20]}"
            )

    print(
        f"SenseFuzeAI windows       : "
        f"{len(dataframe):,}"
    )

    print(
        f"SenseFuzeAI sessions      : "
        f"{dataframe['session_id'].nunique():,}"
    )

    print(
        f"SenseFuzeAI participants  : "
        f"{dataframe['participant_id'].nunique():,}"
    )

    print(
        f"Too-short sessions        : "
        f"{skipped_too_short:,}"
    )

    print(
        f"Low-quality windows       : "
        f"{skipped_low_quality:,}"
    )

    if missing_participant_sessions:

        print()
        print(
            "NOTE:"
        )

        print(
            f"{len(set(missing_participant_sessions))} "
            "sessions did not have explicit participant "
            "metadata."
        )

        print(
            "They were assigned to the declared default "
            f"participant: {default_participant_id!r}"
        )

    return dataframe


# =============================================================================
# Harmonise EmoSurv
# =============================================================================

def emosurv_common_frame(
    dataframe: pd.DataFrame,
    *,
    label_origin: str,
) -> pd.DataFrame:

    rows: list[
        dict[str, Any]
    ] = []

    for _, row in dataframe.iterrows():

        source_type = (
            str(
                row[
                    "source_type"
                ]
            )
        )

        user_id = (
            str(
                row[
                    "user_id"
                ]
            )
        )

        emotion_code = (
            str(
                row[
                    "emotion_code"
                ]
            )
        )

        emotion_label = (
            str(
                row[
                    "emotion_label"
                ]
            )
        )

        window_id = int(
            row[
                "window_id"
            ]
        )

        label = (
            str(
                row[
                    "behaviour_proxy"
                ]
            )
        )

        rows.append(
            {
                "dataset_source":
                    "emosurv",

                "participant_id":
                    (
                        f"emosurv_"
                        f"{user_id}"
                    ),

                "session_id":
                    (
                        f"emosurv_"
                        f"{source_type}_"
                        f"{user_id}_"
                        f"{emotion_code}"
                    ),

                "sample_id":
                    (
                        f"emosurv_"
                        f"{source_type}_"
                        f"window_"
                        f"{window_id:07d}"
                    ),

                "source_type":
                    (
                        f"emosurv_"
                        f"{source_type}"
                    ),

                "label":
                    label,

                "label_origin":
                    label_origin,

                "original_label":
                    emotion_label,

                "window_index":
                    window_id,

                "window_keystrokes":
                    int(
                        row[
                            "window_keystrokes"
                        ]
                    ),

                **{
                    feature:
                        safe_float(
                            row.get(
                                feature
                            )
                        )
                    for feature
                    in FEATURE_COLUMNS
                },
            }
        )

    return pd.DataFrame(
        rows
    )


def build_emosurv_datasets() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    float,
]:

    raw = (
        load_emosurv_datasets()
    )

    windows = (
        build_window_dataset(
            raw
        )
    )

    # -------------------------------------------------------------------------
    # Primary conservative 3-class dataset
    # -------------------------------------------------------------------------

    three_class = (
        windows[
            windows[
                "emotion_code"
            ]
            .isin(
                DIRECT_EMOSURV_PROXY
            )
        ]
        .copy()
    )

    three_class[
        "behaviour_proxy"
    ] = (
        three_class[
            "emotion_code"
        ]
        .map(
            DIRECT_EMOSURV_PROXY
        )
    )

    three_harmonised = (
        emosurv_common_frame(
            three_class,
            label_origin=(
                "emosurv_direct_semantic_proxy"
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Exploratory four-class weakly supervised dataset
    # -------------------------------------------------------------------------

    (
        four_class,
        distraction_threshold,
    ) = (
        assign_behaviour_proxy_labels(
            windows
        )
    )

    four_harmonised = (
        emosurv_common_frame(
            four_class,
            label_origin=(
                "emosurv_weakly_supervised_"
                "four_class_proxy"
            ),
        )
    )

    return (
        three_harmonised,
        four_harmonised,
        distraction_threshold,
    )


# =============================================================================
# Dataset validation
# =============================================================================

def validate_harmonised_dataset(
    dataframe: pd.DataFrame,
    *,
    labels: tuple[str, ...],
    name: str,
) -> None:

    required = {
        *METADATA_COLUMNS,
        *FEATURE_COLUMNS,
    }

    missing = (
        required
        -
        set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            f"{name} is missing columns: "
            f"{sorted(missing)}"
        )

    if dataframe.empty:

        raise ValueError(
            f"{name} is empty."
        )

    if dataframe[
        "sample_id"
    ].duplicated().any():

        raise ValueError(
            f"{name} contains duplicate sample IDs."
        )

    observed_labels = set(
        dataframe[
            "label"
        ]
        .astype(
            str
        )
    )

    unexpected = (
        observed_labels
        -
        set(
            labels
        )
    )

    if unexpected:

        raise ValueError(
            f"{name} contains unexpected labels: "
            f"{sorted(unexpected)}"
        )

    for feature in FEATURE_COLUMNS:

        dataframe[
            feature
        ] = pd.to_numeric(
            dataframe[
                feature
            ],
            errors="coerce",
        )

        values = (
            dataframe[
                feature
            ]
            .to_numpy(
                dtype=float
            )
        )

        invalid_infinite = np.isinf(
            values
        )

        if invalid_infinite.any():

            dataframe.loc[
                invalid_infinite,
                feature,
            ] = np.nan


# =============================================================================
# Statistics
# =============================================================================

def build_statistics(
    datasets: dict[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:

    rows: list[
        dict[str, Any]
    ] = []

    for name, dataframe in datasets.items():

        for source in sorted(
            dataframe[
                "dataset_source"
            ]
            .unique()
        ):

            source_df = (
                dataframe[
                    dataframe[
                        "dataset_source"
                    ]
                    == source
                ]
            )

            for label in sorted(
                source_df[
                    "label"
                ]
                .unique()
            ):

                label_df = (
                    source_df[
                        source_df[
                            "label"
                        ]
                        == label
                    ]
                )

                rows.append(
                    {
                        "dataset":
                            name,

                        "dataset_source":
                            source,

                        "label":
                            label,

                        "windows":
                            len(
                                label_df
                            ),

                        "sessions":
                            label_df[
                                "session_id"
                            ]
                            .nunique(),

                        "participants":
                            label_df[
                                "participant_id"
                            ]
                            .nunique(),
                    }
                )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# Main build
# =============================================================================

def build_all(
    *,
    participant_map_path: Path | None,
    default_participant_id: str,
    strict_participant_map: bool,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    participant_lookup = (
        build_participant_lookup(
            participant_map_path
        )
    )

    sensefuze_all = (
        build_sensefuze_windows(
            participant_lookup=(
                participant_lookup
            ),
            default_participant_id=(
                default_participant_id
            ),
            strict_participant_map=(
                strict_participant_map
            ),
        )
    )

    (
        emosurv_three,
        emosurv_four,
        distraction_threshold,
    ) = (
        build_emosurv_datasets()
    )

    sensefuze_three = (
        sensefuze_all[
            sensefuze_all[
                "label"
            ]
            .isin(
                THREE_CLASS_LABELS
            )
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    sensefuze_four = (
        sensefuze_all.copy()
        .reset_index(
            drop=True
        )
    )

    combined_three = (
        pd.concat(
            [
                emosurv_three,
                sensefuze_three,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    combined_four = (
        pd.concat(
            [
                emosurv_four,
                sensefuze_four,
            ],
            ignore_index=True,
            sort=False,
        )
    )

    datasets = {
        "emosurv_3class":
            emosurv_three,

        "sensefuzeai_3class":
            sensefuze_three,

        "combined_3class":
            combined_three,

        "emosurv_4class":
            emosurv_four,

        "sensefuzeai_4class":
            sensefuze_four,

        "combined_4class":
            combined_four,
    }

    for name, dataframe in datasets.items():

        labels = (
            THREE_CLASS_LABELS
            if "3class"
            in name
            else FOUR_CLASS_LABELS
        )

        validate_harmonised_dataset(
            dataframe,
            labels=labels,
            name=name,
        )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    atomic_write_csv(
        emosurv_three,
        EMOSURV_3_PATH,
    )

    atomic_write_csv(
        sensefuze_three,
        SENSEFUZE_3_PATH,
    )

    atomic_write_csv(
        combined_three,
        COMBINED_3_PATH,
    )

    atomic_write_csv(
        emosurv_four,
        EMOSURV_4_PATH,
    )

    atomic_write_csv(
        sensefuze_four,
        SENSEFUZE_4_PATH,
    )

    atomic_write_csv(
        combined_four,
        COMBINED_4_PATH,
    )

    statistics = (
        build_statistics(
            datasets
        )
    )

    atomic_write_csv(
        statistics,
        STATISTICS_PATH,
    )

    atomic_write_json(
        list(
            FEATURE_COLUMNS
        ),
        FEATURE_SCHEMA_PATH,
    )

    metadata = {
        "project":
            "SenseFuzeAI",

        "purpose":
            (
                "EmoSurv IEEE + SenseFuzeAI "
                "keystroke dataset harmonisation"
            ),

        "window_size":
            WINDOW_SIZE,

        "window_step":
            WINDOW_STEP,

        "minimum_window_size":
            MIN_WINDOW_SIZE,

        "feature_count":
            len(
                FEATURE_COLUMNS
            ),

        "feature_columns":
            list(
                FEATURE_COLUMNS
            ),

        "primary_three_class_labels":
            list(
                THREE_CLASS_LABELS
            ),

        "primary_emosurv_mapping": {
            "Calm":
                "focused",

            "Sad":
                "fatigued",

            "Angry":
                "overloaded",

            "Happy":
                "excluded",

            "Neutral":
                "excluded",
        },

        "exploratory_four_class_status":
            (
                "weakly_supervised_proxy_"
                "not_original_emosurv_ground_truth"
            ),

        "four_class_distraction_threshold":
            float(
                distraction_threshold
            ),

        "participant_map_path":
            (
                str(
                    participant_map_path
                )
                if participant_map_path
                is not None
                else None
            ),

        "default_sensefuze_participant_id":
            default_participant_id,

        "strict_participant_map":
            strict_participant_map,

        "dataset_signatures": {
            name:
                dataframe_signature(
                    dataframe
                )
            for name, dataframe
            in datasets.items()
        },

        "outputs": {
            "emosurv_3class":
                str(
                    EMOSURV_3_PATH
                ),

            "sensefuzeai_3class":
                str(
                    SENSEFUZE_3_PATH
                ),

            "combined_3class":
                str(
                    COMBINED_3_PATH
                ),

            "emosurv_4class":
                str(
                    EMOSURV_4_PATH
                ),

            "sensefuzeai_4class":
                str(
                    SENSEFUZE_4_PATH
                ),

            "combined_4class":
                str(
                    COMBINED_4_PATH
                ),

            "statistics":
                str(
                    STATISTICS_PATH
                ),
        },

        "methodological_notes": [
            (
                "EmoSurv original labels are emotions, "
                "not SenseFuzeAI behavioural-state labels."
            ),
            (
                "The three-class experiment uses only "
                "direct semantic proxy mappings."
            ),
            (
                "The four-class experiment is exploratory "
                "and uses weak supervision."
            ),
            (
                "SenseFuzeAI raw key events are re-extracted "
                "into the exact EmoSurv comparison feature schema."
            ),
            (
                "dataset_source, participant_id, session_id "
                "and label provenance are metadata only and "
                "must never enter classifier feature matrices."
            ),
        ],
    }

    atomic_write_json(
        metadata,
        BUILD_METADATA_PATH,
    )

    # -------------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 88)
    print("KEYSTROKE DATASET HARMONISATION COMPLETE")
    print("=" * 88)

    print()

    for name, dataframe in datasets.items():

        print(
            f"{name:24s}: "
            f"{len(dataframe):,} windows"
        )

    print()
    print(
        f"Common feature count: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {OUTPUT_DIR}"
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "No original EmoSurv or SenseFuzeAI "
        "data files were modified."
    )


# =============================================================================
# CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build harmonised EmoSurv IEEE + "
            "SenseFuzeAI keystroke datasets."
        )
    )

    parser.add_argument(
        "--participant-map",
        type=Path,
        default=(
            DEFAULT_PARTICIPANT_MAP_PATH
            if DEFAULT_PARTICIPANT_MAP_PATH.exists()
            else None
        ),
        help=(
            "Optional CSV containing session_id,participant_id. "
            "Explicit values override participant IDs found in the "
            "temporal manifest."
        ),
    )

    parser.add_argument(
        "--default-participant-id",
        default=(
            "participant_001"
        ),
        help=(
            "Participant ID assigned to historical SenseFuzeAI sessions "
            "without participant metadata. If all historical data came "
            "from you, leave all of them under one participant rather "
            "than inventing a participant per session."
        ),
    )

    parser.add_argument(
        "--strict-participant-map",
        action="store_true",
        help=(
            "Require an explicit participant ID for every SenseFuzeAI "
            "session instead of using the default participant."
        ),
    )

    args = parser.parse_args()

    build_all(
        participant_map_path=(
            args.participant_map
        ),
        default_participant_id=(
            args.default_participant_id
        ),
        strict_participant_map=(
            args.strict_participant_map
        ),
    )


if __name__ == "__main__":

    main()
