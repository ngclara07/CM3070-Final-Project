"""
train_multimodal_comparison.py

SenseFuzeAI
Multimodal Feature-Group, Model, and Temporal-Fusion Comparison

=============================================================================
PURPOSE
=============================================================================

This script provides the principal controlled comparison experiment for the
SenseFuzeAI multimodal behavioural-state classification system.

It evaluates:

    Individual modalities:
        - keystroke_only
        - text_only
        - audio_only
        - image_only

    Two-modality combinations:
        - keystroke_text
        - keystroke_audio
        - keystroke_image
        - text_audio
        - text_image
        - audio_image

    Three-modality combinations:
        - keystroke_text_audio
        - keystroke_text_image
        - keystroke_audio_image
        - text_audio_image

    Complete four-modality system:
        - multimodal_all

Candidate classifiers:
        - Logistic Regression
        - Random Forest
        - RBF SVM
        - XGBoost, when installed
        - LightGBM, when installed
        - CatBoost, when installed

For every candidate model / feature-group combination, the script compares:

    RAW MULTIMODAL CLASSIFICATION
        model.predict_proba(...)

versus:

    CANONICAL TEMPORAL FUSION
        TemporalFusionEngine
            ->
        rolling probability history
            ->
        equal arithmetic probability averaging
            ->
        temporal behavioural prediction

The temporal implementation is imported directly from:

    temporal_fusion.py

No duplicate temporal averaging logic is implemented here.


=============================================================================
SCIENTIFIC EVALUATION DESIGN
=============================================================================

The evaluation is intentionally divided into:

    OUTER HOLD-OUT TEST PARTITION
        Never used for model selection.

    DEVELOPMENT PARTITION
        Used for:
            - cross-validation
            - candidate-model selection
            - permutation/leakage testing

After model selection:

    best candidate for each feature group
        ->
    retrain on complete development partition
        ->
    evaluate once on untouched held-out test partition
        ->
    calculate raw and temporal metrics


=============================================================================
GROUP-AWARE SPLITTING
=============================================================================

When participant/session identifiers are available, observations belonging
to the same group are kept entirely within either development or test.

This prevents leakage such as:

    same participant/session
        partly in training
        partly in test

Candidate split grouping preference:

    participant_id
    participant
    subject_id
    subject
    user_id
    user
    session_id


=============================================================================
TEMPORAL SEQUENCE BOUNDARIES
=============================================================================

Temporal probability histories are reconstructed independently for each:

    session_id
    + trial/condition identifier, when available
    + generation, when available

Generation therefore behaves exactly like the production applications:

    temporal reset
        ->
    generation changes
        ->
    new temporal history


=============================================================================
IMPORTANT
=============================================================================

Temporal fusion is POST-PROCESSING.

The temporal window does NOT retrain the classifier.

The comparison is therefore:

    SAME trained classifier
            |
            +--> raw probabilities
            |
            +--> same probabilities through TemporalFusionEngine

This provides a fair measurement of whether temporal aggregation improves:

    - classification performance
    - prediction stability
    - confidence behaviour


=============================================================================
"""

from __future__ import annotations

import json
import math
import warnings

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    clone,
)

from sklearn.ensemble import (
    RandomForestClassifier,
)

from sklearn.impute import (
    SimpleImputer,
)

from sklearn.linear_model import (
    LogisticRegression,
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.model_selection import (
    StratifiedKFold,
)

try:

    from sklearn.model_selection import (
        StratifiedGroupKFold,
    )

except ImportError:

    StratifiedGroupKFold = None

from sklearn.pipeline import (
    Pipeline,
)

from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler,
)

from sklearn.svm import (
    SVC,
)

from temporal_fusion import (
    LABELS,
    TEMPORAL_PROBABILITY_WINDOW,
    TemporalFusionEngine,
    normalise_probability_dict,
    summarise_probability_dict,
)


# =============================================================================
# WARNINGS / DISPLAY
# =============================================================================

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
)

sns.set_theme(
    style="whitegrid"
)


# =============================================================================
# PROJECT PATHS
# =============================================================================

DATA_PATH = Path(
    "data/processed/multimodal_features.csv"
)

OUTPUT_DIR = Path(
    "data/processed/multimodal_comparison_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# CANONICAL DATA COLUMNS
# =============================================================================

SESSION_COL = "session_id"
LABEL_COL = "label"


# =============================================================================
# OPTIONAL METADATA CANDIDATES
# =============================================================================

SPLIT_GROUP_CANDIDATES = [
    "participant_id",
    "participant",
    "subject_id",
    "subject",
    "user_id",
    "user",
    SESSION_COL,
]

TRIAL_COLUMN_CANDIDATES = [
    "trial_id",
    "condition_id",
    "trial",
    "condition",
    "run_id",
]

GENERATION_COLUMN_CANDIDATES = [
    "generation",
    "reset_generation",
]

ORDER_COLUMN_CANDIDATES = [
    "sequence_index",
    "observation_index",
    "sample_index",
    "timestamp_epoch",
    "timestamp_perf",
    "timestamp",
    "datetime",
    "time",
]


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

RANDOM_STATE = 42

CV_SPLITS = 5

OUTER_HOLDOUT_SPLITS = 5

N_ESTIMATORS = 100

MODEL_SELECTION_METRIC = (
    "cv_macro_f1_mean"
)

PLOT_DPI = 300

PERMUTATION_REPEATS = 5


# =============================================================================
# CANONICAL LABELS
# =============================================================================

CANONICAL_LABELS = tuple(
    LABELS
)


# =============================================================================
# LABEL-ENCODED WRAPPER
#
# Required for classifiers such as XGBoost that operate most reliably
# using integer targets.
# =============================================================================

class LabelEncodedClassifier(
    BaseEstimator,
    ClassifierMixin,
):

    def __init__(
        self,
        classifier: Any,
    ) -> None:

        self.classifier = (
            classifier
        )

    def fit(
        self,
        X: Any,
        y: Any,
    ) -> "LabelEncodedClassifier":

        self.label_encoder_ = (
            LabelEncoder()
        )

        y_encoded = (
            self.label_encoder_
            .fit_transform(
                y
            )
        )

        self.classifier_ = clone(
            self.classifier
        )

        self.classifier_.fit(
            X,
            y_encoded,
        )

        self.classes_ = (
            self.label_encoder_
            .classes_
        )

        return self

    def predict(
        self,
        X: Any,
    ) -> np.ndarray:

        encoded = (
            self.classifier_
            .predict(
                X
            )
        )

        encoded = (
            np.asarray(
                encoded
            )
            .reshape(-1)
            .astype(int)
        )

        return (
            self.label_encoder_
            .inverse_transform(
                encoded
            )
        )

    def predict_proba(
        self,
        X: Any,
    ) -> np.ndarray:

        return (
            self.classifier_
            .predict_proba(
                X
            )
        )


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> Optional[str]:

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    return None


def normalise_label(
    value: Any,
) -> str:

    return (
        str(value)
        .strip()
        .lower()
    )


def safe_float(
    value: Any,
    default: float = math.nan,
) -> float:

    try:

        numeric = float(
            value
        )

        if np.isfinite(
            numeric
        ):

            return numeric

    except (
        TypeError,
        ValueError,
        OverflowError,
    ):

        pass

    return default


def safe_mean(
    values: Iterable[float],
) -> float:

    numeric = np.asarray(
        list(values),
        dtype=np.float64,
    )

    numeric = numeric[
        np.isfinite(
            numeric
        )
    ]

    if numeric.size == 0:

        return math.nan

    return float(
        np.mean(
            numeric
        )
    )


def safe_std(
    values: Iterable[float],
) -> float:

    numeric = np.asarray(
        list(values),
        dtype=np.float64,
    )

    numeric = numeric[
        np.isfinite(
            numeric
        )
    ]

    if numeric.size == 0:

        return math.nan

    return float(
        np.std(
            numeric
        )
    )


def get_model_classes(
    model: Any,
) -> list[str]:

    classes = getattr(
        model,
        "classes_",
        None,
    )

    if classes is not None:

        return [
            normalise_label(
                value
            )
            for value in classes
        ]

    if hasattr(
        model,
        "named_steps",
    ):

        for step in reversed(
            list(
                model.named_steps.values()
            )
        ):

            classes = getattr(
                step,
                "classes_",
                None,
            )

            if classes is not None:

                return [
                    normalise_label(
                        value
                    )
                    for value in classes
                ]

    return []


# =============================================================================
# DATASET
# =============================================================================

def load_dataset() -> pd.DataFrame:

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    dataframe = pd.read_csv(
        DATA_PATH
    )

    if dataframe.empty:

        raise ValueError(
            "Multimodal dataset is empty."
        )

    required = {
        SESSION_COL,
        LABEL_COL,
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:

        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing)}"
        )

    dataframe = (
        dataframe.copy()
        .reset_index(
            drop=True
        )
    )

    dataframe[
        "__source_row"
    ] = np.arange(
        len(
            dataframe
        ),
        dtype=int,
    )

    dataframe[
        LABEL_COL
    ] = (
        dataframe[
            LABEL_COL
        ]
        .apply(
            normalise_label
        )
    )

    observed_labels = set(
        dataframe[
            LABEL_COL
        ].unique()
    )

    unsupported = (
        observed_labels
        - set(
            CANONICAL_LABELS
        )
    )

    if unsupported:

        raise ValueError(
            "Dataset contains unsupported behavioural labels:\n"
            f"{sorted(unsupported)}\n\n"
            "Canonical labels are:\n"
            f"{list(CANONICAL_LABELS)}"
        )

    missing_classes = (
        set(
            CANONICAL_LABELS
        )
        - observed_labels
    )

    if missing_classes:

        raise ValueError(
            "The comparison dataset does not contain all four "
            "canonical behavioural classes.\n"
            f"Missing: {sorted(missing_classes)}"
        )

    if dataframe[
        SESSION_COL
    ].isna().any():

        raise ValueError(
            "session_id contains missing values."
        )

    # Repeated session_id values are now EXPECTED when the dataset
    # contains chronological multimodal observations.
    #
    # Therefore duplicate session IDs must NOT be rejected.

    return dataframe


# =============================================================================
# METADATA RESOLUTION
# =============================================================================

def resolve_metadata_columns(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:

    split_group_column = (
        find_first_existing_column(
            dataframe,
            SPLIT_GROUP_CANDIDATES,
        )
    )

    trial_column = (
        find_first_existing_column(
            dataframe,
            TRIAL_COLUMN_CANDIDATES,
        )
    )

    generation_column = (
        find_first_existing_column(
            dataframe,
            GENERATION_COLUMN_CANDIDATES,
        )
    )

    order_column = (
        find_first_existing_column(
            dataframe,
            ORDER_COLUMN_CANDIDATES,
        )
    )

    temporal_group_columns = [
        SESSION_COL
    ]

    if (
        trial_column
        and
        trial_column
        not in temporal_group_columns
    ):

        temporal_group_columns.append(
            trial_column
        )

    if (
        generation_column
        and
        generation_column
        not in temporal_group_columns
    ):

        temporal_group_columns.append(
            generation_column
        )

    if split_group_column is None:

        split_group_column = (
            SESSION_COL
        )

    return {
        "split_group_column":
            split_group_column,

        "trial_column":
            trial_column,

        "generation_column":
            generation_column,

        "order_column":
            order_column,

        "temporal_group_columns":
            temporal_group_columns,
    }


# =============================================================================
# FEATURE DISCOVERY
# =============================================================================

def metadata_columns_to_exclude(
    dataframe: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> set[str]:

    excluded = {
        SESSION_COL,
        LABEL_COL,
        "__source_row",
    }

    for candidate in (
        SPLIT_GROUP_CANDIDATES
        +
        TRIAL_COLUMN_CANDIDATES
        +
        GENERATION_COLUMN_CANDIDATES
        +
        ORDER_COLUMN_CANDIDATES
    ):

        if candidate in dataframe.columns:

            excluded.add(
                candidate
            )

    for value in metadata.values():

        if isinstance(
            value,
            str,
        ):

            excluded.add(
                value
            )

        elif isinstance(
            value,
            list,
        ):

            excluded.update(
                item
                for item in value
                if isinstance(
                    item,
                    str,
                )
            )

    # Generic *_id columns must not enter model features.
    for column in dataframe.columns:

        if (
            column.endswith(
                "_id"
            )
        ):

            excluded.add(
                column
            )

    return excluded


def get_feature_groups(
    dataframe: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[
    str,
    list[str]
]:

    excluded = (
        metadata_columns_to_exclude(
            dataframe,
            metadata,
        )
    )

    all_feature_columns = [
        column
        for column
        in dataframe.columns
        if (
            column
            not in excluded
            and
            pd.api.types
            .is_numeric_dtype(
                dataframe[
                    column
                ]
            )
        )
    ]

    if not all_feature_columns:

        raise ValueError(
            "No numeric model features were found."
        )

    text_columns = [
        column
        for column
        in all_feature_columns
        if column.startswith(
            "text_"
        )
    ]

    audio_columns = [
        column
        for column
        in all_feature_columns
        if column.startswith(
            "audio_"
        )
    ]

    image_columns = [
        column
        for column
        in all_feature_columns
        if column.startswith(
            "image_"
        )
    ]

    pretrained_columns = set(
        text_columns
        +
        audio_columns
        +
        image_columns
    )

    keystroke_columns = [
        column
        for column
        in all_feature_columns
        if column
        not in pretrained_columns
    ]

    groups = {
        "keystroke_only":
            keystroke_columns,

        "text_only":
            text_columns,

        "audio_only":
            audio_columns,

        "image_only":
            image_columns,

        "keystroke_text":
            (
                keystroke_columns
                + text_columns
            ),

        "keystroke_audio":
            (
                keystroke_columns
                + audio_columns
            ),

        "keystroke_image":
            (
                keystroke_columns
                + image_columns
            ),

        "text_audio":
            (
                text_columns
                + audio_columns
            ),

        "text_image":
            (
                text_columns
                + image_columns
            ),

        "audio_image":
            (
                audio_columns
                + image_columns
            ),

        "keystroke_text_audio":
            (
                keystroke_columns
                + text_columns
                + audio_columns
            ),

        "keystroke_text_image":
            (
                keystroke_columns
                + text_columns
                + image_columns
            ),

        "keystroke_audio_image":
            (
                keystroke_columns
                + audio_columns
                + image_columns
            ),

        "text_audio_image":
            (
                text_columns
                + audio_columns
                + image_columns
            ),

        "multimodal_all":
            (
                keystroke_columns
                + text_columns
                + audio_columns
                + image_columns
            ),
    }

    for group_name, columns in groups.items():

        if not columns:

            raise ValueError(
                f"No features found for feature group: "
                f"{group_name}"
            )

    return groups


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================

def common_preprocessing_steps() -> list[
    tuple[str, Any]
]:

    return [
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),

        (
            "scaler",
            StandardScaler(),
        ),
    ]


def build_models() -> dict[
    str,
    Pipeline
]:

    models: dict[
        str,
        Pipeline
    ] = {}

    models[
        "logistic_regression"
    ] = Pipeline(
        common_preprocessing_steps()
        + [
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=(
                        RANDOM_STATE
                    ),
                ),
            ),
        ]
    )

    models[
        "random_forest"
    ] = Pipeline(
        common_preprocessing_steps()
        + [
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=(
                        N_ESTIMATORS
                    ),
                    class_weight="balanced",
                    random_state=(
                        RANDOM_STATE
                    ),
                    n_jobs=-1,
                ),
            ),
        ]
    )

    models[
        "svm_rbf"
    ] = Pipeline(
        common_preprocessing_steps()
        + [
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=(
                        RANDOM_STATE
                    ),
                ),
            ),
        ]
    )

    try:

        from xgboost import (
            XGBClassifier,
        )

        models[
            "xgboost"
        ] = Pipeline(
            common_preprocessing_steps()
            + [
                (
                    "clf",
                    LabelEncodedClassifier(
                        XGBClassifier(
                            n_estimators=(
                                N_ESTIMATORS
                            ),
                            learning_rate=0.05,
                            max_depth=3,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            objective=(
                                "multi:softprob"
                            ),
                            eval_metric=(
                                "mlogloss"
                            ),
                            random_state=(
                                RANDOM_STATE
                            ),
                            n_jobs=-1,
                        )
                    ),
                ),
            ]
        )

    except ImportError:

        print(
            "XGBoost not installed. "
            "Skipping xgboost."
        )

    try:

        from lightgbm import (
            LGBMClassifier,
        )

        models[
            "lightgbm"
        ] = Pipeline(
            common_preprocessing_steps()
            + [
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=(
                            N_ESTIMATORS
                        ),
                        learning_rate=0.05,
                        class_weight="balanced",
                        random_state=(
                            RANDOM_STATE
                        ),
                        n_jobs=-1,
                        verbose=-1,
                    ),
                ),
            ]
        )

    except ImportError:

        print(
            "LightGBM not installed. "
            "Skipping lightgbm."
        )

    try:

        from catboost import (
            CatBoostClassifier,
        )

        models[
            "catboost"
        ] = Pipeline(
            common_preprocessing_steps()
            + [
                (
                    "clf",
                    CatBoostClassifier(
                        iterations=(
                            N_ESTIMATORS
                        ),
                        learning_rate=0.05,
                        depth=4,
                        loss_function=(
                            "MultiClass"
                        ),
                        auto_class_weights=(
                            "Balanced"
                        ),
                        random_seed=(
                            RANDOM_STATE
                        ),
                        verbose=False,
                        thread_count=-1,
                    ),
                ),
            ]
        )

    except ImportError:

        print(
            "CatBoost not installed. "
            "Skipping catboost."
        )

    return models


# =============================================================================
# CANONICAL PROBABILITY PREDICTION
# =============================================================================

def predict_canonical_probabilities(
    model: Any,
    X: pd.DataFrame,
) -> list[
    dict[str, float]
]:

    if not hasattr(
        model,
        "predict_proba",
    ):

        raise ValueError(
            "Comparison model does not implement predict_proba(...)."
        )

    classes = (
        get_model_classes(
            model
        )
    )

    if not classes:

        raise ValueError(
            "Trained model does not expose class ordering."
        )

    if set(
        classes
    ) != set(
        CANONICAL_LABELS
    ):

        raise ValueError(
            "Trained model classes do not match the "
            "canonical behavioural classes.\n"
            f"Model classes: {classes}\n"
            f"Expected: {list(CANONICAL_LABELS)}"
        )

    matrix = np.asarray(
        model.predict_proba(
            X
        ),
        dtype=np.float64,
    )

    if matrix.ndim != 2:

        raise ValueError(
            "predict_proba(...) did not return "
            "a two-dimensional matrix."
        )

    if (
        matrix.shape[1]
        != len(
            classes
        )
    ):

        raise ValueError(
            "Probability/class dimension mismatch."
        )

    output: list[
        dict[str, float]
    ] = []

    for row in matrix:

        mapping = {
            class_name:
                probability
            for class_name, probability
            in zip(
                classes,
                row,
            )
        }

        output.append(
            normalise_probability_dict(
                mapping,
                labels=(
                    CANONICAL_LABELS
                ),
            )
        )

    return output


# =============================================================================
# SPLITTING HELPERS
# =============================================================================

def normalised_group_values(
    dataframe: pd.DataFrame,
    group_column: str,
) -> pd.Series:

    values = (
        dataframe[
            group_column
        ]
        .astype(
            "string"
        )
        .copy()
    )

    missing = (
        values.isna()
    )

    for index in dataframe.index[
        missing
    ]:

        values.loc[
            index
        ] = (
            "__missing_group_"
            f"{dataframe.loc[index, '__source_row']}"
        )

    return (
        values.astype(
            str
        )
    )


def split_contains_all_classes(
    y: pd.Series,
    indices: np.ndarray,
) -> bool:

    observed = set(
        y.iloc[
            indices
        ].unique()
    )

    return (
        observed
        == set(
            CANONICAL_LABELS
        )
    )


def validate_no_group_leakage(
    groups: pd.Series,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> None:

    train_groups = set(
        groups.iloc[
            train_indices
        ]
    )

    test_groups = set(
        groups.iloc[
            test_indices
        ]
    )

    overlap = (
        train_groups
        &
        test_groups
    )

    if overlap:

        raise RuntimeError(
            "Group leakage detected between training "
            "and validation/test partitions.\n"
            f"Overlapping groups: "
            f"{sorted(list(overlap))[:20]}"
        )


def build_group_aware_splits(
    dataframe: pd.DataFrame,
    *,
    group_column: str,
    desired_splits: int,
    purpose: str,
) -> tuple[
    list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    int,
    str,
]:

    y = (
        dataframe[
            LABEL_COL
        ]
        .reset_index(
            drop=True
        )
    )

    groups = (
        normalised_group_values(
            dataframe.reset_index(
                drop=True
            ),
            group_column,
        )
    )

    unique_group_count = (
        groups.nunique()
    )

    minimum_class_count = int(
        y.value_counts()
        .min()
    )

    maximum_splits = min(
        desired_splits,
        unique_group_count,
        minimum_class_count,
    )

    if maximum_splits < 2:

        raise ValueError(
            f"Insufficient data for {purpose}.\n"
            f"Unique groups: {unique_group_count}\n"
            f"Smallest class count: {minimum_class_count}"
        )

    # -------------------------------------------------------------------------
    # Preferred: StratifiedGroupKFold
    # -------------------------------------------------------------------------

    if (
        StratifiedGroupKFold
        is not None
    ):

        for split_count in range(
            maximum_splits,
            1,
            -1,
        ):

            splitter = (
                StratifiedGroupKFold(
                    n_splits=(
                        split_count
                    ),
                    shuffle=True,
                    random_state=(
                        RANDOM_STATE
                    ),
                )
            )

            try:

                candidate_splits = list(
                    splitter.split(
                        np.zeros(
                            len(
                                dataframe
                            )
                        ),
                        y,
                        groups,
                    )
                )

            except ValueError:

                continue

            valid = True

            for (
                train_indices,
                test_indices,
            ) in candidate_splits:

                validate_no_group_leakage(
                    groups,
                    train_indices,
                    test_indices,
                )

                if not (
                    split_contains_all_classes(
                        y,
                        train_indices,
                    )
                    and
                    split_contains_all_classes(
                        y,
                        test_indices,
                    )
                ):

                    valid = False

                    break

            if valid:

                return (
                    candidate_splits,
                    split_count,
                    "StratifiedGroupKFold",
                )

    # -------------------------------------------------------------------------
    # Safe fallback ONLY when every row already has a unique group.
    # -------------------------------------------------------------------------

    if (
        unique_group_count
        == len(
            dataframe
        )
    ):

        split_count = min(
            maximum_splits,
            desired_splits,
        )

        splitter = (
            StratifiedKFold(
                n_splits=(
                    split_count
                ),
                shuffle=True,
                random_state=(
                    RANDOM_STATE
                ),
            )
        )

        splits = list(
            splitter.split(
                np.zeros(
                    len(
                        dataframe
                    )
                ),
                y,
            )
        )

        return (
            splits,
            split_count,
            "StratifiedKFold_unique_groups",
        )

    raise ValueError(
        f"Could not construct a leakage-safe {purpose} split.\n\n"
        "Repeated observations exist within split groups, so "
        "row-level StratifiedKFold would leak related observations "
        "across partitions.\n\n"
        "Consider collecting more participants/sessions."
    )


# =============================================================================
# TEMPORAL SEQUENCE VALIDATION
# =============================================================================

def report_temporal_sequence_structure(
    dataframe: pd.DataFrame,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:

    group_columns = (
        metadata[
            "temporal_group_columns"
        ]
    )

    grouped = (
        dataframe.groupby(
            group_columns,
            dropna=False,
            sort=False,
        )
    )

    lengths: list[
        int
    ] = []

    mixed_label_sequences = 0

    for _, group in grouped:

        lengths.append(
            len(
                group
            )
        )

        if (
            group[
                LABEL_COL
            ].nunique()
            > 1
        ):

            mixed_label_sequences += 1

    sequence_count = len(
        lengths
    )

    if sequence_count == 0:

        raise ValueError(
            "No temporal sequences were identified."
        )

    full_window_sequences = sum(
        1
        for length
        in lengths
        if length
        >= TEMPORAL_PROBABILITY_WINDOW
    )

    return {
        "temporal_group_columns":
            list(
                group_columns
            ),

        "order_column":
            metadata[
                "order_column"
            ],

        "sequence_count":
            sequence_count,

        "minimum_sequence_length":
            int(
                min(
                    lengths
                )
            ),

        "maximum_sequence_length":
            int(
                max(
                    lengths
                )
            ),

        "mean_sequence_length":
            float(
                np.mean(
                    lengths
                )
            ),

        "sequences_reaching_full_temporal_window":
            int(
                full_window_sequences
            ),

        "mixed_label_sequences":
            int(
                mixed_label_sequences
            ),

        "mixed_label_warning":
            (
                mixed_label_sequences
                > 0
            ),
    }


# =============================================================================
# CANONICAL RAW + TEMPORAL PREDICTION FRAME
# =============================================================================

def create_prediction_frame(
    dataframe: pd.DataFrame,
    probabilities: list[
        dict[str, float]
    ],
    *,
    metadata: Mapping[str, Any],
    feature_group: str,
    model_name: str,
) -> pd.DataFrame:

    if len(
        dataframe
    ) != len(
        probabilities
    ):

        raise ValueError(
            "Prediction probability count does not match "
            "the evaluation dataframe."
        )

    working = (
        dataframe.copy()
        .reset_index(
            drop=True
        )
    )

    working[
        "__raw_probability_object"
    ] = probabilities

    group_columns = list(
        metadata[
            "temporal_group_columns"
        ]
    )

    order_column = (
        metadata[
            "order_column"
        ]
    )

    sort_columns = list(
        group_columns
    )

    if order_column:

        sort_columns.append(
            order_column
        )

    else:

        sort_columns.append(
            "__source_row"
        )

    working = (
        working.sort_values(
            sort_columns,
            kind="mergesort",
        )
        .reset_index(
            drop=True
        )
    )

    rows: list[
        dict[str, Any]
    ] = []

    grouped = (
        working.groupby(
            group_columns,
            dropna=False,
            sort=False,
        )
    )

    metadata_columns = [
        "__source_row",
        SESSION_COL,
        LABEL_COL,
    ]

    for candidate in (
        metadata.get(
            "trial_column"
        ),
        metadata.get(
            "generation_column"
        ),
        metadata.get(
            "order_column"
        ),
        metadata.get(
            "split_group_column"
        ),
    ):

        if (
            candidate
            and
            candidate
            not in metadata_columns
            and
            candidate
            in working.columns
        ):

            metadata_columns.append(
                candidate
            )

    for _, sequence in grouped:

        engine = (
            TemporalFusionEngine(
                window_size=(
                    TEMPORAL_PROBABILITY_WINDOW
                ),
                labels=(
                    CANONICAL_LABELS
                ),
            )
        )

        for _, row in sequence.iterrows():

            raw_probabilities = (
                row[
                    "__raw_probability_object"
                ]
            )

            raw_summary = (
                summarise_probability_dict(
                    raw_probabilities,
                    labels=(
                        CANONICAL_LABELS
                    ),
                )
            )

            temporal_summary = (
                engine.append(
                    raw_probabilities
                )
            )

            result: dict[
                str,
                Any
            ] = {
                column:
                    row[
                        column
                    ]
                for column
                in metadata_columns
            }

            result.update(
                {
                    "feature_group":
                        feature_group,

                    "model":
                        model_name,

                    "raw_fusion_state":
                        raw_summary[
                            "current_state"
                        ],

                    "raw_confidence":
                        raw_summary[
                            "confidence"
                        ],

                    "raw_confidence_gap":
                        raw_summary[
                            "confidence_gap"
                        ],

                    "raw_confidence_level":
                        raw_summary[
                            "confidence_level"
                        ],

                    "raw_probabilities":
                        json.dumps(
                            raw_summary[
                                "probabilities"
                            ]
                        ),

                    "final_state":
                        temporal_summary[
                            "current_state"
                        ],

                    "temporal_confidence":
                        temporal_summary[
                            "confidence"
                        ],

                    "temporal_confidence_gap":
                        temporal_summary[
                            "confidence_gap"
                        ],

                    "temporal_confidence_level":
                        temporal_summary[
                            "confidence_level"
                        ],

                    "temporal_probabilities":
                        json.dumps(
                            temporal_summary[
                                "probabilities"
                            ]
                        ),

                    "temporal_samples":
                        temporal_summary[
                            "temporal_samples"
                        ],

                    "temporal_window":
                        temporal_summary[
                            "temporal_window"
                        ],

                    "temporal_window_full":
                        temporal_summary[
                            "temporal_window_full"
                        ],
                }
            )

            for label in CANONICAL_LABELS:

                result[
                    f"raw_{label}_prob"
                ] = float(
                    raw_summary[
                        "probabilities"
                    ][
                        label
                    ]
                )

                result[
                    f"temporal_{label}_prob"
                ] = float(
                    temporal_summary[
                        "probabilities"
                    ][
                        label
                    ]
                )

            rows.append(
                result
            )

    output = (
        pd.DataFrame(
            rows
        )
    )

    return output


# =============================================================================
# CLASSIFICATION METRICS
# =============================================================================

def classification_metrics(
    prediction_frame: pd.DataFrame,
    *,
    prediction_column: str,
) -> dict[str, float]:

    if prediction_frame.empty:

        return {
            "accuracy":
                math.nan,

            "macro_f1":
                math.nan,
        }

    y_true = (
        prediction_frame[
            LABEL_COL
        ]
    )

    y_pred = (
        prediction_frame[
            prediction_column
        ]
    )

    return {
        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    labels=(
                        CANONICAL_LABELS
                    ),
                    average="macro",
                    zero_division=0,
                )
            ),
    }


# =============================================================================
# SWITCH RATE
# =============================================================================

def calculate_switch_rate(
    prediction_frame: pd.DataFrame,
    *,
    prediction_column: str,
    metadata: Mapping[str, Any],
) -> float:

    group_columns = (
        metadata[
            "temporal_group_columns"
        ]
    )

    total_switches = 0
    total_transitions = 0

    grouped = (
        prediction_frame.groupby(
            group_columns,
            dropna=False,
            sort=False,
        )
    )

    for _, sequence in grouped:

        predictions = (
            sequence[
                prediction_column
            ]
            .tolist()
        )

        possible = max(
            0,
            len(
                predictions
            )
            - 1,
        )

        switches = sum(
            1
            for index in range(
                1,
                len(
                    predictions
                ),
            )
            if (
                predictions[
                    index
                ]
                !=
                predictions[
                    index - 1
                ]
            )
        )

        total_switches += (
            switches
        )

        total_transitions += (
            possible
        )

    if total_transitions == 0:

        return 0.0

    return float(
        total_switches
        / total_transitions
    )


# =============================================================================
# ONE-FOLD EVALUATION
# =============================================================================

def evaluate_prediction_frame(
    prediction_frame: pd.DataFrame,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, float]:

    raw = (
        classification_metrics(
            prediction_frame,
            prediction_column=(
                "raw_fusion_state"
            ),
        )
    )

    temporal = (
        classification_metrics(
            prediction_frame,
            prediction_column=(
                "final_state"
            ),
        )
    )

    full_window = (
        prediction_frame[
            prediction_frame[
                "temporal_window_full"
            ]
            == True  # noqa: E712
        ]
    )

    temporal_full = (
        classification_metrics(
            full_window,
            prediction_column=(
                "final_state"
            ),
        )
    )

    raw_switch_rate = (
        calculate_switch_rate(
            prediction_frame,
            prediction_column=(
                "raw_fusion_state"
            ),
            metadata=metadata,
        )
    )

    temporal_switch_rate = (
        calculate_switch_rate(
            prediction_frame,
            prediction_column=(
                "final_state"
            ),
            metadata=metadata,
        )
    )

    return {
        "raw_accuracy":
            raw[
                "accuracy"
            ],

        "raw_macro_f1":
            raw[
                "macro_f1"
            ],

        "temporal_accuracy":
            temporal[
                "accuracy"
            ],

        "temporal_macro_f1":
            temporal[
                "macro_f1"
            ],

        "temporal_full_accuracy":
            temporal_full[
                "accuracy"
            ],

        "temporal_full_macro_f1":
            temporal_full[
                "macro_f1"
            ],

        "raw_switch_rate":
            raw_switch_rate,

        "temporal_switch_rate":
            temporal_switch_rate,

        "switch_rate_reduction":
            (
                raw_switch_rate
                - temporal_switch_rate
            ),

        "full_window_observations":
            int(
                len(
                    full_window
                )
            ),
    }


# =============================================================================
# DEVELOPMENT CROSS-VALIDATION
# =============================================================================

def cross_validate_group(
    development_dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    models: Mapping[
        str,
        Any
    ],
    feature_group_name: str,
    cv_splits: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    metadata: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    X = (
        development_dataframe[
            feature_columns
        ]
        .reset_index(
            drop=True
        )
    )

    development = (
        development_dataframe
        .reset_index(
            drop=True
        )
    )

    aggregate_rows: list[
        dict[str, Any]
    ] = []

    fold_rows: list[
        dict[str, Any]
    ] = []

    for model_name, model_template in models.items():

        print(
            f"Cross-validating "
            f"{feature_group_name} / "
            f"{model_name}..."
        )

        model_fold_rows: list[
            dict[str, Any]
        ] = []

        for fold_index, (
            train_indices,
            validation_indices,
        ) in enumerate(
            cv_splits,
            start=1,
        ):

            model = clone(
                model_template
            )

            X_train = (
                X.iloc[
                    train_indices
                ]
            )

            X_validation = (
                X.iloc[
                    validation_indices
                ]
            )

            y_train = (
                development.iloc[
                    train_indices
                ][
                    LABEL_COL
                ]
            )

            model.fit(
                X_train,
                y_train,
            )

            probability_vectors = (
                predict_canonical_probabilities(
                    model,
                    X_validation,
                )
            )

            validation_dataframe = (
                development.iloc[
                    validation_indices
                ]
                .copy()
                .reset_index(
                    drop=True
                )
            )

            prediction_frame = (
                create_prediction_frame(
                    validation_dataframe,
                    probability_vectors,
                    metadata=metadata,
                    feature_group=(
                        feature_group_name
                    ),
                    model_name=(
                        model_name
                    ),
                )
            )

            fold_metrics = (
                evaluate_prediction_frame(
                    prediction_frame,
                    metadata=metadata,
                )
            )

            fold_row = {
                "feature_group":
                    feature_group_name,

                "model":
                    model_name,

                "fold":
                    fold_index,

                "num_features":
                    len(
                        feature_columns
                    ),

                "validation_rows":
                    len(
                        validation_dataframe
                    ),

                **fold_metrics,
            }

            fold_rows.append(
                fold_row
            )

            model_fold_rows.append(
                fold_row
            )

        # ---------------------------------------------------------------------
        # Preserve old raw-CV column names for compatibility.
        # ---------------------------------------------------------------------

        raw_accuracy = [
            row[
                "raw_accuracy"
            ]
            for row
            in model_fold_rows
        ]

        raw_macro_f1 = [
            row[
                "raw_macro_f1"
            ]
            for row
            in model_fold_rows
        ]

        temporal_accuracy = [
            row[
                "temporal_accuracy"
            ]
            for row
            in model_fold_rows
        ]

        temporal_macro_f1 = [
            row[
                "temporal_macro_f1"
            ]
            for row
            in model_fold_rows
        ]

        temporal_full_accuracy = [
            row[
                "temporal_full_accuracy"
            ]
            for row
            in model_fold_rows
        ]

        temporal_full_macro_f1 = [
            row[
                "temporal_full_macro_f1"
            ]
            for row
            in model_fold_rows
        ]

        raw_switch_rates = [
            row[
                "raw_switch_rate"
            ]
            for row
            in model_fold_rows
        ]

        temporal_switch_rates = [
            row[
                "temporal_switch_rate"
            ]
            for row
            in model_fold_rows
        ]

        aggregate_rows.append(
            {
                "feature_group":
                    feature_group_name,

                "model":
                    model_name,

                "num_features":
                    len(
                        feature_columns
                    ),

                # Existing compatibility fields = RAW classifier CV.
                "cv_accuracy_mean":
                    safe_mean(
                        raw_accuracy
                    ),

                "cv_accuracy_std":
                    safe_std(
                        raw_accuracy
                    ),

                "cv_macro_f1_mean":
                    safe_mean(
                        raw_macro_f1
                    ),

                "cv_macro_f1_std":
                    safe_std(
                        raw_macro_f1
                    ),

                # Canonical temporal CV.
                "cv_temporal_accuracy_mean":
                    safe_mean(
                        temporal_accuracy
                    ),

                "cv_temporal_accuracy_std":
                    safe_std(
                        temporal_accuracy
                    ),

                "cv_temporal_macro_f1_mean":
                    safe_mean(
                        temporal_macro_f1
                    ),

                "cv_temporal_macro_f1_std":
                    safe_std(
                        temporal_macro_f1
                    ),

                # Full five-observation temporal window only.
                "cv_temporal_full_accuracy_mean":
                    safe_mean(
                        temporal_full_accuracy
                    ),

                "cv_temporal_full_accuracy_std":
                    safe_std(
                        temporal_full_accuracy
                    ),

                "cv_temporal_full_macro_f1_mean":
                    safe_mean(
                        temporal_full_macro_f1
                    ),

                "cv_temporal_full_macro_f1_std":
                    safe_std(
                        temporal_full_macro_f1
                    ),

                # Stability.
                "cv_raw_switch_rate_mean":
                    safe_mean(
                        raw_switch_rates
                    ),

                "cv_temporal_switch_rate_mean":
                    safe_mean(
                        temporal_switch_rates
                    ),

                "cv_switch_rate_reduction":
                    (
                        safe_mean(
                            raw_switch_rates
                        )
                        -
                        safe_mean(
                            temporal_switch_rates
                        )
                    ),
            }
        )

    return (
        aggregate_rows,
        fold_rows,
    )


# =============================================================================
# LABEL-PERMUTATION ROBUSTNESS TEST
# =============================================================================

def run_label_permutation_test(
    development_dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    model_template: Any,
    feature_group_name: str,
    cv_splits: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
    n_repeats: int,
) -> dict[str, Any]:

    X = (
        development_dataframe[
            feature_columns
        ]
        .reset_index(
            drop=True
        )
    )

    original_y = (
        development_dataframe[
            LABEL_COL
        ]
        .reset_index(
            drop=True
        )
    )

    repeat_scores: list[
        float
    ] = []

    for repeat in range(
        n_repeats
    ):

        shuffled_y = (
            original_y
            .sample(
                frac=1.0,
                random_state=(
                    RANDOM_STATE
                    + repeat
                ),
            )
            .reset_index(
                drop=True
            )
        )

        fold_scores: list[
            float
        ] = []

        for (
            train_indices,
            validation_indices,
        ) in cv_splits:

            y_train = (
                shuffled_y.iloc[
                    train_indices
                ]
            )

            # Extremely small datasets can lose a class after
            # permutation within a fold. Such a repeat is invalid.
            if (
                set(
                    y_train.unique()
                )
                != set(
                    CANONICAL_LABELS
                )
            ):

                continue

            model = clone(
                model_template
            )

            model.fit(
                X.iloc[
                    train_indices
                ],
                y_train,
            )

            probabilities = (
                predict_canonical_probabilities(
                    model,
                    X.iloc[
                        validation_indices
                    ],
                )
            )

            predictions = [
                summarise_probability_dict(
                    probability,
                    labels=(
                        CANONICAL_LABELS
                    ),
                )[
                    "current_state"
                ]
                for probability
                in probabilities
            ]

            score = (
                f1_score(
                    shuffled_y.iloc[
                        validation_indices
                    ],
                    predictions,
                    labels=(
                        CANONICAL_LABELS
                    ),
                    average="macro",
                    zero_division=0,
                )
            )

            fold_scores.append(
                float(
                    score
                )
            )

        if fold_scores:

            repeat_scores.append(
                float(
                    np.mean(
                        fold_scores
                    )
                )
            )

    if not repeat_scores:

        raise RuntimeError(
            "No valid folds were produced by the "
            "label-permutation test."
        )

    return {
        "feature_group":
            feature_group_name,

        "permutation_macro_f1_mean":
            float(
                np.mean(
                    repeat_scores
                )
            ),

        "permutation_macro_f1_std":
            float(
                np.std(
                    repeat_scores
                )
            ),

        "repeats_completed":
            len(
                repeat_scores
            ),
    }


def run_robustness_checks(
    development_dataframe: pd.DataFrame,
    *,
    feature_groups: Mapping[
        str,
        list[str]
    ],
    cv_splits: list[
        tuple[
            np.ndarray,
            np.ndarray,
        ]
    ],
) -> None:

    print(
        "\nRunning leakage / "
        "robustness checks..."
    )

    models = (
        build_models()
    )

    baseline_model = (
        models[
            "logistic_regression"
        ]
    )

    groups_to_check = [
        "keystroke_only",
        "text_only",
        "audio_only",
        "image_only",
        "multimodal_all",
    ]

    checks: list[
        dict[str, Any]
    ] = []

    for group_name in groups_to_check:

        if (
            group_name
            not in feature_groups
        ):

            continue

        print(
            "Permutation test: "
            f"{group_name}"
        )

        result = (
            run_label_permutation_test(
                development_dataframe,
                feature_columns=(
                    feature_groups[
                        group_name
                    ]
                ),
                model_template=(
                    baseline_model
                ),
                feature_group_name=(
                    group_name
                ),
                cv_splits=(
                    cv_splits
                ),
                n_repeats=(
                    PERMUTATION_REPEATS
                ),
            )
        )

        checks.append(
            result
        )

    checks_dataframe = (
        pd.DataFrame(
            checks
        )
    )

    output_path = (
        OUTPUT_DIR
        / "leakage_permutation_check.csv"
    )

    checks_dataframe.to_csv(
        output_path,
        index=False,
    )

    notes = [
        "Leakage / robustness validation notes",
        "=====================================",
        "",
        "Method:",
        "- Permutation testing uses only the development partition.",
        "- The untouched held-out test partition is not used.",
        "- Group-aware CV boundaries are retained.",
        "",
        "Interpretation:",
        "- For a reasonably balanced four-class task, chance-like macro-F1 "
        "is expected to be approximately 0.25.",
        "- Permutation performance close to chance is desirable.",
        "- Persistently high permutation performance can indicate leakage "
        "or label-coded artifacts.",
        "- Very high text-only performance may indicate that textual prompts "
        "encode the behavioural label rather than natural behaviour.",
        "",
    ]

    (
        OUTPUT_DIR
        / "robustness_notes.txt"
    ).write_text(
        "\n".join(
            notes
        ),
        encoding="utf-8",
    )

    print()

    print(
        checks_dataframe
    )

    print(
        f"\nSaved to: "
        f"{output_path}"
    )


# =============================================================================
# REPORT / CONFUSION-MATRIX HELPERS
# =============================================================================

def save_classification_report(
    prediction_frame: pd.DataFrame,
    *,
    prediction_column: str,
    output_path: Path,
) -> None:

    report = (
        classification_report(
            prediction_frame[
                LABEL_COL
            ],
            prediction_frame[
                prediction_column
            ],
            labels=(
                CANONICAL_LABELS
            ),
            target_names=(
                CANONICAL_LABELS
            ),
            digits=4,
            zero_division=0,
        )
    )

    output_path.write_text(
        report,
        encoding="utf-8",
    )


def save_confusion_matrix(
    prediction_frame: pd.DataFrame,
    *,
    prediction_column: str,
    title: str,
    output_path: Path,
) -> None:

    matrix = (
        confusion_matrix(
            prediction_frame[
                LABEL_COL
            ],
            prediction_frame[
                prediction_column
            ],
            labels=(
                CANONICAL_LABELS
            ),
        )
    )

    plt.figure(
        figsize=(
            7,
            5,
        )
    )

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=(
            CANONICAL_LABELS
        ),
        yticklabels=(
            CANONICAL_LABELS
        ),
    )

    plt.xlabel(
        "Predicted label"
    )

    plt.ylabel(
        "True label"
    )

    plt.title(
        title
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=PLOT_DPI,
    )

    plt.close()


# =============================================================================
# FINAL HELD-OUT TEST EVALUATION
# =============================================================================

def evaluate_best_model(
    development_dataframe: pd.DataFrame,
    test_dataframe: pd.DataFrame,
    *,
    feature_columns: list[str],
    feature_group_name: str,
    model_name: str,
    model_template: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:

    model = clone(
        model_template
    )

    model.fit(
        development_dataframe[
            feature_columns
        ],
        development_dataframe[
            LABEL_COL
        ],
    )

    probabilities = (
        predict_canonical_probabilities(
            model,
            test_dataframe[
                feature_columns
            ],
        )
    )

    prediction_frame = (
        create_prediction_frame(
            test_dataframe,
            probabilities,
            metadata=metadata,
            feature_group=(
                feature_group_name
            ),
            model_name=(
                model_name
            ),
        )
    )

    metrics = (
        evaluate_prediction_frame(
            prediction_frame,
            metadata=metadata,
        )
    )

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    base_name = (
        f"{feature_group_name}_"
        f"{model_name}"
    )

    save_classification_report(
        prediction_frame,
        prediction_column=(
            "raw_fusion_state"
        ),
        output_path=(
            OUTPUT_DIR
            / (
                f"{base_name}_"
                "raw_classification_report.txt"
            )
        ),
    )

    save_confusion_matrix(
        prediction_frame,
        prediction_column=(
            "raw_fusion_state"
        ),
        title=(
            f"{feature_group_name} / "
            f"{model_name} / RAW"
        ),
        output_path=(
            OUTPUT_DIR
            / (
                f"{base_name}_"
                "raw_confusion_matrix.png"
            )
        ),
    )

    save_classification_report(
        prediction_frame,
        prediction_column=(
            "final_state"
        ),
        output_path=(
            OUTPUT_DIR
            / (
                f"{base_name}_"
                "temporal_classification_report.txt"
            )
        ),
    )

    save_confusion_matrix(
        prediction_frame,
        prediction_column=(
            "final_state"
        ),
        title=(
            f"{feature_group_name} / "
            f"{model_name} / "
            "TEMPORAL"
        ),
        output_path=(
            OUTPUT_DIR
            / (
                f"{base_name}_"
                "temporal_confusion_matrix.png"
            )
        ),
    )

    full_window_frame = (
        prediction_frame[
            prediction_frame[
                "temporal_window_full"
            ]
            == True  # noqa: E712
        ]
        .copy()
    )

    if not full_window_frame.empty:

        save_classification_report(
            full_window_frame,
            prediction_column=(
                "final_state"
            ),
            output_path=(
                OUTPUT_DIR
                / (
                    f"{base_name}_"
                    "temporal_full_window_"
                    "classification_report.txt"
                )
            ),
        )

        save_confusion_matrix(
            full_window_frame,
            prediction_column=(
                "final_state"
            ),
            title=(
                f"{feature_group_name} / "
                f"{model_name} / "
                "TEMPORAL 5/5"
            ),
            output_path=(
                OUTPUT_DIR
                / (
                    f"{base_name}_"
                    "temporal_full_window_"
                    "confusion_matrix.png"
                )
            ),
        )

    # -------------------------------------------------------------------------
    # Row-level prediction export
    #
    # This is intentionally compatible with the later multimodal
    # evaluation/parity script.
    # -------------------------------------------------------------------------

    prediction_frame.to_csv(
        OUTPUT_DIR
        / (
            f"{base_name}_"
            "heldout_predictions.csv"
        ),
        index=False,
    )

    # -------------------------------------------------------------------------
    # Save trained comparison model.
    # -------------------------------------------------------------------------

    joblib.dump(
        model,
        OUTPUT_DIR
        / (
            f"{base_name}.joblib"
        ),
    )

    return {
        "feature_group":
            feature_group_name,

        "best_model":
            model_name,

        "num_features":
            len(
                feature_columns
            ),

        "test_rows":
            int(
                len(
                    test_dataframe
                )
            ),

        # Existing compatibility fields = RAW held-out result.
        "test_accuracy":
            metrics[
                "raw_accuracy"
            ],

        "test_macro_f1":
            metrics[
                "raw_macro_f1"
            ],

        # Temporal all observations.
        "test_temporal_accuracy":
            metrics[
                "temporal_accuracy"
            ],

        "test_temporal_macro_f1":
            metrics[
                "temporal_macro_f1"
            ],

        # Full temporal window.
        "test_temporal_full_accuracy":
            metrics[
                "temporal_full_accuracy"
            ],

        "test_temporal_full_macro_f1":
            metrics[
                "temporal_full_macro_f1"
            ],

        "full_window_test_observations":
            metrics[
                "full_window_observations"
            ],

        # Stability.
        "test_raw_switch_rate":
            metrics[
                "raw_switch_rate"
            ],

        "test_temporal_switch_rate":
            metrics[
                "temporal_switch_rate"
            ],

        "test_switch_rate_reduction":
            metrics[
                "switch_rate_reduction"
            ],

        # Direct temporal deltas.
        "temporal_minus_raw_accuracy":
            (
                metrics[
                    "temporal_accuracy"
                ]
                -
                metrics[
                    "raw_accuracy"
                ]
            ),

        "temporal_minus_raw_macro_f1":
            (
                metrics[
                    "temporal_macro_f1"
                ]
                -
                metrics[
                    "raw_macro_f1"
                ]
            ),
    }


# =============================================================================
# SPLIT ASSIGNMENTS
# =============================================================================

def save_split_assignments(
    dataframe: pd.DataFrame,
    *,
    development_indices: np.ndarray,
    test_indices: np.ndarray,
    metadata: Mapping[str, Any],
) -> None:

    assignments = (
        dataframe.copy()
    )

    assignments[
        "experiment_partition"
    ] = (
        "unassigned"
    )

    assignments.loc[
        dataframe.index[
            development_indices
        ],
        "experiment_partition",
    ] = "development"

    assignments.loc[
        dataframe.index[
            test_indices
        ],
        "experiment_partition",
    ] = "heldout_test"

    columns = [
        "__source_row",
        LABEL_COL,
        SESSION_COL,
        "experiment_partition",
    ]

    for column in (
        metadata.get(
            "split_group_column"
        ),
        metadata.get(
            "trial_column"
        ),
        metadata.get(
            "generation_column"
        ),
        metadata.get(
            "order_column"
        ),
    ):

        if (
            column
            and
            column
            not in columns
        ):

            columns.append(
                column
            )

    assignments[
        columns
    ].to_csv(
        OUTPUT_DIR
        / "data_split_assignments.csv",
        index=False,
    )


# =============================================================================
# RAW-vs-TEMPORAL OVERVIEW PLOT
# =============================================================================

def plot_raw_vs_temporal_results(
    test_results: pd.DataFrame,
) -> None:

    plot_frame = (
        test_results[
            [
                "feature_group",
                "test_macro_f1",
                "test_temporal_macro_f1",
            ]
        ]
        .rename(
            columns={
                "test_macro_f1":
                    "Raw",

                "test_temporal_macro_f1":
                    "Temporal",
            }
        )
        .melt(
            id_vars=[
                "feature_group"
            ],
            var_name=(
                "prediction_stage"
            ),
            value_name=(
                "macro_f1"
            ),
        )
    )

    plt.figure(
        figsize=(
            13,
            8,
        )
    )

    sns.barplot(
        data=plot_frame,
        x="macro_f1",
        y="feature_group",
        hue="prediction_stage",
    )

    plt.xlabel(
        "Held-Out Macro-F1"
    )

    plt.ylabel(
        "Feature Group"
    )

    plt.title(
        "Raw vs Temporal Fusion "
        "Held-Out Macro-F1"
    )

    plt.xlim(
        0,
        1.05,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "raw_vs_temporal_test_macro_f1.png",
        dpi=PLOT_DPI,
    )

    plt.close()


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def main() -> None:

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    dataframe = (
        load_dataset()
    )

    metadata = (
        resolve_metadata_columns(
            dataframe
        )
    )

    feature_groups = (
        get_feature_groups(
            dataframe,
            metadata,
        )
    )

    temporal_structure = (
        report_temporal_sequence_structure(
            dataframe,
            metadata,
        )
    )

    print()

    print(
        "=" * 80
    )

    print(
        "SenseFuzeAI "
        "Multimodal Comparison"
    )

    print(
        "=" * 80
    )

    print(
        f"Observations: "
        f"{len(dataframe)}"
    )

    print(
        "Canonical labels: "
        f"{list(CANONICAL_LABELS)}"
    )

    print(
        "Split group column: "
        f"{metadata['split_group_column']}"
    )

    print(
        "Temporal group columns: "
        f"{metadata['temporal_group_columns']}"
    )

    print(
        "Temporal ordering column: "
        f"{metadata['order_column'] or '__source_row'}"
    )

    print(
        "Temporal window: "
        f"{TEMPORAL_PROBABILITY_WINDOW}"
    )

    print()

    print(
        "Label distribution:"
    )

    print(
        dataframe[
            LABEL_COL
        ]
        .value_counts()
        .reindex(
            CANONICAL_LABELS,
            fill_value=0,
        )
    )

    print()

    print(
        "Temporal sequence structure:"
    )

    for key, value in (
        temporal_structure.items()
    ):

        print(
            f"  {key}: {value}"
        )

    if (
        temporal_structure[
            "mixed_label_warning"
        ]
    ):

        print()

        print(
            "WARNING: At least one temporal sequence contains "
            "multiple ground-truth labels."
        )

        print(
            "This is valid for genuinely changing live behaviour, "
            "but controlled-condition experiments should normally "
            "provide trial/condition boundaries so unrelated states "
            "are not averaged together."
        )

    print()

    print(
        "Feature groups:"
    )

    for group_name, columns in (
        feature_groups.items()
    ):

        print(
            f"  {group_name}: "
            f"{len(columns)} features"
        )

    # -------------------------------------------------------------------------
    # OUTER untouched held-out split
    # -------------------------------------------------------------------------

    (
        outer_splits,
        outer_split_count,
        outer_split_method,
    ) = (
        build_group_aware_splits(
            dataframe,
            group_column=(
                metadata[
                    "split_group_column"
                ]
            ),
            desired_splits=(
                OUTER_HOLDOUT_SPLITS
            ),
            purpose=(
                "outer held-out evaluation"
            ),
        )
    )

    (
        development_indices,
        heldout_indices,
    ) = (
        outer_splits[
            0
        ]
    )

    save_split_assignments(
        dataframe,
        development_indices=(
            development_indices
        ),
        test_indices=(
            heldout_indices
        ),
        metadata=metadata,
    )

    development_dataframe = (
        dataframe.iloc[
            development_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    heldout_dataframe = (
        dataframe.iloc[
            heldout_indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        "Outer split:"
    )

    print(
        f"  method: "
        f"{outer_split_method}"
    )

    print(
        f"  folds available: "
        f"{outer_split_count}"
    )

    print(
        f"  development observations: "
        f"{len(development_dataframe)}"
    )

    print(
        f"  held-out observations: "
        f"{len(heldout_dataframe)}"
    )

    # -------------------------------------------------------------------------
    # INNER development-only CV
    # -------------------------------------------------------------------------

    (
        inner_cv_splits,
        inner_cv_count,
        inner_cv_method,
    ) = (
        build_group_aware_splits(
            development_dataframe,
            group_column=(
                metadata[
                    "split_group_column"
                ]
            ),
            desired_splits=(
                CV_SPLITS
            ),
            purpose=(
                "development cross-validation"
            ),
        )
    )

    print()

    print(
        "Inner model-selection CV:"
    )

    print(
        f"  method: "
        f"{inner_cv_method}"
    )

    print(
        f"  folds: "
        f"{inner_cv_count}"
    )

    # -------------------------------------------------------------------------
    # Leakage / robustness testing
    #
    # IMPORTANT: development partition only.
    # -------------------------------------------------------------------------

    run_robustness_checks(
        development_dataframe,
        feature_groups=(
            feature_groups
        ),
        cv_splits=(
            inner_cv_splits
        ),
    )

    # -------------------------------------------------------------------------
    # Candidate model comparison
    # -------------------------------------------------------------------------

    models = (
        build_models()
    )

    all_cv_rows: list[
        dict[str, Any]
    ] = []

    all_fold_rows: list[
        dict[str, Any]
    ] = []

    for group_name, columns in (
        feature_groups.items()
    ):

        (
            aggregate_rows,
            fold_rows,
        ) = (
            cross_validate_group(
                development_dataframe,
                feature_columns=(
                    columns
                ),
                models=models,
                feature_group_name=(
                    group_name
                ),
                cv_splits=(
                    inner_cv_splits
                ),
                metadata=metadata,
            )
        )

        all_cv_rows.extend(
            aggregate_rows
        )

        all_fold_rows.extend(
            fold_rows
        )

    cv_results = (
        pd.DataFrame(
            all_cv_rows
        )
        .sort_values(
            by=[
                "feature_group",
                MODEL_SELECTION_METRIC,
            ],
            ascending=[
                True,
                False,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    cv_fold_results = (
        pd.DataFrame(
            all_fold_rows
        )
    )

    cv_results.to_csv(
        OUTPUT_DIR
        / "cross_validation_comparison.csv",
        index=False,
    )

    cv_fold_results.to_csv(
        OUTPUT_DIR
        / "cross_validation_fold_details.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Select candidate using RAW CV macro-F1.
    #
    # Temporal fusion is not itself a separately trained model and therefore
    # does not determine classifier selection.
    # -------------------------------------------------------------------------

    best_per_group = (
        cv_results
        .sort_values(
            MODEL_SELECTION_METRIC,
            ascending=False,
        )
        .groupby(
            "feature_group",
            as_index=False,
        )
        .first()
        .reset_index(
            drop=True
        )
    )

    best_per_group.to_csv(
        OUTPUT_DIR
        / "best_model_per_feature_group.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Untouched held-out test
    # -------------------------------------------------------------------------

    test_rows: list[
        dict[str, Any]
    ] = []

    for _, row in (
        best_per_group.iterrows()
    ):

        group_name = str(
            row[
                "feature_group"
            ]
        )

        model_name = str(
            row[
                "model"
            ]
        )

        print()

        print(
            "Held-out evaluation: "
            f"{group_name} / "
            f"{model_name}"
        )

        fresh_models = (
            build_models()
        )

        if (
            model_name
            not in fresh_models
        ):

            raise RuntimeError(
                "Selected model is unavailable "
                f"during final evaluation: "
                f"{model_name}"
            )

        result = (
            evaluate_best_model(
                development_dataframe,
                heldout_dataframe,
                feature_columns=(
                    feature_groups[
                        group_name
                    ]
                ),
                feature_group_name=(
                    group_name
                ),
                model_name=(
                    model_name
                ),
                model_template=(
                    fresh_models[
                        model_name
                    ]
                ),
                metadata=metadata,
            )
        )

        test_rows.append(
            result
        )

    test_results = (
        pd.DataFrame(
            test_rows
        )
        .sort_values(
            by=(
                "test_macro_f1"
            ),
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    test_results.to_csv(
        OUTPUT_DIR
        / "test_set_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Explicit RAW vs TEMPORAL comparison table
    # -------------------------------------------------------------------------

    comparison_columns = [
        "feature_group",
        "best_model",
        "num_features",
        "test_accuracy",
        "test_macro_f1",
        "test_temporal_accuracy",
        "test_temporal_macro_f1",
        "test_temporal_full_accuracy",
        "test_temporal_full_macro_f1",
        "full_window_test_observations",
        "test_raw_switch_rate",
        "test_temporal_switch_rate",
        "test_switch_rate_reduction",
        "temporal_minus_raw_accuracy",
        "temporal_minus_raw_macro_f1",
    ]

    raw_vs_temporal = (
        test_results[
            comparison_columns
        ]
        .copy()
    )

    raw_vs_temporal.to_csv(
        OUTPUT_DIR
        / "raw_vs_temporal_test_comparison.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Four-modality result extracted separately for dissertation reporting.
    # -------------------------------------------------------------------------

    multimodal_all_rows = (
        raw_vs_temporal[
            raw_vs_temporal[
                "feature_group"
            ]
            == "multimodal_all"
        ]
    )

    if not multimodal_all_rows.empty:

        multimodal_all_record = (
            multimodal_all_rows.iloc[
                0
            ]
            .to_dict()
        )

        with (
            OUTPUT_DIR
            / "multimodal_all_raw_vs_temporal.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as file_handle:

            json.dump(
                multimodal_all_record,
                file_handle,
                indent=4,
                default=lambda value:
                    (
                        float(value)
                        if isinstance(
                            value,
                            np.floating,
                        )
                        else
                        int(value)
                        if isinstance(
                            value,
                            np.integer,
                        )
                        else value
                    ),
            )

    # -------------------------------------------------------------------------
    # Feature-group definition
    # -------------------------------------------------------------------------

    with (
        OUTPUT_DIR
        / "feature_groups.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file_handle:

        json.dump(
            feature_groups,
            file_handle,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    plot_raw_vs_temporal_results(
        test_results
    )

    # -------------------------------------------------------------------------
    # Experiment metadata
    # -------------------------------------------------------------------------

    split_group_column = (
        metadata[
            "split_group_column"
        ]
    )

    development_group_count = (
        development_dataframe[
            split_group_column
        ]
        .nunique()
    )

    test_group_count = (
        heldout_dataframe[
            split_group_column
        ]
        .nunique()
    )

    comparison_metadata = {
        "dataset":
            str(
                DATA_PATH
            ),

        "num_observations":
            int(
                len(
                    dataframe
                )
            ),

        "num_development_observations":
            int(
                len(
                    development_dataframe
                )
            ),

        "num_heldout_observations":
            int(
                len(
                    heldout_dataframe
                )
            ),

        "canonical_labels":
            list(
                CANONICAL_LABELS
            ),

        "temporal_backend":
            (
                "temporal_fusion."
                "TemporalFusionEngine"
            ),

        "temporal_probability_window":
            TEMPORAL_PROBABILITY_WINDOW,

        "temporal_aggregation":
            (
                "equal arithmetic mean of "
                "canonical class probabilities"
            ),

        "temporal_retraining":
            False,

        "model_selection_metric":
            MODEL_SELECTION_METRIC,

        "outer_split": {
            "method":
                outer_split_method,

            "fold_count":
                outer_split_count,

            "group_column":
                split_group_column,

            "development_groups":
                int(
                    development_group_count
                ),

            "heldout_groups":
                int(
                    test_group_count
                ),
        },

        "inner_cross_validation": {
            "method":
                inner_cv_method,

            "fold_count":
                inner_cv_count,

            "development_partition_only":
                True,
        },

        "temporal_sequence_structure":
            temporal_structure,

        "metadata_columns":
            metadata,

        "n_estimators_or_iterations":
            N_ESTIMATORS,

        "permutation_repeats":
            PERMUTATION_REPEATS,

        "models":
            list(
                models.keys()
            ),

        "feature_groups": {
            name:
                len(
                    columns
                )
            for name, columns
            in feature_groups.items()
        },

        "methodological_notes": [
            (
                "The held-out test partition is not "
                "used for candidate-model selection."
            ),
            (
                "Where repeated participant/session "
                "observations exist, group-aware "
                "splitting prevents train/test leakage."
            ),
            (
                "Temporal aggregation uses exactly "
                "the shared TemporalFusionEngine."
            ),
            (
                "Raw and temporal metrics use the "
                "same trained classifier probabilities."
            ),
            (
                "Full-window temporal metrics include "
                "only observations where the canonical "
                "five-observation window is full."
            ),
        ],
    }

    with (
        OUTPUT_DIR
        / "comparison_metadata.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file_handle:

        json.dump(
            comparison_metadata,
            file_handle,
            indent=4,
        )

    # -------------------------------------------------------------------------
    # Console results
    # -------------------------------------------------------------------------

    print()

    print(
        "=" * 80
    )

    print(
        "CROSS-VALIDATION COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        cv_results
    )

    print()

    print(
        "=" * 80
    )

    print(
        "BEST MODEL PER FEATURE GROUP"
    )

    print(
        "=" * 80
    )

    print(
        best_per_group
    )

    print()

    print(
        "=" * 80
    )

    print(
        "UNTOUCHED HELD-OUT TEST"
    )

    print(
        "=" * 80
    )

    print(
        test_results
    )

    print()

    print(
        "=" * 80
    )

    print(
        "RAW vs TEMPORAL TEST COMPARISON"
    )

    print(
        "=" * 80
    )

    print(
        raw_vs_temporal
    )

    print()

    print(
        "Outputs saved to:"
    )

    print(
        OUTPUT_DIR
    )


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    main()
