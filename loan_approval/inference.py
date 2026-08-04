from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "loan_approval_model.joblib"
)


class LoanApprovalService:
    """Load, validate, and serve the loan approval model."""

    REQUIRED_BUNDLE_KEYS = {
        "model_version",
        "pipeline",
        "classification_threshold",
        "feature_columns",
        "positive_label",
        "negative_label",
    }

    ALLOWED_CATEGORIES = {
        "Gender": {"Male", "Female"},
        "Married": {"Yes", "No"},
        "Dependents": {"0", "1", "2", "3+"},
        "Education": {
            "Graduate",
            "Not Graduate",
        },
        "Self_Employed": {"Yes", "No"},
        "Property_Area": {
            "Rural",
            "Semiurban",
            "Urban",
        },
    }

    NUMERIC_FEATURES = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
    ]

    def __init__(
        self,
        artifact_path: Path | str = DEFAULT_MODEL_PATH,
    ) -> None:
        self.artifact_path = Path(
            artifact_path
        ).resolve()

        if not self.artifact_path.is_file():
            raise FileNotFoundError(
                f"Model artifact not found: "
                f"{self.artifact_path}"
            )

        # Only load Joblib files from trusted sources.
        self.bundle = joblib.load(
            self.artifact_path
        )

        if not isinstance(self.bundle, dict):
            raise TypeError(
                "The model artifact must contain a dictionary."
            )

        missing_keys = (
            self.REQUIRED_BUNDLE_KEYS
            - set(self.bundle)
        )

        if missing_keys:
            raise ValueError(
                f"Model bundle is missing keys: "
                f"{sorted(missing_keys)}"
            )

        self.pipeline = self.bundle["pipeline"]

        self.threshold = float(
            self.bundle["classification_threshold"]
        )

        self.feature_columns = list(
            self.bundle["feature_columns"]
        )

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True

        missing_result = pd.isna(value)

        return bool(missing_result)

    def validate_application(
        self,
        application: Mapping[str, Any],
    ) -> pd.DataFrame:
        if not isinstance(application, Mapping):
            raise TypeError(
                "Application must be a mapping."
            )

        expected_features = set(
            self.feature_columns
        )

        provided_features = set(application)

        missing_features = sorted(
            expected_features - provided_features
        )

        unexpected_features = sorted(
            provided_features - expected_features
        )

        if missing_features:
            raise ValueError(
                f"Missing required features: "
                f"{missing_features}"
            )

        if unexpected_features:
            raise ValueError(
                f"Unexpected features: "
                f"{unexpected_features}"
            )

        application_frame = pd.DataFrame(
            [dict(application)],
            columns=self.feature_columns,
        )

        application_frame = (
            application_frame.replace(
                r"^\s*$",
                np.nan,
                regex=True,
            )
        )

        self._validate_categories(
            application_frame
        )

        self._validate_numeric_features(
            application_frame
        )

        return application_frame

    def _validate_categories(
        self,
        application_frame: pd.DataFrame,
    ) -> None:
        for column, allowed_values in (
            self.ALLOWED_CATEGORIES.items()
        ):
            value = application_frame.at[
                0,
                column,
            ]

            if self._is_missing(value):
                continue

            normalized_value = str(value).strip()

            if normalized_value not in allowed_values:
                raise ValueError(
                    f"Invalid value for {column}: "
                    f"{normalized_value!r}. "
                    f"Expected one of "
                    f"{sorted(allowed_values)}."
                )

            application_frame.at[
                0,
                column,
            ] = normalized_value

    def _validate_numeric_features(
        self,
        application_frame: pd.DataFrame,
    ) -> None:
        for column in self.NUMERIC_FEATURES:
            value = application_frame.at[
                0,
                column,
            ]

            if self._is_missing(value):
                continue

            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"{column} must be numeric."
                ) from error

            if not np.isfinite(numeric_value):
                raise ValueError(
                    f"{column} must be finite."
                )

            application_frame.at[
                0,
                column,
            ] = numeric_value

        for column in [
            "ApplicantIncome",
            "CoapplicantIncome",
        ]:
            value = application_frame.at[
                0,
                column,
            ]

            if (
                not self._is_missing(value)
                and value < 0
            ):
                raise ValueError(
                    f"{column} cannot be negative."
                )

        for column in [
            "LoanAmount",
            "Loan_Amount_Term",
        ]:
            value = application_frame.at[
                0,
                column,
            ]

            if (
                not self._is_missing(value)
                and value <= 0
            ):
                raise ValueError(
                    f"{column} must be greater than zero."
                )

        credit_history = application_frame.at[
            0,
            "Credit_History",
        ]

        if (
            not self._is_missing(credit_history)
            and credit_history not in {0.0, 1.0}
        ):
            raise ValueError(
                "Credit_History must be 0, 1, or missing."
            )

        application_frame[
            self.NUMERIC_FEATURES
        ] = application_frame[
            self.NUMERIC_FEATURES
        ].apply(
            pd.to_numeric
        )

    def predict(
        self,
        application: Mapping[str, Any],
    ) -> dict[str, Any]:
        application_frame = (
            self.validate_application(application)
        )

        approval_score = float(
            self.pipeline.predict_proba(
                application_frame
            )[0, 1]
        )

        encoded_prediction = int(
            approval_score >= self.threshold
        )

        if encoded_prediction == 1:
            prediction = "Approved"
            predicted_label = self.bundle[
                "positive_label"
            ]
        else:
            prediction = "Rejected"
            predicted_label = self.bundle[
                "negative_label"
            ]

        return {
            "prediction": prediction,
            "predicted_label": predicted_label,
            "encoded_prediction": encoded_prediction,
            "approval_score": round(
                approval_score,
                4,
            ),
            "classification_threshold": (
                self.threshold
            ),
            "model_version": self.bundle[
                "model_version"
            ],
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "model_version": self.bundle[
                "model_version"
            ],
            "selected_configuration": (
                self.bundle.get(
                    "selected_configuration",
                    "Unknown",
                )
            ),
            "classification_threshold": (
                self.threshold
            ),
            "feature_count": len(
                self.feature_columns
            ),
            "feature_columns": (
                self.feature_columns.copy()
            ),
            "artifact_filename": (
                self.artifact_path.name
            ),
        }