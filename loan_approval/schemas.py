from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class LoanApplicationRequest(BaseModel):
    """Validated public API request."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "gender": "Male",
                    "married": "Yes",
                    "dependents": "1",
                    "education": "Graduate",
                    "self_employed": "No",
                    "applicant_income": 5000,
                    "coapplicant_income": 1500,
                    "loan_amount": 130,
                    "loan_amount_term": 360,
                    "credit_history": 1,
                    "property_area": "Semiurban",
                }
            ]
        },
    )

    gender: Literal["Male", "Female"] | None
    married: Literal["Yes", "No"] | None
    dependents: Literal["0", "1", "2", "3+"] | None
    education: Literal[
        "Graduate",
        "Not Graduate",
    ]
    self_employed: Literal["Yes", "No"] | None

    applicant_income: float = Field(
        ge=0,
        description="Primary applicant income.",
    )
    coapplicant_income: float = Field(
        ge=0,
        description="Coapplicant income.",
    )
    loan_amount: float | None = Field(
        gt=0,
        description="Requested loan amount.",
    )
    loan_amount_term: float | None = Field(
        gt=0,
        description="Loan duration.",
    )
    credit_history: Literal[0, 1] | None

    property_area: Literal[
        "Rural",
        "Semiurban",
        "Urban",
    ]

    def to_model_features(self) -> dict[str, object]:
        return {
            "Gender": self.gender,
            "Married": self.married,
            "Dependents": self.dependents,
            "Education": self.education,
            "Self_Employed": self.self_employed,
            "ApplicantIncome": self.applicant_income,
            "CoapplicantIncome": self.coapplicant_income,
            "LoanAmount": self.loan_amount,
            "Loan_Amount_Term": self.loan_amount_term,
            "Credit_History": self.credit_history,
            "Property_Area": self.property_area,
        }


class PredictionResponse(BaseModel):
    prediction: Literal["Approved", "Rejected"]
    predicted_label: Literal["Y", "N"]
    encoded_prediction: Literal[0, 1]

    approval_score: float = Field(
        ge=0,
        le=1,
        description=(
            "Model approval score; not a guaranteed "
            "probability of repayment."
        ),
    )

    classification_threshold: float = Field(
        ge=0,
        le=1,
    )

    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_version: str
    selected_configuration: str
    classification_threshold: float
    feature_count: int
    feature_columns: list[str]
    artifact_filename: str