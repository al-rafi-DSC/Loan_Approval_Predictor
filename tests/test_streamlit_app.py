from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from loan_approval.api_client import LoanApprovalAPIClient


APP_PATH = (
    Path(__file__).resolve().parents[1]
    / "streamlit_app.py"
)


@pytest.fixture()
def tested_app(monkeypatch):
    captured_application = {}

    def fake_health(self):
        return {
            "status": "healthy",
            "model_loaded": True,
            "model_version": "1.0.0",
        }

    def fake_predict(self, application):
        captured_application.update(application)

        return {
            "prediction": "Approved",
            "predicted_label": "Y",
            "encoded_prediction": 1,
            "approval_score": 0.81,
            "classification_threshold": 0.34,
            "model_version": "1.0.0",
            "disclaimer": "Educational prediction only.",
        }

    monkeypatch.setattr(
        LoanApprovalAPIClient,
        "health",
        fake_health,
    )

    monkeypatch.setattr(
        LoanApprovalAPIClient,
        "predict",
        fake_predict,
    )

    app = AppTest.from_file(
        APP_PATH,
        default_timeout=10,
    ).run()

    return app, captured_application


def test_streamlit_app_loads(tested_app):
    app, _ = tested_app

    assert not app.exception
    assert app.title[0].value == "Loan approval predictor"
    assert len(app.button) == 1
    assert app.button[0].label == "Evaluate application"

    assert any(
        message.value == "API connected"
        for message in app.success
    )


def test_prediction_form_submission(tested_app):
    app, captured_application = tested_app

    app.button[0].click().run()

    assert not app.exception

    metrics = {
        metric.label: metric.value
        for metric in app.metric
    }

    assert metrics["Prediction"] == "Approved"
    assert metrics["Model approval score"] == "81.0%"
    assert metrics["Decision threshold"] == "34%"

    assert captured_application["gender"] == "Male"
    assert captured_application["married"] == "Yes"
    assert captured_application["credit_history"] == 1
    assert captured_application["property_area"] == "Semiurban"

    assert any(
        "may be approved" in message.value
        for message in app.success
    )