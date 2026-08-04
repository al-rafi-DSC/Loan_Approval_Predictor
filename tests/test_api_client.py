import httpx
import pytest

from loan_approval.api_client import (
    LoanApprovalAPIClient,
    LoanApprovalAPIError,
)


@pytest.fixture()
def api_client():
    client = LoanApprovalAPIClient(
        base_url="http://test-api"
    )

    yield client

    client.close()


def test_health_returns_response(
    api_client,
    monkeypatch,
):
    expected_response = {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1.0.0",
    }

    def fake_request(
        self,
        method,
        endpoint,
        **kwargs,
    ):
        assert method == "GET"
        assert endpoint == "/health"

        return httpx.Response(
            status_code=200,
            json=expected_response,
        )

    monkeypatch.setattr(
        httpx.Client,
        "request",
        fake_request,
    )

    result = api_client.health()

    assert result == expected_response


def test_predict_sends_application(
    api_client,
    monkeypatch,
):
    application = {
        "gender": "Male",
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 5500,
        "coapplicant_income": 2000,
        "loan_amount": 120,
        "loan_amount_term": 360,
        "credit_history": 1,
        "property_area": "Semiurban",
    }

    expected_response = {
        "prediction": "Approved",
        "predicted_label": "Y",
        "encoded_prediction": 1,
        "approval_score": 0.81,
        "classification_threshold": 0.34,
        "model_version": "1.0.0",
        "disclaimer": "Educational prediction only.",
    }

    def fake_request(
        self,
        method,
        endpoint,
        **kwargs,
    ):
        assert method == "POST"
        assert endpoint == "/predict"
        assert kwargs["json"] == application

        return httpx.Response(
            status_code=200,
            json=expected_response,
        )

    monkeypatch.setattr(
        httpx.Client,
        "request",
        fake_request,
    )

    result = api_client.predict(application)

    assert result == expected_response
    assert result["prediction"] == "Approved"
    assert result["classification_threshold"] == 0.34


def test_validation_error_is_converted(
    api_client,
    monkeypatch,
):
    def fake_request(
        self,
        method,
        endpoint,
        **kwargs,
    ):
        return httpx.Response(
            status_code=422,
            json={
                "detail": "Applicant income cannot be negative."
            },
        )

    monkeypatch.setattr(
        httpx.Client,
        "request",
        fake_request,
    )

    with pytest.raises(
        LoanApprovalAPIError,
        match="status 422",
    ):
        api_client.predict({})


def test_connection_error_is_converted(
    api_client,
    monkeypatch,
):
    def fake_request(
        self,
        method,
        endpoint,
        **kwargs,
    ):
        request = httpx.Request(
            method,
            f"http://test-api{endpoint}",
        )

        raise httpx.ConnectError(
            "Connection failed",
            request=request,
        )

    monkeypatch.setattr(
        httpx.Client,
        "request",
        fake_request,
    )

    with pytest.raises(
        LoanApprovalAPIError,
        match="Cannot connect to the API",
    ):
        api_client.health()