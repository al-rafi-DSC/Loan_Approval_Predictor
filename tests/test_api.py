import pytest

from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def valid_request():
    return {
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


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "model_loaded": True,
        "model_version": "1.0.0",
    }


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    body = response.json()

    assert body["model_version"] == "1.0.0"
    assert body["feature_count"] == 11
    assert body["classification_threshold"] == 0.34


def test_valid_prediction(
    client,
    valid_request,
):
    response = client.post(
        "/predict",
        json=valid_request,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["prediction"] in {
        "Approved",
        "Rejected",
    }
    assert body["predicted_label"] in {
        "Y",
        "N",
    }
    assert 0 <= body["approval_score"] <= 1
    assert body["classification_threshold"] == 0.34
    assert body["model_version"] == "1.0.0"
    assert body["disclaimer"]


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("credit_history", 4),
        ("applicant_income", -100),
        ("loan_amount", 0),
        ("gender", "Unknown"),
    ],
)
def test_invalid_values_return_422(
    client,
    valid_request,
    field,
    invalid_value,
):
    invalid_request = valid_request.copy()
    invalid_request[field] = invalid_value

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422


def test_missing_field_returns_422(
    client,
    valid_request,
):
    invalid_request = valid_request.copy()
    invalid_request.pop("education")

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422


def test_extra_field_returns_422(
    client,
    valid_request,
):
    invalid_request = valid_request.copy()
    invalid_request["customer_name"] = "Example"

    response = client.post(
        "/predict",
        json=invalid_request,
    )

    assert response.status_code == 422


def test_nullable_fields_are_accepted(
    client,
    valid_request,
):
    request_with_missing_values = valid_request.copy()

    request_with_missing_values.update({
        "gender": None,
        "married": None,
        "dependents": None,
        "self_employed": None,
        "loan_amount": None,
        "loan_amount_term": None,
        "credit_history": None,
    })

    response = client.post(
        "/predict",
        json=request_with_missing_values,
    )

    assert response.status_code == 200