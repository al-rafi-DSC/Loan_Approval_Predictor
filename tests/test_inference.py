import pytest

from loan_approval import LoanApprovalService


@pytest.fixture(scope="module")
def service():
    return LoanApprovalService()


@pytest.fixture()
def valid_model_input():
    return {
        "Gender": "Male",
        "Married": "Yes",
        "Dependents": "1",
        "Education": "Graduate",
        "Self_Employed": "No",
        "ApplicantIncome": 5000,
        "CoapplicantIncome": 1500,
        "LoanAmount": 130,
        "Loan_Amount_Term": 360,
        "Credit_History": 1,
        "Property_Area": "Semiurban",
    }


def test_model_information(service):
    information = service.model_info()

    assert information["model_version"] == "1.0.0"
    assert information["feature_count"] == 11
    assert information["classification_threshold"] == 0.34


def test_valid_prediction(
    service,
    valid_model_input,
):
    prediction = service.predict(
        valid_model_input
    )

    assert prediction["prediction"] in {
        "Approved",
        "Rejected",
    }
    assert prediction["predicted_label"] in {
        "Y",
        "N",
    }
    assert 0 <= prediction["approval_score"] <= 1
    assert (
        prediction["classification_threshold"]
        == 0.34
    )


def test_prediction_is_deterministic(
    service,
    valid_model_input,
):
    first_prediction = service.predict(
        valid_model_input
    )
    second_prediction = service.predict(
        valid_model_input
    )

    assert first_prediction == second_prediction


def test_invalid_credit_history(
    service,
    valid_model_input,
):
    invalid_input = valid_model_input.copy()
    invalid_input["Credit_History"] = 4

    with pytest.raises(
        ValueError,
        match="Credit_History",
    ):
        service.predict(invalid_input)


def test_missing_feature(
    service,
    valid_model_input,
):
    invalid_input = valid_model_input.copy()
    invalid_input.pop("LoanAmount")

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        service.predict(invalid_input)


def test_unexpected_feature(
    service,
    valid_model_input,
):
    invalid_input = valid_model_input.copy()
    invalid_input["CustomerName"] = "Example"

    with pytest.raises(
        ValueError,
        match="Unexpected features",
    ):
        service.predict(invalid_input)


def test_negative_income(
    service,
    valid_model_input,
):
    invalid_input = valid_model_input.copy()
    invalid_input["ApplicantIncome"] = -100

    with pytest.raises(
        ValueError,
        match="cannot be negative",
    ):
        service.predict(invalid_input)