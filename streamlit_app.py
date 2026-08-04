import os
from typing import Any

import streamlit as st

from loan_approval.api_client import (
    LoanApprovalAPIClient,
    LoanApprovalAPIError,
)


API_BASE_URL = os.getenv(
    "LOAN_API_URL",
    "http://127.0.0.1:8000",
)


st.set_page_config(
    page_title="Loan approval predictor",
    page_icon=":material/account_balance:",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        "About": (
            "An educational machine-learning application that "
            "demonstrates historical loan approval prediction."
        ),
    },
)


@st.cache_resource(show_spinner=False)
def get_api_client(base_url: str) -> LoanApprovalAPIClient:
    return LoanApprovalAPIClient(base_url=base_url)


@st.cache_data(
    ttl="30s",
    max_entries=5,
    show_spinner=False,
)
def get_api_health(base_url: str) -> dict[str, Any]:
    return get_api_client(base_url).health()


def display_application_summary(
    application: dict[str, Any],
) -> None:
    summary = [
        {
            "Field": "Applicant profile",
            "Value": (
                f"{application['gender']}, "
                f"{application['education']}"
            ),
        },
        {
            "Field": "Household",
            "Value": (
                f"Married: {application['married']}; "
                f"Dependents: {application['dependents']}"
            ),
        },
        {
            "Field": "Employment",
            "Value": (
                "Self-employed"
                if application["self_employed"] == "Yes"
                else "Not self-employed"
            ),
        },
        {
            "Field": "Applicant income",
            "Value": f"{application['applicant_income']:,.0f} per month",
        },
        {
            "Field": "Co-applicant income",
            "Value": f"{application['coapplicant_income']:,.0f} per month",
        },
        {
            "Field": "Loan request",
            "Value": (
                f"{application['loan_amount']:,.0f} thousand over "
                f"{application['loan_amount_term']:,.0f} months"
            ),
        },
        {
            "Field": "Credit history",
            "Value": (
                "Meets guidelines"
                if application["credit_history"] == 1
                else "Does not meet guidelines"
            ),
        },
        {
            "Field": "Property area",
            "Value": application["property_area"],
        },
    ]

    st.table(summary)


def display_prediction(
    result: dict[str, Any],
    application: dict[str, Any],
) -> None:
    prediction = result["prediction"]
    approval_score = float(result["approval_score"])
    threshold = float(result["classification_threshold"])
    distance_from_threshold = abs(
        approval_score - threshold
    )
    relationship = (
        "above"
        if approval_score >= threshold
        else "below"
    )

    with st.container(border=True):
        st.markdown(
            "### :material/analytics: Prediction result"
        )

        if prediction == "Approved":
            st.success(
                "The model predicts that this application may be approved.",
                icon=":material/check_circle:",
            )
        else:
            st.warning(
                "The model predicts that this application may be rejected.",
                icon=":material/warning:",
            )

        metric_columns = st.columns(
            3,
            gap="small",
        )

        metric_columns[0].metric(
            "Prediction",
            prediction,
        )
        metric_columns[1].metric(
            "Model approval score",
            f"{approval_score:.1%}",
        )
        metric_columns[2].metric(
            "Decision threshold",
            f"{threshold:.0%}",
        )

        st.progress(
            approval_score,
            text=f"Approval score: {approval_score:.1%}",
        )
        st.caption(
            f"The score is {distance_from_threshold:.1%} "
            f"{relationship} the model's decision threshold. "
            f"Model version: {result['model_version']}."
        )

        with st.expander(
            "How to interpret this result",
            icon=":material/lightbulb:",
        ):
            st.markdown(
                "- The **approval score** measures how strongly the "
                "application resembles historically approved cases.\n"
                "- The model returns **Approved** when the score is at "
                "or above the configured threshold.\n"
                "- This score is not a guarantee of approval, repayment, "
                "creditworthiness, or financial eligibility."
            )

        with st.expander(
            "Review submitted details",
            icon=":material/receipt_long:",
        ):
            display_application_summary(application)

        st.info(
            result["disclaimer"],
            icon=":material/info:",
        )


client = get_api_client(API_BASE_URL)


with st.container(horizontal_alignment="center"):
    st.badge(
        "Educational ML demo",
        icon=":material/science:",
        color="blue",
    )
    st.title(
        "Loan approval predictor",
        text_alignment="center",
    )
    st.caption(
        "Explore how a trained machine-learning model evaluates "
        "an application based on historical approval patterns.",
        text_alignment="center",
    )


st.warning(
    "Use this tool for learning and demonstration only. It must not "
    "be used to make or support a real lending decision.",
    icon=":material/gavel:",
)


with st.expander(
    "How to use this predictor",
    icon=":material/help:",
):
    st.markdown(
        "1. Complete the applicant and financial details below.\n"
        "2. Select **Evaluate application** to send the information "
        "to the prediction API.\n"
        "3. Review the model score, decision threshold, and responsible-"
        "use notice before interpreting the result."
    )
    st.caption(
        "Income values are monthly. Loan amount is entered in thousands "
        "of the same currency—for example, 130 represents 130,000."
    )


with st.form(
    "loan_application_form",
    border=True,
):
    st.subheader("Application details")
    st.caption(
        "All fields are required. Hover over the help icons for guidance."
    )

    left_column, right_column = st.columns(
        2,
        gap="large",
    )

    with left_column:
        st.markdown(
            "#### :material/person: Applicant profile"
        )

        gender = st.segmented_control(
            "Gender",
            options=["Male", "Female"],
            default="Male",
            selection_mode="single",
            help=(
                "Category recorded in the original training dataset. "
                "Sensitive attributes should not be used in real lending."
            ),
        )

        married = st.segmented_control(
            "Marital status",
            options=["Yes", "No"],
            default="Yes",
            selection_mode="single",
            format_func=lambda value: (
                "Married" if value == "Yes" else "Not married"
            ),
            help=(
                "Whether the applicant was recorded as married in the "
                "source dataset."
            ),
        )

        dependents = st.selectbox(
            "Number of dependents",
            options=["0", "1", "2", "3+"],
            help=(
                "People who financially depend on the applicant. "
                "Choose 3+ for three or more dependents."
            ),
        )

        education = st.segmented_control(
            "Education",
            options=["Graduate", "Not Graduate"],
            default="Graduate",
            selection_mode="single",
            help=(
                "Select Graduate if the applicant has completed a "
                "degree-level qualification."
            ),
        )

        self_employed = st.segmented_control(
            "Employment type",
            options=["Yes", "No"],
            default="No",
            selection_mode="single",
            format_func=lambda value: (
                "Self-employed"
                if value == "Yes"
                else "Not self-employed"
            ),
            help=(
                "Choose Self-employed when the applicant primarily "
                "works for their own business."
            ),
        )

        property_area = st.segmented_control(
            "Property area",
            options=["Urban", "Semiurban", "Rural"],
            default="Semiurban",
            selection_mode="single",
            help=(
                "The general area classification of the property "
                "connected to the loan request."
            ),
        )

    with right_column:
        st.markdown(
            "#### :material/payments: Financial details"
        )

        applicant_income = st.number_input(
            "Applicant income (monthly)",
            min_value=0.0,
            value=5000.0,
            step=100.0,
            format="%.0f",
            help=(
                "Applicant's gross monthly income before deductions, "
                "using your chosen currency."
            ),
        )

        coapplicant_income = st.number_input(
            "Co-applicant income (monthly)",
            min_value=0.0,
            value=1500.0,
            step=100.0,
            format="%.0f",
            help=(
                "Co-applicant's gross monthly income. Enter 0 when "
                "there is no co-applicant income."
            ),
        )

        loan_amount = st.number_input(
            "Loan amount (thousands)",
            min_value=1.0,
            value=130.0,
            step=5.0,
            format="%.0f",
            help=(
                "Requested principal in thousands. For example, enter "
                "130 for a requested amount of 130,000."
            ),
        )

        loan_amount_term = st.selectbox(
            "Loan term",
            options=[
                12.0,
                36.0,
                60.0,
                84.0,
                120.0,
                180.0,
                240.0,
                300.0,
                360.0,
                480.0,
            ],
            index=8,
            format_func=lambda value: f"{int(value)} months",
            help=(
                "The planned repayment period in months. "
                "For example, 360 months equals 30 years."
            ),
        )

        credit_history = st.segmented_control(
            "Credit history",
            options=[1, 0],
            default=1,
            selection_mode="single",
            format_func=lambda value: (
                "Meets guidelines"
                if value == 1
                else "Does not meet"
            ),
            help=(
                "Whether the recorded credit history meets the "
                "guidelines represented in the training dataset."
            ),
        )

        st.space("small")
        st.caption(
            "Check every value carefully before requesting a prediction."
        )

    submitted = st.form_submit_button(
        "Evaluate application",
        type="primary",
        icon=":material/analytics:",
        width="stretch",
    )


result_container = st.container()


with st.sidebar:
    st.markdown(
        "### :material/account_balance: Model service"
    )
    st.caption(
        "Live connection information for the prediction API."
    )

    try:
        health_information = get_api_health(API_BASE_URL)

        st.success(
            "API connected",
            icon=":material/cloud_done:",
        )
        st.caption(
            f"Model version: "
            f"{health_information.get('model_version', 'Unknown')}"
        )

    except LoanApprovalAPIError as error:
        st.error(
            "API unavailable",
            icon=":material/cloud_off:",
        )
        st.caption(str(error))

    with st.expander(
        "About the model",
        icon=":material/model_training:",
    ):
        st.write(
            "The model compares an application with patterns learned "
            "from historical approval data."
        )
        st.caption(
            "It does not measure repayment ability or replace a "
            "qualified human decision-maker."
        )

    with st.expander(
        "Technical details",
        icon=":material/settings:",
    ):
        st.caption(f"API address: {API_BASE_URL}")


if submitted:
    application = {
        "gender": gender,
        "married": married,
        "dependents": dependents,
        "education": education,
        "self_employed": self_employed,
        "applicant_income": applicant_income,
        "coapplicant_income": coapplicant_income,
        "loan_amount": loan_amount,
        "loan_amount_term": loan_amount_term,
        "credit_history": credit_history,
        "property_area": property_area,
    }

    try:
        with result_container.skeleton(height=240):
            prediction_result = client.predict(application)

        with result_container:
            display_prediction(
                prediction_result,
                application,
            )

    except LoanApprovalAPIError as error:
        with result_container:
            st.error(
                str(error),
                icon=":material/error:",
            )
