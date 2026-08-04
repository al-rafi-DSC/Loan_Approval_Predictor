import logging

from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)

from loan_approval import (
    HealthResponse,
    LoanApplicationRequest,
    LoanApprovalService,
    ModelInfoResponse,
    PredictionResponse,
)


logger = logging.getLogger(__name__)

RESPONSIBLE_USE_NOTICE = (
    "This prediction reflects patterns in historical approval "
    "decisions. It does not measure repayment ability and must "
    "not be used as an autonomous lending decision."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trusted model artifact once at startup."""

    app.state.loan_service = LoanApprovalService()

    yield

    del app.state.loan_service


app = FastAPI(
    title="Loan Approval Predictor API",
    summary="Serve the validated loan approval model.",
    description=RESPONSIBLE_USE_NOTICE,
    version="1.0.0",
    lifespan=lifespan,
)


def get_loan_service(
    request: Request,
) -> LoanApprovalService:
    return request.app.state.loan_service


LoanServiceDependency = Annotated[
    LoanApprovalService,
    Depends(get_loan_service),
]


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Check service health",
)
def health_check(
    service: LoanServiceDependency,
) -> HealthResponse:
    information = service.model_info()

    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_version=information["model_version"],
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get deployed model information",
)
def model_information(
    service: LoanServiceDependency,
) -> ModelInfoResponse:
    return ModelInfoResponse(
        **service.model_info()
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Prediction"],
    summary="Predict a historical approval outcome",
)
def predict_loan_approval(
    application: LoanApplicationRequest,
    service: LoanServiceDependency,
) -> PredictionResponse:
    try:
        prediction = service.predict(
            application.to_model_features()
        )

        return PredictionResponse(
            **prediction,
            disclaimer=RESPONSIBLE_USE_NOTICE,
        )

    except (TypeError, ValueError) as error:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(error),
        ) from error

    except Exception as error:
        logger.exception(
            "Unexpected prediction failure."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction could not be completed.",
        ) from error