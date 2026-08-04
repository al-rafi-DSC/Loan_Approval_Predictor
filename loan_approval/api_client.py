from typing import Any

import httpx


class LoanApprovalAPIError(RuntimeError):
    """Raised when communication with the prediction API fails."""


class LoanApprovalAPIClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")

        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout),
        )

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def model_info(self) -> dict[str, Any]:
        return self._request("GET", "/model-info")

    def predict(self, application: dict[str, Any]) -> dict[str, Any]:
        return self._request(
            "POST",
            "/predict",
            json=application,
        )

    def close(self) -> None:
        self._client.close()

    def _request(
        self,
        method: str,
        endpoint: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            response = self._client.request(
                method,
                endpoint,
                **kwargs,
            )

        except httpx.ConnectError as error:
            raise LoanApprovalAPIError(
                f"Cannot connect to the API at {self.base_url}. "
                "Make sure FastAPI is running."
            ) from error

        except httpx.TimeoutException as error:
            raise LoanApprovalAPIError(
                "The prediction API took too long to respond."
            ) from error

        except httpx.HTTPError as error:
            raise LoanApprovalAPIError(
                f"An API communication error occurred: {error}"
            ) from error

        if response.is_success:
            return response.json()

        try:
            response_body = response.json()
            detail = response_body.get("detail", response_body)
        except ValueError:
            detail = response.text or "Unknown API error"

        raise LoanApprovalAPIError(
            f"API request failed with status "
            f"{response.status_code}: {detail}"
        )