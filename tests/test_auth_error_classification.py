"""Auth-failure classification, including gateway-specific recovery hints."""

from __future__ import annotations

import httpx
import pytest

from clawagents.errors.taxonomy import ErrorClass, classify_error

MANTLE_URL = "https://bedrock-mantle.us-east-1.api.aws/anthropic/v1/messages"
# Verbatim shape of a Mantle bearer rejection — note it never names the host.
MANTLE_401_BODY = (
    "Error code: 401 - {'type': 'error', 'request_id': 'req_abc123', 'error': "
    "{'type': 'authentication_error', 'message': 'Invalid bearer token'}}"
)


class FakeAPIStatusError(Exception):
    """Stand-in for anthropic/openai ``APIStatusError`` (same duck type)."""

    def __init__(self, message: str, url: str, status_code: int = 401):
        super().__init__(message)
        self.status_code = status_code
        self.request = httpx.Request("POST", url)
        self.response = httpx.Response(status_code, request=self.request)


def test_mantle_401_hint_points_at_bedrock_key_not_anthropic_key():
    """The generic hint used to send users to ANTHROPIC_API_KEY, which also 401s."""
    descriptor = classify_error(FakeAPIStatusError(MANTLE_401_BODY, MANTLE_URL))

    assert descriptor.error_class is ErrorClass.PROVIDER_AUTH
    assert descriptor.retryable is False
    hint = descriptor.recovery_hint
    assert "BEDROCK_API_KEY" in hint
    assert "MANTLE_API_KEY" in hint
    assert "not valid on Mantle" in hint


def test_mantle_401_without_request_url_still_reads_as_bearer_problem():
    descriptor = classify_error(Exception(MANTLE_401_BODY))

    assert descriptor.error_class is ErrorClass.PROVIDER_AUTH
    assert "Bearer" in descriptor.recovery_hint


def test_mantle_x_api_key_error_keeps_its_own_hint():
    """Bearer and X-Api-Key failures need different fixes; don't collapse them."""
    err = FakeAPIStatusError(
        "Error code: 401 - {'error': {'message': 'invalid x-api-key'}}",
        MANTLE_URL,
    )
    descriptor = classify_error(err)

    assert descriptor.error_class is ErrorClass.PROVIDER_AUTH
    assert "anthropic>=0.95.0" in descriptor.recovery_hint


def test_plain_anthropic_401_keeps_generic_hint():
    err = FakeAPIStatusError(
        "Error code: 401 - {'error': {'message': 'invalid x-api-key'}}",
        "https://api.anthropic.com/v1/messages",
    )
    descriptor = classify_error(err)

    assert descriptor.error_class is ErrorClass.PROVIDER_AUTH
    assert "ANTHROPIC_API_KEY" in descriptor.recovery_hint
    assert "Mantle" not in descriptor.recovery_hint


def test_gemini_pydantic_extra_forbidden_is_not_auth():
    """6.20.62 sent FunctionResponse.call_id; google-genai rejected it as extra_forbidden."""
    err = Exception(
        "43 validation errors for _GenerateContentParameters\n"
        "contents.list[union[Content,str]].4.Content.parts.0.function_response.call_id\n"
        "  Extra inputs are not permitted [type=extra_forbidden, "
        "input_value='call_8652019', input_type=str]"
    )
    descriptor = classify_error(err)
    assert descriptor.error_class is not ErrorClass.PROVIDER_AUTH
    assert "API key" not in descriptor.recovery_hint
    assert "not an API-key" in descriptor.recovery_hint


@pytest.mark.parametrize(
    "message",
    [
        "Error code: 429 - insufficient_quota: You exceeded your current quota",
        "Your credit balance is too low to access the API",
    ],
)
def test_quota_errors_return_a_descriptor(message: str):
    """Regression: this branch returned a bare ErrorClass, breaking callers."""
    descriptor = classify_error(Exception(message))

    assert descriptor.error_class is ErrorClass.PROVIDER_QUOTA
    assert isinstance(descriptor.recovery_hint, str) and descriptor.recovery_hint
