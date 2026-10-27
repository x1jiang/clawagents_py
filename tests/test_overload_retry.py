import pytest

from clawagents.errors.taxonomy import ErrorClass, classify_error
from clawagents.providers.llm import _is_retryable


@pytest.mark.parametrize("message", [
    "Our servers are currently overloaded. Please try again later.",
    "server_is_overloaded",
    "Model temporarily overloaded",
])
def test_unstructured_overload_is_retryable_provider_error(message):
    error = RuntimeError(message)
    assert _is_retryable(error)
    assert classify_error(error).error_class == ErrorClass.PROVIDER_INTERNAL
    assert classify_error(error).retryable


def test_unrelated_error_does_not_become_retryable():
    error = RuntimeError("invalid model name")
    assert not _is_retryable(error)
    assert not classify_error(error).retryable
