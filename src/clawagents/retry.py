"""Composable retry policy built on top of the existing ErrorClass taxonomy.

``LLMProvider._with_retry`` already implements jittered exponential backoff
for a small hard-coded set of errors. ``RetryPolicy`` promotes that to a
first-class, user-configurable object:

* Decide **which** :class:`~clawagents.errors.taxonomy.ErrorClass` values
  to retry.
* Configure ``max_retries``, ``base_delay``, ``max_delay``, and ``jitter``.
* Optionally cap retries per-error-class (e.g. one retry for auth errors,
  six for rate limits).

Providers still see a single knob — pass a ``RetryPolicy`` to
``LLMProvider(..., retry_policy=...)`` and the internal retry loop asks the
policy whether to retry and how long to wait.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from clawagents.errors.taxonomy import ErrorClass, ErrorDescriptor, classify_error


@dataclass
class RetryPolicy:
    """Policy deciding if a failed LLM call should be retried.

    Attributes:
        max_retries: global cap on retry attempts per call.
        retry_on: set of error classes eligible for retry.
        base_delay: first backoff, in seconds.
        max_delay: upper bound on the backoff.
        jitter: multiplicative uniform jitter factor (``0`` disables).
        per_class_max: optional per-:class:`ErrorClass` override of
            ``max_retries``. Unset classes use the global cap.
    """
    max_retries: int = 6
    retry_on: frozenset[ErrorClass] = field(
        default_factory=lambda: frozenset({
            ErrorClass.PROVIDER_RATE_LIMIT,
            ErrorClass.PROVIDER_INTERNAL,
            ErrorClass.PROVIDER_TRANSPORT,
        })
    )
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: float = 0.25
    per_class_max: dict[ErrorClass, int] = field(default_factory=dict)

    def classify(self, exc: BaseException) -> ErrorDescriptor:
        return classify_error(exc)

    def should_retry(
        self,
        exc: BaseException,
        attempt: int,
        *,
        descriptor: ErrorDescriptor | None = None,
    ) -> bool:
        """Return True if ``attempt`` (1-indexed) may be retried."""
        descriptor = descriptor or self.classify(exc)
        if descriptor.error_class not in self.retry_on:
            return False
        cap = self.per_class_max.get(descriptor.error_class, self.max_retries)
        return attempt <= cap

    def compute_delay(
        self,
        attempt: int,
        *,
        retry_after: float | None = None,
    ) -> float:
        """Return the seconds to sleep before retrying ``attempt+1``.

        If the error descriptor surfaces a ``retry_after`` hint (e.g. from
        a provider ``Retry-After`` header), it is honoured directly, capped
        by ``max_delay``.
        """
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.max_delay)
        delay = min(self.base_delay * (2 ** max(0, attempt - 1)), self.max_delay)
        if self.jitter > 0:
            factor = 1.0 + random.uniform(-self.jitter, self.jitter)
            delay *= max(0.0, factor)
        return delay


DEFAULT_RETRY_POLICY = RetryPolicy()
