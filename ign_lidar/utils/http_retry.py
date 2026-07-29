"""Retry-with-backoff helper for IGN WMS requests.

The IGN Géoplateforme WMS (data.geopf.fr) occasionally returns transient
502/503/504 under load. Callers that treated any non-2xx response as a
permanent failure (RGB/NIR orthophoto fetchers, RGE ALTI/LiDAR HD MNT
fetcher) silently fell back to a default/degraded value (gray color, no
height correction) on what was often just a momentary hiccup — this is
fine occasionally, but on a run doing one WMS call per (small) patch across
tens of thousands of patches, a non-trivial fraction of the dataset would
end up degraded for no real reason.
"""

import logging
import random
import time
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

#: Status codes worth retrying — transient server-side/rate-limit conditions.
#: Anything else (400, 404, ...) means the request itself is wrong; retrying
#: would just waste time.
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def get_with_retry(
    url: str,
    params: Dict[str, Any],
    timeout: float = 30.0,
    max_retries: int = 3,
    backoff_base: float = 1.0,
    operation_name: str = "WMS request",
) -> requests.Response:
    """
    GET with exponential backoff + jitter on transient errors.

    Retries on connection errors, timeouts, and RETRYABLE_STATUS_CODES.
    Does NOT retry other 4xx errors (bad request/layer/params) — those are
    permanent, not transient, and raise immediately via raise_for_status().

    Raises the underlying `requests` exception if all attempts fail.
    """
    last_exc: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            _sleep_backoff(attempt, backoff_base, operation_name, type(e).__name__)
            continue

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            _sleep_backoff(attempt, backoff_base, operation_name, f"HTTP {response.status_code}")
            continue

        response.raise_for_status()
        return response

    # Unreachable in practice (loop always returns or raises), kept for mypy.
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{operation_name}: exhausted retries with no response")


def _sleep_backoff(attempt: int, backoff_base: float, operation_name: str, reason: str) -> None:
    delay = backoff_base * (2 ** attempt) + random.uniform(0, backoff_base)
    logger.warning(
        f"{operation_name}: {reason}, retrying in {delay:.1f}s (attempt {attempt + 1})"
    )
    time.sleep(delay)


__all__ = ["get_with_retry", "RETRYABLE_STATUS_CODES"]
