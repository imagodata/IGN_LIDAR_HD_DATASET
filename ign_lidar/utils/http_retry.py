"""Retry-with-backoff helper for IGN WMS requests.

The IGN Géoplateforme WMS (data.geopf.fr) occasionally returns transient
502/503/504 under load. Callers that treated any non-2xx response as a
permanent failure (RGB/NIR orthophoto fetchers, RGE ALTI/LiDAR HD MNT
fetcher) silently fell back to a default/degraded value (gray color, no
height correction) on what was often just a momentary hiccup — this is
fine occasionally, but on a run doing one WMS call per (small) patch across
tens of thousands of patches, a non-trivial fraction of the dataset would
end up degraded for no real reason.

403 is handled separately from the other retryable codes (see
RETRYABLE_STATUS_CODES): these WMS/WFS layers are public and unauthenticated,
so a 403 here is not "permanently forbidden" — it is IGN rate-limiting or
temporarily IP-blocking the caller (observed after sustained high-concurrency
load, e.g. 16 parallel workers over several hours). Silently falling back on
this would degrade RGB/NIR/height for every patch until the ban lifts, which
can be a large fraction of a run. Instead, `get_with_retry` blocks and retries
indefinitely on 403 until the request succeeds, with an escalating backoff
capped at `_BLOCK_MAX_BACKOFF`, coordinated across worker processes via a
shared state file so 16 workers don't independently hammer the endpoint every
retry (which would likely prolong the ban rather than let it clear).
"""

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

#: Status codes worth retrying — transient server-side/rate-limit conditions.
#: Anything else (400, 404, ...) means the request itself is wrong; retrying
#: would just waste time. 403 is handled separately (see module docstring).
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

#: Cross-process coordination file for the 403 circuit breaker. Whichever
#: worker hits a 403 first writes `blocked_until` here; every worker
#: (including itself) checks this before its next attempt instead of racing
#: to retry independently. Best-effort — no file locking — a few redundant
#: probes from a race are harmless.
_BLOCK_STATE_FILE = Path(
    os.environ.get("IGN_WMS_BLOCK_STATE_FILE", "/tmp/ign_wms_block_state.json")
)
_BLOCK_INITIAL_BACKOFF = 30.0
_BLOCK_MAX_BACKOFF = 600.0  # 10 min cap — gentle enough not to re-trigger the ban


def _read_block_state() -> Dict[str, float]:
    try:
        return json.loads(_BLOCK_STATE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _write_block_state(blocked_until: float, backoff: float) -> None:
    try:
        tmp = _BLOCK_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps({"blocked_until": blocked_until, "backoff": backoff}))
        tmp.replace(_BLOCK_STATE_FILE)
    except OSError as e:
        logger.debug(f"Could not persist WMS block state (continuing in-process only): {e}")


def _wait_for_wms_unblock(operation_name: str) -> None:
    """Block until data.geopf.fr is no longer known to be rate-limiting us."""
    state = _read_block_state()
    blocked_until = state.get("blocked_until", 0.0)
    remaining = blocked_until - time.time()
    if remaining > 0:
        jitter = random.uniform(0, 5.0)
        logger.warning(
            f"{operation_name}: IGN WMS still rate-limited (403) — "
            f"waiting {remaining + jitter:.0f}s before next attempt "
            f"(another worker recorded this block; not falling back to defaults)"
        )
        time.sleep(remaining + jitter)


def _record_wms_block(operation_name: str) -> float:
    """Record a fresh 403 and return the backoff duration used."""
    state = _read_block_state()
    prev_backoff = state.get("backoff", _BLOCK_INITIAL_BACKOFF / 2)
    backoff = min(prev_backoff * 2, _BLOCK_MAX_BACKOFF)
    _write_block_state(time.time() + backoff, backoff)
    logger.error(
        f"{operation_name}: 403 Forbidden from IGN WMS (data.geopf.fr) — "
        f"treating as rate-limit/IP-block, not a permanent error. "
        f"Waiting {backoff:.0f}s then retrying (will NOT fall back to a "
        f"default/degraded value)."
    )
    return backoff


def _record_wms_success() -> None:
    """Clear any recorded block once a request succeeds."""
    if _BLOCK_STATE_FILE.exists():
        try:
            _BLOCK_STATE_FILE.unlink()
        except OSError:
            pass


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

    Retries on connection errors, timeouts, and RETRYABLE_STATUS_CODES up to
    `max_retries` times. 403 is retried indefinitely (not counted against
    `max_retries`) with a cross-process-coordinated escalating backoff — see
    module docstring for why.

    Raises the underlying `requests` exception if all bounded-retry attempts
    fail. Does NOT retry other 4xx errors (bad request/layer/params) — those
    are permanent, not transient, and raise immediately via
    raise_for_status().
    """
    last_exc: Optional[Exception] = None
    attempt = 0

    while True:
        _wait_for_wms_unblock(operation_name)

        try:
            response = requests.get(url, params=params, timeout=timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            last_exc = e
            if attempt >= max_retries:
                raise
            _sleep_backoff(attempt, backoff_base, operation_name, type(e).__name__)
            attempt += 1
            continue

        if response.status_code == 403:
            _record_wms_block(operation_name)
            continue  # unbounded — never falls through to raise_for_status()

        if response.status_code in RETRYABLE_STATUS_CODES and attempt < max_retries:
            _sleep_backoff(attempt, backoff_base, operation_name, f"HTTP {response.status_code}")
            attempt += 1
            continue

        response.raise_for_status()
        _record_wms_success()
        return response


def _sleep_backoff(attempt: int, backoff_base: float, operation_name: str, reason: str) -> None:
    delay = backoff_base * (2 ** attempt) + random.uniform(0, backoff_base)
    logger.warning(
        f"{operation_name}: {reason}, retrying in {delay:.1f}s (attempt {attempt + 1})"
    )
    time.sleep(delay)


__all__ = ["get_with_retry", "RETRYABLE_STATUS_CODES"]
