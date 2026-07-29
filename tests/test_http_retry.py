"""Regression tests for get_with_retry's 403 handling.

Covers an incident found running a full FRACTAL conversion at scale: IGN's
WMS (data.geopf.fr) started returning 403 Forbidden after ~3h of sustained
load from 16 parallel workers (an IP-level rate-limit/block, confirmed by
querying the same endpoint from a different network and getting 200). Before
this fix, 403 was treated like any other non-retryable error: the RGB/NIR/DTM
fetchers logged a warning and fell back to a default/degraded value (gray
color, height via the coarser 'ground_plane' method) — silently degrading
~30% of the shards produced while the ban was in effect and continuing to
degrade more for as long as the run kept going.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from ign_lidar.utils import http_retry
from ign_lidar.utils.http_retry import get_with_retry


@pytest.fixture(autouse=True)
def isolated_block_state(tmp_path, monkeypatch):
    """Point the cross-process block-state file at a throwaway path per test."""
    monkeypatch.setattr(http_retry, "_BLOCK_STATE_FILE", tmp_path / "block.json")


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Tests assert on retry/backoff behavior, not wall-clock time."""
    monkeypatch.setattr(http_retry.time, "sleep", MagicMock())


def _response(status_code):
    resp = MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.raise_for_status.side_effect = (
        None
        if status_code < 400
        else requests.exceptions.HTTPError(f"{status_code} error")
    )
    return resp


def test_success_first_try():
    with patch.object(http_retry.requests, "get", return_value=_response(200)) as get:
        result = get_with_retry("http://x", {}, operation_name="test")
    assert result.status_code == 200
    assert get.call_count == 1
    http_retry.time.sleep.assert_not_called()


def test_retryable_5xx_then_success():
    responses = [_response(503), _response(503), _response(200)]
    with patch.object(http_retry.requests, "get", side_effect=responses) as get:
        result = get_with_retry("http://x", {}, max_retries=3, operation_name="test")
    assert result.status_code == 200
    assert get.call_count == 3


def test_non_retryable_404_raises_immediately():
    with patch.object(http_retry.requests, "get", return_value=_response(404)) as get:
        with pytest.raises(requests.exceptions.HTTPError):
            get_with_retry("http://x", {}, operation_name="test")
    assert get.call_count == 1


def test_403_retries_indefinitely_instead_of_giving_up():
    """A 403 must never surface as a raised error / give the caller a reason
    to fall back — it should keep retrying past what max_retries would allow
    for any other status code."""
    responses = [_response(403)] * 10 + [_response(200)]
    with patch.object(http_retry.requests, "get", side_effect=responses) as get:
        result = get_with_retry("http://x", {}, max_retries=3, operation_name="test")
    assert result.status_code == 200
    assert get.call_count == 11  # far beyond max_retries=3


def test_403_records_and_clears_shared_block_state():
    responses = [_response(403), _response(200)]
    with patch.object(http_retry.requests, "get", side_effect=responses):
        get_with_retry("http://x", {}, operation_name="test")
    # success clears the block so the next unrelated call doesn't wait
    assert not http_retry._BLOCK_STATE_FILE.exists()


def test_second_caller_waits_out_a_block_recorded_by_first():
    """Simulates two worker processes sharing the block-state file: the
    first records a 403 block, the second (checked independently, as it
    would be from a different process) must wait instead of hitting the
    endpoint immediately."""
    http_retry._record_wms_block("first-worker")
    state = json.loads(http_retry._BLOCK_STATE_FILE.read_text())
    assert state["blocked_until"] > time.time()

    http_retry._wait_for_wms_unblock("second-worker")
    http_retry.time.sleep.assert_called()


def test_backoff_escalates_and_caps():
    first = http_retry._record_wms_block("test")
    second = http_retry._record_wms_block("test")
    third_and_beyond = http_retry._record_wms_block("test")
    assert first == http_retry._BLOCK_INITIAL_BACKOFF
    assert second > first
    for _ in range(20):
        third_and_beyond = http_retry._record_wms_block("test")
    assert third_and_beyond == http_retry._BLOCK_MAX_BACKOFF
