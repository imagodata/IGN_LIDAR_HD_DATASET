"""Structural regression tests for GPUStrategy / GPUChunkedStrategy.

These classes require CuPy + a real GPU to *run*, which is not available in
this CI/dev environment -- but their `compute()` methods contained two
unconditional AttributeErrors (`self.gpu_manager` never set;
`GPUProcessor.compute_normals()` never existed) that made them 100% dead
code regardless of hardware. Everything downstream of GPU_AVAILABLE in this
module (BatchTransferContext, GPUMemoryPoolIntegration, the stream
optimizer) does its own independent hardware detection and gracefully
CPU-falls-back, so a fake GPUProcessor stub plus patching this module's own
`GPU_AVAILABLE`/`GPUProcessor` names is enough to exercise the full
`compute()` code path -- and would have caught both bugs immediately.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


def _fake_points(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform([650_000, 6_860_000, 100.0], [650_100, 6_860_100, 130.0],
                       size=(n, 3))


class _FakeGPUProcessor:
    """Minimal stand-in with exactly the attributes/methods GPUStrategy
    and GPUChunkedStrategy actually call on `self.gpu_processor`."""

    def __init__(self, *args, **kwargs):
        self.use_gpu = True
        self.gpu_cache = None
        self.chunk_threshold = 10_000_000
        self.chunk_size = kwargs.get("chunk_size", 1_000_000)

    def compute_features(self, points, feature_types=None, k=20, show_progress=False):
        n = len(points)
        rng = np.random.default_rng(0)
        normals = rng.normal(size=(n, 3)).astype(np.float32)
        normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
        out = {}
        if feature_types is None or "normals" in feature_types:
            out["normals"] = normals
        if feature_types is None or "curvature" in feature_types:
            out["curvature"] = rng.uniform(0.0, 0.33, n).astype(np.float32)
        return out


@pytest.mark.parametrize("module_name, class_name", [
    ("ign_lidar.features.strategy_gpu", "GPUStrategy"),
    ("ign_lidar.features.strategy_gpu_chunked", "GPUChunkedStrategy"),
])
def test_gpu_strategy_compute_does_not_raise_attributeerror(module_name, class_name):
    """Regression test for two confirmed-dead bugs (pre-fix):
    - `self.gpu_manager` was never set (only a module-level `_gpu_manager`,
      which has no `gpu_pool` attribute either) -> AttributeError on every call.
    - GPUStrategy (not GPUChunkedStrategy) called the nonexistent
      `GPUProcessor.compute_normals()` -> AttributeError.
    Both are structural bugs independent of whether real GPU hardware/CuPy
    is present, so a fake GPUProcessor is sufficient to catch them.
    """
    import importlib
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    points = _fake_points()
    rgb = np.random.default_rng(1).uniform(0, 1, size=(len(points), 3)).astype(np.float32)

    with patch.object(module, "GPU_AVAILABLE", True), \
         patch.object(module, "GPUProcessor", _FakeGPUProcessor):
        strategy = cls(k_neighbors=8, verbose=False)
        result = strategy.compute(points, rgb=rgb, classification=None)

    for key in ("normals", "curvature", "verticality", "horizontality",
                "planarity", "sphericity", "height"):
        assert key in result, f"missing '{key}' in GPU strategy result"
    assert result["normals"].shape == (len(points), 3)
    assert result["horizontality"].shape == (len(points),)
    assert np.isfinite(result["normals"]).all()


def test_multi_scale_gpu_path_does_not_call_nonexistent_compute_normals():
    """Same bug, different call site: `MultiScaleFeatureComputer.
    _compute_single_scale_gpu()` also called the nonexistent
    `GPUProcessor.compute_normals()`. Only reachable when a real
    GPUProcessor is passed in AND use_gpu=True AND GPU_AVAILABLE, so this
    exercises the fixed private method directly instead.
    """
    from ign_lidar.features.compute.multi_scale import (
        MultiScaleFeatureComputer,
        ScaleConfig,
    )

    with patch("ign_lidar.features.compute.multi_scale.GPU_AVAILABLE", True):
        computer = MultiScaleFeatureComputer(
            scales=[
                ScaleConfig(name="fine", k_neighbors=8, search_radius=0.5, weight=1.0),
                ScaleConfig(name="coarse", k_neighbors=16, search_radius=1.5, weight=1.0),
            ],
            gpu_processor=_FakeGPUProcessor(),
            use_gpu=True,
        )
    assert computer.use_gpu is True

    points = _fake_points(n=100)
    result = computer._compute_single_scale_gpu(
        points, features=["normals", "curvature", "horizontality"],
        k_neighbors=8, search_radius=0.5,
    )
    assert "normals" in result and "curvature" in result and "horizontality" in result
    assert result["normals"].shape == (100, 3)
