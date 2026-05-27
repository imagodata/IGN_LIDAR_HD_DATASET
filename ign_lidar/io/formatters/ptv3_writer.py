"""
Dataset writer for the PTv3 / Pointcept export layout.

Wraps :class:`Ptv3Formatter` with the orchestration concerns it deliberately
does not own: split assignment, per-patch writes into ``train/val/test``
subdirectories, and the final ``meta.json`` summary.

Typical use::

    splits = assign_splits_by_tile(
        tile_ids=["0942_6543", "0942_6544", "0943_6543"],
        train=0.8, val=0.1, test=0.1, seed=42,
    )
    writer = Ptv3DatasetWriter(formatter, output_root, splits)
    for patch in patches:
        writer.write_patch(patch)
    writer.finalize()
"""

from __future__ import annotations

import hashlib
import logging
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from .ptv3_formatter import Ptv3Formatter

logger = logging.getLogger(__name__)

VALID_SPLITS: Tuple[str, ...] = ("train", "val", "test")


def assign_splits_by_tile(
    tile_ids: Sequence[str],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """Deterministically assign each unique tile to ``train``/``val``/``test``.

    Splitting at the tile level (not patch level) prevents spatial leakage:
    no patch from a given dalle ever appears in two splits at once.

    Args:
        tile_ids: Iterable of tile identifiers. Duplicates are collapsed.
        train, val, test: Ratios in [0, 1]. Must sum to 1 within 1e-6.
        seed: RNG seed for reproducible assignment.

    Returns:
        Dict mapping each unique tile id to its split label.
    """
    total = train + val + test
    if not (1.0 - 1e-6 <= total <= 1.0 + 1e-6):
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
    if any(r < 0 for r in (train, val, test)):
        raise ValueError("split ratios must be non-negative")

    unique_tiles = sorted({str(t) for t in tile_ids})
    n = len(unique_tiles)
    if n == 0:
        return {}

    rng = np.random.default_rng(seed)
    shuffled = list(unique_tiles)
    rng.shuffle(shuffled)

    # Compute integer counts; assign any rounding remainder to train.
    n_val = int(round(n * val))
    n_test = int(round(n * test))
    n_train = n - n_val - n_test
    if n_train < 0:
        # Pathological ratios — clip val/test back.
        n_train = 0
        n_val = min(n_val, n)
        n_test = n - n_val

    assignment: Dict[str, str] = {}
    cursor = 0
    for split_name, count in (("train", n_train), ("val", n_val), ("test", n_test)):
        for tile in shuffled[cursor : cursor + count]:
            assignment[tile] = split_name
        cursor += count
    return assignment


def assign_splits_by_patch(
    patch_ids: Sequence[str],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
) -> Dict[str, str]:
    """Patch-level split — use only when tile_id is unavailable or for sanity checks.

    Spatial leakage is likely; prefer :func:`assign_splits_by_tile` whenever
    tile identifiers are known.
    """
    return assign_splits_by_tile(patch_ids, train=train, val=val, test=test, seed=seed)


class HashSplitAssigner:
    """Stateless, deterministic tile -> split mapping driven by MD5(seed:tile_id).

    Useful for streaming pipelines where the full tile list is not known
    upfront. A given tile_id always maps to the same split for a given seed,
    and assignments are stable when new tiles are added later (no rebalance).

    The empirical distribution converges to the requested ratios as the
    number of tiles grows but is not exact for small N. Implements just
    enough of the Mapping protocol for :class:`Ptv3DatasetWriter` to consume.
    """

    def __init__(
        self,
        train: float = 0.8,
        val: float = 0.1,
        test: float = 0.1,
        seed: int = 42,
    ):
        total = train + val + test
        if not (1.0 - 1e-6 <= total <= 1.0 + 1e-6):
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        if any(r < 0 for r in (train, val, test)):
            raise ValueError("split ratios must be non-negative")
        self._train = train
        self._val = val
        self._test = test
        self._seed = seed

    def _resolve(self, tile_id: str) -> str:
        # 32 bits of MD5 are enough for a stable uniform [0,1) draw.
        h = hashlib.md5(f"{self._seed}:{tile_id}".encode()).hexdigest()
        p = int(h[:8], 16) / 0xFFFFFFFF
        if p < self._train:
            return "train"
        if p < self._train + self._val:
            return "val"
        return "test"

    def get(self, tile_id: str, default=None) -> str:
        # `default` is ignored — every tile resolves to a valid split.
        return self._resolve(str(tile_id))

    def __getitem__(self, tile_id: str) -> str:
        return self._resolve(str(tile_id))

    def __contains__(self, tile_id: str) -> bool:  # pragma: no cover - trivial
        return True

    def values(self) -> Iterable[str]:
        # Exposed for Ptv3DatasetWriter's __init__ validation. The assigner
        # only ever produces splits in VALID_SPLITS, so this satisfies it.
        return VALID_SPLITS


class Ptv3DatasetWriter:
    """Drives ``Ptv3Formatter`` over a stream of patches into a Pointcept layout."""

    def __init__(
        self,
        formatter: Ptv3Formatter,
        output_root: Path,
        split_assignment: Union[Mapping[str, str], "HashSplitAssigner"],
        split_key: str = "tile_id",
        on_unknown: str = "error",
    ):
        """
        Args:
            formatter: Configured :class:`Ptv3Formatter` instance.
            output_root: Root directory for the dataset. Created if missing.
            split_assignment: Mapping from the key (tile_id by default) to a
                split label among ``train``/``val``/``test``.
            split_key: Which patch key to look up in ``split_assignment``.
                Use ``"patch_id"`` when working with patch-level splits.
            on_unknown: ``"error"`` (default) raises if a patch's key is
                missing from ``split_assignment``; ``"skip"`` drops it with
                a warning; ``"test"`` routes unknown patches to the test set.
        """
        if on_unknown not in {"error", "skip", "test"}:
            raise ValueError(f"on_unknown must be error|skip|test, got {on_unknown!r}")
        unknown_split = {v for v in split_assignment.values()} - set(VALID_SPLITS)
        if unknown_split:
            raise ValueError(
                f"split_assignment contains invalid splits {unknown_split}; "
                f"allowed: {VALID_SPLITS}"
            )

        self.formatter = formatter
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Store by reference: HashSplitAssigner is stateless, plain dicts
        # shouldn't be mutated by the caller after handing them over.
        self.split_assignment = split_assignment
        self.split_key = split_key
        self.on_unknown = on_unknown
        self._counts: Counter = Counter()
        self._finalized = False

    def write_patch(self, patch: Dict[str, np.ndarray]) -> Optional[Path]:
        """Format and persist one patch into its assigned split subdir.

        Returns the path written, or ``None`` if the patch was skipped.
        """
        if self._finalized:
            raise RuntimeError("write_patch called after finalize()")

        key_value = patch.get(self.split_key)
        if key_value is None:
            raise KeyError(
                f"patch is missing split key {self.split_key!r}; "
                f"available keys: {sorted(patch.keys())}"
            )

        split = self.split_assignment.get(str(key_value))
        if split is None:
            if self.on_unknown == "error":
                raise KeyError(
                    f"{self.split_key}={key_value!r} not found in split_assignment; "
                    f"call assign_splits_by_tile() first or set on_unknown='skip'/'test'"
                )
            if self.on_unknown == "skip":
                logger.warning("ptv3_writer: skipping patch with unknown %s=%s",
                               self.split_key, key_value)
                return None
            split = "test"
            logger.warning("ptv3_writer: routing unknown %s=%s to 'test'",
                           self.split_key, key_value)

        formatted = self.formatter.format_patch(patch)
        target = self.formatter.write_patch(formatted, self.output_root / split)
        self._counts[split] += 1
        return target

    def finalize(self) -> Dict[str, int]:
        """Write ``meta.json`` and return the per-split patch counts."""
        counts = {split: self._counts.get(split, 0) for split in VALID_SPLITS}
        if self._finalized:
            return counts
        self.formatter.write_meta(self.output_root, num_patches_per_split=counts)
        self._finalized = True
        return counts

    @property
    def counts(self) -> Dict[str, int]:
        """Current per-split write counts (live; reflects writes so far)."""
        return {split: self._counts.get(split, 0) for split in VALID_SPLITS}
