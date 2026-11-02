"""Statistical dataset sampling strategies.

Provides uniform random sampling, stratified sampling across categorical metadata,
and deterministic systematic stride sampling.
"""
from __future__ import annotations

from collections import defaultdict
import random
from typing import Optional

from veriflux.datasets.loader import Dataset, TestCase


class Sampler:
    """Samples subsets of test cases for efficient evaluation rounds."""

    def random_sample(self, dataset: Dataset, n: int, seed: Optional[int] = None) -> Dataset:
        """Extract a uniform random sample of size n."""
        if n >= len(dataset):
            return Dataset(list(dataset.cases), name=f"{dataset.name}_sampled")
        rng = random.Random(seed)
        sampled = rng.sample(dataset.cases, n)
        return Dataset(sampled, name=f"{dataset.name}_sample_{n}")

    def stratified_sample(self, dataset: Dataset, n: int, key: str, seed: Optional[int] = None) -> Dataset:
        """Extract stratified sample maintaining category distribution."""
        groups = defaultdict(list)
        for c in dataset.cases:
            grp = c.metadata.get(key, "default")
            groups[grp].append(c)

        rng = random.Random(seed)
        total = len(dataset)
        sampled: list[TestCase] = []

        for grp_items in groups.values():
            grp_n = max(1, int(round((len(grp_items) / total) * n)))
            sampled.extend(rng.sample(grp_items, min(grp_n, len(grp_items))))

        return Dataset(sampled[:n], name=f"{dataset.name}_stratified")

    def systematic_sample(self, dataset: Dataset, n: int) -> Dataset:
        """Select every k-th item deterministically."""
        total = len(dataset)
        if n >= total or n <= 0:
            return Dataset(list(dataset.cases), name=dataset.name)
        step = total / n
        indices = [int(i * step) for i in range(n)]
        return Dataset([dataset.cases[i] for i in indices], name=f"{dataset.name}_systematic")

