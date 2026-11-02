"""Datasets management and synthetic data generation for Veriflux.

Provides dataset loaders, golden datasets, statistical samplers, and adversarial generators.
"""
from __future__ import annotations

from veriflux.datasets.generator import SyntheticGenerator
from veriflux.datasets.golden import GoldenDataset
from veriflux.datasets.loader import Dataset, DatasetLoader, TestCase
from veriflux.datasets.sampler import Sampler

__all__ = [
    # Loader
    "DatasetLoader",
    "Dataset",
    "TestCase",
    # Generator
    "SyntheticGenerator",
    # Golden
    "GoldenDataset",
    # Sampler
    "Sampler",
]

