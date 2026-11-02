"""Built-in pytest fixtures for AI evaluation.

Provides standard fixtures for configs, tracers, mock providers, and quality gates.
"""
from __future__ import annotations

from typing import Any, Generator

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

from veriflux._types import VerifluxConfig
from veriflux.cicd.gates import QualityGate
from veriflux.core.config import get_config
from veriflux.providers.mock import MockProvider
from veriflux.trajectory.tracer import Tracer

if pytest is not None:
    @pytest.fixture
    def veriflux_config() -> VerifluxConfig:
        """Provide active Veriflux global configuration."""
        return get_config()

    @pytest.fixture
    def veriflux_mock_provider() -> MockProvider:
        """Provide a fresh deterministic mock provider."""
        return MockProvider(default_response="Mock response for pytest")

    @pytest.fixture
    def veriflux_tracer() -> Generator[Tracer, None, None]:
        """Provide an active Tracer session for an individual test case."""
        with Tracer(name="pytest_case") as tracer:
            yield tracer

    @pytest.fixture
    def veriflux_gate() -> QualityGate:
        """Provide a standard QualityGate factory instance."""
        return QualityGate(name="pytest_gate")

