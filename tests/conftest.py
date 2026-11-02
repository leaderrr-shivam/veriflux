"""Pytest configuration and global test fixtures for Veriflux test suite."""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is in sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import pytest
import veriflux
from veriflux.providers.mock import MockProvider


@pytest.fixture
def mock_provider() -> MockProvider:
    return MockProvider()


@pytest.fixture
def sample_context() -> str:
    return (
        "The Apollo program was the third United States human spaceflight program carried out by NASA. "
        "First conceived in 1960 during the Eisenhower administration, Apollo succeeded in landing the "
        "first humans on the Moon in 1969 with Apollo 11. Neil Armstrong and Buzz Aldrin walked on the lunar surface."
    )

