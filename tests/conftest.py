"""Pytest configuration and shared fixtures."""

import os

import pytest


# Custom marker for tests that require Ollama
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_ollama: mark test as requiring Ollama to be running",
    )


# Skip decorator for use in tests
skip_if_github_action = pytest.mark.skipif(
    os.getenv("CI", "false").lower() == "true",
    reason="Skipping in CI environment (Ollama not available)",
)
