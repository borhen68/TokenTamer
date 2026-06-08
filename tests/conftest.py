"""Pytest fixtures for TokenTamer."""

import pytest
from token_tamer.skeletonizer import Skeletonizer


@pytest.fixture
def skeletonizer():
    return Skeletonizer(keep_docstrings=False, keep_class_attrs=True)
