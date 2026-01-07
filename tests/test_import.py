"""Test angst."""

import angst


def test_import() -> None:
    """Test that the package can be imported."""
    assert isinstance(angst.__name__, str)
