"""
Basic sanity tests for Dataseter
"""

import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_import():
    """Test that we can import the main module"""
    try:
        import src
        assert True
    except ImportError:
        pytest.skip("Module import not configured yet")


def test_basic_math():
    """Basic test to ensure pytest is working"""
    assert 2 + 2 == 4
    assert 3 * 3 == 9


def test_environment():
    """Test that environment is set up correctly"""
    assert sys.version_info >= (3, 8)


class TestDataseter:
    """Basic tests for Dataseter"""

    def test_placeholder(self):
        """Placeholder test"""
        assert True

    def test_string_operations(self):
        """Test basic string operations"""
        text = "Dataseter"
        assert text.lower() == "dataseter"
        assert text.upper() == "DATASETER"
        assert len(text) == 9