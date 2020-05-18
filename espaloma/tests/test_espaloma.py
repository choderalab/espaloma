"""
Unit and regression test for the espaloma package.
"""

# Import package, test suite, and other packages as needed
import espaloma
import pytest
import sys

def test_espaloma_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "espaloma" in sys.modules
