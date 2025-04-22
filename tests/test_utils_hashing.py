# tests/test_utils_hashing.py
import pytest
import json
from unittest.mock import patch
import logging
import time

from synthetic_data_generator.utils import hashing

def test_hash_item_stable():
    """Test that identical dicts produce the same hash."""
    item1 = {"b": 2, "a": 1, "c": [3, 1]}
    item2 = {"a": 1, "b": 2, "c": [3, 1]} # Same content, different order
    assert hashing.hash_item(item1) == hashing.hash_item(item2)

def test_hash_item_different():
    """Test that different dicts produce different hashes."""
    item1 = {"a": 1, "b": 2}
    item2 = {"a": 1, "b": 3}
    item3 = {"a": 1, "c": 2}
    assert hashing.hash_item(item1) != hashing.hash_item(item2)
    assert hashing.hash_item(item1) != hashing.hash_item(item3)

def test_hash_item_nested():
    """Test hashing with nested structures."""
    item1 = {"a": 1, "b": {"c": 2, "d": [3, 4]}}
    item2 = {"a": 1, "b": {"d": [3, 4], "c": 2}}
    item3 = {"a": 1, "b": {"c": 2, "d": [3, 5]}}
    assert hashing.hash_item(item1) == hashing.hash_item(item2)
    assert hashing.hash_item(item1) != hashing.hash_item(item3)

def test_hash_item_unicode():
    """Test hashing with unicode characters."""
    item1 = {"name": "你好", "value": 1}
    item2 = {"name": "你好", "value": 1}
    item3 = {"name": "世界", "value": 1}
    assert hashing.hash_item(item1) == hashing.hash_item(item2)
    assert hashing.hash_item(item1) != hashing.hash_item(item3)

def test_hash_item_type_error_fallback_repr(mocker, caplog):
    """Test fallback to repr when json.dumps fails."""
    caplog.set_level(logging.WARNING)
    mock_dumps = mocker.patch('json.dumps', side_effect=TypeError("Cannot serialize"))
    # Mock time to prevent it being used in the final fallback if repr also fails
    mocker.patch('time.time', return_value=12345.0)

    # Use a simple dict that should be repr-able
    item = {"a": 1, "b": object()} # object() is not JSON serializable

    h = hashing.hash_item(item)

    mock_dumps.assert_called_once()
    assert "Fallback hash used: Could not serialize item for hashing." in caplog.text
    # Check if the hash is based on the repr (difficult to assert exact value)
    assert isinstance(h, str)
    assert len(h) == 32 # MD5 hash length

def test_hash_item_type_error_fallback_time(mocker, caplog):
    """Test final fallback to time+str when json.dumps and repr fails."""
    caplog.set_level(logging.WARNING)
    mock_dumps = mocker.patch('json.dumps', side_effect=TypeError("Cannot serialize"))
    # Mock sorted().items() inside the fallback to raise another error
    mocker.patch('builtins.sorted', side_effect=Exception("Cannot sort"))
    mock_time = mocker.patch('time.time', return_value=12345.0)

    item = {"a": 1}

    h = hashing.hash_item(item)

    mock_dumps.assert_called_once()
    mock_time.assert_called_once()
    assert "Fallback hash used" in caplog.text
    assert "Final hash fallback used" in caplog.text
    assert isinstance(h, str)
    assert len(h) == 32

def test_hash_item_unexpected_error(mocker, caplog):
    """Test fallback when an unexpected error occurs during hashing."""
    caplog.set_level(logging.ERROR)
    # Simulate an error other than TypeError during dumps
    mock_dumps = mocker.patch('json.dumps', side_effect=ValueError("Unexpected issue"))
    mock_time = mocker.patch('time.time', return_value=12345.0)

    item = {"a": 1}
    h = hashing.hash_item(item)

    mock_dumps.assert_called_once()
    mock_time.assert_called_once()
    assert "Unexpected error during item hashing" in caplog.text
    assert isinstance(h, str)
    assert len(h) == 32
