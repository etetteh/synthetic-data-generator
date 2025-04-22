# tests/test_config.py
import pytest
import os
from unittest.mock import patch, MagicMock
import logging

# Import the module to test
from synthetic_data_generator import config
from synthetic_data_generator import exceptions

# Reload config module for specific tests if needed, but often manage env vars is enough
# import importlib
# importlib.reload(config)

def test_load_environment_success(mocker, caplog):
    """Test load_environment when API key is set."""
    caplog.set_level(logging.INFO)
    # Ensure API key is set (handled by manage_environment_variables fixture)
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}, clear=True)
    mock_dotenv = mocker.patch('synthetic_data_generator.config.load_dotenv', return_value=True)

    config.load_environment()

    assert mock_dotenv.called # Check if dotenv was attempted
    assert ".env file loaded if found." in caplog.text
    # No ValueError should be raised

def test_load_environment_missing_key(mocker, capsys):
    """Test load_environment when API key is NOT set."""
    # Ensure API key is NOT set
    mocker.patch.dict(os.environ, {}, clear=True)
    mock_dotenv = mocker.patch('synthetic_data_generator.config.load_dotenv', return_value=True)

    with pytest.raises(ValueError, match="Please set the GOOGLE_API_KEY"):
        config.load_environment()

    captured = capsys.readouterr()
    assert "CRITICAL: GOOGLE_API_KEY environment variable not set." in captured.err
    assert mock_dotenv.called

def test_load_environment_dotenv_unavailable(mocker, caplog):
    """Test load_environment when python-dotenv is not installed."""
    caplog.set_level(logging.INFO)
    mocker.patch.dict(os.environ, {"GOOGLE_API_KEY": "fake_key"}, clear=True)
    # Simulate dotenv not being available
    mocker.patch('synthetic_data_generator.config.DOTENV_AVAILABLE', False)
    mocker.patch('synthetic_data_generator.config.load_dotenv', None) # Ensure it's None

    config.load_environment()

    assert ".env file loaded if found." not in caplog.text # Should not log loading
    # No ValueError should be raised

def test_constants_exist():
    """Check that key constants are defined."""
    assert isinstance(config.DEFAULT_MODEL_NAME, str)
    assert isinstance(config.DEFAULT_SAMPLES, int)
    assert isinstance(config.OUTPUT_FILE_FORMATS, list)
    assert isinstance(config.TYPE_MAP, dict)
    assert "string" in config.TYPE_MAP
    assert config.TYPE_MAP["string"] == str

# Test critical import failure (less common to test directly, but possible)
# You might need to manipulate sys.modules or use complex mocking
# def test_critical_import_failure(mocker):
#     mocker.patch.dict(sys.modules, {'langchain_google_genai': None})
#     with pytest.raises(SystemExit):
#          # How to trigger the import again? Requires careful setup or separate process.
#          pass
