# tests/test_loading_document_loader.py
import pytest
from unittest.mock import patch, MagicMock
import os
import logging

# Import the module to test
from synthetic_data_generator.loading import document_loader
from synthetic_data_generator import exceptions

# Mock the external dependency AutoDocumentLoader
@pytest.fixture
def mock_ad_loader(mocker):
    mock_instance = MagicMock()
    mock_class = MagicMock(return_value=mock_instance)
    # Patch within the document_loader module where it's imported
    mocker.patch('synthetic_data_generator.loading.document_loader.AutoDocumentLoader', mock_class)
    # Assume loading is enabled for most tests
    mocker.patch('synthetic_data_generator.loading.document_loader.DOCUMENT_LOADING_ENABLED', True)
    return mock_instance, mock_class

@pytest.fixture
def mock_os_path(mocker):
    return mocker.patch('os.path.exists')

def test_load_document_texts_success(mock_ad_loader, mock_os_path):
    """Test successful document loading and text extraction."""
    mock_loader_instance, _ = mock_ad_loader
    mock_os_path.return_value = True # Path exists

    # Simulate ADLoader returning Document-like objects
    mock_doc1 = MagicMock()
    mock_doc1.page_content = " Content of doc 1.  "
    mock_doc2 = MagicMock()
    mock_doc2.page_content = " Content of doc 2. "
    mock_loader_instance.load.return_value = [mock_doc1, mock_doc2]

    texts = document_loader.load_document_texts("dummy/path")

    mock_os_path.assert_called_once_with("dummy/path")
    mock_loader_instance.load.assert_called_once()
    assert texts == ["Content of doc 1.", "Content of doc 2."] # Check stripping

def test_load_document_texts_path_not_found(mock_os_path):
    """Test error when document path does not exist."""
    mock_os_path.return_value = False # Path does not exist

    with pytest.raises(exceptions.ConfigurationError, match="Document path not found"):
        document_loader.load_document_texts("invalid/path")

    mock_os_path.assert_called_once_with("invalid/path")

def test_load_document_texts_loading_disabled(mocker, mock_os_path):
    """Test error when document loading is disabled."""
    mock_os_path.return_value = True
    # Disable loading
    mocker.patch('synthetic_data_generator.loading.document_loader.DOCUMENT_LOADING_ENABLED', False)

    with pytest.raises(exceptions.LoaderError, match="Document loading is disabled"):
        document_loader.load_document_texts("dummy/path")

def test_load_document_texts_loader_returns_empty(mock_ad_loader, mock_os_path):
    """Test error when the loader returns an empty list."""
    mock_loader_instance, _ = mock_ad_loader
    mock_os_path.return_value = True
    mock_loader_instance.load.return_value = [] # Loader returns empty

    with pytest.raises(exceptions.LoaderError, match="No documents successfully loaded"):
        document_loader.load_document_texts("dummy/path")

def test_load_document_texts_no_valid_content(mock_ad_loader, mock_os_path, caplog):
    """Test error when loaded documents have no valid page_content."""
    caplog.set_level(logging.WARNING)
    mock_loader_instance, _ = mock_ad_loader
    mock_os_path.return_value = True

    mock_doc1 = MagicMock(page_content="   ") # Whitespace only
    mock_doc2 = MagicMock(page_content=None) # None content
    mock_doc3 = MagicMock() # No page_content attribute
    del mock_doc3.page_content # Ensure attribute is missing
    mock_doc4 = MagicMock(page_content=123) # Non-string content

    mock_loader_instance.load.return_value = [mock_doc1, mock_doc2, mock_doc3, mock_doc4]

    with pytest.raises(exceptions.LoaderError, match="No non-empty text content found"):
        document_loader.load_document_texts("dummy/path")

    # Check warnings for skipped docs
    assert "empty 'page_content'. Skipping." in caplog.text
    assert "missing, non-string, or empty 'page_content'. Skipping." in caplog.text # Covers None, missing, non-string

def test_load_document_texts_loader_exception(mock_ad_loader, mock_os_path):
    """Test wrapping exceptions raised by the underlying loader."""
    mock_loader_instance, _ = mock_ad_loader
    mock_os_path.return_value = True
    mock_loader_instance.load.side_effect = RuntimeError("ADLoader internal error")

    with pytest.raises(exceptions.LoaderError, match="Document loading or processing failed: ADLoader internal error"):
        document_loader.load_document_texts("dummy/path")

def test_load_document_texts_large_doc_warning(mock_ad_loader, mock_os_path, mocker, caplog):
    """Test the warning for potentially large document content."""
    caplog.set_level(logging.WARNING)
    mock_loader_instance, _ = mock_ad_loader
    mock_os_path.return_value = True

    # Create mock docs with large content to trigger the warning
    # Threshold is 500k tokens, approx 2M chars
    large_content = "a" * 2_000_001
    mock_doc1 = MagicMock(page_content=large_content)
    mock_loader_instance.load.return_value = [mock_doc1]

    # Mock the config threshold if needed, or just ensure content is large enough
    # mocker.patch('synthetic_data_generator.loading.document_loader.context_warning_threshold_tokens', 100)

    document_loader.load_document_texts("large/doc/path")

    assert "Combined document text is large" in caplog.text
    assert "approx. 500,000 tokens" in caplog.text # Based on char/4 heuristic
