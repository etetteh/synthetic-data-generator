# tests/conftest.py
import pytest
import os
from unittest.mock import MagicMock

# Mock LangChain classes if needed globally, or within specific tests
# Example: Mock the LLM client
@pytest.fixture
def mock_llm_client():
    """Fixture for a mocked ChatGoogleGenerativeAI client."""
    mock_client = MagicMock()
    # Configure mock responses as needed for specific tests
    mock_response = MagicMock()
    mock_response.content = '[]' # Default empty JSON list response
    mock_response.response_metadata = {'finish_reason': 'STOP', 'block_reason': 'N/A'}
    mock_client.invoke.return_value = mock_response
    # Mock attributes accessed in the generator init
    mock_client.model = "mock-gemini-model"
    mock_client.temperature = 0.5
    mock_client.top_p = 0.9
    mock_client.top_k = None
    return mock_client

@pytest.fixture
def sample_custom_format_def():
    """Fixture for a sample valid custom format definition."""
    return {
        "name": "sample_custom",
        "description": "A sample custom format.",
        "fields": {
            "id": {"type": "integer", "description": "Unique ID", "required": True, "example": 1},
            "text": {"type": "string", "description": "Some text content", "required": True},
            "score": {"type": "float", "description": "A score", "required": False, "example": 0.85},
            "verified": {"type": "boolean", "description": "Verification status", "required": False}
        }
    }

@pytest.fixture(autouse=True)
def manage_environment_variables():
    """Ensure GOOGLE_API_KEY is set for tests, reset afterwards."""
    original_value = os.environ.get("GOOGLE_API_KEY")
    # Set a dummy key for tests if not already set
    if not original_value:
        os.environ["GOOGLE_API_KEY"] = "test_api_key_for_pytest"
    yield # Run the test
    # Restore original value or unset if it wasn't set before
    if original_value:
        os.environ["GOOGLE_API_KEY"] = original_value
    elif "GOOGLE_API_KEY" in os.environ:
        del os.environ["GOOGLE_API_KEY"]
