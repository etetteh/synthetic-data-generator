# tests/test_formats_utils.py
import pytest
import json
import logging
from synthetic_data_generator.formats import utils
from synthetic_data_generator import exceptions
from synthetic_data_generator import config # For TYPE_MAP

# Sample valid JSON content
VALID_FORMAT_JSON = """
{
  "name": "test_format",
  "description": "A test format.",
  "fields": {
    "field1": {
      "type": "string",
      "description": "First field",
      "required": true
    },
    "field2": {
      "type": "integer",
      "description": "Second field",
      "required": false,
      "example": 100
    }
  }
}
"""

# Sample invalid JSON content
INVALID_JSON = """
{
  "name": "test_format",
  "fields": {
    "field1": { "type": "string", "description": "Missing comma" }
    "field2": { "type": "integer", "description": "Another field" }
  }
}
"""

# Sample structurally invalid content
INVALID_SCHEMA_NO_FIELDS = '{"name": "test"}'
INVALID_SCHEMA_BAD_FIELD_TYPE = '{"fields": {"f1": "not_a_dict"}}'
INVALID_SCHEMA_MISSING_TYPE = '{"fields": {"f1": {"description": "desc"}}}'
INVALID_SCHEMA_INVALID_TYPE = '{"fields": {"f1": {"type": "invalid", "description": "desc"}}}'
INVALID_SCHEMA_MISSING_DESC = '{"fields": {"f1": {"type": "string"}}}'

def test_load_valid_format(tmp_path):
    """Test loading a valid custom format JSON file."""
    p = tmp_path / "valid_format.json"
    p.write_text(VALID_FORMAT_JSON, encoding='utf-8')

    definition = utils.load_and_validate_custom_format(str(p))

    assert isinstance(definition, dict)
    assert definition["name"] == "test_format"
    assert "fields" in definition
    assert "field1" in definition["fields"]
    assert definition["fields"]["field1"]["type"] == "string"
    assert definition["fields"]["field2"]["required"] is False # Check default handling if needed
    assert definition["fields"]["field2"]["example"] == 100

def test_load_file_not_found(tmp_path):
    """Test loading a non-existent file."""
    non_existent_path = tmp_path / "not_a_file.json"
    with pytest.raises(exceptions.ConfigurationError, match="Custom format definition file not found"):
        utils.load_and_validate_custom_format(str(non_existent_path))

def test_load_invalid_json(tmp_path):
    """Test loading a file with invalid JSON syntax."""
    p = tmp_path / "invalid.json"
    p.write_text(INVALID_JSON, encoding='utf-8')
    with pytest.raises(exceptions.ConfigurationError, match="Invalid JSON"):
        utils.load_and_validate_custom_format(str(p))

@pytest.mark.parametrize("content, error_msg_part", [
    (INVALID_SCHEMA_NO_FIELDS, "must contain a 'fields' key"),
    (INVALID_SCHEMA_BAD_FIELD_TYPE, "must be a dictionary"),
    (INVALID_SCHEMA_MISSING_TYPE, "missing or invalid 'type'"),
    (INVALID_SCHEMA_INVALID_TYPE, f"Must be one of: {', '.join(config.TYPE_MAP.keys())}"),
    (INVALID_SCHEMA_MISSING_DESC, "must have a non-empty string 'description'"),
    ('{"fields": {}}', "'fields' dictionary cannot be empty"),
    ('{"fields": {"f1": {"type": "string", "description": "ok", "required": "yes"}}}', "non-boolean 'required' value"),
])
def test_load_invalid_schema(tmp_path, content, error_msg_part):
    """Test loading files with various schema validation errors."""
    p = tmp_path / "invalid_schema.json"
    p.write_text(content, encoding='utf-8')
    with pytest.raises(exceptions.ConfigurationError, match="Invalid structure"):
        try:
            utils.load_and_validate_custom_format(str(p))
        except exceptions.ConfigurationError as e:
            # Check the underlying validation error message
            assert error_msg_part in str(e.__cause__) or error_msg_part in str(e)
            raise e # Re-raise to satisfy pytest.raises

def test_load_default_name_description(tmp_path):
    """Test default name/description assignment when missing."""
    content = """
    {
      "fields": {
        "data": {"type": "string", "description": "Some data"}
      }
    }
    """
    p = tmp_path / "minimal_format.json"
    p.write_text(content, encoding='utf-8')

    definition = utils.load_and_validate_custom_format(str(p))

    assert definition["name"] == "minimal_format"
    assert "minimal_format" in definition["description"]

def test_load_example_type_warning(tmp_path, caplog):
    """Test warning for inconsistent example type."""
    caplog.set_level(logging.WARNING)
    content = """
    {
      "fields": {
        "count": {"type": "integer", "description": "A count", "example": "not_an_int"}
      }
    }
    """
    p = tmp_path / "bad_example.json"
    p.write_text(content, encoding='utf-8')

    utils.load_and_validate_custom_format(str(p))

    assert "inconsistent with declared type (integer)" in caplog.text
