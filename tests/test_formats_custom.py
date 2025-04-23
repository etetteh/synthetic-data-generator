# tests/test_formats_custom.py
import pytest
import logging
from synthetic_data_generator.formats import custom
from synthetic_data_generator import exceptions
from synthetic_data_generator import config # For TYPE_MAP

# Use the fixture from conftest.py
def test_custom_handler_init(sample_custom_format_def):
    """Test successful initialization with a valid definition."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    assert handler.get_format_name() == "sample_custom"
    assert "sample custom format" in handler.get_description()
    assert handler.get_field_names() == ["id", "text", "score", "verified"]

def test_custom_handler_prompts(sample_custom_format_def):
    """Test prompt building for custom formats."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    system_prompt = handler.build_system_prompt()
    query_prompt = handler.build_query_prompt("custom query", 10)
    doc_prompt = handler.build_document_prompt(["custom doc"], 2)

    # Check system prompt includes field definitions
    assert "**Field Definitions:**" in system_prompt
    assert "`id` (integer, (Required))" in system_prompt
    assert "Unique ID (e.g., `1`)" in system_prompt # Check example inclusion
    assert "`score` (float, (Optional))" in system_prompt
    assert "A score (e.g., `0.85`)" in system_prompt
    assert "MATRIX-Inspired" in system_prompt

    # Check user prompts include format name and example
    example_str = handler.get_example_structure_string()
    assert handler.get_format_name() in query_prompt
    assert example_str in query_prompt
    assert "custom query" in query_prompt
    assert "generate 10" in query_prompt

    assert handler.get_format_name() in doc_prompt
    assert example_str in doc_prompt
    assert "custom doc" in doc_prompt
    assert "generate 2" in doc_prompt.lower()

def test_custom_validate_item_success(sample_custom_format_def):
    """Test successful validation for a custom format."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    valid_item = {"id": 10, "text": "Valid text", "score": 0.5, "verified": True}
    validated = handler.validate_item(valid_item.copy(), 0)
    assert validated == valid_item

    # Test with optional fields missing
    valid_item_missing_optional = {"id": 11, "text": "More text"}
    validated = handler.validate_item(valid_item_missing_optional.copy(), 0)
    # The returned dict should have None for missing optional fields defined in the spec
    assert validated == {"id": 11, "text": "More text", "score": None, "verified": None}

def test_custom_validate_item_not_dict(sample_custom_format_def):
    """Test validation when item is not a dictionary."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    with pytest.raises(exceptions.ValidationError, match="not a dictionary"):
        handler.validate_item("not a dict", 0)

def test_custom_validate_item_missing_required(sample_custom_format_def):
    """Test validation failure due to missing required fields."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    invalid_item = {"id": 1} # Missing 'text'
    with pytest.raises(exceptions.ValidationError, match="missing required field\\(s\\): {'text'}"):
        handler.validate_item(invalid_item, 0)

def test_custom_validate_item_wrong_type(sample_custom_format_def):
    """Test validation failure due to incorrect data types."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    invalid_item = {"id": "not_an_int", "text": "Some text"}
    with pytest.raises(exceptions.ValidationError, match="field 'id': Expected type int, got str"):
        handler.validate_item(invalid_item, 0)

    invalid_item_2 = {"id": 1, "text": "Some text", "score": "high"}
    with pytest.raises(exceptions.ValidationError, match="field 'score': Expected type float, got str"):
        handler.validate_item(invalid_item_2, 0)

def test_custom_validate_item_extra_keys(sample_custom_format_def, caplog):
    """Test validation handles extra keys with a warning and ignores them."""
    caplog.set_level(logging.WARNING)
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    item_with_extra = {"id": 1, "text": "abc", "extra": "ignored", "score": 0.1}
    validated = handler.validate_item(item_with_extra, 0)

    assert "extra" not in validated # Extra key should not be in the result
    assert validated["id"] == 1
    assert validated["text"] == "abc"
    assert validated["score"] == 0.1
    assert validated["verified"] is None # Optional missing field
    assert "has extra field(s) not in definition: {'extra'}" in caplog.text

def test_custom_validate_item_empty_required_string(sample_custom_format_def, caplog):
    """Test warning for empty required string fields."""
    caplog.set_level(logging.WARNING)
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    item_with_empty = {"id": 1, "text": "  ", "score": 0.5} # Whitespace only
    handler.validate_item(item_with_empty, 0)
    assert "required string field 'text' is empty or whitespace" in caplog.text

def test_custom_validate_item_int_to_float_coercion(sample_custom_format_def):
    """Test automatic coercion from int to float for custom float fields."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    item = {"id": 1, "text": "abc", "score": 1} # Score is int
    validated = handler.validate_item(item, 0)
    assert isinstance(validated["score"], float)
    assert validated["score"] == 1.0

def test_custom_validate_item_optional_none(sample_custom_format_def):
    """Test validation allows None for optional fields."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    item = {"id": 1, "text": "abc", "score": None, "verified": None}
    validated = handler.validate_item(item, 0)
    assert validated["score"] is None
    assert validated["verified"] is None

def test_custom_validate_item_required_none_fails(sample_custom_format_def):
    """Test validation fails if a required field is explicitly None."""
    handler = custom.CustomFormatHandler(sample_custom_format_def)
    item = {"id": None, "text": "abc"} # id is required
    
    # It first fails the type check because None is not int
    expected_error_msg = r"Item \d+, required field 'id' is missing or None."
    with pytest.raises(exceptions.ValidationError, match=expected_error_msg):
         handler.validate_item(item, 0)
    # If type was 'any' or similar, the explicit None check would trigger:
    # with pytest.raises(exceptions.ValidationError, match="required field 'id' is missing or None"):
    #     handler.validate_item(item, 0)
