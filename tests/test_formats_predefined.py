# tests/test_formats_predefined.py
import pytest
import logging
from synthetic_data_generator.formats import predefined
from synthetic_data_generator import exceptions
from synthetic_data_generator import config

# Get all predefined format names dynamically
PREDEFINED_FORMATS = list(config.PredefinedDataFormat.__args__) if hasattr(config.PredefinedDataFormat, '__args__') else []

@pytest.mark.parametrize("format_name", PREDEFINED_FORMATS)
def test_predefined_handler_init_success(format_name):
    """Test successful initialization for all predefined formats."""
    handler = predefined.PredefinedFormatHandler(format_name)
    assert handler.get_format_name() == format_name
    assert isinstance(handler.get_description(), str)
    assert isinstance(handler.get_example_structure_string(), str)
    assert isinstance(handler.get_field_names(), list)
    assert isinstance(handler.build_system_prompt(), str)
    assert "Format Specific Guidance" in handler.build_system_prompt()

def test_predefined_handler_init_failure():
    """Test initialization with an invalid format name."""
    with pytest.raises(exceptions.ConfigurationError, match="Unknown predefined format name"):
        predefined.PredefinedFormatHandler("invalid-format-name")

@pytest.mark.parametrize("format_name", PREDEFINED_FORMATS)
def test_predefined_prompts(format_name):
    """Test prompt building methods for predefined formats."""
    handler = predefined.PredefinedFormatHandler(format_name)
    system_prompt = handler.build_system_prompt()
    query_prompt = handler.build_query_prompt("test query", 5)
    doc_prompt = handler.build_document_prompt(["doc1 text", "doc2 text"], 3)

    assert handler.get_format_name() in system_prompt
    assert handler.get_description() in system_prompt
    assert handler.get_example_structure_string() in system_prompt
    assert "MATRIX-Inspired" in system_prompt

    assert "test query" in query_prompt
    assert "generate 5" in query_prompt
    assert handler.get_format_name() in query_prompt
    assert handler.get_example_structure_string() in query_prompt

    assert "doc1 text" in doc_prompt
    assert "doc2 text" in doc_prompt
    assert "generate 3" in doc_prompt.lower()
    assert handler.get_format_name() in doc_prompt
    assert handler.get_example_structure_string() in doc_prompt

# --- Validation Tests ---

# Example valid items for each format
VALID_ITEMS = {
    "qa": {"context": "c", "question": "q", "answer": "a"},
    "triplet": {"anchor": "a", "positive": "p", "negative": "n"},
    "pair": {"anchor": "a", "positive": "p"},
    "pair-score": {"sentence1": "s1", "sentence2": "s2", "score": 0.75},
    "pair-class": {"premise": "p", "hypothesis": "h", "label": 1}, # Assuming 0, 1, 2 are valid labels
}

# Example invalid items
INVALID_ITEMS_MISSING_KEY = {
    "qa": {"context": "c", "question": "q"}, # Missing answer
    "triplet": {"anchor": "a", "positive": "p"}, # Missing negative
    "pair": {"anchor": "a"}, # Missing positive
    "pair-score": {"sentence1": "s1", "score": 0.5}, # Missing sentence2
    "pair-class": {"premise": "p", "label": 0}, # Missing hypothesis
}
INVALID_ITEMS_WRONG_TYPE = {
    "qa": {"context": "c", "question": "q", "answer": 123}, # Answer not string
    "triplet": {"anchor": 1, "positive": "p", "negative": "n"}, # Anchor not string
    "pair": {"anchor": "a", "positive": True}, # Positive not string
    "pair-score": {"sentence1": "s1", "sentence2": "s2", "score": "high"}, # Score not float
    "pair-class": {"premise": "p", "hypothesis": "h", "label": "entailment"}, # Label not int
}
INVALID_ITEMS_CONSTRAINT = {
    "pair-score": {"sentence1": "s1", "sentence2": "s2", "score": 1.5}, # Score > 1.0
    "pair-class": {"premise": "p", "hypothesis": "h", "label": 5}, # Label not 0, 1, 2
}

@pytest.mark.parametrize("format_name", PREDEFINED_FORMATS)
def test_validate_item_success(format_name):
    """Test successful validation for each format."""
    handler = predefined.PredefinedFormatHandler(format_name)
    valid_item = VALID_ITEMS[format_name]
    validated = handler.validate_item(valid_item.copy(), 0) # Use copy
    assert validated == valid_item # Should return the same structure/values

def test_validate_item_not_dict():
    """Test validation when item is not a dictionary."""
    handler = predefined.PredefinedFormatHandler("qa")
    with pytest.raises(exceptions.ValidationError, match="not a dictionary"):
        handler.validate_item(["list", "item"], 0)

@pytest.mark.parametrize("format_name", INVALID_ITEMS_MISSING_KEY.keys())
def test_validate_item_missing_key(format_name):
    """Test validation failure due to missing keys."""
    handler = predefined.PredefinedFormatHandler(format_name)
    invalid_item = INVALID_ITEMS_MISSING_KEY[format_name]
    with pytest.raises(exceptions.ValidationError, match="missing required field"):
        handler.validate_item(invalid_item, 0)

@pytest.mark.parametrize("format_name", INVALID_ITEMS_WRONG_TYPE.keys())
def test_validate_item_wrong_type(format_name):
    """Test validation failure due to incorrect data types."""
    handler = predefined.PredefinedFormatHandler(format_name)
    invalid_item = INVALID_ITEMS_WRONG_TYPE[format_name]
    with pytest.raises(exceptions.ValidationError, match="Expected type"):
        handler.validate_item(invalid_item, 0)

@pytest.mark.parametrize("format_name", INVALID_ITEMS_CONSTRAINT.keys())
def test_validate_item_constraint_fail(format_name):
    """Test validation failure due to constraint violations (e.g., score range)."""
    handler = predefined.PredefinedFormatHandler(format_name)
    invalid_item = INVALID_ITEMS_CONSTRAINT[format_name]
    with pytest.raises(exceptions.ValidationError, match="failed constraint check"):
        handler.validate_item(invalid_item, 0)

def test_validate_item_extra_keys(caplog):
    """Test validation handles extra keys with a warning."""
    caplog.set_level(logging.WARNING)
    handler = predefined.PredefinedFormatHandler("qa")
    item_with_extra = {"context": "c", "question": "q", "answer": "a", "extra_field": "ignore"}
    validated = handler.validate_item(item_with_extra, 0)
    assert "extra_field" not in validated # Extra field should be removed/ignored
    assert "has unexpected extra field(s): {'extra_field'}" in caplog.text

def test_validate_item_empty_string_warning(caplog):
    """Test warning for empty string fields."""
    caplog.set_level(logging.WARNING)
    handler = predefined.PredefinedFormatHandler("qa")
    item_with_empty = {"context": "c", "question": "", "answer": "a"}
    handler.validate_item(item_with_empty, 0)
    assert "field 'question' is an empty or whitespace-only string" in caplog.text

def test_validate_item_int_to_float_coercion():
    """Test automatic coercion from int to float for scores."""
    handler = predefined.PredefinedFormatHandler("pair-score")
    item = {"sentence1": "s1", "sentence2": "s2", "score": 1} # Score is int
    validated = handler.validate_item(item, 0)
    assert isinstance(validated["score"], float)
    assert validated["score"] == 1.0
