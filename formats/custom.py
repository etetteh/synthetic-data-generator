"""
Concrete implementation of DataFormatHandler for custom formats defined by JSON schema.
"""
import logging
import json
from typing import List, Dict, Any, Optional, Type

from .base import DataFormatHandler # Import abstract base class
from .. import exceptions # Import custom exceptions
from .. import config # Import config constants and types

logger = logging.getLogger(__name__)

class CustomFormatHandler(DataFormatHandler):
    """
    Handles specifics for custom data formats defined via a JSON file.

    Reads the format definition upon initialization and uses it to generate
    prompts and validate generated data items.
    """

    def __init__(self, format_definition: config.CustomFormatDefinition):
        """
        Initializes the handler with a validated custom format definition.

        Args:
            format_definition: The dictionary loaded and validated from the
                               custom format JSON file by
                               `formats.utils.load_and_validate_custom_format`.
        """
        self._definition = format_definition
        self._fields_spec = format_definition.get("fields", {})
        # Pre-calculate required and optional keys for efficiency during validation
        self._required_keys = {name for name, f_def in self._fields_spec.items() if f_def.get("required", True)}
        self._optional_keys = set(self._fields_spec.keys()) - self._required_keys

    def get_format_name(self) -> str:
        """Returns the name defined in the custom format file (or default 'custom')."""
        return self._definition.get("name", "custom")

    def get_description(self) -> str:
        """Returns the description defined in the custom format file (or a default)."""
        return self._definition.get("description", "Generate data according to a custom structure.")

    def get_field_names(self) -> Optional[List[str]]:
        """Returns the field names defined in the custom format definition."""
        # Return keys in the order they likely appear in the definition (Python 3.7+)
        return list(self._fields_spec.keys())

    def _get_default_example(self, type_str: Optional[str]) -> Any:
        """Provides a default example value based on type string, used for constructing the example structure string."""
        if type_str == "string": return "sample text"
        if type_str == "integer": return 123
        if type_str == "float": return 1.23
        if type_str == "boolean": return True
        return None # Default for unknown or complex types not handled here

    def get_example_structure_string(self) -> str:
        """Generates a JSON example string based on the format definition."""
        example_obj = {}
        for name, f_def in self._fields_spec.items():
            # Use provided example from definition or generate a default based on type
            example_obj[name] = f_def.get("example", self._get_default_example(f_def.get("type")))
        # Format as a JSON string representation of a list containing such an object
        # Use compact separators for the example string
        return f"`[{json.dumps(example_obj, ensure_ascii=False, separators=(',', ':'))}]` (List containing objects like this)"

    def build_system_prompt(self) -> str:
        """Builds the system prompt, including detailed field definitions from the custom format."""
        base = self._get_common_system_prompt_base()
        fields_desc = []
        # Generate detailed field descriptions from the definition
        for name, f_def in self._fields_spec.items():
            type_str = f_def.get('type', 'unknown')
            req_str = "(Required)" if f_def.get('required', True) else "(Optional)"
            desc = f_def.get('description', 'No description provided.')
            # Include simple examples directly in the description for clarity
            example_str = ""
            if "example" in f_def and isinstance(f_def["example"], (str, int, float, bool, type(None))):
                 # Safely format example for inclusion in markdown-like list
                 try:
                     example_str = f" (e.g., `{json.dumps(f_def['example'], ensure_ascii=False, separators=(',', ':'))}`)"
                 except Exception:
                     # Fallback if example itself is not JSON serializable
                     logger.warning(f"Field '{name}' has 'example' that is not JSON serializable for prompt: {f_def['example']}")
                     example_str = f" (e.g., {repr(f_def['example'])})" # Use repr as fallback

            fields_desc.append(f"- `{name}` ({type_str}, {req_str}): {desc}{example_str}")

        fields_str = "\n".join(fields_desc) if fields_desc else "  (No specific field definitions provided in the custom format file.)"
        # Append the field definitions section to the common base prompt
        return f"{base}\n\n**Field Definitions:**\n{fields_str}\n"

    def build_query_prompt(self, query: str, num_samples: int) -> str:
        """Builds the query-based user prompt (uses common implementation)."""
        return self._get_common_query_prompt(query, num_samples)

    def build_document_prompt(self, documents: List[str], num_samples: int) -> str:
        """Builds the document-based user prompt (uses common implementation)."""
        return self._get_common_document_prompt(documents, num_samples)

    def validate_item(self, item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
        """Validates a single item against the custom format definition."""
        if not isinstance(item, dict): raise exceptions.ValidationError(f"Item {item_index} is not a dictionary (got {type(item)}).")
        item_keys = set(item.keys())

        # Check for missing required keys
        missing = self._required_keys - item_keys
        if missing: raise exceptions.ValidationError(f"Item {item_index} (format '{self.get_format_name()}') missing required field(s): {missing}. Item keys: {item_keys}")

        # Check for unexpected extra keys (issue warning)
        allowed_keys = self._required_keys.union(self._optional_keys)
        extra_keys = item_keys - allowed_keys
        if extra_keys:
             logger.warning(f"Item {item_index} (format '{self.get_format_name()}') has extra field(s) not in definition: {extra_keys}. Ignoring them.")

        validated_item = {}
        # Validate types and constraints for keys defined in the spec
        for name in allowed_keys: # Iterate over defined keys to ensure order and include optional missing keys
            field_def = self._fields_spec[name]
            expected_type_str = field_def.get("type")
            expected_py_type = config.TYPE_MAP.get(expected_type_str)
            is_required = field_def.get("required", True)

            value = item.get(name) # Use .get() to handle missing optional keys gracefully

            # Handle optional fields being null/None explicitly
            if not is_required and value is None:
                validated_item[name] = None # Store None if optional and value is None
                continue

            # If a required field is None or missing, it should have been caught by missing_keys check, but double check
            if is_required and value is None:
                 # This case should ideally not happen if missing_keys check is correct
                 raise exceptions.ValidationError(f"Item {item_index}, required field '{name}' is missing or None.")


            # Check if type is defined in our simple map
            if expected_py_type is None:
                logger.warning(f"Cannot validate type for field '{name}' (unknown type '{expected_type_str}'). Skipping type check.")
                validated_item[name] = value # Pass value through without type check
                continue

            current_value = value
            # Type Coercion (int -> float) - perform only if needed
            if expected_py_type == float and isinstance(current_value, int):
                try:
                    current_value = float(current_value)
                except (ValueError, TypeError):
                    raise exceptions.ValidationError(f"Item {item_index}, field '{name}': Could not coerce integer '{value}' to float.")

            # Primary Type Check
            if not isinstance(current_value, expected_py_type):
                raise exceptions.ValidationError(f"Item {item_index}, field '{name}': Expected type {expected_py_type.__name__}, got {type(current_value).__name__}. Value: '{str(current_value)[:50]}'.")

            # Basic Constraints (e.g., check for empty required strings)
            if expected_py_type == str and not current_value.strip() and is_required:
                logger.warning(f"Item {item_index}, required string field '{name}' is empty or whitespace.")
            # Add more custom constraints here if needed based on format_definition (e.g., range checks, regex patterns)

            validated_item[name] = current_value # Store validated/coerced value

        # Return dictionary containing only the validated fields according to the spec
        # This ensures the output dictionary structure matches the definition order and fields
        return {name: validated_item.get(name) for name in self._fields_spec.keys()}
