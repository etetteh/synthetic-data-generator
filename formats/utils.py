"""
Utility functions related to data formats.
"""
import json
import os
import logging
from typing import Dict, Any

from .. import exceptions # Import custom exceptions
from .. import config # Import config constants and types

logger = logging.getLogger(__name__)

def load_and_validate_custom_format(file_path: str) -> config.CustomFormatDefinition:
    """
    Loads a custom data format definition from a JSON file and validates its structure.

    Ensures the file exists, is valid JSON, and adheres to the expected schema
    (e.g., contains 'fields' dictionary with 'type' and 'description').

    Args:
        file_path: The path to the JSON custom format definition file.

    Returns:
        The loaded and validated custom format definition dictionary.

    Raises:
        exceptions.ConfigurationError: If the file is not found, invalid JSON, structurally
                           invalid, or causes an OS error during loading.
        exceptions.ValidationError: If the content of the JSON file does not adhere to the
                         expected schema for a custom format definition.
    """
    logger.info(f"Loading custom format definition from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            definition = json.load(f)

        # --- Schema Validation ---
        if not isinstance(definition, dict): raise exceptions.ValidationError("Format definition root must be a JSON object.")
        if "fields" not in definition or not isinstance(definition["fields"], dict): raise exceptions.ValidationError("Format definition must contain a 'fields' key with a dictionary value.")
        if not definition["fields"]: raise exceptions.ValidationError("'fields' dictionary cannot be empty.")

        # Validate individual field definitions more strictly
        for name, field_def in definition["fields"].items():
            if not isinstance(field_def, dict): raise exceptions.ValidationError(f"Definition for field '{name}' must be a dictionary.")
            if "type" not in field_def or field_def["type"] not in config.TYPE_MAP: raise exceptions.ValidationError(f"Field '{name}' missing or invalid 'type'. Must be one of: {', '.join(config.TYPE_MAP.keys())}.")
            if "description" not in field_def or not isinstance(field_def["description"], str) or not field_def["description"].strip(): raise exceptions.ValidationError(f"Field '{name}' must have a non-empty string 'description'.")
            if "required" in field_def and not isinstance(field_def["required"], bool): raise exceptions.ValidationError(f"Field '{name}' has non-boolean 'required' value.")
            # Set default for required if missing
            field_def.setdefault("required", True)
            # Example validation: check type consistency if example is provided
            if "example" in field_def:
                 expected_py_type = config.TYPE_MAP.get(field_def["type"])
                 if expected_py_type is not None and not isinstance(field_def["example"], (expected_py_type, type(None))):
                      # Allow None for optional fields even if example type doesn't match
                      if field_def.get("required", True) or field_def["example"] is not None:
                         logger.warning(f"Field '{name}' has 'example' with type ({type(field_def['example']).__name__}) inconsistent with declared type ({field_def['type']}).")


        # Add default name/description if missing (using filename)
        definition.setdefault("name", os.path.splitext(os.path.basename(file_path))[0])
        definition.setdefault("description", f"Generate data according to the structure defined for '{definition['name']}'.")

        logger.info(f"Successfully loaded and validated custom format '{definition['name']}'.")
        return definition

    except FileNotFoundError: raise exceptions.ConfigurationError(f"Custom format definition file not found: {file_path}")
    except json.JSONDecodeError as e: raise exceptions.ConfigurationError(f"Invalid JSON in custom format file {file_path}: {e}") from e
    except exceptions.ValidationError as e: raise exceptions.ConfigurationError(f"Invalid structure in custom format file {file_path}: {e}") from e
    except OSError as e: raise exceptions.ConfigurationError(f"OS error loading custom format file {file_path}: {e}") from e
