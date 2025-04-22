"""
Synthetic Data Generator Package.

This package provides tools for generating synthetic NLP training data
using Google Gemini models based on predefined or custom formats.
"""

# Import key components to make them accessible directly under the package name
# This defines the public API of the package (if it were used as a library)
# For a script, it helps organize imports in main.py
from .config import __version__, PredefinedDataFormat, OUTPUT_FILE_FORMATS, load_environment
from .exceptions import (
    SyntheticDataGeneratorError, ConfigurationError, GenerationError,
    ValidationError, LoaderError, OutputParserError
)
# No need to import all classes/functions, main.py will import what it needs
# from specific modules (e.g., from .llm.generator import SyntheticDataGenerator)
# This keeps the __init__.py clean and forces explicit imports elsewhere.

# Define __all__ to specify what is considered public if someone does `from synthetic_data_generator import *`
__all__ = [
    "__version__",
    "PredefinedDataFormat",
    "OUTPUT_FILE_FORMATS",
    "load_environment",
    "SyntheticDataGeneratorError",
    "ConfigurationError",
    "GenerationError",
    "ValidationError",
    "LoaderError",
    "OutputParserError",
    # Add other public components if needed, e.g., "SyntheticDataGenerator"
]

# Optional: Perform environment loading automatically when the package is imported
# This can be useful if the package is used as a library, but for a script
# entry point, it's better to control this in main.py.
# load_environment()
