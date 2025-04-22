"""
Configuration and constants for the synthetic data generator.

Handles environment variable loading and defines default values
and shared configuration settings.
"""
import os
import logging
import sys
from typing import Literal, Dict, Any, Type, List

# --- Optional Dependency Handling for Config ---
try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    load_dotenv = None
    DOTENV_AVAILABLE = False

# LangChain imports (Critical for LLM config)
try:
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        HarmBlockThreshold,
        HarmCategory,
    )
    # from langchain_core.messages import SystemMessage, HumanMessage # Not needed in config
    # from langchain_core.exceptions import OutputParserException # Not needed in config
    # from google.api_core import exceptions as google_api_exceptions # Not needed in config
except ImportError as e:
    # This is a critical dependency, fail fast if not available
    print(f"ERROR: Critical LangChain/Google components not found: {e}", file=sys.stderr)
    print("Please ensure langchain-google-genai and langchain-core are installed:", file=sys.stderr)
    print("  pip install langchain-google-genai langchain-core", file=sys.stderr)
    sys.exit(1)

# Define package version (example)
__version__ = "0.1.0"

# --- Environment Variable Loading ---
def load_environment():
    """
    Loads environment variables from a .env file if available and checks for GOOGLE_API_KEY.

    Raises:
        ValueError: If GOOGLE_API_KEY is not set after loading environment variables.
    """
    if DOTENV_AVAILABLE and load_dotenv:
        load_dotenv()
        logging.info(".env file loaded if found.")

    # API Key Check (Fail Fast)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # Use basic print/sys.stderr here as logging might not be fully configured yet
        print("CRITICAL: GOOGLE_API_KEY environment variable not set.", file=sys.stderr)
        raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
    # Store API key internally if needed by LLM client init, but avoid global variable if possible
    # LangChain client handles reading from env var or passing directly, so check is sufficient here.

# --- Type Definitions ---
PredefinedDataFormat = Literal["pair-class", "pair-score", "pair", "triplet", "qa"]
"""Type hint for supported predefined format names."""
CustomFormatDefinition = Dict[str, Any]
"""Type hint for the structure of a loaded custom format definition (dictionary)."""
TYPE_MAP: Dict[str, Type] = {"string": str, "integer": int, "float": float, "boolean": bool}
"""Mapping from JSON type strings (in custom format def) to Python types."""
OUTPUT_FILE_FORMATS = ['jsonl', 'csv', 'parquet', 'all']
"""List of supported output file formats."""

# --- Default Generation Parameters ---
DEFAULT_MODEL_NAME = "gemini-2.0-flash"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_TOP_P = 0.95
DEFAULT_TOP_K = None # Explicitly None means not used by default
DEFAULT_BATCH_SIZE = 25
DEFAULT_SAMPLES = 50
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 5.0 # Use float for sleep
DEFAULT_OUTPUT_PREFIX = "synthetic_data"
DEFAULT_LOG_LEVEL = "INFO"
PREVIEW_SAMPLE_COUNT = 3 # Number of samples to show in the final preview

# --- Logging Configuration ---
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# --- Other Configuration ---
LARGE_RUN_WARNING_THRESHOLD = 2000 # Samples threshold for warning user about potential cost/time

# Re-export critical types/classes needed by other modules for clarity
# This avoids other modules needing to import from langchain_google_genai directly
# if they only need these specific types for type hinting or configuration.
HarmBlockThreshold = HarmBlockThreshold
HarmCategory = HarmCategory
ChatGoogleGenerativeAI = ChatGoogleGenerativeAI # Re-export the class for main.py to instantiate