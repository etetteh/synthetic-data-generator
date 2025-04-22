"""
Custom exception hierarchy for the synthetic data generator.

Provides specific error types for different failure scenarios.
"""

class SyntheticDataGeneratorError(Exception):
    """Base exception for all errors in the synthetic data generator."""
    pass

class ConfigurationError(SyntheticDataGeneratorError, ValueError):
    """Indicates an error in the provided configuration or arguments."""
    pass

class GenerationError(SyntheticDataGeneratorError, RuntimeError):
    """Indicates a non-recoverable error during the LLM generation process."""
    pass

class ValidationError(SyntheticDataGeneratorError, ValueError):
    """Indicates that generated data failed validation against the expected format."""
    pass

class LoaderError(SyntheticDataGeneratorError, RuntimeError):
    """Indicates an error during document loading."""
    pass

class OutputParserError(SyntheticDataGeneratorError, ValueError):
    """Indicates that the LLM response could not be parsed into the expected structure."""
    pass

# You could add more specific exceptions if needed, e.g.:
# class LLMAPIError(GenerationError): ...
# class DuplicateItemError(ValidationError): ...
