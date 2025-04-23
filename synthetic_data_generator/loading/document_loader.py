"""
Handles loading document content for context-based generation.

Wraps the AutoDocumentLoader dependency.
"""
import logging
import os
from typing import List, Any

from .. import exceptions # Import custom exceptions

logger = logging.getLogger(__name__)

# --- Optional Dependency Handling for Document Loading ---
try:
    # Assuming ADLoader.py contains class AutoDocumentLoader
    # Note: In a real project, ADLoader would likely be installed via pip
    # and imported like `from my_document_loader import AutoDocumentLoader`
    # For this example, we keep the try/except pattern.
    from .ADLoader import AutoDocumentLoader
    DOCUMENT_LOADING_ENABLED = True
except ImportError:
    DOCUMENT_LOADING_ENABLED = False
    # Define a placeholder if not available to prevent NameError later
    class AutoDocumentLoader:
        """Placeholder class if ADLoader.py is not found."""
        def __init__(self, path: str):
            """Initializes the placeholder."""
            # Raise error only if actually used later
            pass
        def load(self) -> List[Any]: # Simulate return type
             """Simulates loading, raises error as AutoDocumentLoader is unavailable."""
             # This will be checked via DOCUMENT_LOADING_ENABLED flag before instantiation
             raise NotImplementedError("AutoDocumentLoader is unavailable.")
    logger.warning("AutoDocumentLoader dependency not found. Document loading functionality is disabled.")


def load_document_texts(document_path: str) -> List[str]:
    """
    Loads documents from a given path and extracts their text content.

    Args:
        document_path: Path to a document file or directory.

    Returns:
        A list of strings, where each string is the text content of a document
        or document chunk.

    Raises:
        exceptions.LoaderError: If document loading fails or is disabled.
        exceptions.ConfigurationError: If the document path is not found.
    """
    if not DOCUMENT_LOADING_ENABLED:
         raise exceptions.LoaderError("Document loading is disabled (AutoDocumentLoader dependency missing).")

    logger.info(f"Attempting to load documents from: {document_path}")
    if not os.path.exists(document_path):
        raise exceptions.ConfigurationError(f"Document path not found: {document_path}")

    try:
        # Instantiate the loader (ensure ADLoader handles paths correctly)
        document_loader = AutoDocumentLoader(document_path)
        # Load documents (expecting list of LangChain Document-like objects with .page_content)
        documents = document_loader.load()

        if not documents: raise exceptions.LoaderError(f"No documents successfully loaded from path: {document_path}.")

        # Extract non-empty text content, assuming .page_content attribute exists
        document_texts = []
        for i, doc in enumerate(documents):
            if hasattr(doc, 'page_content') and isinstance(doc.page_content, str) and doc.page_content.strip():
                document_texts.append(doc.page_content.strip())
            else:
                 logger.warning(f"Document at index {i} has missing, non-string, or empty 'page_content'. Skipping.")


        if not document_texts: raise exceptions.LoaderError(f"No non-empty text content found in loaded documents from {document_path}.")

        logger.info(f"Successfully loaded {len(documents)} Document object(s), using {len(document_texts)} non-empty text chunk(s).")

        # Estimate token count and warn if potentially very large
        total_chars = sum(len(text) for text in document_texts)
        estimated_tokens = total_chars / 4 # Very rough heuristic
        # Make threshold configurable? Use model context window size?
        context_warning_threshold_tokens = 500_000 # Example threshold (adjust as needed)
        if estimated_tokens > context_warning_threshold_tokens:
             logger.warning(f"Combined document text is large (approx. {estimated_tokens:,.0f} tokens). May exceed context limits or be slow/expensive.")

        return document_texts

    except NotImplementedError:
         # This should be caught by the initial DOCUMENT_LOADING_ENABLED check, but included for robustness
         raise exceptions.LoaderError("AutoDocumentLoader is unavailable.")
    except Exception as e:
         # Catch any other exceptions during loading (e.g., parsing errors in ADLoader)
         logger.error(f"Document loading or processing failed: {e}", exc_info=True)
         raise exceptions.LoaderError(f"Document loading or processing failed: {e}") from e
