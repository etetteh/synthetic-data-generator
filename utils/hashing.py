"""
Utility functions for hashing data items.
"""
import json
import hashlib
import time
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

def hash_item(item: Dict[str, Any]) -> str:
    """
    Creates a stable MD5 hash for a dictionary item.

    Handles potential TypeError during JSON serialization and uses fallbacks.

    Args:
        item: The dictionary to hash.

    Returns:
        A hexadecimal MD5 hash string.
    """
    try:
        # Ensure consistent ordering and encoding for stable hashing
        # Use separators=(',', ':') to remove whitespace for more compact hash
        item_str = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(',', ':'))
        return hashlib.md5(item_str.encode('utf-8')).hexdigest()
    except TypeError as e:
        logger.warning(f"Fallback hash used: Could not serialize item for hashing. Error: {e}. Item: {str(item)[:100]}...")
        # Fallback using string representation of sorted items (more robust than just str(item))
        try:
            # Use repr() for more reliable string conversion of complex types within the dict
            stable_repr = repr(sorted(item.items()))
            return hashlib.md5(stable_repr.encode('utf-8')).hexdigest()
        except Exception: # Handle potential errors in sorting/string conversion
             # Final fallback with timestamp for very problematic items
             logger.error(f"Final hash fallback used for item: {str(item)[:100]}...", exc_info=False)
             return hashlib.md5(f"{time.time()}-{str(item)}".encode('utf-8')).hexdigest()
    except Exception as e:
        logger.error(f"Unexpected error during item hashing: {e}. Item: {str(item)[:100]}...", exc_info=False)
        # Fallback with timestamp to ensure uniqueness in error cases
        return hashlib.md5(f"{time.time()}-{str(item)}".encode('utf-8')).hexdigest()