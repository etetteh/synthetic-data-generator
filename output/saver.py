"""
Handles saving generated data to various file formats.
"""
import json
import csv
import logging
import sys
import os
from typing import List, Dict, Any, Optional

from .. import exceptions # Import custom exceptions
from .. import config # Import config constants

logger = logging.getLogger(__name__)

# --- Optional Dependency Handling for Saving ---
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    logger.warning("Pandas library not found. Parquet saving will be disabled.")

# Note: pyarrow is needed by pandas for parquet, but pandas import check is sufficient for now.


def save_data(data: List[Dict], filename_base: str, output_formats: List[str], csv_fieldnames: Optional[List[str]] = None) -> None:
    """
    Saves the generated data to one or more specified file formats.

    Args:
        data: The list of generated data dictionaries to save.
        filename_base: The base path and prefix for output files (e.g., "output/mydata").
                       Extensions (.jsonl, .csv, .parquet) will be appended.
        output_formats: A list of format strings (e.g., ['jsonl', 'csv']) to save as.
        csv_fieldnames: Optional list of field names for CSV header. If None,
                        inferred from the first data item.
    """
    if not data:
        logger.warning("No data provided to save_data function. Skipping saving.")
        return
    if not output_formats:
        logger.warning("No output formats specified for saving.")
        return

    logger.info(f"Saving {len(data)} items to base filename '{filename_base}' in formats: {output_formats}")

    # Ensure output directory exists
    output_dir = os.path.dirname(filename_base)
    if output_dir and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Created output directory: {output_dir}")
        except OSError as e:
            logger.error(f"Failed to create output directory {output_dir}: {e}. Cannot save files.", exc_info=True)
            return # Stop saving if directory creation fails


    # Determine fieldnames for CSV if needed and not provided
    effective_csv_fieldnames = csv_fieldnames
    if 'csv' in output_formats and effective_csv_fieldnames is None:
         effective_csv_fieldnames = list(data[0].keys()) if data else None
         if effective_csv_fieldnames:
             logger.warning(f"CSV fieldnames not provided by format handler. Inferring from first data item: {effective_csv_fieldnames}")
         else:
            # Cannot save CSV without headers
            logger.error("Cannot determine CSV fieldnames (no data and handler provided none). Skipping CSV save.")
            output_formats = [f for f in output_formats if f != 'csv'] # Remove csv from list


    # Call specific save methods for requested formats
    if 'jsonl' in output_formats: save_as_jsonl(data, f"{filename_base}.jsonl")
    if 'csv' in output_formats and effective_csv_fieldnames: save_as_csv(data, f"{filename_base}.csv", effective_csv_fieldnames)
    if 'parquet' in output_formats: save_as_parquet(data, f"{filename_base}.parquet")


def save_as_jsonl(data: List[Dict], filename: str) -> None:
    """Saves data as a JSON Lines file (one JSON object per line)."""
    logger.debug(f"Attempting to save {len(data)} items as JSONL to {filename}")
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            for i, item in enumerate(data):
                try:
                    # ensure_ascii=False preserves non-ASCII characters
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
                except Exception as e: # Catch errors writing individual items
                    logger.error(f"Error writing item #{i} to JSONL: {e}. Item snippet: {str(item)[:100]}...", exc_info=False)
                    # Consider writing a placeholder error line? Or just skip? Skipping for now.
        logger.info(f"Finished saving data as JSONL to {filename}")
    except OSError as e: logger.error(f"Failed to open/write JSONL file {filename}: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error during JSONL saving to {filename}: {e}", exc_info=True)

def load_from_jsonl(filename: str) -> List[Dict]:
    """Loads data from a JSON Lines file."""
    logger.debug(f"Attempting to load data from JSONL file: {filename}")
    data: List[Dict] = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    logger.error(f"JSON Decode Error on line {i+1} in {filename}: {e}. Skipping line.", exc_info=False)
                except Exception as e:
                    logger.error(f"Unexpected error reading line {i+1} from {filename}: {e}. Skipping line.", exc_info=True)
        logger.info(f"Successfully loaded {len(data)} items from {filename}")
        return data
    except FileNotFoundError:
        logger.error(f"JSONL file not found: {filename}")
        raise # Re-raise FileNotFoundError
    except OSError as e:
        logger.error(f"OS error reading JSONL file {filename}: {e}", exc_info=True)
        raise # Re-raise OS errors
    except Exception as e:
        logger.error(f"Unexpected error during JSONL loading from {filename}: {e}", exc_info=True)
        raise # Re-raise other errors


def save_as_csv(data: List[Dict], filename: str, fieldnames: List[str]) -> None:
    """Saves data as a CSV file using the provided fieldnames for the header."""
    logger.debug(f"Attempting to save {len(data)} items as CSV to {filename} using fields: {fieldnames}")
    if not data: logger.warning("No data provided to save_as_csv."); return
    if not fieldnames: logger.error("No fieldnames provided for CSV saving. Skipping."); return

    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            # extrasaction='ignore' prevents errors if data items have keys not in fieldnames
            # QUOTE_MINIMAL is usually sufficient, consider QUOTE_NONNUMERIC or QUOTE_ALL for complex strings
            writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, extrasaction='ignore')
            writer.writeheader()
            # Filter data items to only include keys present in fieldnames before writing
            # This ensures consistency and matches the header
            filtered_data = [{k: item.get(k, '') for k in fieldnames} for item in data]
            writer.writerows(filtered_data)
        logger.info(f"Successfully saved data as CSV to {filename}")
    except KeyError as e: # Should be rare with extrasaction='ignore' but possible if header fails?
        logger.error(f"KeyError during CSV writing: {e}. Fields expected: {fieldnames}", exc_info=False)
    except OSError as e: logger.error(f"Failed to open/write CSV file {filename}: {e}", exc_info=True)
    except Exception as e: logger.error(f"Unexpected error during CSV saving to {filename}: {e}", exc_info=True)


def save_as_parquet(data: List[Dict], filename: str) -> None:
    """Saves data as a Parquet file using pandas and pyarrow."""
    if not PANDAS_AVAILABLE or pd is None:
        logger.error("Pandas library not found. Cannot save as Parquet.")
        print("Install pandas & pyarrow: pip install pandas pyarrow", file=sys.stderr)
        return
    logger.debug(f"Attempting to save {len(data)} items as Parquet to {filename}")
    if not data: logger.warning("No data provided to save_as_parquet."); return
    try:
        # Create DataFrame - pandas infers schema by default
        df = pd.DataFrame(data)
        # Optional: Add schema validation or explicit type casting here if needed before saving
        # e.g., if 'score' should always be float: df['score'] = df['score'].astype(float)
        # logger.debug(f"DataFrame Info before Parquet save:\n{df.info()}") # Check inferred types
        df.to_parquet(filename, index=False, engine='pyarrow') # 'pyarrow' is the standard engine
        logger.info(f"Successfully saved data as Parquet to {filename}")
    except ImportError: # Catch if pyarrow is missing at runtime (pandas might be installed, but not pyarrow)
         logger.error("Error saving to Parquet: pyarrow library not found or failed to import."); print("Install pyarrow: pip install pyarrow", file=sys.stderr)
    except Exception as e: # Catch pandas or pyarrow specific errors
         logger.error(f"Failed to create/write Parquet file {filename}: {e}", exc_info=True)
