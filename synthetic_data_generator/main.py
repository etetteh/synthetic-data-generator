#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry point for the synthetic data generation application.

Handles command-line argument parsing, configuration loading,
initialization of core components, and orchestration of the
data generation pipeline.
"""

import argparse
import logging
import sys
import time
import os
import json
from typing import List, Dict, Any

# Import modules from the refactored structure
from . import config
from . import exceptions
from .formats import base as format_base
from .formats import predefined as format_predefined
from .formats import custom as format_custom
from .formats import utils as format_utils
from .llm import generator as llm_generator
from .loading import document_loader as doc_loader
from .output import saver as data_saver

# Get logger for this module
logger = logging.getLogger(__name__)

def setup_logging(log_level_str: str):
    """
    Configures the root logger level and format based on the provided string.

    Args:
        log_level_str: The desired logging level as a string (e.g., "INFO", "DEBUG").
    """
    log_level_int = getattr(logging, log_level_str.upper(), logging.INFO)
    root_logger = logging.getLogger() # Get the root logger
    root_logger.setLevel(logging.DEBUG) # Set root to DEBUG to allow handlers to filter
    # Clear existing handlers to prevent duplicate output if run multiple times
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Add a console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level_int) # Set handler level based on user input
    formatter = logging.Formatter(config.LOG_FORMAT, datefmt=config.LOG_DATE_FORMAT)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Suppress verbose logs from dependencies unless DEBUG level is set for this script
    if log_level_int > logging.DEBUG:
        logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("google.api_core").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("fsspec").setLevel(logging.WARNING) # Added for Parquet/Pandas

    logger.info(f"Logging level set to {log_level_str.upper()}")


def run_generation_pipeline(args: argparse.Namespace):
    """
    Orchestrates the data generation process based on parsed arguments.

    Initializes the appropriate DataFormatHandler, LLM client, and
    SyntheticDataGenerator, performs generation, and handles saving
    and previewing the results.

    Args:
        args: The namespace object containing parsed command-line arguments.

    Raises:
        exceptions.ConfigurationError: If format loading or argument validation fails.
        exceptions.LoaderError: If document loading fails.
        exceptions.GenerationError: If the generation process fails after retries or
                                    encounters a critical error.
    """

    # --- Determine Format Handler ---
    format_handler: format_base.DataFormatHandler
    try:
        if args.custom_format_file:
            # Load and validate the custom format definition
            custom_format_definition = format_utils.load_and_validate_custom_format(args.custom_format_file)
            format_handler = format_custom.CustomFormatHandler(custom_format_definition)
            logger.info(f"Using Custom data format handler from: {args.custom_format_file}")
        elif args.format:
            # Use a predefined format handler
            format_handler = format_predefined.PredefinedFormatHandler(args.format)
            logger.info(f"Using Predefined data format handler: {args.format}")
        else:
            # This case should be prevented by argparse's mutually exclusive group
            raise exceptions.ConfigurationError("No data format specified (--format or --custom_format_file is required).")
    except exceptions.ConfigurationError as e:
         logger.critical(f"Format Configuration Error: {e}", exc_info=False)
         raise # Re-raise for main to catch and exit

    # --- Initialize LLM Client ---
    # Initialize the LLM client here and pass it to the generator (Dependency Injection)
    try:
        # Safety settings can be moved to config.py if needed
        safety_settings = {
            config.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: config.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            config.HarmCategory.HARM_CATEGORY_HATE_SPEECH: config.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            config.HarmCategory.HARM_CATEGORY_HARASSMENT: config.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            config.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: config.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }
        llm_client = config.ChatGoogleGenerativeAI(
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            safety_settings=safety_settings,
            # convert_system_message_to_human=True # Consider for older models compatibility
        )
        logger.info(f"Initialized LLM client: {args.model}")
    except Exception as e:
        # Catch potential initialization errors (e.g., invalid model name, API key issues)
        logger.critical(f"Failed to initialize LLM client for model '{args.model}': {e}", exc_info=True)
        raise exceptions.GenerationError(f"LLM client initialization failed: {e}") from e


    # --- Initialize Generator ---
    try:
        generator = llm_generator.SyntheticDataGenerator(
            format_handler=format_handler,
            llm_client=llm_client, # Inject the initialized client
            max_retries=args.max_retries,
            retry_delay=config.DEFAULT_RETRY_DELAY_SECONDS # Use constant
        )
    except (exceptions.ConfigurationError, exceptions.GenerationError) as e:
         # Catch errors during generator setup
         logger.critical(f"Generator Initialization Error: {e}", exc_info=False)
         raise

    # --- Prepare Output Formats ---
    # Handle 'all' option and check for dependencies
    selected_output_formats = sorted(list(set(args.output_format))) # Deduplicate
    if 'all' in selected_output_formats:
        selected_output_formats = [f for f in config.OUTPUT_FILE_FORMATS if f != 'all']

    # Check for Parquet dependency specifically
    if 'parquet' in selected_output_formats and not data_saver.PANDAS_AVAILABLE:
        logger.warning("Parquet output requested, but pandas/pyarrow not found. Skipping Parquet.")
        selected_output_formats = [f for f in selected_output_formats if f != 'parquet']

    if not selected_output_formats and not args.incremental_save:
         logger.warning("No valid output file formats specified and incremental save is off. Data will only be generated, not saved to final files.")

    # --- Perform Generation ---
    generated_data: List[Dict] = [] # Data held in memory if not incremental saving
    final_unique_count = 0
    jsonl_output_filename = None # Track filename if incremental save is used

    logger.info(f"Starting generation process...")
    # Log key parameters for the run
    logger.info(f"Target samples: {args.samples:,}, Format: '{format_handler.get_format_name()}'")
    logger.info(f"Source: {'Query' if args.query else 'Documents'}")
    logger.info(f"Batch Size: {args.batch_size}, Max Retries/Batch: {args.max_retries}")
    logger.info(f"Incremental saving: {'Enabled' if args.incremental_save else 'Disabled'}")
    logger.info(f"Output formats: {selected_output_formats or ('None' if not args.incremental_save else 'JSONL (incremental)')}")

    generation_start_time = time.time()

    try:
        # Call the appropriate generator method based on input source
        if args.query:
            refine_flag = not args.no_refine
            logger.info(f"Generating from query (Refinement {'enabled' if refine_flag else 'disabled'})...")
            generated_data, final_unique_count = generator.generate_from_query(
                query=args.query,
                num_samples=args.samples,
                batch_size=args.batch_size,
                refine=refine_flag,
                output_file_prefix=args.output_prefix if args.incremental_save else None,
                incremental_save=args.incremental_save
            )
        elif args.documents:
            logger.info(f"Generating from documents at '{args.documents}'...")
            generated_data, final_unique_count = generator.generate_from_documents(
                document_path=args.documents,
                num_samples=args.samples,
                batch_size=args.batch_size,
                output_file_prefix=args.output_prefix if args.incremental_save else None,
                incremental_save=args.incremental_save
            )

        generation_end_time = time.time(); duration = generation_end_time - generation_start_time
        logger.info(f"Generation phase finished in {duration:.2f} seconds.")
        logger.info(f"Target samples requested: {args.samples:,}, Actual unique samples generated: {final_unique_count:,}.")

    except (exceptions.ConfigurationError, exceptions.LoaderError, exceptions.GenerationError, exceptions.ValidationError) as e:
        # Catch known errors from the generation process
        logger.critical(f"Generation failed: {e}", exc_info=False) # Log clearly without full traceback for known issues
        raise # Propagate for main to handle exit code
    except Exception as e:
         # Catch truly unexpected errors during the core generation loop
         logger.critical(f"An unexpected critical error occurred during generation: {e}", exc_info=True) # Log full traceback for debugging
         raise exceptions.GenerationError("Unexpected generation failure") from e


    # --- Saving Final Outputs ---
    if final_unique_count == 0:
        logger.warning("No unique data generated. Skipping final saving.")
        return # Exit the pipeline function successfully, nothing more to do

    # Determine base filename for output files
    source_indicator = "_from_docs" if args.documents else "_from_query"
    safe_format_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in format_handler.get_format_name())
    base_filename = f"{args.output_prefix}_{safe_format_name}{source_indicator}"

    data_to_save_for_conversion: List[Dict] = [] # Data needed for non-JSONL formats

    # Logic to handle data source for saving (memory vs. incremental file)
    if args.incremental_save:
        # Assume JSONL was written during the loop
        jsonl_output_filename = f"{base_filename}.jsonl"
        logger.info(f"Incremental saving was used. Primary output is assumed to be in: {jsonl_output_filename}")
        # Check if other formats are needed, if so, load data from JSONL
        needs_other_formats = any(fmt in selected_output_formats for fmt in ['csv', 'parquet'])
        if needs_other_formats:
            other_formats_str = ', '.join([f for f in selected_output_formats if f != 'jsonl'])
            logger.info(f"Loading full dataset from {jsonl_output_filename} to save to: {other_formats_str}")
            try:
                # Use the saver's load method for consistency
                data_to_save_for_conversion = data_saver.load_from_jsonl(jsonl_output_filename)
                # Optional: Sanity check count against final_unique_count
                if len(data_to_save_for_conversion) != final_unique_count:
                     logger.warning(f"Loaded {len(data_to_save_for_conversion):,} items from incremental file, expected {final_unique_count:,}. Mismatch may occur if script was interrupted or file modified.")
                logger.info(f"Successfully loaded {len(data_to_save_for_conversion):,} items for conversion.")
            except FileNotFoundError:
                 logger.error(f"Incremental file {jsonl_output_filename} not found. Cannot save to other formats.")
                 selected_output_formats = ['jsonl'] # Mark only JSONL as potentially existing
            except Exception as e:
                 logger.error(f"Error reading incremental file {jsonl_output_filename} for conversion: {e}. Cannot save to other formats.", exc_info=True)
                 data_to_save_for_conversion = [] # Prevent saving partial/incorrect data
                 selected_output_formats = ['jsonl'] # Mark only JSONL as potentially existing
    else:
        # Not incremental save, data should be in memory from the loop return
        data_to_save_for_conversion = generated_data
        # Save JSONL now if it was requested and not done incrementally
        if 'jsonl' in selected_output_formats:
            data_saver.save_as_jsonl(data_to_save_for_conversion, f"{base_filename}.jsonl")
            # Remove jsonl from list to avoid double processing by _save_data later
            selected_output_formats = [f for f in selected_output_formats if f != 'jsonl']

    # Save remaining formats (CSV, Parquet) from the prepared data list
    formats_to_save_now = [f for f in selected_output_formats if f in ['csv', 'parquet']]
    if data_to_save_for_conversion and formats_to_save_now:
         logger.info(f"Saving data to formats: {', '.join(formats_to_save_now)}")
         # Pass the data loaded/held and the remaining formats to the saver module
         # Determine CSV fieldnames via the format handler
         csv_fieldnames = format_handler.get_field_names()
         data_saver.save_data(data_to_save_for_conversion, base_filename, formats_to_save_now, csv_fieldnames)
    elif not data_to_save_for_conversion and formats_to_save_now:
        # This case usually occurs if incremental save was on, other formats requested, but reading the jsonl failed
        logger.warning(f"No data loaded/available to save in formats: {formats_to_save_now} (likely due to error reading incremental file). Skipping.")


    # --- Preview Data ---
    # Decide where to efficiently read the preview data from
    preview_source: List[Dict] = []
    if not args.incremental_save:
        preview_source = generated_data[:config.PREVIEW_SAMPLE_COUNT] # Use slice of in-memory data
    elif jsonl_output_filename and os.path.exists(jsonl_output_filename): # If incremental, try reading first few lines from the file
         logger.info(f"Reading preview from start of {jsonl_output_filename}")
         try:
             with open(jsonl_output_filename, 'r', encoding='utf-8') as f:
                 for i, line in enumerate(f):
                     if i >= config.PREVIEW_SAMPLE_COUNT: break # Limit preview read
                     try:
                         preview_source.append(json.loads(line))
                     except json.JSONDecodeError:
                         logger.warning(f"Could not parse line {i+1} from {jsonl_output_filename} for preview.")
                         continue # Skip to next line
         except Exception as e: logger.warning(f"Could not read preview from {jsonl_output_filename}: {e}")

    logger.info(f"\n--- Data Preview (First {min(config.PREVIEW_SAMPLE_COUNT, final_unique_count)} Generated Samples) ---")
    preview_count = len(preview_source) # Use actual length read/sliced
    if preview_count > 0:
        for i, item in enumerate(preview_source): # Iterate over the actual preview data
            print("-" * 20 + f" Sample {i+1} " + "-" * 20)
            # Use pretty JSON printing for a clear, structured preview
            try:
                print(json.dumps(item, indent=2, ensure_ascii=False))
            except Exception as e:
                 print(f"  Error formatting preview for item {i+1}: {e}")
                 print(f"  Raw item data: {str(item)[:200]}...") # Fallback
        print("-" * (40 + len(f" Sample {preview_count} "))) # Dynamic separator length
        if final_unique_count > preview_count: logger.info(f"... and {final_unique_count - preview_count:,} more unique samples generated.")
    else:
        logger.info("(No samples available to preview)") # If preview source is empty


def main():
    """
    Main entry point for the script.

    Parses command-line arguments, sets up logging, runs the generation pipeline,
    and handles top-level exceptions and exit codes.
    """
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Generate synthetic NLP training data using Google Gemini models.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help message
    )

    # Add version argument
    parser.add_argument(
        '-v', '--version', action='version',
        version=f'%(prog)s {config.__version__}',
        help="Show program's version number and exit."
    )

    # --- Define Argument Groups ---
    input_group = parser.add_argument_group("Input Source (Required: Choose Query or Documents)")
    format_group = parser.add_argument_group("Data Format Definition (Required: Choose One)")
    gen_group = parser.add_argument_group("Generation Parameters")
    model_group = parser.add_argument_group("Model Parameters")
    output_group = parser.add_argument_group("Output Parameters")
    adv_group = parser.add_argument_group("Advanced Parameters")

    # --- Input Source Arguments ---
    input_src_mx_group = input_group.add_mutually_exclusive_group(required=True)
    input_src_mx_group.add_argument('--query', type=str, help='Natural language query to generate data from.')
    input_src_mx_group.add_argument('--documents', type=str, metavar='PATH', help=f'Path to document file/directory for context. {"(DISABLED)" if not doc_loader.DOCUMENT_LOADING_ENABLED else ""}')

    # --- Data Format Arguments ---
    format_mx_group = format_group.add_mutually_exclusive_group(required=True)
    predefined_choices = list(config.PredefinedDataFormat.__args__) if hasattr(config.PredefinedDataFormat, '__args__') else [] # Get choices dynamically
    format_mx_group.add_argument('--format', type=str, choices=predefined_choices, help='Use a predefined logical format.')
    format_mx_group.add_argument('--custom_format_file', type=str, metavar='PATH', help='Path to a JSON file defining a custom data format structure.')

    # --- Generation Parameter Arguments ---
    gen_group.add_argument('--samples', type=int, default=config.DEFAULT_SAMPLES, metavar='N', help='Target number of *unique* samples.')
    gen_group.add_argument('--no-refine', action='store_true', help='Disable automatic query refinement.')

    # --- Model Parameter Arguments ---
    model_group.add_argument('--model', type=str, default=config.DEFAULT_MODEL_NAME, help='Google Gemini model name.')
    model_group.add_argument('--temperature', type=float, default=config.DEFAULT_TEMPERATURE, metavar='T', help='Generation temperature (0.0-2.0).')
    model_group.add_argument('--top_p', type=float, default=config.DEFAULT_TOP_P, metavar='P', help='Nucleus sampling threshold (>0.0-1.0).')
    model_group.add_argument('--top_k', type=int, default=config.DEFAULT_TOP_K, metavar='K', help='Sample from top K tokens. Default: None (disabled).')

    # --- Output Parameter Arguments ---
    output_group.add_argument('--output_prefix', type=str, default=config.DEFAULT_OUTPUT_PREFIX, metavar='PREFIX', help='Prefix for output filenames.')
    output_group.add_argument('--output_format', choices=config.OUTPUT_FILE_FORMATS, default=['all'], nargs='+', metavar='FMT', help='Output file format(s). Use "all" for jsonl, csv, parquet (if available).')
    output_group.add_argument('--incremental_save', action='store_true', help='Save unique samples incrementally to JSONL as they are generated. Requires --output_prefix.')

    # --- Advanced Parameter Arguments ---
    adv_group.add_argument('--batch_size', type=int, default=config.DEFAULT_BATCH_SIZE, metavar='B', help='Number of samples aimed for per API call (actual request may vary slightly due to diversity goals).')
    adv_group.add_argument('--max_retries', type=int, default=config.DEFAULT_MAX_RETRIES, metavar='R', help='Max retries per batch for API/parsing/validation failures.')
    adv_group.add_argument('--log_level', type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default=config.DEFAULT_LOG_LEVEL, help="Set console logging level.")

    # Parse arguments from command line
    args = parser.parse_args()

    # --- Setup Logging Level based on args ---
    # Done after parsing so user can control verbosity
    setup_logging(args.log_level)

    # --- Load Environment Variables ---
    config.load_environment() # Load .env file and check API key

    # --- Validate Arguments ---
    exit_code = 0
    try:
        # Perform critical argument validation after parsing and logging setup
        if args.samples <= 0: raise exceptions.ConfigurationError("--samples must be positive.")
        if args.batch_size <= 0: raise exceptions.ConfigurationError("--batch_size must be positive.")
        if args.max_retries < 0: raise exceptions.ConfigurationError("--max_retries must be non-negative.")
        if args.incremental_save and not args.output_prefix: raise exceptions.ConfigurationError("--incremental_save requires --output_prefix.")
        if args.documents and not doc_loader.DOCUMENT_LOADING_ENABLED: raise exceptions.ConfigurationError("--documents specified, but document loading dependency (ADLoader) is missing or failed to import.")
        # Model parameter range checks
        if not (0.0 <= args.temperature <= 2.0): raise exceptions.ConfigurationError(f"Temperature must be between 0.0 and 2.0, got {args.temperature}")
        if not (0.0 < args.top_p <= 1.0): raise exceptions.ConfigurationError(f"top_p must be > 0.0 and <= 1.0, got {args.top_p}")
        if args.top_k is not None and args.top_k < 1: raise exceptions.ConfigurationError(f"top_k must be None or >= 1, got {args.top_k}")

        # User confirmation for potentially large/costly runs
        if args.samples > config.LARGE_RUN_WARNING_THRESHOLD:
            logger.warning(f"Requesting large number of samples ({args.samples:,}). This may be slow and/or costly.")
            # Check if running interactively before prompting
            if sys.stdout.isatty():
                try:
                    response = input("Continue? (y/n): ").strip().lower()
                    if response != 'y':
                        logger.info("Operation cancelled by user.")
                        sys.exit(0) # Clean exit code 0 for user cancellation
                except EOFError: # Handle non-interactive execution (e.g., piping, background job)
                    logger.warning("Non-interactive mode detected (EOFError). Proceeding with large sample request...")
            else:
                 logger.warning("Non-interactive mode detected (not a TTY). Proceeding with large sample request...")

        # --- Run Pipeline ---
        # All core logic is within this function call
        run_generation_pipeline(args)
        logger.info("Synthetic data generation process finished successfully.")

    except (exceptions.ConfigurationError, exceptions.LoaderError, exceptions.GenerationError, exceptions.ValidationError) as e:
        # Catch specific, known error types from the pipeline
        logger.critical(f"Error: {e}", exc_info=False) # Log clearly without full traceback for known issues
        exit_code = 1
    except Exception as e:
        # Catch any other unexpected critical errors during setup or execution
        logger.critical(f"An unexpected critical error occurred: {e}", exc_info=True) # Log full traceback for debugging
        exit_code = 1
    finally:
        # Ensure all log messages (especially to file handlers if added later) are written
        logging.shutdown()

    # Use print for final status message that's always visible regardless of logging level
    print(f"\nScript finished with exit code {exit_code}.")
    sys.exit(exit_code) # Exit with the appropriate code


if __name__ == "__main__":
    main()