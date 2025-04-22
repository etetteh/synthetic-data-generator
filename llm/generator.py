"""
Core logic for generating synthetic data using an LLM.

Handles LLM invocation, batching, retries, parsing, validation,
and duplicate detection.
"""
import logging
import json
import time
import random
from tqdm import tqdm
from typing import List, Dict, Union, Optional, Set, Any, Tuple, Callable

# Import types and exceptions from the local package structure
from .. import exceptions
from .. import config
from ..utils import hashing # Import hashing utility
from ..formats.base import DataFormatHandler # Import abstract base class
from langchain_core.messages import SystemMessage, HumanMessage # Re-export or import specific LangChain types
from langchain_core.exceptions import OutputParserException # Re-export or import specific LangChain exceptions
from langchain_google_genai import ChatGoogleGenerativeAI # Import the concrete class for type hinting

logger = logging.getLogger(__name__)

class SyntheticDataGenerator:
    """
    Orchestrates the synthetic data generation process.

    This class manages the interaction with the LLM, handles batching,
    retries on failure, checks for duplicate generated items, and coordinates
    saving the results. It utilizes a DataFormatHandler strategy object
    to delegate format-specific tasks like prompt building and validation.
    It receives the LLM client and format handler via dependency injection.
    """
    def __init__(self,
                 format_handler: DataFormatHandler,
                 llm_client: ChatGoogleGenerativeAI, # Inject the LLM client instance
                 max_retries: int,
                 retry_delay: float):
        """
        Initializes the SyntheticDataGenerator.

        Args:
            format_handler: An instance of a DataFormatHandler subclass
                            (PredefinedFormatHandler or CustomFormatHandler)
                            to manage format-specific logic.
            llm_client: An initialized instance of the LLM client (e.g., ChatGoogleGenerativeAI).
            max_retries: Maximum number of retry attempts for failed LLM calls
                         or parsing/validation errors per batch.
            retry_delay: Initial delay in seconds between retries (uses exponential backoff).

        Raises:
            exceptions.ConfigurationError: If the format_handler or llm_client is invalid.
        """
        if not isinstance(format_handler, DataFormatHandler):
            raise exceptions.ConfigurationError("Generator requires a valid DataFormatHandler instance.")
        self.format_handler = format_handler

        if not isinstance(llm_client, ChatGoogleGenerativeAI): # Or a more abstract LLM interface if created
             raise exceptions.ConfigurationError("Generator requires a valid LLM client instance.")
        self.model = llm_client # Store the injected LLM client

        # Store validated parameters
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        logger.info(f"Initializing generator for format '{self.format_handler.get_format_name()}'")
        # Access model parameters from the injected client for logging
        # CORRECTED: Use self.model.model instead of self.model.model_name
        logger.info(f"Model: {self.model.model}, Temp: {self.model.temperature}, Top-P: {self.model.top_p}, Top-K: {self.model.top_k}")
        logger.info(f"Retries per batch: {self.max_retries}, Initial retry delay: {self.retry_delay}s")

        # State for tracking generated items
        self.unique_entries: Set[str] = set()
        # Build system prompt once using the handler, store for reuse
        self.system_prompt: str = self.format_handler.build_system_prompt()

    def refine_query(self, query: str) -> str:
        """
        Attempts to refine the user's input query using the LLM.

        Uses a dedicated prompt asking the LLM to enhance the query based on the
        target data format and MATRIX-inspired principles (compositionality,
        specificity, multimodal description), aiming for better diversity and
        quality in the generated data. If refinement fails or doesn't change
        the query, the original query is returned.

        Args:
            query: The original user query.

        Returns:
            The refined query string, or the original query if refinement fails
            or is ineffective.
        """
        format_name = self.format_handler.get_format_name()
        format_desc = self.format_handler.get_description()
        logger.info(f"Attempting to refine query for '{format_name}' format: '{query}'")

        # System prompt specific to the refinement task, incorporating MATRIX ideas
        refine_system_prompt = f"""
            You are an expert query analyst and prompt engineer specializing in refining prompts for synthetic NLP data generation, potentially for multimodal scenarios. Your goal is to enhance the user's query to elicit diverse, high-quality, and compositionally rich examples conforming to the '{format_name}' format.

            Format Description: {format_desc}

            Follow these steps mentally:
            1.  **Analyze Intent & Domain:** What is the core goal? What specific domain(s) should be covered (e.g., "recent news," "scientific papers," "product reviews")?
            2.  **Assess Specificity:** Is the query too broad? How can it be made more concrete?
            3.  **Consider Format:** How can the query better align with the fields and purpose of the '{format_name}' format?
            4.  **Inject Compositionality:** How can the query request examples involving interactions between multiple entities, steps in a process, or combinations of concepts?
            5.  **Introduce Multimodal Aspects (Textually):** Can the query ask for descriptions that imply visual scenes, sounds, or interactions, even if only text is generated? (e.g., "Generate QA pairs about *images depicting* historical events", "Create descriptions of *dialogue scenarios including background sounds*").
            6.  **Enhance Diversity:** Add constraints or keywords to encourage variation in topics, perspectives, complexity, and linguistic style.
            7.  **Synthesize:** Rewrite the query clearly and concisely, embedding these enhancements.

            Output ONLY the refined query text, without any explanations, preamble, or markdown formatting.
        """
        # Human prompt guiding the refinement
        refine_human_prompt = f"""
            Original Query:
            "{query}"

            Target Data Format: {format_name}
            Format Purpose: {format_desc}

            Refine this query based on your analysis to maximize the generation of diverse, compositionally rich, and high-quality synthetic data examples for this format. Consider adding specificity, requesting interactions, or implying multimodal contexts where relevant. Provide ONLY the refined query text.
        """
        try:
            # Use LLM to refine the query
            response = self._invoke_llm_with_retry(
                messages=[SystemMessage(content=refine_system_prompt), HumanMessage(content=refine_human_prompt)],
                purpose="query refinement"
            )
            refined_query = response.content.strip()

            # Basic checks on the refined query
            if not refined_query:
                logger.warning("Query refinement returned empty string. Using original query.")
                return query
            # Check if refinement actually changed the query meaningfully
            # Use case-insensitive comparison after stripping
            if refined_query.lower() == query.strip().lower():
                logger.info("Query refinement did not significantly change the query. Using original.")
                return query

            logger.info(f"Refined query: '{refined_query}'")
            return refined_query
        except Exception as e:
            # Log refinement errors but proceed with original query
            logger.error(f"Non-critical error during query refinement: {e}", exc_info=True)
            logger.warning("Proceeding with the original query due to refinement error.")
            return query

    def _invoke_llm_with_retry(self, messages: List[Union[SystemMessage, HumanMessage]], purpose: str = "data generation") -> Any:
        """
        Invokes the configured LLM with robust retry logic.

        Handles transient errors (like timeouts or rate limits, depending on API)
        using exponential backoff.

        Args:
            messages: A list of SystemMessage and HumanMessage objects for the LLM call.
            purpose: A string describing the purpose of the LLM call (for logging).

        Returns:
            The response object from the successful LLM invocation (typically includes
            `content` and `response_metadata`).

        Raises:
            exceptions.GenerationError: If the LLM call fails permanently after all retries,
                             or if the LLM returns empty content unexpectedly.
        """
        retries = 0
        last_exception = None
        while retries <= self.max_retries:
            try:
                logger.debug(f"Invoking LLM for {purpose} (Attempt {retries + 1}/{self.max_retries + 1})")
                response = self.model.invoke(messages)

                # --- Check for Empty or Blocked Response ---
                # Check if content is missing or empty string
                if not response or not hasattr(response, 'content') or not response.content or response.content.strip() == "":
                    # Try to get more details from metadata if available
                    block_reason = "N/A"
                    finish_reason = "N/A"
                    if hasattr(response, 'response_metadata'):
                        block_reason = response.response_metadata.get('block_reason', 'N/A')
                        finish_reason = response.response_metadata.get('finish_reason', 'N/A')
                        # If specifically blocked by safety filters, maybe don't retry?
                        if finish_reason == 'SAFETY':
                             logger.error(f"LLM call for {purpose} blocked due to safety settings (Block Reason: {block_reason}). Cannot retry this prompt.")
                             raise exceptions.GenerationError(f"LLM call blocked by safety settings (Reason: {block_reason})")

                    logger.warning(f"LLM returned empty content for {purpose} (Attempt {retries+1}). Finish Reason: {finish_reason}, Block Reason: {block_reason}.")
                    # Raise error to trigger retry for potentially transient empty responses
                    raise exceptions.GenerationError(f"LLM returned empty content (Finish: {finish_reason}, Block: {block_reason})")

                # --- Success ---
                return response

            # --- Error Handling & Retry Logic ---
            except Exception as e:
                 last_exception = e
                 log_msg = f"LLM invocation failed for {purpose} (Attempt {retries + 1}/{self.max_retries + 1}): {type(e).__name__}: {e}"

                 # TODO: Implement more specific checks for retryable API errors if possible
                 # Example: Check for specific Google API error codes or HTTP status codes if accessible
                 # is_retryable = isinstance(e, (google_api_exceptions.InternalServerError, ...))
                 is_retryable = True # Defaulting to retry most exceptions for simplicity

                 if is_retryable and retries < self.max_retries:
                    logger.warning(log_msg)
                    retries += 1
                    # Calculate sleep time using exponential backoff + jitter
                    sleep_time = (self.retry_delay * (2 ** (retries - 1))) + (random.uniform(0, 0.5))
                    logger.info(f"Waiting {sleep_time:.2f} seconds before retrying LLM call...")
                    time.sleep(sleep_time)
                 else: # Non-retryable error or max retries reached
                     logger.error(f"LLM invocation failed permanently for {purpose} after {retries + 1} attempts. Last error: {e}")
                     # Raise the final error, preserving the original exception context
                     raise exceptions.GenerationError(f"LLM call failed after multiple retries for {purpose}") from last_exception

        # This line should theoretically be unreachable if max_retries >= 0
        # Added a final check just in case, though the loop logic should prevent it
        if last_exception:
             raise exceptions.GenerationError(f"LLM call failed after multiple retries for {purpose}") from last_exception
        else:
             raise exceptions.GenerationError(f"Exited LLM retry loop unexpectedly for {purpose}.")


    def _parse_and_validate_llm_response(self, response_text: str, item_index_offset: int = 0) -> List[Dict]:
        """
        Parses the LLM's text response expecting JSON, and validates each item.

        Attempts to robustly extract JSON even if surrounded by markdown or other text.
        Uses the configured DataFormatHandler to validate each JSON object found.

        Args:
            response_text: The raw text response from the LLM.
            item_index_offset: The starting index for items in this batch (for logging).

        Returns:
            A list of validated dictionary items. Items failing validation are logged
            and excluded from the returned list.

        Raises:
            exceptions.OutputParserError: If the response cannot be parsed as JSON.
            exceptions.ValidationError: If the overall structure is not a list/dict.
            exceptions.GenerationError: For other unexpected parsing/validation errors.
        """
        logger.debug(f"Attempting to parse LLM Raw Response Snippet: {response_text[:500]}...")
        json_str = response_text.strip()
        data_items: Any = None # Initialize

        # Robust JSON extraction logic
        try:
            # 1. Handle potential markdown code blocks
            if json_str.startswith("```json") and json_str.endswith("```"):
                json_str = json_str[len("```json"):-len("```")].strip()
                logger.debug("Extracted JSON from ```json block.")
            elif json_str.startswith("```") and json_str.endswith("```"):
                 json_str = json_str[len("```"):-len("```")].strip()
                 logger.debug("Extracted JSON from generic ``` block.")
            # Add more robust extraction if needed (e.g., regex to find first/last brace/bracket)
            # For now, assume the model is mostly compliant with code blocks or raw JSON.

            # 2. Attempt JSON parsing
            data_items = json.loads(json_str)

            # 3. Handle potential single object return
            if isinstance(data_items, dict):
                logger.warning("LLM returned a single JSON object, wrapping it in a list.")
                data_items = [data_items]

            # 4. Ensure the result is now a list
            if not isinstance(data_items, list):
                 raise exceptions.ValidationError(f"Expected JSON list after parsing and potential wrapping, got {type(data_items)}")

            # 5. Validate each item individually using the format handler
            validated_items: List[Dict] = []
            for i, item in enumerate(data_items):
                try:
                    # Delegate validation to the strategy object
                    validated_item = self.format_handler.validate_item(item, item_index=i + item_index_offset)
                    validated_items.append(validated_item)
                except exceptions.ValidationError as ve:
                     # Log validation error for the specific item but allow loop to continue
                     logger.error(f"Validation Error for item #{i + item_index_offset}: {ve}. Skipping item.")
                except Exception as e:
                     # Catch unexpected errors during validation of a single item
                     logger.error(f"Unexpected error validating item #{i + item_index_offset}: {e}. Skipping item.", exc_info=True)

            # Log summary if items were skipped
            if len(validated_items) < len(data_items):
                skipped_count = len(data_items) - len(validated_items)
                logger.warning(f"Validation completed. {skipped_count} item(s) failed validation and were skipped from this batch.")

            return validated_items # Return only the successfully validated items

        except json.JSONDecodeError as e:
            # Provide context for JSON parsing errors
            error_snippet = json_str[max(0, e.pos-40):min(len(json_str), e.pos+40)] # Wider snippet
            logger.error(f"JSON Decode Error: {e}. Snippet around position {e.pos}: '...{error_snippet}...'", exc_info=False)
            # Avoid logging full response in production unless debugging
            # logger.debug(f"Full response text causing JSON error:\n{response_text}")
            raise exceptions.OutputParserError(f"JSON parsing failed: {e}") from e
        except exceptions.ValidationError as e: # Catch structural validation errors (e.g., not a list)
             logger.error(f"Data Structure Validation Failed: {e}.", exc_info=False)
             raise # Re-raise to be caught by retry logic
        except Exception as e:
            # Catch any other unexpected errors during this process
            logger.error(f"Unexpected error during response parsing/validation: {e}", exc_info=True)
            raise exceptions.GenerationError("Unexpected parsing/validation error") from e


    def _generate_one_batch_with_retry(self, human_prompt: str) -> List[Dict]:
        """
        Generates a single batch of data, handling retries for LLM calls, parsing, and validation.

        This method orchestrates the call to the LLM, parses the response, validates
        the structure and content against the format handler's rules, and implements
        retry logic for recoverable errors.

        Args:
            human_prompt: The fully constructed user prompt for the LLM.

        Returns:
            A list of validated data dictionaries generated in the batch. Returns an
            empty list if the batch fails permanently after all retries.

        Raises:
            exceptions.GenerationError: If an unexpected critical error occurs during generation
                             that prevents continuation.
        """
        batch_results: List[Dict] = []
        batch_success = False
        last_error: Optional[Exception] = None # Keep track of the last error for logging

        # Use a separate retry counter for parsing/validation errors vs LLM invocation errors
        # This allows retrying a batch if parsing fails, even if the LLM call itself succeeded
        parsing_validation_retries = 0
        max_parsing_validation_retries = self.max_retries # Can be different from LLM retries if desired

        for attempt in range(1, self.max_retries + 2): # Allows max_retries + 1 attempts total for LLM call
            try:
                logger.debug(f"Generating batch for '{self.format_handler.get_format_name()}' (Attempt {attempt}/{self.max_retries + 1})")

                # Step 1: Invoke LLM (includes its own internal retry logic for API errors)
                # _invoke_llm_with_retry handles retries for transient API issues.
                # If it raises GenerationError, it means LLM call failed permanently after its retries.
                response = self._invoke_llm_with_retry(
                    messages=[SystemMessage(content=self.system_prompt), HumanMessage(content=human_prompt)],
                    purpose=f"'{self.format_handler.get_format_name()}' data batch"
                )
                response_text = response.content

                # Step 2: Parse and Validate Response
                # This step can raise OutputParserError or ValidationError
                batch_results = self._parse_and_validate_llm_response(response_text)

                # If parsing and validation succeeded (even if list is empty), batch is considered successful
                batch_success = True
                logger.debug(f"Batch attempt {attempt} successful, received {len(batch_results)} valid items.")
                break # Exit retry loop on success

            except (exceptions.OutputParserError, exceptions.ValidationError) as e:
                # Catch errors specifically related to parsing or validation of the response
                last_error = e
                parsing_validation_retries += 1
                log_msg = f"Batch attempt {attempt} failed parsing/validation: {type(e).__name__}: {e}"

                if parsing_validation_retries <= max_parsing_validation_retries:
                    logger.warning(log_msg)
                    # Wait before next retry
                    sleep_time = (self.retry_delay * (2 ** (parsing_validation_retries - 1))) + (random.uniform(0, 0.5)) # Exponential backoff + jitter
                    logger.info(f"Waiting {sleep_time:.2f} seconds before retry attempt {attempt + 1} (parsing/validation)...")
                    time.sleep(sleep_time)
                    # Continue the loop to make another LLM call and try parsing again
                else:
                    logger.error(f"Batch failed parsing/validation permanently after {parsing_validation_retries} attempts. Last error: {e}")
                    batch_success = False # Ensure batch_success is False
                    break # Exit retry loop, batch failed

            except exceptions.GenerationError as e:
                 # This exception comes from _invoke_llm_with_retry, meaning the LLM call itself failed permanently
                 last_error = e
                 logger.error(f"Batch attempt {attempt} failed LLM invocation permanently: {e}")
                 batch_success = False # Ensure batch_success is False
                 break # Exit retry loop, batch failed

            except Exception as e: # Catch any other unexpected errors during the process
                 last_error = e
                 logger.critical(f"Unexpected critical error processing batch attempt {attempt}: {e}", exc_info=True)
                 batch_success = False # Ensure batch_success is False
                 break # Don't retry unexpected errors; break the loop for this batch

        if not batch_success:
            logger.error(f"Could not generate a valid batch after all retries. Last error: {last_error}")
            # Return empty list, indicating failure for this batch request
            return []

        # Return the successfully generated and validated results for this batch
        return batch_results


    def _run_generation_loop(self,
                             prompt_builder: Callable[..., str],
                             prompt_args: Dict[str, Any],
                             num_samples: int,
                             batch_size: int,
                             output_file_prefix: Optional[str] = None,
                             incremental_save: bool = False
                            ) -> Tuple[List[Dict], int]:
        """
        Runs the main loop to generate the target number of unique samples.

        Handles batching requests, checking for duplicates, incremental saving (optional),
        and progress tracking. Delegates prompt building and validation.

        Args:
            prompt_builder: The method from the format handler used to build the human prompt
                            (e.g., build_query_prompt or build_document_prompt).
            prompt_args: Arguments needed by the prompt_builder (e.g., query or documents).
            num_samples: The target number of unique samples to generate.
            batch_size: The requested number of samples per LLM API call.
            output_file_prefix: Prefix for output filenames, required for incremental saving.
            incremental_save: If True, save unique samples to JSONL incrementally.

        Returns:
            A tuple containing:
            - A list of unique generated data items held in memory (only if
              incremental_save is False).
            - The final count of unique items generated.

        Raises:
            exceptions.GenerationError: If multiple consecutive batch generation attempts fail,
                             indicating a persistent problem.
        """
        self.unique_entries.clear() # Ensure fresh start for duplicate tracking

        all_unique_results_in_memory: List[Dict] = []
        generated_count = 0
        total_successful_batches = 0
        consecutive_failed_batches = 0 # Track consecutive failures
        # Heuristic threshold: stop if N consecutive batches fail permanently
        max_consecutive_failures = max(3, self.max_retries + 1) # At least 3, or max retries + 1

        output_file_handle = None
        jsonl_filename = None
        format_name = self.format_handler.get_format_name()

        # Setup incremental saving file handle if needed
        if incremental_save and output_file_prefix:
            source_indicator = "_from_docs" if "documents" in prompt_args else "_from_query"
            safe_format_name = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in format_name)
            jsonl_filename = f"{output_file_prefix}_{safe_format_name}{source_indicator}.jsonl"
            try:
                # Open in write mode ('w') for a fresh start each run
                output_file_handle = open(jsonl_filename, 'w', encoding='utf-8')
                logger.info(f"Incremental saving enabled. Writing unique samples to: {jsonl_filename}")
            except OSError as e:
                logger.error(f"Failed to open {jsonl_filename} for incremental saving: {e}. Disabling.", exc_info=False)
                incremental_save = False # Disable if file cannot be opened
        elif incremental_save:
             # This case should be caught by main's argument validation, but safeguard
             logger.warning("`incremental_save` requires `output_prefix`. Disabling incremental save.")
             incremental_save = False


        # Initialize progress bar
        # Use a context manager for tqdm to ensure it's closed properly
        with tqdm(total=num_samples, desc=f"Generating '{format_name}'", unit=" samples", smoothing=0.1) as pbar:
            start_time = time.time() # Start time after setup

            while generated_count < num_samples:
                # --- Calculate Request Size ---
                remaining_needed = num_samples - generated_count
                # Request slightly more to account for duplicates, bounded sensibly
                # Requesting up to batch_size + 50% buffer, but not more than needed + 10% buffer
                buffer_factor = 0.5 # Request up to 50% more than batch_size
                needed_buffer_factor = 0.1 # Request up to 10% more than remaining needed
                request_size = min(
                    int(batch_size * (1 + buffer_factor)),
                    remaining_needed + max(1, int(remaining_needed * needed_buffer_factor)) # Ensure at least 1 buffer
                )
                current_batch_request_size = max(1, request_size) # Ensure at least 1 sample requested

                logger.debug(f"Target: {num_samples}, Generated: {generated_count}. Requesting batch for ~{current_batch_request_size} samples.")

                # --- Build Prompt ---
                current_prompt_args = {**prompt_args, "num_samples": current_batch_request_size}
                try:
                    human_prompt = prompt_builder(**current_prompt_args)
                except Exception as e:
                    logger.critical(f"Error building human prompt: {e}. Cannot continue generation.", exc_info=True)
                    raise exceptions.GenerationError("Failed to build human prompt.") from e


                # --- Generate Batch (with internal retries for LLM/parsing) ---
                batch_results = self._generate_one_batch_with_retry(human_prompt)

                # --- Handle Batch Outcome ---
                if not batch_results:
                    # Batch failed permanently or returned zero valid items after retries
                    consecutive_failed_batches += 1
                    logger.error(f"Batch generation failed or yielded no valid items. Consecutive failures: {consecutive_failed_batches}.")
                    # Stopping condition: If too many consecutive batches fail
                    if consecutive_failed_batches >= max_consecutive_failures:
                       logger.critical(f"Stopping generation due to {consecutive_failed_batches} consecutive batch failures.")
                       raise exceptions.GenerationError(f"Stopping generation due to {consecutive_failed_batches} consecutive batch failures.")
                    else:
                       continue # Skip processing this batch and try the next one
                else:
                    # Reset failure count on success
                    consecutive_failed_batches = 0
                    total_successful_batches += 1

                # --- Process Unique Items ---
                newly_added_items = []
                duplicates_in_batch = 0
                for item in batch_results: # Iterate through the (potentially empty) list
                    item_hash = hashing.hash_item(item) # Use the utility function
                    # Check uniqueness AND if we still need more samples
                    if item_hash not in self.unique_entries and generated_count < num_samples:
                        self.unique_entries.add(item_hash)
                        newly_added_items.append(item)
                        generated_count += 1
                        pbar.update(1) # Update progress bar only for new unique items
                    elif item_hash in self.unique_entries:
                        duplicates_in_batch += 1
                    # Stop adding if we hit the target number of samples mid-batch
                    if generated_count >= num_samples:
                        break

                # Log batch summary
                log_msg = f"Batch {total_successful_batches} yielded {len(batch_results)} valid items. Added {len(newly_added_items)} unique."
                if duplicates_in_batch > 0:
                    log_msg += f" Found {duplicates_in_batch} duplicates."
                logger.debug(log_msg)


                # --- Save Unique Items (if any) ---
                if newly_added_items:
                    if incremental_save and output_file_handle:
                        try:
                            for item in newly_added_items:
                                # Use json.dumps directly for incremental JSONL saving
                                output_file_handle.write(json.dumps(item, ensure_ascii=False) + '\n')
                            output_file_handle.flush() # Write to disk periodically
                            # os.fsync(output_file_handle.fileno()) # Optional: ensure data is physically written
                        except Exception as e:
                            logger.error(f"Error writing batch to incremental file {jsonl_filename}: {e}.", exc_info=True)
                            # Decide how to handle this: stop? continue? For now, log and continue.
                    else:
                        # Append to in-memory list if not saving incrementally
                        all_unique_results_in_memory.extend(newly_added_items)

                # --- Check Termination Condition ---
                if generated_count >= num_samples:
                    logger.info(f"Target number of {num_samples} unique samples reached.")
                    break

                # --- Optional: Stagnation Check ---
                # If a batch generated valid items, but none were unique, it might indicate stagnation.
                # Add a check here if needed, e.g., stop after N consecutive batches with 0 new unique items.
                if batch_results and not newly_added_items and len(self.unique_entries) > 0:
                     logger.warning("Batch generated valid items, but all were duplicates of previously generated samples. Consider adjusting prompt or parameters.")


        # End of while loop
        end_time = time.time(); duration = end_time - start_time
        final_count = len(self.unique_entries) # Final count from the set
        logger.info(f"Generation loop finished in {duration:.2f} seconds.")
        logger.info(f"Total unique samples generated: {final_count:,} (Target: {num_samples:,})")
        logger.info(f"Total successful batches processed: {total_successful_batches}, Total permanently failed batches: {consecutive_failed_batches}.")

        # Ensure incremental file is closed if it was opened
        if output_file_handle:
            try:
                output_file_handle.close()
                logger.info(f"Closed incremental output file: {jsonl_filename}")
            except Exception as e:
                logger.error(f"Error closing incremental file {jsonl_filename}: {e}", exc_info=False)


        # Return data stored in memory (only populated if incremental save is off) and the final count
        return all_unique_results_in_memory, final_count


    def generate_from_query(self, query: str,
                            num_samples: int,
                            batch_size: int,
                            refine: bool = True,
                            output_file_prefix: Optional[str] = None,
                            incremental_save: bool = False
                           ) -> Tuple[List[Dict], int]:
        """
        Generates synthetic data based on a natural language query.

        Args:
            query: The input query to guide generation.
            num_samples: The target number of unique samples.
            batch_size: The number of samples to request per API call.
            refine: Whether to refine the query using the LLM before generation.
            output_file_prefix: Prefix for output files (required for incremental save).
            incremental_save: Whether to save data incrementally to JSONL.

        Returns:
            A tuple containing:
                - List of generated items (if incremental_save is False).
                - The total count of unique items generated.
        """
        effective_query = query
        if refine:
            effective_query = self.refine_query(query) # Use refined query

        # Arguments needed by the format handler's prompt builder
        prompt_args = {"query": effective_query}

        return self._run_generation_loop(
            prompt_builder=self.format_handler.build_query_prompt, # Delegate prompt building
            prompt_args=prompt_args,
            num_samples=num_samples,
            batch_size=batch_size,
            output_file_prefix=output_file_prefix,
            incremental_save=incremental_save
        )

    def generate_from_documents(self, document_path: str,
                                num_samples: int,
                                batch_size: int,
                                output_file_prefix: Optional[str] = None,
                                incremental_save: bool = False
                               ) -> Tuple[List[Dict], int]:
        """
        Generates synthetic data based on the content of provided documents.

        Args:
            document_path: Path to a document file or directory.
            num_samples: The target number of unique samples.
            batch_size: The number of samples to request per API call.
            output_file_prefix: Prefix for output files (required for incremental save).
            incremental_save: Whether to save data incrementally to JSONL.

        Returns:
            A tuple containing:
                - List of generated items (if incremental_save is False).
                - The total count of unique items generated.

        Raises:
            exceptions.LoaderError: If document loading fails or is disabled.
            exceptions.ConfigurationError: If the document path is not found.
        """
        # Document loading logic is now delegated to the loading module
        try:
            document_texts = doc_loader.load_document_texts(document_path)
        except (exceptions.LoaderError, exceptions.ConfigurationError) as e:
            logger.critical(f"Document loading failed: {e}", exc_info=False)
            raise # Re-raise the specific error

        # Arguments needed by the format handler's prompt builder
        prompt_args = {"documents": document_texts}

        return self._run_generation_loop(
            prompt_builder=self.format_handler.build_document_prompt, # Delegate prompt building
            prompt_args=prompt_args,
            num_samples=num_samples,
            batch_size=batch_size,
            output_file_prefix=output_file_prefix,
            incremental_save=incremental_save
        )