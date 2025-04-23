#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AutoDocumentLoader: A production-ready document loading utility for LangChain.

This module provides a robust, efficient, and flexible document loader class,
`AutoDocumentLoader`, designed to simplify the process of loading various
document types into LangChain `Document` objects. It automatically detects
the input source type (file, directory, URL) and selects the most appropriate
LangChain document loader based on file extension or source characteristics.

Key Features:
- Supports a wide range of file formats (PDF, TXT, CSV, JSON, JSONL, HTML,
  Parquet, EPub, Office documents via Unstructured, etc.).
- Handles web URLs for fetching and loading content.
- Processes entire directories, optionally recursively, loading all supported
  files within.
- Offers multithreading for faster directory processing.
- Provides a fallback mechanism using `UnstructuredFileLoader` for otherwise
  unsupported file types (requires `unstructured` library and its extras).
- Configurable options for encoding, CSV parsing, JSON parsing (jq schema),
  web loading, and directory scanning.
- Includes methods for standard loading (`load`), lazy loading (`lazy_load`),
  and integrated loading and splitting (`load_and_split`).
- Defines custom exceptions for clearer error handling.
- Includes a parallel batch loading capability (`batch_load`).

Dependencies:
- Required: langchain-community, langchain-core
- Optional but Recommended:
    - unstructured, python-magic-bin, and format-specific extras (e.g.,
      "unstructured[docx,pptx,xlsx,epub,odt,rst,rtf,md,tsv,xml]") for
      fallback loading and Office/EPub/etc. support.
    - beautifulsoup4 for HTML loading (`BSHTMLLoader`).
    - pymupdf for PDF loading (`PyMuPDFLoader`).
    - requests for web loading (`WebBaseLoader`).
    - jq for advanced JSON processing (`JSONLoader` with `jq_schema`).
    - pandas, pyarrow for Parquet loading (`ParquetLoader`).
"""

import os
import re
import sys  # Add missing sys import
import logging
from pathlib import Path
from typing import Optional, List, Any, Union, Dict, Type, Tuple, Iterator, Callable
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# --- Dependencies ---
# Import necessary loaders and core components
# Attempt to import all supported loaders, errors will be handled if features are used without deps
try:
    from langchain_community.document_loaders.base import BaseLoader
    from langchain_community.document_loaders import (
        PyMuPDFLoader,
        TextLoader,
        CSVLoader,
        JSONLoader,
        BSHTMLLoader,
        WebBaseLoader,
        UnstructuredFileLoader, # Used for fallback
        DirectoryLoader,
        # Specific Unstructured Loaders for better control/performance:
        UnstructuredEPubLoader,
        UnstructuredExcelLoader,
        UnstructuredMarkdownLoader,
        UnstructuredODTLoader,
        UnstructuredPowerPointLoader,
        UnstructuredRSTLoader,
        UnstructuredRTFLoader,
        UnstructuredTSVLoader,
        UnstructuredWordDocumentLoader,
        UnstructuredXMLLoader,
    )
    from langchain_core.documents import Document
except ImportError as e:
    print(f"ERROR: Failed to import core LangChain components: {e}", file=sys.stderr)
    print("Please ensure 'langchain-community' and 'langchain-core' are installed.", file=sys.stderr)
    sys.exit(1)

# Attempt to import unstructured cleaner (optional)
try:
    from unstructured.cleaners.core import clean_extra_whitespace
    CLEAN_EXTRA_WHITESPACE_AVAILABLE = True
except ImportError:
    clean_extra_whitespace = lambda x: x # Define as identity function if unavailable
    CLEAN_EXTRA_WHITESPACE_AVAILABLE = False


# --- Logging Setup ---
logger = logging.getLogger(__name__)
# Avoid setting basicConfig here if this module is imported elsewhere
# Rely on the calling application to configure logging


# --- Custom Exceptions ---
class AutoLoaderError(Exception):
    """Base exception class for errors originating from AutoDocumentLoader."""
    pass


class InvalidInputError(AutoLoaderError):
    """Raised when the input path or URL is invalid, not found, or inaccessible."""
    pass


class UnsupportedFormatError(AutoLoaderError):
    """Raised when a file format is not supported and fallback loading is disabled or failed."""
    pass


class LoadingError(AutoLoaderError):
    """Raised when an error occurs during the document loading or parsing process."""
    pass


# --- Custom Parquet Loader ---
class ParquetLoader(BaseLoader):
    """
    Loads Parquet files into LangChain Document objects using pandas.

    This loader reads a Parquet file, processing each row into a separate
    Document. The content of the document can be configured to be either
    a specific column or the entire row represented as a JSON string.
    Metadata includes the source file path, row number, and optionally
    other specified columns. Requires `pandas` and `pyarrow` to be installed.

    Attributes:
        file_path (str): Path to the Parquet file.
        content_column (Optional[str]): Name of the column to use for
            `page_content`. If None, the entire row (as JSON) is used.
        metadata_columns (Optional[List[str]]): List of column names to
            include in the metadata. If None, all columns except the
            `content_column` (if specified) are included.
        kwargs (Dict[str, Any]): Additional keyword arguments passed directly
            to `pandas.read_parquet`.
    """

    def __init__(
        self,
        file_path: str,
        content_column: Optional[str] = None,
        metadata_columns: Optional[List[str]] = None,
        **kwargs: Any # To pass arguments to pd.read_parquet
    ):
        """
        Initializes the ParquetLoader.

        Args:
            file_path: The path to the Parquet file.
            content_column: The name of the column to use as the main content
                            of the loaded Documents. If None, each row is
                            serialized to JSON for the content.
            metadata_columns: A list of column names to include in the metadata
                              of each Document. If None, all columns not used
                              as the content_column will be included.
            **kwargs: Additional keyword arguments to pass to `pandas.read_parquet`.
        """
        self.file_path = file_path
        self.content_column = content_column
        self.metadata_columns = metadata_columns
        self.kwargs = kwargs

        # Dependency check during initialization
        try:
            import pandas as pd
            import pyarrow
        except ImportError:
            raise ImportError(
                "Could not import pandas or pyarrow. Please install them with "
                "`pip install pandas pyarrow` to use ParquetLoader."
            )

    def load(self) -> List[Document]:
        """
        Loads the Parquet file into a list of Document objects.

        Reads the file row by row, creating a Document for each. Handles content
        and metadata extraction based on the initialization parameters.

        Returns:
            A list of LangChain Document objects, one for each row.

        Raises:
            ValueError: If the specified `content_column` does not exist in the file,
                        or if an error occurs during file reading or processing.
        """
        try:
            import pandas as pd

            logger.debug(f"Reading Parquet file: {self.file_path}")
            df = pd.read_parquet(self.file_path, **self.kwargs)
            logger.debug(f"Read {len(df)} rows from Parquet file.")

            documents: List[Document] = []

            # Determine which columns to include in metadata
            if self.metadata_columns is not None:
                # User specified exact metadata columns
                metadata_keys = [col for col in self.metadata_columns if col in df.columns]
                if len(metadata_keys) != len(self.metadata_columns):
                    missing = set(self.metadata_columns) - set(df.columns)
                    logger.warning(f"Metadata columns not found in Parquet file: {missing}")
            else:
                # Default: Use all columns not designated as content_column
                metadata_keys = [col for col in df.columns if col != self.content_column]

            # Iterate through DataFrame rows
            for i, row in df.iterrows():
                # Determine page content
                if self.content_column is not None:
                    if self.content_column not in df.columns:
                        # This check is important before accessing the column
                        raise ValueError(f"Content column '{self.content_column}' not found in Parquet file '{self.file_path}'")
                    content = str(row[self.content_column])
                else:
                    # Default: Convert the entire row to a JSON string
                    try:
                        content = row.to_json(date_format='iso')
                    except Exception as json_err:
                        logger.warning(f"Could not convert row {i} to JSON: {json_err}. Using string representation as fallback.")
                        content = str(row.to_dict()) # Fallback content

                # Construct metadata dictionary
                metadata: Dict[str, Any] = {"source": self.file_path, "row": int(i)} # Ensure row index is int
                for key in metadata_keys:
                    value = row[key]
                    # Convert numpy/pandas types to Python native types for JSON compatibility
                    # Handles common types like numpy int64, float64, bool_, pandas Timestamp
                    if hasattr(value, 'item'): # Check for numpy types
                         try:
                             metadata[key] = value.item()
                         except ValueError: # Handle potential conversion issues (e.g., object arrays)
                             metadata[key] = str(value)
                    elif pd is not None and isinstance(value, pd.Timestamp): # Check for pandas Timestamp
                         metadata[key] = value.isoformat()
                    else: # Assume it's a native Python type or fallback to string
                         metadata[key] = value if isinstance(value, (str, int, float, bool, list, dict, type(None))) else str(value)

                documents.append(Document(page_content=content, metadata=metadata))

            logger.debug(f"Created {len(documents)} Documents from Parquet file.")
            return documents

        except ImportError:
            # Should have been caught in __init__, but check again
             raise ImportError("pandas or pyarrow not installed. `pip install pandas pyarrow`")
        except FileNotFoundError:
            raise ValueError(f"Parquet file not found: {self.file_path}")
        except Exception as e:
            # Catch other potential pandas/pyarrow errors during read or processing
            logger.error(f"Error processing Parquet file {self.file_path}: {e}", exc_info=True)
            raise ValueError(f"Error loading parquet file '{self.file_path}': {e}") from e


# --- Main AutoLoader Class ---
class AutoDocumentLoader:
    """
    Automatically selects and loads documents from various sources.

    This class inspects the input `path_or_url` to determine if it's a file,
    directory, or web URL. It then instantiates the appropriate LangChain
    `BaseLoader` subclass to load the content into `Document` objects.
    It supports numerous file types directly and can use `UnstructuredFileLoader`
    as a fallback for others.

    Attributes:
        input_source (str): The file path, directory path, or web URL provided
                            during initialization.
        encoding (str): The default encoding used for text-based file loaders.
        jq_schema (str): The jq schema used by the `JSONLoader`.
        csv_loader_kwargs (Dict): Keyword arguments passed to `CSVLoader`.
        web_loader_kwargs (Dict): Keyword arguments passed to `WebBaseLoader`.
        parquet_loader_kwargs (Dict): Keyword arguments passed to `ParquetLoader`.
        dir_loader_kwargs (Dict): Keyword arguments passed to `DirectoryLoader`
                                 (excluding args managed directly like 'recursive').
        recursive (bool): Whether directory loading should be recursive.
        use_multithreading (bool): Whether to use multithreading for directory loading.
        max_concurrency (Optional[int]): Maximum workers for multithreaded directory loading.
        allow_unstructured_fallback (bool): Whether to attempt using
                                            `UnstructuredFileLoader` for unknown file types.
        unstructured_fallback_kwargs (Dict): Keyword arguments passed to the fallback
                                            `UnstructuredFileLoader`.
        single_file_extra_kwargs (Dict): Extra kwargs passed to single file loaders.
        loader (Optional[BaseLoader]): The instantiated LangChain loader instance,
                                       or None if initialization failed.
        input_type (Optional[str]): Detected type of the input ('url', 'directory', 'file').
        loader_class (Optional[Type[BaseLoader]]): The class of the loader used.
    """

    # Mapping from file extensions (lowercase) to LangChain loader classes
    # and default initialization arguments for that loader.
    _DEFAULT_FILE_LOADER_MAPPING: Dict[str, Tuple[Type[BaseLoader], Dict[str, Any]]] = {
        # Standard Loaders (Generally preferred for these types)
        ".pdf": (PyMuPDFLoader, {}),
        ".txt": (TextLoader, {"encoding": None}), # Encoding set in _configure_loader_args
        ".csv": (CSVLoader, {"encoding": None, "csv_args": {"delimiter": ",", "quotechar": '"'}}), # Encoding and args set later
        ".json": (JSONLoader, {"jq_schema": None, "text_content": False}), # Schema set later
        ".jsonl": (JSONLoader, {"jq_schema": None, "text_content": False, "json_lines": True}), # Schema set later
        ".html": (BSHTMLLoader, {"open_encoding": None}), # Encoding set later
        ".htm": (BSHTMLLoader, {"open_encoding": None}), # Encoding set later

        # Unstructured Loaders (for specific complex formats)
        # Apply whitespace cleaning as a basic post-processor
        ".epub": (UnstructuredEPubLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".xls": (UnstructuredExcelLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".xlsx": (UnstructuredExcelLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".md": (UnstructuredMarkdownLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".odt": (UnstructuredODTLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".ppt": (UnstructuredPowerPointLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".pptx": (UnstructuredPowerPointLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".rst": (UnstructuredRSTLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".rtf": (UnstructuredRTFLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".tsv": (UnstructuredTSVLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".doc": (UnstructuredWordDocumentLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".docx": (UnstructuredWordDocumentLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),
        ".xml": (UnstructuredXMLLoader, {"mode": "single", "post_processors": [clean_extra_whitespace] if CLEAN_EXTRA_WHITESPACE_AVAILABLE else []}),

        # Custom Parquet Loader
        ".parquet": (ParquetLoader, {}), # Args (content_column etc) passed via parquet_loader_kwargs
    }

    def __init__(
        self,
        path_or_url: Union[str, Path],
        encoding: str = 'utf-8',
        jq_schema: str = '.',
        csv_loader_kwargs: Optional[Dict[str, Any]] = None,
        web_loader_kwargs: Optional[Dict[str, Any]] = None,
        dir_loader_kwargs: Optional[Dict[str, Any]] = None,
        parquet_loader_kwargs: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        use_multithreading: bool = False,
        max_concurrency: Optional[int] = None,
        allow_unstructured_fallback: bool = True,
        unstructured_fallback_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any # Catches extra args for single file loaders
    ) -> None:
        """
        Initializes the AutoDocumentLoader, detecting the source type and selecting a loader.

        Args:
            path_or_url: The file path, directory path, or web URL to load data from.
            encoding: The default encoding to use for text-based files (e.g., '.txt', '.csv', '.html').
                      Defaults to 'utf-8'.
            jq_schema: The jq schema to apply when loading JSON files. Defaults to '.',
                       which loads the entire file content.
            csv_loader_kwargs: A dictionary of keyword arguments to pass directly to the
                               `CSVLoader`. Overrides defaults like delimiter.
            web_loader_kwargs: A dictionary of keyword arguments to pass directly to the
                               `WebBaseLoader`.
            dir_loader_kwargs: A dictionary of keyword arguments to pass directly to the
                               `DirectoryLoader`, excluding args like `path`, `recursive`, etc.,
                               which are handled by top-level arguments. Example: `silent_errors`.
            parquet_loader_kwargs: A dictionary of keyword arguments to pass directly to the
                                   `ParquetLoader` (e.g., `content_column`, `metadata_columns`).
            recursive: If True, load files recursively from subdirectories when the input
                       is a directory. Defaults to False.
            use_multithreading: If True, use multiple threads to load files when the input
                                is a directory. Defaults to False.
            max_concurrency: The maximum number of concurrent threads to use for directory
                             loading. Defaults to a system-dependent value if None.
            allow_unstructured_fallback: If True, attempt to use `UnstructuredFileLoader`
                                         for file types not explicitly mapped. Requires the
                                         `unstructured` library and relevant extras. Defaults to True.
            unstructured_fallback_kwargs: A dictionary of keyword arguments to pass directly
                                         to the fallback `UnstructuredFileLoader`.
            **kwargs: Additional keyword arguments that will be passed to the specific
                      loader chosen for a *single file*. These are ignored for URL and
                      directory loading.

        Raises:
            InvalidInputError: If the provided `path_or_url` is invalid, cannot be resolved,
                              or does not exist (for local paths).
            UnsupportedFormatError: If the input is a file with an unsupported extension
                                   and `allow_unstructured_fallback` is False or fails.
            AutoLoaderError: For other unexpected initialization errors.
            ImportError: If a required dependency for a specific loader is missing.
        """
        self.input_source = str(path_or_url)
        self.encoding = encoding
        self.jq_schema = jq_schema
        self.csv_loader_kwargs = csv_loader_kwargs or {}
        self.web_loader_kwargs = web_loader_kwargs or {}
        self.parquet_loader_kwargs = parquet_loader_kwargs or {}

        # Filter out keys managed by top-level args from dir_loader_kwargs provided by user
        # This prevents conflicts and ensures top-level args take precedence.
        managed_dir_keys = {'path', 'glob', 'recursive', 'use_multithreading', 'max_concurrency', 'loader_kwargs'}
        self.dir_loader_kwargs = {k: v for k, v in (dir_loader_kwargs or {}).items()
                                 if k not in managed_dir_keys}

        self.recursive = recursive
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.allow_unstructured_fallback = allow_unstructured_fallback
        self.unstructured_fallback_kwargs = unstructured_fallback_kwargs or {}
        self.single_file_extra_kwargs = kwargs # Store extra kwargs for single file loaders

        # Attributes to be set during initialization
        self.loader: Optional[BaseLoader] = None
        self.input_type: Optional[str] = None
        self.loader_class: Optional[Type[BaseLoader]] = None

        logger.info(f"Initializing AutoDocumentLoader for source: {self.input_source}")

        try:
            # Core logic to determine input type and select loader
            self._initialize_loader()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError) as e:
            # Catch specific path-related errors
            logger.error(f"Path validation failed for '{self.input_source}': {e}")
            raise InvalidInputError(f"Input path '{self.input_source}' is invalid or not accessible: {e}") from e
        except ValueError as e:
            # Catch value errors during initialization (e.g., invalid URL format)
            logger.error(f"Initialization value error for '{self.input_source}': {e}")
            raise InvalidInputError(f"Invalid value during initialization for '{self.input_source}': {e}") from e
        except AutoLoaderError:
            # Re-raise specific AutoLoader errors directly
            raise
        except Exception as e:
            # Catch any other unexpected errors during setup
            logger.exception(f"Unexpected error initializing loader for '{self.input_source}'")
            raise AutoLoaderError(f"Unexpected initialization error for '{self.input_source}': {e}") from e

        # Final check if loader was successfully initialized
        if self.loader is None:
            # This indicates an issue in the _initialize_loader logic if no exception was raised
            logger.error(f"Loader initialization completed without setting a loader for {self.input_source}")
            raise AutoLoaderError("Failed to initialize a document loader for an unknown reason.")

        logger.info(f"Successfully initialized loader '{self.loader.__class__.__name__}' for {self.input_type} source.")

    @lru_cache(maxsize=128) # Cache results for potentially repeated checks
    def _is_valid_url(self, text: str) -> bool:
        """
        Checks if the input string is a valid HTTP or HTTPS URL using urlparse.

        Args:
            text: The string to check.

        Returns:
            True if the string is a valid HTTP/HTTPS URL, False otherwise.
        """
        if not isinstance(text, str):
             return False
        try:
            result = urllib.parse.urlparse(text)
            # Check for scheme (http/https) and network location (domain name/IP)
            return all([result.scheme in ['http', 'https'], result.netloc])
        except ValueError:
            # urlparse can raise ValueError for severely malformed inputs
            return False

    def _initialize_loader(self) -> None:
        """
        Determines the input source type and initializes the appropriate loader.

        Checks for URL, then resolves local paths, checks existence, and finally
        determines if the existing path is a file or directory.

        Raises:
            InvalidInputError: If the input source cannot be identified as a valid
                               URL, directory, or file, if the path does not exist,
                               or if path resolution fails.
            # Other exceptions like UnsupportedFormatError are raised by specific methods.
        """
        # 1. Check for URL first
        if self._is_valid_url(self.input_source):
            self._initialize_url_loader()
            return

        # 2. Assume local path: attempt to create Path object and resolve
        try:
            # Create Path object first to handle potential invalid chars early
            path_obj = Path(self.input_source)
            # Resolve the path to get the absolute path and handle symlinks etc.
            # `strict=True` would raise FileNotFoundError here if path doesn't exist,
            # but we want to provide our custom error message, so resolve first, then check exists().
            resolved_path_obj = path_obj.resolve(strict=False)
        except OSError as e:
            # Handle OS-level errors during path resolution (e.g., invalid characters)
            raise InvalidInputError(f"Cannot resolve path '{self.input_source}': {e}") from e
        except Exception as e: # Catch other potential Path() instantiation/resolution errors
             raise InvalidInputError(f"Error processing path '{self.input_source}': {e}") from e

        # --- Corrected Logic: Check Existence FIRST ---
        # 3. Check if the resolved path actually exists on the filesystem
        if not resolved_path_obj.exists():
            # If it doesn't exist, raise the specific error immediately
            raise InvalidInputError(f"Input path does not exist: '{self.input_source}' (Resolved: '{resolved_path_obj}')")

        # --- Path Exists: Now determine if it's a file or directory ---
        # 4. Check if it's a directory
        if resolved_path_obj.is_dir():
            self._initialize_directory_loader(resolved_path_obj)
            return

        # 5. Check if it's a file
        elif resolved_path_obj.is_file():
            self._initialize_file_loader(resolved_path_obj)
            return

        # 6. Handle if it exists but isn't a file or directory
        else:
            # This state is rare but possible (e.g., sockets, broken symlinks after resolve)
            raise InvalidInputError(
                f"Input path '{self.input_source}' exists but is neither a recognizable file nor a directory."
            )

    def _initialize_url_loader(self) -> None:
        """Initializes the WebBaseLoader for URL sources."""
        self.input_type = 'url'
        logger.debug(f"Input identified as URL: {self.input_source}")
        self.loader_class = WebBaseLoader
        # Prepare arguments, prioritizing user-provided kwargs
        init_args = {'web_paths': [self.input_source]} # WebBaseLoader expects a list
        init_args.update(self.web_loader_kwargs) # Add user kwargs
        try:
            self.loader = self.loader_class(**init_args)
        except Exception as e:
            logger.error(f"Failed to initialize WebBaseLoader for {self.input_source}: {e}", exc_info=True)
            raise AutoLoaderError(f"Error initializing WebBaseLoader: {e}") from e

    def _initialize_directory_loader(self, path_obj: Path) -> None:
        """
        Initializes the DirectoryLoader for directory sources.

        Constructs the necessary `loader_kwargs` based on supported file types
        and applies user configurations for recursion, multithreading, etc.

        Args:
            path_obj: A Path object representing the directory.

        Raises:
            AutoLoaderError: If DirectoryLoader initialization fails.
        """
        self.input_type = 'directory'
        logger.debug(f"Input identified as directory: {path_obj}")
        self.loader_class = DirectoryLoader

        # Build the configuration dictionary mapping glob patterns to loader classes/kwargs
        constructed_loader_configs = self._build_directory_loader_configs()

        # Check if any loaders were actually configured
        if not constructed_loader_configs:
             logger.warning(f"No specific file loaders configured for directory '{path_obj}'. DirectoryLoader might not load any files unless a fallback 'glob' is provided.")
             # Depending on desired behavior, could raise an error here if no loaders are set up.

        # Prepare arguments for the DirectoryLoader constructor
        # User-provided glob takes precedence if specified
        final_glob = self.dir_loader_kwargs.pop('glob', None) # Remove glob from general kwargs if present

        dir_init_args = {
            'path': str(path_obj),
            'loader_kwargs': constructed_loader_configs, # Map of globs to configs
            'recursive': self.recursive,
            'use_multithreading': self.use_multithreading,
            'silent_errors': self.dir_loader_kwargs.get('silent_errors', False), # Default to False
            **self.dir_loader_kwargs # Add remaining user-provided kwargs
        }

        # Set glob only if explicitly provided by user or constructed configs exist
        if final_glob:
            dir_init_args['glob'] = final_glob
            logger.info(f"Using user-provided glob pattern for directory loading: '{final_glob}'")
        elif not constructed_loader_configs:
            # If no specific loaders AND no user glob, DirectoryLoader might default to '**/[!.]*'
            # which could be okay, but log a warning.
            logger.warning("No specific file type loaders configured and no 'glob' pattern provided. DirectoryLoader might use its default glob.")
        # else: If constructed_loader_configs exists, DirectoryLoader uses those globs implicitly.


        # Set max_concurrency only if multithreading is enabled
        if self.use_multithreading and self.max_concurrency is not None:
            if self.max_concurrency > 0:
                dir_init_args['max_concurrency'] = self.max_concurrency
                logger.info(f"Setting max_concurrency for directory loading to {self.max_concurrency}")
            else:
                 logger.warning(f"Invalid max_concurrency value ({self.max_concurrency}) provided. Using default.")


        # Instantiate the DirectoryLoader
        try:
            self.loader = DirectoryLoader(**dir_init_args)
        except Exception as e:
            logger.error(f"Failed to initialize DirectoryLoader for {path_obj}: {e}", exc_info=True)
            raise AutoLoaderError(f"Error initializing DirectoryLoader: {e}") from e

    def _build_directory_loader_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Builds the `loader_kwargs` dictionary for `DirectoryLoader`.

        This dictionary maps glob patterns (like `*.txt` or `**/*.pdf`) to
        specific loader configurations (loader class and its arguments).

        Returns:
            A dictionary suitable for `DirectoryLoader(loader_kwargs=...)`.
        """
        constructed_loader_configs: Dict[str, Dict[str, Any]] = {}

        # Iterate through explicitly supported file types
        for ext, (loader_cls, default_args) in self._DEFAULT_FILE_LOADER_MAPPING.items():
            # Create a fresh copy of default args for this loader type
            specific_loader_args = default_args.copy()

            # Apply general configurations (like encoding) and specific ones (like csv_kwargs)
            # Modifies specific_loader_args in place
            self._configure_loader_args(loader_cls, specific_loader_args)

            # Define the glob pattern based on recursion setting
            # Use **/* for recursive, * for non-recursive
            glob_pattern = f"**/*{ext}" if self.recursive else f"*{ext}"

            # Assign the loader class and its configured arguments to this glob pattern
            constructed_loader_configs[glob_pattern] = {
                "loader_cls": loader_cls,
                "loader_kwargs": specific_loader_args,
                # Pass silent_errors from top-level dir_kwargs if needed per loader config
                # "silent_errors": self.dir_loader_kwargs.get('silent_errors', False), # Example
            }
            logger.debug(f"Mapping glob '{glob_pattern}' to {loader_cls.__name__} with args: {specific_loader_args}")

        # Note: Unstructured fallback for directories is handled differently by DirectoryLoader.
        # If a file doesn't match any specific glob, DirectoryLoader might ignore it or raise
        # an error depending on its `silent_errors` setting. Explicitly adding a fallback
        # glob for UnstructuredFileLoader might be needed if that behavior is desired.
        # Example (if needed):
        # if self.allow_unstructured_fallback:
        #     fallback_glob = "**/*" if self.recursive else "*" # Catch-all (careful with precedence)
        #     constructed_loader_configs[fallback_glob] = { ... UnstructuredFileLoader config ... }
        # However, relying on DirectoryLoader's handling is usually simpler.

        if self.allow_unstructured_fallback:
            logger.info("Unstructured fallback is conceptually enabled (individual file errors might still occur).")
        else:
            logger.info("Unstructured fallback is disabled. Files not matching defined globs will be ignored by DirectoryLoader or may raise errors if `silent_errors` is False.")

        return constructed_loader_configs

    def _configure_loader_args(self, loader_cls: Type[BaseLoader], args: Dict[str, Any]) -> None:
        """
        Applies instance-level configurations (encoding, jq_schema, etc.)
        to the specific arguments dictionary for a given loader class.

        Modifies the `args` dictionary in place.

        Args:
            loader_cls: The class of the loader being configured (e.g., `TextLoader`).
            args: The dictionary of arguments for this loader instance, potentially
                  containing default values from `_DEFAULT_FILE_LOADER_MAPPING`.
        """
        # Apply encoding where relevant (TextLoader, BSHTMLLoader, CSVLoader)
        if loader_cls is TextLoader and args.get("encoding") is None:
            args["encoding"] = self.encoding
        elif loader_cls is BSHTMLLoader and args.get("open_encoding") is None:
            args["open_encoding"] = self.encoding
        elif loader_cls is CSVLoader:
            # Start with default encoding, then update with user's csv_loader_kwargs
            csv_final_kwargs = {"encoding": self.encoding}
            csv_final_kwargs.update(self.csv_loader_kwargs) # User kwargs override defaults/encoding
            args.update(csv_final_kwargs)
        # Apply jq_schema for JSONLoader
        elif loader_cls is JSONLoader and args.get("jq_schema") is None:
            args["jq_schema"] = self.jq_schema
            # Keep default text_content=False unless user overrides in single_file_extra_kwargs
        # Apply parquet kwargs
        elif loader_cls is ParquetLoader:
            args.update(self.parquet_loader_kwargs)
        # Add configurations for other loaders if needed (e.g., specific Unstructured modes)

    def _initialize_file_loader(self, path_obj: Path) -> None:
        """
        Initializes the appropriate loader for a single file source.

        Determines the loader based on the file extension, configures it,
        and attempts fallback using `UnstructuredFileLoader` if enabled and needed.

        Args:
            path_obj: A Path object representing the file.

        Raises:
            UnsupportedFormatError: If the file extension is not supported and
                                   fallback is disabled or fails.
            AutoLoaderError: If loader instantiation fails for other reasons.
        """
        self.input_type = 'file'
        file_ext = path_obj.suffix.lower() # Normalize to lowercase
        logger.debug(f"Input identified as file: {path_obj} (extension: {file_ext})")

        # Base arguments for most file loaders
        init_args = {'file_path': str(path_obj)}

        # Find specific loader based on extension
        loader_info = self._DEFAULT_FILE_LOADER_MAPPING.get(file_ext)

        if loader_info:
            # Found an explicitly mapped loader
            self.loader_class, default_args = loader_info
            logger.debug(f"Found specific loader for '{file_ext}': {self.loader_class.__name__}")
            # Start with default args, then apply instance configs, then extra single-file kwargs
            current_loader_args = default_args.copy()
            self._configure_loader_args(self.loader_class, current_loader_args)
            current_loader_args.update(self.single_file_extra_kwargs) # User's direct kwargs override all
            init_args.update(current_loader_args)

        elif self.allow_unstructured_fallback:
            # No specific loader found, attempt fallback
            logger.warning(f"Unsupported file extension '{file_ext}'. Attempting fallback with UnstructuredFileLoader.")
            try:
                # UnstructuredFileLoader doesn't usually take many specific init args besides file_path
                # but we allow passing them via unstructured_fallback_kwargs
                self.loader_class = UnstructuredFileLoader
                fallback_args = self.unstructured_fallback_kwargs.copy()
                # Apply general single file kwargs to fallback too? Might be unexpected.
                # fallback_args.update(self.single_file_extra_kwargs)
                init_args.update(fallback_args)
                logger.debug(f"Using UnstructuredFileLoader with effective args: {init_args}")
            except ImportError:
                # This typically means 'unstructured' itself is missing
                logger.error("Fallback failed: 'unstructured' package is not installed or importable.")
                raise UnsupportedFormatError(
                    f"File extension '{file_ext}' is not explicitly supported, and fallback failed "
                    "because the 'unstructured' package is not available. Install it (`pip install unstructured`) "
                    "and required extras (e.g., `pip install \"unstructured[docx]\"`)."
                )
            except Exception as e:
                 logger.error(f"Unexpected error initializing fallback UnstructuredFileLoader: {e}", exc_info=True)
                 raise AutoLoaderError(f"Failed to initialize fallback loader: {e}") from e
        else:
            # Fallback is disabled and no specific loader found
            logger.error(f"Unsupported file extension '{file_ext}' and fallback is disabled.")
            raise UnsupportedFormatError(
                f"File extension '{file_ext}' is not supported, and unstructured fallback is disabled."
            )

        # Final check for required args (should always have file_path)
        if 'file_path' not in init_args:
            # This would be an internal logic error
            raise AutoLoaderError("Internal error: 'file_path' missing from loader init args.")

        # Instantiate the chosen loader
        try:
            if self.loader_class is None: # Should not happen if logic above is correct
                 raise AutoLoaderError(f"Loader class could not be determined for file '{path_obj}'.")
            self.loader = self.loader_class(**init_args)
        except Exception as e:
            logger.error(f"Failed to instantiate loader {self.loader_class.__name__} for {path_obj} with args {init_args}: {e}", exc_info=True)
            raise AutoLoaderError(f"Error initializing loader {self.loader_class.__name__}: {e}") from e

    def load(self) -> List[Document]:
        """
        Loads the document(s) from the configured source using the selected loader.

        Returns:
            A list of loaded LangChain `Document` objects.

        Raises:
            LoadingError: If any error occurs during the underlying loader's
                          `load()` process (e.g., file parsing errors, network issues).
            AutoLoaderError: If the loader was not properly initialized before calling `load`.
        """
        if not self.loader:
            # Should ideally be caught by __init__, but defensive check
            raise AutoLoaderError("Loader was not initialized correctly before calling load().")

        loader_name = self.loader.__class__.__name__
        logger.info(f"Attempting to load {self.input_type} '{self.input_source}' using {loader_name}...")

        try:
            # Delegate the actual loading to the instantiated LangChain loader
            documents = self.loader.load()
            count = len(documents)
            logger.info(f"Successfully loaded {count} document{'s' if count != 1 else ''} from '{self.input_source}'.")
            # Perform basic validation on returned structure
            if not isinstance(documents, list) or (count > 0 and not isinstance(documents[0], Document)):
                 logger.warning(f"Loader {loader_name} returned unexpected type ({type(documents)}). Expected List[Document].")
                 # Attempt conversion or raise error depending on strictness needed
            return documents
        except Exception as e:
            # Catch errors from the specific loader's load() method
            logger.error(f"Error loading {self.input_type} '{self.input_source}' with {loader_name}: {e}", exc_info=True)
            # Wrap in a custom exception for consistent error handling
            raise LoadingError(f"Failed to load source '{self.input_source}' using {loader_name}.") from e

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily loads document(s), yielding them one by one as an iterator.

        This method is memory-efficient for large sources, especially single large
        files or directories where loaders support iteration. If the underlying
        loader does not natively support lazy loading, this method will fall back
        to loading all documents into memory first and then yielding them.

        Yields:
            `Document`: Loaded LangChain Document objects iteratively.

        Raises:
            AutoLoaderError: If the loader was not properly initialized.
            LoadingError: If an error occurs during the underlying loader's
                          lazy loading process or the fallback `load()` process.
        """
        if not self.loader:
            raise AutoLoaderError("Loader was not initialized correctly before calling lazy_load().")

        loader_name = self.loader.__class__.__name__
        logger.info(f"Attempting to lazy load {self.input_type} '{self.input_source}' using {loader_name}...")

        # Check if the instantiated loader has a 'lazy_load' method
        if hasattr(self.loader, 'lazy_load') and callable(self.loader.lazy_load):
            try:
                # Delegate directly to the loader's lazy_load method
                yield from self.loader.lazy_load()
                logger.debug(f"Finished lazy loading stream from '{self.input_source}'.")
            except Exception as e:
                logger.error(f"Error during native lazy loading with {loader_name}: {e}", exc_info=True)
                raise LoadingError(f"Failed during native lazy load from '{self.input_source}' using {loader_name}.") from e
        else:
            # Loader does not support lazy loading, fall back to loading all then iterating
            logger.warning(f"Loader {loader_name} does not support native lazy_load(). Loading all documents into memory first.")
            try:
                # Load all documents using the standard load method
                documents = self.load()
                # Iterate over the loaded documents
                yield from documents
                logger.debug(f"Finished yielding documents after fallback load for '{self.input_source}'.")
            except LoadingError:
                # Re-raise loading errors from the fallback load() call
                raise
            except Exception as e:
                # Catch unexpected errors during the iteration phase of the fallback
                logger.error(f"Unexpected error during fallback lazy load iteration for '{self.input_source}': {e}", exc_info=True)
                raise LoadingError(f"Unexpected error during fallback lazy load for '{self.input_source}'.") from e

    def load_and_split(self, text_splitter: Any) -> List[Document]:
        """
        Loads document(s) from the source and splits them using a text splitter.

        Optimizes by using the loader's native `load_and_split` method if available.
        Otherwise, falls back to loading all documents first, then applying the splitter.

        Args:
            text_splitter: An initialized LangChain text splitter instance (e.g.,
                           `RecursiveCharacterTextSplitter`). Must have a
                           `split_documents` method.

        Returns:
            A list of LangChain `Document` objects representing the split chunks.

        Raises:
            ValueError: If `text_splitter` is not provided or is not a valid splitter instance.
            AutoLoaderError: If the loader was not properly initialized.
            LoadingError: If an error occurs during the document loading phase (either native
                          `load_and_split` or the fallback `load`).
            RuntimeError: If an error occurs specifically during the splitting phase in the fallback.
        """
        # --- Input Validation ---
        if not text_splitter:
             raise ValueError("A text_splitter instance must be provided.")
        if not hasattr(text_splitter, 'split_documents') or not callable(text_splitter.split_documents):
             raise ValueError("The provided text_splitter must have a callable 'split_documents' method.")

        if not self.loader:
            raise AutoLoaderError("Loader was not initialized correctly before calling load_and_split().")

        loader_name = self.loader.__class__.__name__
        splitter_name = text_splitter.__class__.__name__

        # --- Attempt Native load_and_split (Optimization) ---
        if hasattr(self.loader, 'load_and_split') and callable(self.loader.load_and_split):
            logger.info(f"Attempting native load_and_split of {loader_name} using {splitter_name} for '{self.input_source}'...")
            try:
                # Delegate to the loader's combined method
                split_documents = self.loader.load_and_split(text_splitter=text_splitter)
                count = len(split_documents)
                logger.info(f"Successfully loaded and split into {count} chunk{'s' if count != 1 else ''} using native method.")
                # Validate return type (optional but good practice)
                if not isinstance(split_documents, list) or (count > 0 and not isinstance(split_documents[0], Document)):
                    logger.warning(f"Loader's native load_and_split returned unexpected type ({type(split_documents)}).")
                return split_documents
            except NotImplementedError:
                # Loader has the method name but doesn't implement it (should be rare)
                logger.warning(f"{loader_name} has 'load_and_split' attribute but raised NotImplementedError. Falling back.")
            except Exception as e:
                # Catch errors specific to the native load_and_split call
                logger.error(f"Error during native load_and_split with {loader_name}: {e}", exc_info=True)
                raise LoadingError(f"Failed during native load_and_split with {loader_name}.") from e

        # --- Fallback: Load then Split ---
        logger.info(f"Falling back to load() then split() for '{self.input_source}' using {splitter_name}...")
        # Step 1: Load all documents using the regular load method (handles its own errors)
        try:
             docs = self.load()
        except LoadingError:
             raise # Propagate loading errors immediately

        if not docs:
            logger.warning(f"Loading returned no documents from '{self.input_source}'. Splitting resulted in 0 chunks.")
            return []

        # Step 2: Split the loaded documents
        logger.info(f"Splitting {len(docs)} loaded document{'s' if len(docs) != 1 else ''} using {splitter_name}...")
        try:
            split_documents = text_splitter.split_documents(docs)
            count = len(split_documents)
            logger.info(f"Successfully split documents into {count} chunk{'s' if count != 1 else ''}.")
            # Validate return type
            if not isinstance(split_documents, list) or (count > 0 and not isinstance(split_documents[0], Document)):
                 logger.warning(f"Text splitter returned unexpected type ({type(split_documents)}).")
            return split_documents
        except Exception as e:
            # Catch errors specifically from the text_splitter
            logger.error(f"Error splitting documents from '{self.input_source}' using {splitter_name}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to split documents from '{self.input_source}' using {splitter_name}.") from e

    @classmethod
    def batch_load(cls,
                   source_list: List[Union[str, Path]],
                   max_workers: Optional[int] = None,
                   **common_kwargs: Any
                   ) -> List[Document]:
        """
        Loads documents from a list of sources potentially in parallel using threads.

        Instantiates an `AutoDocumentLoader` for each source and calls its `load()`
        method concurrently. Aggregates results. Errors during loading of individual
        sources are logged but do not stop the processing of others.

        Args:
            source_list: A list of input sources (file paths, directory paths, or URLs).
            max_workers: The maximum number of threads to use. If None, defaults to a
                         reasonable number based on CPU cores or number of sources.
            **common_kwargs: Common keyword arguments to pass to the `__init__` method
                             of each `AutoDocumentLoader` instance created (e.g., `encoding`,
                             `recursive`, `allow_unstructured_fallback`).

        Returns:
            A single list containing all successfully loaded `Document` objects
            from all sources, potentially in an arbitrary order due to concurrency.
        """
        if not source_list:
            logger.warning("batch_load called with an empty source list.")
            return []

        # Function to load a single source, handling potential errors
        def load_single_source(source: Union[str, Path]) -> List[Document]:
            """Loads a single source, returning empty list on error."""
            source_str = str(source) # Ensure string representation for logging
            try:
                # Instantiate loader with common arguments for this source
                loader = cls(source_str, **common_kwargs)
                # Load the documents for this source
                return loader.load()
            except Exception as e:
                # Log the error for the specific source but allow others to continue
                logger.error(f"Error loading source '{source_str}' during batch load: {e}", exc_info=False) # Avoid overly verbose tracebacks for single failures
                return [] # Return empty list for this failed source

        # Determine the number of workers for the thread pool
        if max_workers is None:
            # Default heuristic: min(32, os.cpu_count() + 4, len(source_list))
            # Caps threads, uses slightly more than CPU count for I/O bound tasks,
            # and doesn't create more threads than sources.
            cpu_count = os.cpu_count() or 1
            effective_max_workers = min(32, cpu_count + 4, len(source_list))
        else:
            effective_max_workers = min(max(1, max_workers), len(source_list)) # Ensure at least 1 worker, max is number of sources

        logger.info(f"Starting batch load for {len(source_list)} sources using up to {effective_max_workers} worker threads.")

        all_documents: List[Document] = []
        # Use ThreadPoolExecutor for concurrent loading
        with ThreadPoolExecutor(max_workers=effective_max_workers) as executor:
            # Map the loading function to the list of sources
            # executor.map preserves order of inputs corresponding to outputs (though completion order varies)
            results_iterator = executor.map(load_single_source, source_list)

            # Process results as they complete (or after all complete if map is used)
            for doc_list in results_iterator:
                # Extend the main list with documents from this source (if any)
                all_documents.extend(doc_list)

        total_docs = len(all_documents)
        logger.info(f"Batch loading finished. Loaded a total of {total_docs} document{'s' if total_docs != 1 else ''} from {len(source_list)} sources.")
        return all_documents


# --- Example Usage (for demonstration or testing) ---
if __name__ == "__main__":
    # Configure logging for standalone execution
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")
    logger.info("Running AutoDocumentLoader example...")

    # --- Create Dummy Files for Testing ---
    TEST_DIR = Path("_autoloader_test_dir")
    TEST_DIR.mkdir(exist_ok=True)
    (TEST_DIR / "sample.txt").write_text("This is a sample text file.", encoding="utf-8")
    (TEST_DIR / "sample.csv").write_text("col1,col2\nval1,val2\nval3,val4", encoding="utf-8")
    (TEST_DIR / "sample.json").write_text('{"name": "example", "value": 123}', encoding="utf-8")
    (TEST_DIR / "unsupported.xyz").write_text("Some content in an unknown file.", encoding="utf-8")
    # Create a subdirectory for recursive test
    SUB_DIR = TEST_DIR / "subdir"
    SUB_DIR.mkdir(exist_ok=True)
    (SUB_DIR / "nested.txt").write_text("This is nested text.", encoding="utf-8")

    # --- Test Cases ---
    test_sources = {
        "Single TXT File": str(TEST_DIR / "sample.txt"),
        "Single CSV File": str(TEST_DIR / "sample.csv"),
        "Single JSON File": str(TEST_DIR / "sample.json"),
        "Web URL (Example)": "https://example.com", # Requires internet
        "Directory (Non-Recursive)": str(TEST_DIR),
        "Directory (Recursive)": str(TEST_DIR),
        "Unsupported File (Fallback Enabled)": str(TEST_DIR / "unsupported.xyz"),
        "Unsupported File (Fallback Disabled)": str(TEST_DIR / "unsupported.xyz"),
        "Invalid Path": "_non_existent_file.abc",
    }

    results = {}

    for name, source in test_sources.items():
        print(f"\n--- Testing: {name} ---")
        print(f"Source: {source}")
        kwargs = {}
        if name == "Directory (Recursive)":
            kwargs['recursive'] = True
        if name == "Unsupported File (Fallback Disabled)":
            kwargs['allow_unstructured_fallback'] = False

        try:
            loader = AutoDocumentLoader(source, **kwargs)
            # Test lazy loading
            print("Testing lazy_load()...")
            loaded_docs_lazy = []
            for i, doc in enumerate(loader.lazy_load()):
                 print(f"  Lazy loaded doc {i+1}: {str(doc.page_content)[:80]}...")
                 loaded_docs_lazy.append(doc)
            results[name] = f"Lazy loaded {len(loaded_docs_lazy)} docs."

            # Test standard loading (should be same result)
            # print("Testing load()...")
            # loaded_docs = loader.load()
            # print(f"  Loaded {len(loaded_docs)} docs.")
            # if len(loaded_docs) != len(loaded_docs_lazy):
            #     print(f"  WARNING: load() count ({len(loaded_docs)}) differs from lazy_load() count ({len(loaded_docs_lazy)})")
            # results[name] = f"Loaded {len(loaded_docs)} docs."

            # Optional: Test load_and_split if a splitter is available/needed
            # from langchain.text_splitter import CharacterTextSplitter
            # splitter = CharacterTextSplitter(chunk_size=50, chunk_overlap=0)
            # print("Testing load_and_split()...")
            # split_docs = loader.load_and_split(splitter)
            # print(f"  Split into {len(split_docs)} chunks.")

        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            results[name] = f"Failed ({type(e).__name__})"

    # --- Test Batch Loading ---
    print("\n--- Testing: Batch Load ---")
    batch_sources = [
        str(TEST_DIR / "sample.txt"),
        str(TEST_DIR / "sample.csv"),
        "_non_existent_file.pdf", # Include a failing one
        str(SUB_DIR / "nested.txt")
    ]
    print(f"Sources: {batch_sources}")
    try:
        batch_docs = AutoDocumentLoader.batch_load(batch_sources)
        results["Batch Load"] = f"Loaded {len(batch_docs)} docs."
        print(f"  Batch loaded {len(batch_docs)} docs total.")
    except Exception as e:
         print(f"  ERROR during batch load: {type(e).__name__}: {e}")
         results["Batch Load"] = f"Failed ({type(e).__name__})"


    # --- Print Summary ---
    print("\n--- Test Summary ---")
    for name, result in results.items():
        print(f"- {name:<35}: {result}")

    # --- Cleanup ---
    print("\n--- Cleaning up test files ---")
    try:
        import shutil
        shutil.rmtree(TEST_DIR)
        print(f"Removed test directory: {TEST_DIR}")
    except Exception as e:
        print(f"Error cleaning up test directory {TEST_DIR}: {e}")    
    