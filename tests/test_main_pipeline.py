# tests/test_main_pipeline.py
import pytest
import argparse
from unittest.mock import patch, MagicMock, ANY, mock_open
import logging
import json
import os

# Import parts of main needed for testing run_generation_pipeline
from synthetic_data_generator.main import run_generation_pipeline, setup_logging, parse_arguments
from synthetic_data_generator import exceptions
from synthetic_data_generator import config

# --- Fixtures ---

@pytest.fixture
def mock_args_parser(mocker):
    """Mocks the ArgumentParser and returns the mock object."""
    mock_parser = MagicMock(spec=argparse.ArgumentParser)
    mock_parsed_args = MagicMock(spec=argparse.Namespace)
    mock_parser.parse_args.return_value = mock_parsed_args
    mocker.patch('argparse.ArgumentParser', return_value=mock_parser)

    # Set default values on the mock parsed args
    mock_parsed_args.query = None
    mock_parsed_args.documents = None
    mock_parsed_args.format = None
    mock_parsed_args.custom_format_file = None
    mock_parsed_args.samples = config.DEFAULT_SAMPLES
    mock_parsed_args.no_refine = False
    mock_parsed_args.model = config.DEFAULT_MODEL_NAME
    mock_parsed_args.temperature = config.DEFAULT_TEMPERATURE
    mock_parsed_args.top_p = config.DEFAULT_TOP_P
    mock_parsed_args.top_k = config.DEFAULT_TOP_K
    mock_parsed_args.output_prefix = config.DEFAULT_OUTPUT_PREFIX
    mock_parsed_args.output_format = ['all']
    mock_parsed_args.incremental_save = False
    mock_parsed_args.batch_size = config.DEFAULT_BATCH_SIZE
    mock_parsed_args.max_retries = config.DEFAULT_MAX_RETRIES
    mock_parsed_args.log_level = config.DEFAULT_LOG_LEVEL

    return mock_parsed_args # Return the mock Namespace object

# --- Mocks for Dependencies ---

@pytest.fixture
def mock_dependencies(mocker, mock_llm_client, sample_custom_format_def):
    """Mocks all major dependencies called by run_generation_pipeline."""
    mocks = {
        'load_validate_custom': mocker.patch('synthetic_data_generator.formats.utils.load_and_validate_custom_format', return_value=sample_custom_format_def),
        'PredefinedHandler': mocker.patch('synthetic_data_generator.formats.predefined.PredefinedFormatHandler'),
        'CustomHandler': mocker.patch('synthetic_data_generator.formats.custom.CustomFormatHandler'),
        'ChatGoogleGenerativeAI': mocker.patch('synthetic_data_generator.config.ChatGoogleGenerativeAI', return_value=mock_llm_client),
        'SyntheticDataGenerator': mocker.patch('synthetic_data_generator.llm.generator.SyntheticDataGenerator'),
        'save_data': mocker.patch('synthetic_data_generator.output.saver.save_data'),
        'save_as_jsonl': mocker.patch('synthetic_data_generator.output.saver.save_as_jsonl'),
        'load_from_jsonl': mocker.patch('synthetic_data_generator.output.saver.load_from_jsonl'),
        'os_path_exists': mocker.patch('os.path.exists', return_value=True), # Assume files exist for preview/load
        'open': mocker.patch('builtins.open', mock_open(read_data='{"id":1}\n{"id":2}\n{"id":3}\n')), # Mock file reading for preview
        'json_loads': mocker.patch('json.loads', side_effect=json.loads), # Use real json.loads
        'json_dumps': mocker.patch('json.dumps', side_effect=json.dumps), # Use real json.dumps
        'setup_logging': mocker.patch('synthetic_data_generator.main.setup_logging'), # Mock logging setup
        'load_environment': mocker.patch('synthetic_data_generator.config.load_environment'), # Mock env loading
    }
    # Mock the generator instance methods
    mock_generator_instance = mocks['SyntheticDataGenerator'].return_value
    mock_generator_instance.generate_from_query.return_value = ([{"id": 1}], 1) # (data, count)
    mock_generator_instance.generate_from_documents.return_value = ([{"id": 2}], 1)

    # Mock format handler instances returned by the mocked classes
    mock_predefined_handler_instance = mocks['PredefinedHandler'].return_value
    mock_predefined_handler_instance.get_format_name.return_value = "mock_qa"
    mock_predefined_handler_instance.get_field_names.return_value = ["context", "question", "answer"]

    mock_custom_handler_instance = mocks['CustomHandler'].return_value
    mock_custom_handler_instance.get_format_name.return_value = "mock_custom"
    mock_custom_handler_instance.get_field_names.return_value = ["id", "text", "score", "verified"]

    return mocks, mock_generator_instance

# --- Tests ---

def test_run_pipeline_predefined_query(mock_args_parser, mock_dependencies, caplog):
    """Test pipeline with predefined format and query."""
    caplog.set_level(logging.INFO)
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    # Set specific args for this test
    args.query = "Generate QA pairs"
    args.format = "qa"
    args.custom_format_file = None
    args.output_format = ["jsonl"]
    args.incremental_save = False
    args.no_refine = False # Explicitly set refine=True behavior

    run_generation_pipeline(args)

    mocks['load_environment'].assert_called_once()
    mocks['setup_logging'].assert_called_once_with(args.log_level)
    mocks['PredefinedHandler'].assert_called_once_with("qa")
    mocks['CustomHandler'].assert_not_called()
    mocks['ChatGoogleGenerativeAI'].assert_called_once_with(
        model=args.model, temperature=args.temperature, top_p=args.top_p, top_k=args.top_k, safety_settings=ANY
    )
    mocks['SyntheticDataGenerator'].assert_called_once()
    mock_gen_instance.generate_from_query.assert_called_once_with(
        query="Generate QA pairs", num_samples=args.samples, batch_size=args.batch_size, refine=True, output_file_prefix=None, incremental_save=False
    )
    mock_gen_instance.generate_from_documents.assert_not_called()
    mocks['save_as_jsonl'].assert_called_once() # Called because incremental=False
    mocks['save_data'].assert_not_called() # Only jsonl requested
    assert "Data Preview" in caplog.text
    assert "Generation complete. 1 unique samples generated." in caplog.text

def test_run_pipeline_custom_docs_incremental(mock_args_parser, mock_dependencies, mocker, caplog):
    """Test pipeline with custom format, documents, and incremental save."""
    caplog.set_level(logging.INFO)
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    # Set specific args
    args.query = None
    args.format = None
    args.documents = "path/to/docs"
    args.custom_format_file = "my_format.json"
    args.output_format = ["csv", "parquet"] # Request other formats
    args.incremental_save = True
    args.output_prefix = "output/my_custom_data"

    # Simulate generator returning data count
    mock_gen_instance.generate_from_documents.return_value = ([], 5) # No data in memory, 5 generated

    # Simulate loading from incremental file for conversion
    mocks['load_from_jsonl'].return_value = [{"id": i} for i in range(5)]
    # Mock pandas availability for parquet
    mocker.patch('synthetic_data_generator.output.saver.PANDAS_AVAILABLE', True)
    # Mock os.path.basename for filename generation
    mocker.patch('os.path.basename', return_value='my_custom_data')

    run_generation_pipeline(args)

    mocks['load_validate_custom'].assert_called_once_with("my_format.json")
    mocks['CustomHandler'].assert_called_once()
    mocks['PredefinedHandler'].assert_not_called()
    mocks['ChatGoogleGenerativeAI'].assert_called_once()
    mocks['SyntheticDataGenerator'].assert_called_once()
    mock_gen_instance.generate_from_documents.assert_called_once_with(
        document_path="path/to/docs", num_samples=args.samples, batch_size=args.batch_size, output_file_prefix=args.output_prefix, incremental_save=True
    )
    mock_gen_instance.generate_from_query.assert_not_called()

    mocks['save_as_jsonl'].assert_not_called() # Not called directly when incremental=True
    mocks['load_from_jsonl'].assert_called_once() # Called to load for conversion
    # Check the filename passed to load_from_jsonl
    expected_inc_file = f"{args.output_prefix}_mock_custom_from_documents.jsonl"
    mocks['load_from_jsonl'].assert_called_once_with(expected_inc_file)

    mocks['save_data'].assert_called_once() # Called for csv/parquet
    # Check args passed to save_data
    save_data_args = mocks['save_data'].call_args[0]
    assert save_data_args[0] == [{"id": i} for i in range(5)] # Data
    assert save_data_args[1] == expected_inc_file.replace(".jsonl", "") # Base filename
    assert save_data_args[2] == ['csv', 'parquet'] # Formats
    assert save_data_args[3] == ["id", "text", "score", "verified"] # Fieldnames from custom handler

    assert "Data Preview" in caplog.text
    assert "Generation complete. 5 unique samples generated." in caplog.text
    assert f"Saved incrementally to {expected_inc_file}" in caplog.text
    assert "Converting incremental JSONL to requested formats: csv, parquet" in caplog.text

def test_run_pipeline_format_load_fail(mock_args_parser, mock_dependencies):
    """Test pipeline exits if format loading fails."""
    mocks, _ = mock_dependencies
    args = mock_args_parser
    args.format = None
    args.custom_format_file = "bad_format.json"
    mocks['load_validate_custom'].side_effect = exceptions.ConfigurationError("Bad format file")

    with pytest.raises(exceptions.ConfigurationError, match="Bad format file"):
        run_generation_pipeline(args)

def test_run_pipeline_llm_init_fail(mock_args_parser, mock_dependencies):
    """Test pipeline exits if LLM initialization fails."""
    mocks, _ = mock_dependencies
    args = mock_args_parser
    args.format = "qa"
    args.custom_format_file = None
    mocks['ChatGoogleGenerativeAI'].side_effect = Exception("Invalid API Key")

    with pytest.raises(exceptions.GenerationError, match="LLM client initialization failed"):
        run_generation_pipeline(args)

def test_run_pipeline_generator_fail(mock_args_parser, mock_dependencies):
    """Test pipeline exits if generator execution fails."""
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    args.format = "qa"
    args.query = "test"
    mock_gen_instance.generate_from_query.side_effect = exceptions.GenerationError("LLM failed permanently")

    with pytest.raises(exceptions.GenerationError, match="LLM failed permanently"):
        run_generation_pipeline(args)

def test_run_pipeline_no_unique_data(mock_args_parser, mock_dependencies, caplog):
    """Test pipeline skips saving if no unique data is generated."""
    caplog.set_level(logging.INFO)
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    args.format = "qa"
    args.query = "test"
    # Simulate generator returning 0 unique samples
    mock_gen_instance.generate_from_query.return_value = ([], 0)

    run_generation_pipeline(args)

    assert "No unique data generated. Skipping final saving." in caplog.text
    mocks['save_as_jsonl'].assert_not_called()
    mocks['save_data'].assert_not_called()
    # Preview should also indicate no samples
    assert "(No samples available to preview)" in caplog.text

def test_run_pipeline_parquet_unavailable(mock_args_parser, mock_dependencies, mocker, caplog):
    """Test parquet saving is skipped if pandas is unavailable."""
    caplog.set_level(logging.INFO)
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    args.format = "qa"
    args.query = "test"
    args.output_format = ["parquet", "jsonl"] # Request parquet
    args.incremental_save = False

    # Mock pandas as unavailable
    mocker.patch('synthetic_data_generator.output.saver.PANDAS_AVAILABLE', False)

    run_generation_pipeline(args)

    assert "Skipping Parquet output: pandas library not found or failed to import." in caplog.text
    # save_as_jsonl should still be called
    mocks['save_as_jsonl'].assert_called_once()
    # save_data should NOT be called for parquet
    mocks['save_data'].assert_not_called()

def test_run_pipeline_incremental_load_fail(mock_args_parser, mock_dependencies, caplog):
    """Test pipeline handles failure when loading incremental file for conversion."""
    caplog.set_level(logging.ERROR)
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    args.format = "qa"
    args.query = "test"
    args.output_format = ["csv"] # Request conversion
    args.incremental_save = True
    args.output_prefix = "output/inc_fail"

    # Simulate generator completing
    mock_gen_instance.generate_from_query.return_value = ([], 5)
    # Simulate load_from_jsonl failing
    mocks['load_from_jsonl'].side_effect = FileNotFoundError("Cannot find incremental file")

    # Should not raise an exception, but log an error
    run_generation_pipeline(args)

    mocks['load_from_jsonl'].assert_called_once()
    assert "Failed to load incremental data from" in caplog.text
    assert "Cannot find incremental file" in caplog.text
    assert "Skipping conversion to other formats." in caplog.text
    mocks['save_data'].assert_not_called() # Conversion skipped

def test_run_pipeline_preview_fail(mock_args_parser, mock_dependencies, caplog):
    """Test pipeline handles failure during data preview."""
    caplog.set_level(logging.WARNING)
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    args.format = "qa"
    args.query = "test"
    args.incremental_save = False # Preview reads from memory

    # Simulate generator returning data
    mock_gen_instance.generate_from_query.return_value = ([{"id": 1}], 1)
    # Simulate json.dumps failing during preview formatting
    mocks['json_dumps'].side_effect = TypeError("Cannot dump")

    # Should not raise an exception, but log a warning
    run_generation_pipeline(args)

    assert "Could not generate preview for item 0" in caplog.text
    assert "Cannot dump" in caplog.text
    # Saving should still proceed
    mocks['save_as_jsonl'].assert_called_once()

def test_run_pipeline_preview_incremental_file_fail(mock_args_parser, mock_dependencies, caplog):
    """Test pipeline handles failure reading incremental file for preview."""
    caplog.set_level(logging.WARNING)
    mocks, mock_gen_instance = mock_dependencies
    args = mock_args_parser
    args.format = "qa"
    args.query = "test"
    args.incremental_save = True # Preview reads from file
    args.output_prefix = "output/preview_fail"

    # Simulate generator completing
    mock_gen_instance.generate_from_query.return_value = ([], 5)
    # Simulate file opening failing during preview
    mocks['open'].side_effect = OSError("Cannot open preview file")

    # Should not raise an exception, but log a warning
    run_generation_pipeline(args)

    assert "Could not open incremental file" in caplog.text
    assert "for preview" in caplog.text
    assert "Cannot open preview file" in caplog.text
    # Conversion should still be attempted (if requested)
    mocks['load_from_jsonl'].assert_called_once() # Assumes default output format 'all' includes jsonl

# --- setup_logging tests ---

@patch('logging.basicConfig')
def test_setup_logging_levels(mock_basic_config):
    """Test setup_logging configures correct level."""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    for level_name, level_const in level_map.items():
        mock_basic_config.reset_mock()
        setup_logging(level_name)
        mock_basic_config.assert_called_once()
        # Check the level argument passed to basicConfig
        assert mock_basic_config.call_args.kwargs.get('level') == level_const

@patch('logging.basicConfig')
def test_setup_logging_invalid_level(mock_basic_config, caplog):
    """Test setup_logging defaults to INFO for invalid level."""
    caplog.set_level(logging.WARNING)
    setup_logging("INVALID_LEVEL")
    mock_basic_config.assert_called_once()
    assert mock_basic_config.call_args.kwargs.get('level') == logging.INFO
    assert "Invalid log level 'INVALID_LEVEL'" in caplog.text

# --- parse_arguments tests (Optional - requires more setup or testing via main entry point) ---
# Testing argparse directly can be complex. Often covered by testing run_generation_pipeline
# with different mock_args_parser configurations, or via integration tests.
# Example sketch:
# def test_parse_arguments_requires_input_source():
#     with pytest.raises(SystemExit): # Argparse exits on error
#         parse_arguments(['--format', 'qa']) # Missing --query or --documents

# def test_parse_arguments_requires_format():
#     with pytest.raises(SystemExit):
#         parse_arguments(['--query', 'test']) # Missing --format or --custom_format_file
