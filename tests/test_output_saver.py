# tests/test_output_saver.py
import pytest
import json
import csv
import os
from unittest.mock import patch, mock_open, MagicMock, call, ANY
import logging

# Import the module to test
from synthetic_data_generator.output import saver
from synthetic_data_generator import config # For OUTPUT_FILE_FORMATS

# Sample data
SAMPLE_DATA = [
    {"col_a": 1, "col_b": "hello", "col_c": 1.1},
    {"col_a": 2, "col_b": "world", "col_c": 2.2},
    {"col_a": 3, "col_b": "pytest", "col_c": 3.3},
]
CSV_FIELDNAMES = ["col_a", "col_b", "col_c"]

# Capture original json.dumps before any patches
original_json_dumps = json.dumps

# --- save_data tests ---

def test_save_data_no_data(mocker, caplog):
    """Test save_data skips if data list is empty."""
    mock_save_jsonl = mocker.patch('synthetic_data_generator.output.saver.save_as_jsonl')
    mock_save_csv = mocker.patch('synthetic_data_generator.output.saver.save_as_csv')
    mock_save_parquet = mocker.patch('synthetic_data_generator.output.saver.save_as_parquet')
    mock_makedirs = mocker.patch('os.makedirs')

    saver.save_data([], "output/test", ["jsonl", "csv"])

    assert "No data provided to save_data" in caplog.text
    mock_makedirs.assert_not_called()
    mock_save_jsonl.assert_not_called()
    mock_save_csv.assert_not_called()
    mock_save_parquet.assert_not_called()

def test_save_data_no_formats(mocker, caplog):
    """Test save_data skips if output_formats list is empty."""
    mock_save_jsonl = mocker.patch('synthetic_data_generator.output.saver.save_as_jsonl')
    mock_makedirs = mocker.patch('os.makedirs')

    saver.save_data(SAMPLE_DATA, "output/test", [])

    assert "No output formats specified" in caplog.text
    mock_makedirs.assert_not_called()
    mock_save_jsonl.assert_not_called()

def test_save_data_creates_dir(mocker):
    """Test save_data creates the output directory if it doesn't exist."""
    mock_exists = mocker.patch('os.path.exists', return_value=False)
    mock_makedirs = mocker.patch('os.makedirs')
    # Mock the actual save functions to prevent file writing
    mocker.patch('synthetic_data_generator.output.saver.save_as_jsonl')

    output_dir = "new_output_dir"
    base_filename = os.path.join(output_dir, "my_data")
    saver.save_data(SAMPLE_DATA, base_filename, ["jsonl"])

    mock_exists.assert_called_once_with(output_dir)
    mock_makedirs.assert_called_once_with(output_dir, exist_ok=True)

def test_save_data_dir_creation_fails(mocker, caplog):
    """Test save_data handles directory creation failure."""
    mock_exists = mocker.patch('os.path.exists', return_value=False)
    mock_makedirs = mocker.patch('os.makedirs', side_effect=OSError("Permission denied"))
    mock_save_jsonl = mocker.patch('synthetic_data_generator.output.saver.save_as_jsonl')

    output_dir = "unwritable_dir"
    base_filename = os.path.join(output_dir, "my_data")
    saver.save_data(SAMPLE_DATA, base_filename, ["jsonl"])

    assert f"Failed to create output directory {output_dir}" in caplog.text
    mock_save_jsonl.assert_not_called() # Should not attempt saving

def test_save_data_calls_specific_savers(mocker):
    """Test save_data calls the correct specific save functions."""
    mock_save_jsonl = mocker.patch('synthetic_data_generator.output.saver.save_as_jsonl')
    mock_save_csv = mocker.patch('synthetic_data_generator.output.saver.save_as_csv')
    mock_save_parquet = mocker.patch('synthetic_data_generator.output.saver.save_as_parquet')
    mocker.patch('os.path.exists', return_value=True) # Assume dir exists

    base = "output/data"
    formats = ["jsonl", "csv", "parquet"]
    saver.save_data(SAMPLE_DATA, base, formats, CSV_FIELDNAMES)

    mock_save_jsonl.assert_called_once_with(SAMPLE_DATA, f"{base}.jsonl")
    mock_save_csv.assert_called_once_with(SAMPLE_DATA, f"{base}.csv", CSV_FIELDNAMES)
    mock_save_parquet.assert_called_once_with(SAMPLE_DATA, f"{base}.parquet")

def test_save_data_csv_header_inference(mocker, caplog):
    """Test save_data infers CSV headers if not provided."""
    mock_save_csv = mocker.patch('synthetic_data_generator.output.saver.save_as_csv')
    mocker.patch('os.path.exists', return_value=True)

    base = "output/data"
    saver.save_data(SAMPLE_DATA, base, ["csv"], csv_fieldnames=None) # Explicitly None

    assert "Inferring from first data item" in caplog.text
    inferred_fieldnames = list(SAMPLE_DATA[0].keys())
    mock_save_csv.assert_called_once_with(SAMPLE_DATA, f"{base}.csv", inferred_fieldnames)


def test_save_data_csv_header_inference_fails(mocker, caplog):
    """Test save_data skips CSV and logs error if headers cannot be inferred."""
    caplog.set_level(logging.ERROR) # Check for the error log
    # Patch save_as_csv in the context of the saver module
    mock_save_csv = mocker.patch('synthetic_data_generator.output.saver.save_as_csv')
    mocker.patch('os.path.exists', return_value=True)
    # Suppress the initial "No data" warning if testing the empty list case first
    # mocker.patch('synthetic_data_generator.output.saver.logger.warning')

    base = "output/data"

    # Test with data that leads to empty headers: [{}]
    saver.save_data([{}], base, ["csv"], csv_fieldnames=None)

    # Assert that save_as_csv was NOT called because 'csv' was removed from formats
    mock_save_csv.assert_not_called()
    # Assert that the specific error was logged
    assert "Cannot determine CSV fieldnames (no data and handler provided none). Skipping CSV save." in caplog.text



# --- save_as_jsonl tests ---

def test_save_as_jsonl_success(mocker):
    """Test successful saving to JSONL."""
    mock_file = mock_open()
    mocker.patch('builtins.open', mock_file)
    # Correct the mock for json.dumps to handle keyword arguments
    # Use the captured original_json_dumps to maintain behavior but allow mocking
    mock_dumps = mocker.patch('json.dumps', side_effect=lambda d, **kwargs: original_json_dumps(d, **kwargs))

    filename = "test.jsonl"
    saver.save_as_jsonl(SAMPLE_DATA, filename)

    mock_file.assert_called_once_with(filename, 'w', encoding='utf-8')
    handle = mock_file()

    # Check that json.dumps was called correctly for each item
    assert mock_dumps.call_count == len(SAMPLE_DATA)
    mock_dumps.assert_called_with(SAMPLE_DATA[-1], ensure_ascii=False) # Check last call args

    # Check if write was called for each item + newline
    assert handle.write.call_count == len(SAMPLE_DATA) # This assertion should now pass
    first_call_args = handle.write.call_args_list[0].args[0]
    assert first_call_args.startswith('{')
    assert first_call_args.endswith('}\n')
    # Use json.loads to check content reliably
    assert json.loads(first_call_args.strip()) == SAMPLE_DATA[0]

def test_save_as_jsonl_os_error(mocker, caplog):
    """Test handling OSError during file open/write."""
    mocker.patch('builtins.open', side_effect=OSError("Cannot write"))

    saver.save_as_jsonl(SAMPLE_DATA, "test.jsonl")

    assert "Failed to open/write JSONL file" in caplog.text

# --- load_from_jsonl tests ---

def test_load_from_jsonl_success(mocker):
    """Test successful loading from JSONL."""
    jsonl_content = '\n'.join(json.dumps(item) for item in SAMPLE_DATA) + '\n'
    mock_file = mock_open(read_data=jsonl_content)
    mocker.patch('builtins.open', mock_file)

    filename = "test.jsonl"
    loaded_data = saver.load_from_jsonl(filename)

    mock_file.assert_called_once_with(filename, 'r', encoding='utf-8')
    assert loaded_data == SAMPLE_DATA

def test_load_from_jsonl_file_not_found(mocker):
    """Test handling FileNotFoundError."""
    mocker.patch('builtins.open', side_effect=FileNotFoundError("No such file"))

    with pytest.raises(FileNotFoundError):
        saver.load_from_jsonl("nonexistent.jsonl")

def test_load_from_jsonl_decode_error(mocker, caplog):
    """Test handling JSONDecodeError on a specific line."""
    jsonl_content = '{"col_a": 1}\n{"invalid json\n{"col_a": 3}\n'
    mock_file = mock_open(read_data=jsonl_content)
    mocker.patch('builtins.open', mock_file)

    loaded_data = saver.load_from_jsonl("bad.jsonl")

    assert len(loaded_data) == 2 # Should skip the bad line
    assert loaded_data[0] == {"col_a": 1}
    assert loaded_data[1] == {"col_a": 3}
    assert "JSON Decode Error on line 2" in caplog.text

# --- save_as_csv tests ---

def test_save_as_csv_success(mocker):
    """Test successful saving to CSV."""
    mock_file = mock_open()
    mocker.patch('builtins.open', mock_file)
    # More robust mocking for csv.DictWriter
    mock_dict_writer_instance = MagicMock()
    mock_dict_writer_class = MagicMock(return_value=mock_dict_writer_instance)
    # Patch where csv is used inside the saver module
    mocker.patch('synthetic_data_generator.output.saver.csv.DictWriter', mock_dict_writer_class)
    # Also mock the constant if needed, though patching DictWriter is usually enough
    mocker.patch('synthetic_data_generator.output.saver.csv.QUOTE_MINIMAL', csv.QUOTE_MINIMAL)


    filename = "test.csv"
    saver.save_as_csv(SAMPLE_DATA, filename, CSV_FIELDNAMES)

    mock_file.assert_called_once_with(filename, 'w', newline='', encoding='utf-8')
    # Check that DictWriter was instantiated correctly
    mock_dict_writer_class.assert_called_once_with(
        ANY, # The file handle
        fieldnames=CSV_FIELDNAMES,
        quoting=csv.QUOTE_MINIMAL,
        extrasaction='ignore'
    )
    # Check methods called on the instance
    mock_dict_writer_instance.writeheader.assert_called_once()
    mock_dict_writer_instance.writerows.assert_called_once()
    # Check data passed to writerows (ensure it's filtered - already correct in original)
    passed_data = mock_dict_writer_instance.writerows.call_args[0][0]
    assert passed_data == SAMPLE_DATA # In this case, all keys are in fieldnames

def test_save_as_csv_filters_data(mocker):
    """Test that save_as_csv filters data to match fieldnames."""
    mock_file = mock_open()
    mocker.patch('builtins.open', mock_file)
    mock_dict_writer = MagicMock()
    mock_csv_module = MagicMock()
    mock_csv_module.DictWriter.return_value = mock_dict_writer
    mocker.patch('synthetic_data_generator.output.saver.csv', mock_csv_module)

    data_with_extra = [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "d": 6}]
    fieldnames = ["a", "b"]
    expected_filtered_data = [{"a": 1, "b": 2}, {"a": 4, "b": 5}]

    saver.save_as_csv(data_with_extra, "test.csv", fieldnames)

    mock_dict_writer.writerows.assert_called_once_with(expected_filtered_data)


def test_save_as_csv_no_data_or_fieldnames(caplog):
    """Test save_as_csv handles empty data or fieldnames."""
    saver.save_as_csv([], "test.csv", CSV_FIELDNAMES)
    assert "No data provided to save_as_csv" in caplog.text

    caplog.clear()
    saver.save_as_csv(SAMPLE_DATA, "test.csv", [])
    assert "No fieldnames provided for CSV saving" in caplog.text

# --- save_as_parquet tests ---

# Mock pandas DataFrame and to_parquet
@pytest.fixture
def mock_pandas(mocker):
    mock_df_instance = MagicMock()
    mock_df_class = MagicMock(return_value=mock_df_instance)
    mock_pd = MagicMock()
    mock_pd.DataFrame = mock_df_class
    mocker.patch('synthetic_data_generator.output.saver.pd', mock_pd)
    mocker.patch('synthetic_data_generator.output.saver.PANDAS_AVAILABLE', True)
    return mock_pd, mock_df_instance

def test_save_as_parquet_success(mock_pandas):
    """Test successful saving to Parquet."""
    mock_pd, mock_df_instance = mock_pandas
    filename = "test.parquet"

    saver.save_as_parquet(SAMPLE_DATA, filename)

    mock_pd.DataFrame.assert_called_once_with(SAMPLE_DATA)
    mock_df_instance.to_parquet.assert_called_once_with(filename, index=False, engine='pyarrow')

def test_save_as_parquet_pandas_unavailable(mocker, caplog, capsys):
    """Test save_as_parquet when pandas is not available."""
    mocker.patch('synthetic_data_generator.output.saver.PANDAS_AVAILABLE', False)
    mocker.patch('synthetic_data_generator.output.saver.pd', None)

    saver.save_as_parquet(SAMPLE_DATA, "test.parquet")

    assert "Pandas library not found. Cannot save as Parquet." in caplog.text
    captured = capsys.readouterr()
    assert "pip install pandas pyarrow" in captured.err

def test_save_as_parquet_pyarrow_import_error(mock_pandas, caplog, capsys):
    """Test save_as_parquet when pyarrow import fails at runtime."""
    mock_pd, mock_df_instance = mock_pandas
    # Simulate pyarrow import error during to_parquet call
    mock_df_instance.to_parquet.side_effect = ImportError("Cannot import pyarrow")

    saver.save_as_parquet(SAMPLE_DATA, "test.parquet")

    assert "pyarrow library not found or failed to import" in caplog.text
    captured = capsys.readouterr()
    assert "pip install pyarrow" in captured.err

def test_save_as_parquet_write_error(mock_pandas, caplog):
    """Test handling errors during DataFrame creation or writing."""
    mock_pd, mock_df_instance = mock_pandas
    mock_df_instance.to_parquet.side_effect = Exception("Disk full")

    saver.save_as_parquet(SAMPLE_DATA, "test.parquet")

    assert "Failed to create/write Parquet file" in caplog.text
    assert "Disk full" in caplog.text
