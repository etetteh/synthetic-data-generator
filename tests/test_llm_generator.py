# tests/test_llm_generator.py
import pytest
import json
import time
from unittest.mock import patch, MagicMock, call, mock_open
from synthetic_data_generator.llm import generator
from synthetic_data_generator.formats import predefined, custom
from synthetic_data_generator import exceptions
from synthetic_data_generator import config
import logging

# --- Fixtures ---

@pytest.fixture
def predefined_handler():
    return predefined.PredefinedFormatHandler("qa")

@pytest.fixture
def custom_handler(sample_custom_format_def):
    return custom.CustomFormatHandler(sample_custom_format_def)

@pytest.fixture
def gen_instance(predefined_handler, mock_llm_client):
    """Basic generator instance with predefined handler."""
    return generator.SyntheticDataGenerator(
        format_handler=predefined_handler,
        llm_client=mock_llm_client,
        max_retries=2,
        retry_delay=0.1
    )

# --- Initialization Tests ---

def test_generator_init_success(predefined_handler, mock_llm_client):
    """Test successful generator initialization."""
    gen = generator.SyntheticDataGenerator(
        format_handler=predefined_handler,
        llm_client=mock_llm_client,
        max_retries=3,
        retry_delay=0.5
    )
    assert gen.format_handler == predefined_handler
    assert gen.model == mock_llm_client
    assert gen.max_retries == 3
    assert gen.retry_delay == 0.5
    assert isinstance(gen.system_prompt, str)
    assert "qa" in gen.system_prompt # Check format name is in prompt

def test_generator_init_invalid_handler(mock_llm_client):
    """Test init fails with invalid format handler."""
    with pytest.raises(exceptions.ConfigurationError, match="valid DataFormatHandler"):
        generator.SyntheticDataGenerator("not_a_handler", mock_llm_client, 1, 0.1)

def test_generator_init_invalid_llm(predefined_handler):
    """Test init fails with invalid LLM client."""
    with pytest.raises(exceptions.ConfigurationError, match="valid LLM client"):
        generator.SyntheticDataGenerator(predefined_handler, "not_an_llm", 1, 0.1)

# --- _invoke_llm_with_retry Tests ---

def test_invoke_llm_success(gen_instance, mock_llm_client):
    """Test successful LLM invocation on first try."""
    mock_response = MagicMock()
    mock_response.content = '[{"data": 1}]'
    mock_response.response_metadata = {'finish_reason': 'STOP'}
    mock_llm_client.invoke.return_value = mock_response

    messages = [MagicMock(spec=config.SystemMessage), MagicMock(spec=config.HumanMessage)]
    response = gen_instance._invoke_llm_with_retry(messages, "test_purpose")

    mock_llm_client.invoke.assert_called_once_with(messages)
    assert response == mock_response

def test_invoke_llm_retry_once(gen_instance, mock_llm_client, mocker):
    """Test successful LLM invocation after one retry."""
    mock_sleep = mocker.patch('time.sleep')
    error_response = Exception("Temporary API error")
    success_response = MagicMock()
    success_response.content = '[{"data": 1}]'
    success_response.response_metadata = {'finish_reason': 'STOP'}

    mock_llm_client.invoke.side_effect = [error_response, success_response]

    messages = [MagicMock(), MagicMock()]
    response = gen_instance._invoke_llm_with_retry(messages, "retry_test")

    assert mock_llm_client.invoke.call_count == 2
    mock_sleep.assert_called_once_with(gen_instance.retry_delay) # Check that sleep was called with correct delay
    assert response == success_response

def test_invoke_llm_permanent_failure(gen_instance, mock_llm_client, mocker):
    """Test permanent failure after max retries."""
    mock_sleep = mocker.patch('time.sleep')
    error_response = Exception("Permanent API error")
    # max_retries is 2, so 3 calls total
    mock_llm_client.invoke.side_effect = [error_response, error_response, error_response]

    messages = [MagicMock(), MagicMock()]
    with pytest.raises(exceptions.GenerationError, match="LLM call failed after multiple retries"):
        gen_instance._invoke_llm_with_retry(messages, "fail_test")

    assert mock_llm_client.invoke.call_count == 3
    assert mock_sleep.call_count == 2 # Slept twice before final failure

def test_invoke_llm_empty_content_retry(gen_instance, mock_llm_client, mocker):
    """Test retry when LLM returns empty content."""
    mock_sleep = mocker.patch('time.sleep')
    empty_response = MagicMock()
    empty_response.content = "" # Empty string
    empty_response.response_metadata = {'finish_reason': 'STOP'}
    success_response = MagicMock()
    success_response.content = '[{"data": 1}]'
    success_response.response_metadata = {'finish_reason': 'STOP'}

    mock_llm_client.invoke.side_effect = [empty_response, success_response]

    messages = [MagicMock(), MagicMock()]
    response = gen_instance._invoke_llm_with_retry(messages, "empty_content_test")

    assert mock_llm_client.invoke.call_count == 2
    mock_sleep.assert_called_once()
    assert response == success_response

def test_invoke_llm_safety_block_no_retry(gen_instance, mock_llm_client, mocker):
    """Test no retry when LLM response is blocked by safety."""
    mock_sleep = mocker.patch('time.sleep')
    blocked_response = MagicMock()
    blocked_response.content = ""
    blocked_response.response_metadata = {'finish_reason': 'SAFETY', 'block_reason': 'DANGEROUS'}

    mock_llm_client.invoke.return_value = blocked_response

    messages = [MagicMock(), MagicMock()]
    with pytest.raises(exceptions.GenerationError, match="LLM call blocked by safety settings"):
        gen_instance._invoke_llm_with_retry(messages, "safety_block_test")

    mock_llm_client.invoke.assert_called_once()
    mock_sleep.assert_not_called() # Should not sleep/retry

# --- _parse_and_validate_llm_response Tests ---

def test_parse_valid_json_list(gen_instance, predefined_handler, mocker):
    """Test parsing a valid JSON list response."""
    response_text = '[{"context": "c1", "question": "q1", "answer": "a1"}, {"context": "c2", "question": "q2", "answer": "a2"}]'
    # Mock the handler's validate_item to just return the item
    mocker.patch.object(predefined_handler, 'validate_item', side_effect=lambda item, idx: item)

    parsed_data = gen_instance._parse_and_validate_llm_response(response_text)

    assert len(parsed_data) == 2
    assert parsed_data[0] == {"context": "c1", "question": "q1", "answer": "a1"}
    assert predefined_handler.validate_item.call_count == 2

def test_parse_json_with_markdown(gen_instance, predefined_handler, mocker):
    """Test parsing JSON enclosed in markdown code blocks."""
    response_text = '```json\n[{"context": "c", "question": "q", "answer": "a"}]\n```'
    mocker.patch.object(predefined_handler, 'validate_item', side_effect=lambda item, idx: item)
    parsed_data = gen_instance._parse_and_validate_llm_response(response_text)
    assert len(parsed_data) == 1
    assert parsed_data[0] == {"context": "c", "question": "q", "answer": "a"}

    response_text_generic = '```\n[{"context": "c", "question": "q", "answer": "a"}]\n```'
    parsed_data_generic = gen_instance._parse_and_validate_llm_response(response_text_generic)
    assert len(parsed_data_generic) == 1

def test_parse_single_json_object(gen_instance, predefined_handler, mocker):
    """Test parsing when LLM returns a single object instead of a list."""
    response_text = '{"context": "c", "question": "q", "answer": "a"}'
    mocker.patch.object(predefined_handler, 'validate_item', side_effect=lambda item, idx: item)
    parsed_data = gen_instance._parse_and_validate_llm_response(response_text)
    assert len(parsed_data) == 1
    assert parsed_data[0] == {"context": "c", "question": "q", "answer": "a"}

def test_parse_invalid_json(gen_instance):
    """Test parsing failure with invalid JSON."""
    response_text = '[{"context": "c", "question": "q", "answer": "a"}, {"invalid]'
    with pytest.raises(exceptions.OutputParserError, match="JSON parsing failed"):
        gen_instance._parse_and_validate_llm_response(response_text)

def test_parse_not_a_list_after_wrap(gen_instance):
    """Test failure if result isn't a list after potential wrapping."""
    response_text = '"just a string"' # Valid JSON, but not a list/dict
    with pytest.raises(exceptions.ValidationError, match="Expected JSON list"):
        gen_instance._parse_and_validate_llm_response(response_text)

def test_parse_validation_skips_items(gen_instance, predefined_handler, mocker, caplog):
    """Test that validation errors skip individual items."""
    caplog.set_level(logging.WARNING)
    response_text = '[{"context": "c1", "question": "q1", "answer": "a1"}, {"context": "c2"}, {"context": "c3", "question": "q3", "answer": "a3"}]'
    # Mock validate_item to fail for the second item
    def mock_validate(item, idx):
        if "question" not in item:
            raise exceptions.ValidationError("Missing question")
        return item
    mocker.patch.object(predefined_handler, 'validate_item', side_effect=mock_validate)

    parsed_data = gen_instance._parse_and_validate_llm_response(response_text, item_index_offset=10)

    assert len(parsed_data) == 2
    assert parsed_data[0]["context"] == "c1"
    assert parsed_data[1]["context"] == "c3"
    assert "Validation Error for item #11: Missing question. Skipping item." in caplog.text
    assert "Validation completed. 1 item(s) failed validation" in caplog.text

# --- _generate_one_batch_with_retry Tests ---
# These require mocking the invoke and parse/validate methods

@patch.object(generator.SyntheticDataGenerator, '_invoke_llm_with_retry')
@patch.object(generator.SyntheticDataGenerator, '_parse_and_validate_llm_response')
def test_generate_batch_success(mock_parse, mock_invoke, gen_instance):
    """Test successful batch generation."""
    mock_invoke.return_value = MagicMock(content='some response')
    mock_parse.return_value = [{"data": 1}, {"data": 2}]

    results = gen_instance._generate_one_batch_with_retry("human prompt")

    mock_invoke.assert_called_once()
    mock_parse.assert_called_once_with('some response', item_index_offset=0) # Check offset
    assert results == [{"data": 1}, {"data": 2}]

@patch.object(generator.SyntheticDataGenerator, '_invoke_llm_with_retry')
@patch.object(generator.SyntheticDataGenerator, '_parse_and_validate_llm_response')
@patch('time.sleep') # Mock sleep
def test_generate_batch_retry_parse_fail(mock_sleep, mock_parse, mock_invoke, gen_instance):
    """Test retry logic when parsing/validation fails once."""
    mock_invoke.return_value = MagicMock(content='response text')
    # Fail parse first time, succeed second time
    mock_parse.side_effect = [exceptions.OutputParserError("Bad JSON"), [{"data": 1}]]

    results = gen_instance._generate_one_batch_with_retry("human prompt", item_index_offset=5)

    assert mock_invoke.call_count == 2 # LLM called again on retry
    assert mock_parse.call_count == 2
    # Check offset passed correctly on both calls
    mock_parse.assert_has_calls([call('response text', item_index_offset=5), call('response text', item_index_offset=5)])
    mock_sleep.assert_called_once_with(gen_instance.retry_delay)
    assert results == [{"data": 1}]

@patch.object(generator.SyntheticDataGenerator, '_invoke_llm_with_retry')
@patch.object(generator.SyntheticDataGenerator, '_parse_and_validate_llm_response')
@patch('time.sleep') # Mock sleep
def test_generate_batch_permanent_parse_fail(mock_sleep, mock_parse, mock_invoke, gen_instance, caplog):
    """Test permanent failure after parse/validation retries."""
    caplog.set_level(logging.WARNING)
    mock_invoke.return_value = MagicMock(content='response text')
    # Fail parse/validate repeatedly (max_retries=2 -> 3 attempts)
    mock_parse.side_effect = [
        exceptions.ValidationError("Bad structure"),
        exceptions.OutputParserError("Bad JSON again"),
        exceptions.ValidationError("Still bad")
    ]

    results = gen_instance._generate_one_batch_with_retry("human prompt")

    assert mock_invoke.call_count == 3 # LLM called 3 times
    assert mock_parse.call_count == 3
    assert mock_sleep.call_count == 2
    assert results == [] # Should return empty list on permanent failure
    assert "Batch attempt 1 failed parsing/validation: Bad structure. Retrying..." in caplog.text
    assert "Batch attempt 2 failed parsing/validation: Bad JSON again. Retrying..." in caplog.text
    assert "Batch attempt 3 failed parsing/validation: Still bad. Max retries reached." in caplog.text
    assert "Could not generate a valid batch after all retries" in caplog.text

@patch.object(generator.SyntheticDataGenerator, '_invoke_llm_with_retry')
@patch.object(generator.SyntheticDataGenerator, '_parse_and_validate_llm_response')
def test_generate_batch_llm_permanent_fail(mock_parse, mock_invoke, gen_instance, caplog):
    """Test batch failure when LLM invocation fails permanently."""
    caplog.set_level(logging.ERROR)
    # Simulate _invoke_llm_with_retry raising GenerationError after its internal retries
    mock_invoke.side_effect = exceptions.GenerationError("LLM failed permanently")

    results = gen_instance._generate_one_batch_with_retry("human prompt")

    mock_invoke.assert_called_once() # Only called once from batch perspective
    mock_parse.assert_not_called()
    assert results == []
    assert "Batch attempt 1 failed LLM invocation permanently: LLM failed permanently" in caplog.text
    assert "Could not generate a valid batch after all retries" in caplog.text


# --- _run_generation_loop Tests (More complex mocking) ---

@patch.object(generator.SyntheticDataGenerator, '_generate_one_batch_with_retry')
@patch('synthetic_data_generator.utils.hashing.hash_item') # Mock hashing
@patch('synthetic_data_generator.llm.generator.tqdm') # Mock tqdm
def test_run_loop_basic(mock_tqdm, mock_hash, mock_generate_batch, gen_instance, predefined_handler):
    """Test basic generation loop reaching target samples."""
    mock_tqdm_instance = MagicMock()
    mock_tqdm.return_value = mock_tqdm_instance
    # Mock batch generation to return unique items
    mock_generate_batch.side_effect = [
        [{"context": "c1", "question": "q1", "answer": "a1"}], # Batch 1
        [{"context": "c2", "question": "q2", "answer": "a2"}], # Batch 2
    ]
    # Mock hashing to return unique hashes
    mock_hash.side_effect = ["hash1", "hash2"]

    results_mem, final_count = gen_instance._run_generation_loop(
        prompt_builder=predefined_handler.build_query_prompt,
        prompt_args={"query": "test"},
        num_samples=2,
        batch_size=1,
        incremental_save=False
    )

    mock_tqdm.assert_called_once_with(total=2, desc="Generating Data", unit=" samples")
    assert mock_generate_batch.call_count == 2
    # Check item_index_offset passed to batch generator
    mock_generate_batch.assert_has_calls([
        call(ANY, item_index_offset=0),
        call(ANY, item_index_offset=1)
    ])
    assert final_count == 2
    assert len(results_mem) == 2
    assert results_mem[0]["context"] == "c1"
    assert results_mem[1]["context"] == "c2"
    assert len(gen_instance.unique_entries) == 2
    assert mock_tqdm_instance.update.call_count == 2
    mock_tqdm_instance.close.assert_called_once()

@patch.object(generator.SyntheticDataGenerator, '_generate_one_batch_with_retry')
@patch('synthetic_data_generator.utils.hashing.hash_item')
@patch('synthetic_data_generator.llm.generator.tqdm')
def test_run_loop_duplicates(mock_tqdm, mock_hash, mock_generate_batch, gen_instance, predefined_handler):
    """Test loop handling duplicate items."""
    mock_tqdm_instance = MagicMock()
    mock_tqdm.return_value = mock_tqdm_instance
    mock_generate_batch.side_effect = [
        [{"id": 1}], # Batch 1 (1 unique)
        [{"id": 1}, {"id": 2}], # Batch 2 (1 unique, 1 duplicate)
        [{"id": 3}], # Batch 3 (1 unique)
    ]
    mock_hash.side_effect = ["hash1", "hash1", "hash2", "hash3"] # Simulate duplicate hash

    results_mem, final_count = gen_instance._run_generation_loop(
        prompt_builder=predefined_handler.build_query_prompt,
        prompt_args={"query": "test"},
        num_samples=3,
        batch_size=2, # Request 2 per batch
        incremental_save=False
    )

    assert mock_generate_batch.call_count == 3
    # Check batch sizes requested
    mock_generate_batch.assert_has_calls([
        call(predefined_handler.build_query_prompt(query='test', num_samples=2), item_index_offset=0), # Request 2 initially
        call(predefined_handler.build_query_prompt(query='test', num_samples=2), item_index_offset=1), # Request 2 again (1 needed)
        call(predefined_handler.build_query_prompt(query='test', num_samples=1), item_index_offset=2)  # Request 1 (1 needed)
    ])
    assert final_count == 3
    assert len(results_mem) == 3
    assert results_mem == [{"id": 1}, {"id": 2}, {"id": 3}]
    assert len(gen_instance.unique_entries) == 3
    assert mock_tqdm_instance.update.call_count == 3 # Updated 3 times for unique items

@patch.object(generator.SyntheticDataGenerator, '_generate_one_batch_with_retry')
@patch('builtins.open', new_callable=mock_open) # Mock file opening
@patch('json.dumps', side_effect=lambda d, ensure_ascii: json.dumps(d)) # Mock json dumps
@patch('synthetic_data_generator.utils.hashing.hash_item')
@patch('synthetic_data_generator.llm.generator.tqdm')
def test_run_loop_incremental_save(mock_tqdm, mock_hash, mock_dumps, mock_file_open, mock_generate_batch, gen_instance, predefined_handler):
    """Test loop with incremental saving enabled."""
    mock_tqdm_instance = MagicMock()
    mock_tqdm.return_value = mock_tqdm_instance
    mock_generate_batch.side_effect = [
        [{"id": 1, "text": "a"}],
        [{"id": 2, "text": "b"}],
    ]
    mock_hash.side_effect = ["hash1", "hash2"]

    results_mem, final_count = gen_instance._run_generation_loop(
        prompt_builder=predefined_handler.build_query_prompt,
        prompt_args={"query": "test"},
        num_samples=2,
        batch_size=1,
        output_file_prefix="output/inc_test",
        incremental_save=True
    )

    assert final_count == 2
    assert len(results_mem) == 0 # No data kept in memory
    # Check file interactions
    mock_file_open.assert_called_once_with('output/inc_test_qa_from_query.jsonl', 'w', encoding='utf-8')
    handle = mock_file_open()
    assert handle.write.call_count == 2
    handle.write.assert_any_call('{"id": 1, "text": "a"}\n')
    handle.write.assert_any_call('{"id": 2, "text": "b"}\n')
    handle.flush.call_count == 2
    handle.close.assert_called_once()
    assert mock_tqdm_instance.update.call_count == 2

@patch.object(generator.SyntheticDataGenerator, '_generate_one_batch_with_retry')
@patch('synthetic_data_generator.llm.generator.tqdm')
def test_run_loop_consecutive_failures(mock_tqdm, mock_generate_batch, gen_instance, predefined_handler, caplog):
    """Test loop stops after too many consecutive batch failures."""
    caplog.set_level(logging.WARNING)
    mock_tqdm_instance = MagicMock()
    mock_tqdm.return_value = mock_tqdm_instance
    # max_retries=2 -> max_consecutive_failures = 3
    mock_generate_batch.side_effect = [
        [], # Fail 1
        [], # Fail 2
        [], # Fail 3
        [{"id": 1}] # Should not be reached
    ]

    with pytest.raises(exceptions.GenerationError, match="Stopping generation due to 3 consecutive batch failures"):
        gen_instance._run_generation_loop(
            prompt_builder=predefined_handler.build_query_prompt,
            prompt_args={"query": "test"},
            num_samples=5,
            batch_size=1,
            incremental_save=False
        )

    assert mock_generate_batch.call_count == 3
    assert "Batch generation failed or yielded no valid items. Consecutive failures: 1" in caplog.text
    assert "Batch generation failed or yielded no valid items. Consecutive failures: 2" in caplog.text
    assert "Batch generation failed or yielded no valid items. Consecutive failures: 3" in caplog.text
    assert "Stopping generation due to 3 consecutive batch failures." in caplog.text
    mock_tqdm_instance.close.assert_called_once() # Ensure tqdm is closed even on failure

@patch.object(generator.SyntheticDataGenerator, '_generate_one_batch_with_retry')
@patch('synthetic_data_generator.llm.generator.tqdm')
def test_run_loop_batch_size_adjustment(mock_tqdm, mock_generate_batch, gen_instance, predefined_handler):
    """Test that the requested batch size adjusts based on remaining samples."""
    mock_tqdm_instance = MagicMock()
    mock_tqdm.return_value = mock_tqdm_instance
    mock_generate_batch.side_effect = [
        [{"id": i} for i in range(5)], # Batch 1 (5 unique)
        [{"id": i} for i in range(5, 8)], # Batch 2 (3 unique)
    ]
    mock_hash.side_effect = [f"hash{i}" for i in range(8)]

    gen_instance._run_generation_loop(
        prompt_builder=predefined_handler.build_query_prompt,
        prompt_args={"query": "test"},
        num_samples=8,
        batch_size=5, # Initial batch size
        incremental_save=False
    )

    assert mock_generate_batch.call_count == 2
    # Check requested num_samples in prompt builder calls
    mock_generate_batch.assert_has_calls([
        call(predefined_handler.build_query_prompt(query='test', num_samples=5), item_index_offset=0), # Request 5 initially
        call(predefined_handler.build_query_prompt(query='test', num_samples=3), item_index_offset=5)  # Request 3 (8 total - 5 done)
    ])
    assert mock_tqdm_instance.update.call_count == 8


# --- generate_from_query / generate_from_documents Tests ---
# These mainly test the delegation to _run_generation_loop and query refinement/doc loading

@patch.object(generator.SyntheticDataGenerator, '_run_generation_loop')
@patch.object(generator.SyntheticDataGenerator, 'refine_query')
def test_generate_from_query_with_refine(mock_refine, mock_run_loop, gen_instance):
    """Test generate_from_query calls refine and run_loop."""
    mock_refine.return_value = "refined query"
    mock_run_loop.return_value = ([{"id": 1}], 1) # Mock loop result

    results, count = gen_instance.generate_from_query(
        query="original query", num_samples=1, batch_size=1, refine=True
    )

    mock_refine.assert_called_once_with("original query")
    mock_run_loop.assert_called_once()
    # Check that refined query was passed to run_loop's prompt_args
    call_args = mock_run_loop.call_args[1] # Get kwargs
    assert call_args['prompt_builder'] == gen_instance.format_handler.build_query_prompt
    assert call_args['prompt_args'] == {"query": "refined query"}
    assert call_args['num_samples'] == 1
    assert call_args['batch_size'] == 1
    assert call_args['incremental_save'] is False
    assert call_args['output_file_prefix'] is None
    assert results == [{"id": 1}]
    assert count == 1

@patch.object(generator.SyntheticDataGenerator, '_run_generation_loop')
@patch.object(generator.SyntheticDataGenerator, 'refine_query')
def test_generate_from_query_no_refine(mock_refine, mock_run_loop, gen_instance):
    """Test generate_from_query skips refine."""
    mock_run_loop.return_value = ([{"id": 1}], 1)

    gen_instance.generate_from_query(
        query="original query", num_samples=1, batch_size=1, refine=False # refine=False
    )

    mock_refine.assert_not_called()
    mock_run_loop.assert_called_once()
    call_args = mock_run_loop.call_args[1]
    assert call_args['prompt_args'] == {"query": "original query"}

@patch.object(generator.SyntheticDataGenerator, '_run_generation_loop')
@patch('synthetic_data_generator.loading.document_loader.load_document_texts')
def test_generate_from_documents(mock_load_docs, mock_run_loop, gen_instance):
    """Test generate_from_documents calls doc loader and run_loop."""
    mock_load_docs.return_value = ["doc content 1", "doc content 2"]
    mock_run_loop.return_value = ([{"id": 1}], 1)

    results, count = gen_instance.generate_from_documents(
        document_path="path/to/docs", num_samples=1, batch_size=1, output_file_prefix="out", incremental_save=True
    )

    mock_load_docs.assert_called_once_with("path/to/docs")
    mock_run_loop.assert_called_once()
    call_args = mock_run_loop.call_args[1]
    assert call_args['prompt_builder'] == gen_instance.format_handler.build_document_prompt
    assert call_args['prompt_args'] == {"documents": ["doc content 1", "doc content 2"]}
    assert call_args['num_samples'] == 1
    assert call_args['batch_size'] == 1
    assert call_args['incremental_save'] is True
    assert call_args['output_file_prefix'] == "out"
    assert results == [{"id": 1}]
    assert count == 1

@patch('synthetic_data_generator.loading.document_loader.load_document_texts')
def test_generate_from_documents_loader_fails(mock_load_docs, gen_instance):
    """Test generate_from_documents handles loader errors."""
    mock_load_docs.side_effect = exceptions.LoaderError("Failed to load")

    with pytest.raises(exceptions.LoaderError, match="Failed to load"):
        gen_instance.generate_from_documents(
            document_path="bad/path", num_samples=1, batch_size=1
        )

# --- refine_query Tests ---

@patch.object(generator.SyntheticDataGenerator, '_invoke_llm_with_retry')
def test_refine_query_success(mock_invoke, gen_instance):
    """Test successful query refinement."""
    mock_response = MagicMock()
    mock_response.content = "This is the refined query."
    mock_response.response_metadata = {'finish_reason': 'STOP'}
    mock_invoke.return_value = mock_response

    refined = gen_instance.refine_query("original query")

    mock_invoke.assert_called_once()
    messages = mock_invoke.call_args[0][0]
    assert len(messages) == 1
    assert isinstance(messages[0], config.HumanMessage)
    assert "Refine the following user query" in messages[0].content
    assert "original query" in messages[0].content
    assert refined == "This is the refined query."

@patch.object(generator.SyntheticDataGenerator, '_invoke_llm_with_retry')
def test_refine_query_llm_fails(mock_invoke, gen_instance):
    """Test query refinement failure when LLM call fails."""
    mock_invoke.side_effect = exceptions.GenerationError("LLM refine failed")

    with pytest.raises(exceptions.GenerationError, match="LLM refine failed"):
        gen_instance.refine_query("original query")

@patch.object(generator.SyntheticDataGenerator, '_invoke_llm_with_retry')
def test_refine_query_empty_response(mock_invoke, gen_instance, caplog):
    """Test query refinement returns original query if LLM response is empty."""
    caplog.set_level(logging.WARNING)
    mock_response = MagicMock()
    mock_response.content = "" # Empty response
    mock_response.response_metadata = {'finish_reason': 'STOP'}
    mock_invoke.return_value = mock_response

    refined = gen_instance.refine_query("original query")

    assert "LLM returned empty response for query refinement" in caplog.text
    assert refined == "original query" # Should fallback to original
