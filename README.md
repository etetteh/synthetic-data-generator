# Synthetic Data Generator

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- Add CI/CD badge here if you set one up, e.g., GitHub Actions -->
<!-- [![CI Status](https://github.com/etetteh/synthetic_data_generator/actions/workflows/ci.yml/badge.svg)](https://github.com/etetteh/synthetic_data_generator/actions/workflows/ci.yml) -->


## Table of Contents

1.  [Introduction](#introduction)
2.  [Features](#features)
3.  [Architecture Overview](#architecture-overview)
4.  [Setup and Installation](#setup-and-installation)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Configuration (API Key)](#configuration-api-key)
5.  [Usage](#usage)
    *   [Basic Command Structure](#basic-command-structure)
        *   [Input Source](#input-source)
        *   [Data Format](#data-format)
        *   [Generation Parameters](#generation-parameters)
        *   [Model Parameters](#model-parameters)
        *   [Output Parameters](#output-parameters)
        *   [Advanced Parameters](#advanced-parameters)
        *   [Examples](#examples)
6.  [Custom Format Specification](#custom-format-specification)
7.  [Document Loading](#document-loading)
8.  [Testing](#8-testing)
9.  [Contributing](#contributing)
10. [Versioning](#versioning)
11. [License](#license)
12. [Contact](#contact)
13. [Acknowledgements](#acknowledgements)

## 1. Introduction

This project provides an enterprise-grade Python application for generating synthetic training data for various Natural Language Processing (NLP) tasks. It leverages Google's Gemini models via the LangChain library to create diverse, high-quality datasets based on natural language queries or provided document contexts.

The application is designed with maintainability, testability, and scalability in mind, following principles of modular architecture, clear separation of concerns, and robust error handling.

## 2. Features

*   **LLM Integration:** Utilizes Google Gemini models via LangChain for data generation.
*   **Flexible Input:** Generate data from natural language queries or document contexts.
*   **Multiple Formats:** Supports predefined formats (QA, pairs, triplets) and custom formats defined via JSON schema.
*   **Customizable Generation:** Control model parameters like `temperature`, `top_p`, and `top_k`.
*   **Query Refinement:** Optional LLM-based refinement of input queries for potentially better results.
*   **Batch Processing:** Efficiently generates data in batches.
*   **Duplicate Detection:** Ensures uniqueness of generated samples using hashing.
*   **Robust Error Handling:** Custom exception hierarchy and retry logic for transient API/parsing errors.
*   **Input Validation:** Checks command-line arguments and custom format definitions.
*   **Comprehensive Logging:** Detailed logging with configurable levels.
*   **Multiple Output Formats:** Save generated data as JSON Lines (`jsonl`), CSV (`csv`), and Parquet (`parquet`).
*   **Incremental Saving:** Option to save data incrementally to JSONL for large datasets, reducing memory usage.
*   **Modular Design:** Well-structured codebase promoting maintainability and testability.

## 3. Architecture Overview

The project follows a layered and modular architecture to separate concerns and improve maintainability. The following is the project structure:

```
synthetic-data-generator/
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Fixtures and common setup
│   ├── test_config.py
│   ├── test_formats_utils.py
│   ├── test_formats_predefined.py
│   ├── test_formats_custom.py
│   ├── test_llm_generator.py
│   ├── test_loading_document_loader.py
│   ├── test_output_saver.py
│   ├── test_utils_hashing.py
│   └── test_main_pipeline.py # Tests for run_generation_pipeline
├── synthetic_data_generator/
│   ├── __init__.py
│   ├── config.py
│   ├── exceptions.py
│   ├── main.py
│   ├── formats/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── custom.py
│   │   ├── predefined.py
│   │   └── utils.py
│   ├── llm/
│   │   ├── __init__.py
│   │   └── generator.py
│   ├── loading/
│   │   ├── __init__.py
│   │   └── document_loader.py
│   ├── output/
│   │   ├── __init__.py
│   │   └── saver.py
│   └── utils/
│       ├── __init__.py
│       └── hashing.py
├── .env 
├── .gitignore
├── README.md                
└── requirements.txt                  
```

*   **Presentation Layer (`main.py`):** Handles command-line interface, parses arguments, sets up logging, initializes core components (injecting dependencies), and orchestrates the generation and saving process.
*   **Business Logic Layer (`llm/generator.py`):** Contains the main `SyntheticDataGenerator` class. It manages the generation loop, interacts with the LLM client, handles batching, duplicate checking, and delegates format-specific tasks to the `DataFormatHandler`.
*   **Infrastructure/Data Access Layers (`formats/`, `llm/client.py`, `loading/`, `output/`):** These modules encapsulate external interactions and data handling logic.
    *   `formats/`: Defines the interface (`DataFormatHandler`) and concrete implementations for different data structures, including prompt building and validation logic.
    *   `llm/`: Manages the interaction with the LLM API. `generator.py` uses an injected LLM client instance.
    *   `loading/`: Handles loading data from external document sources.
    *   `output/`: Manages saving the generated data to various file formats.
*   **Shared Components (`config.py`, `exceptions.py`, `utils/`):** Provide cross-cutting concerns like configuration, custom errors, and general utility functions.

Dependency Injection is used to provide the `SyntheticDataGenerator` with instances of the `DataFormatHandler` and the LLM client, making the generator class more testable and loosely coupled. The Strategy pattern is applied in the `formats` module to handle different data structures polymorphically.

## 4. Setup and Installation

### Prerequisites

*   Python 3.12.9 (or compatible 3.8+)
*   Access to Google Gemini API and a corresponding API key.

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/etetteh/synthetic_data_generator.git
    cd synthetic_data_generator
    ```
2.  (Recommended) Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
    ```
3.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt 
    ```
4.  Install optional dependencies as needed:
    * For Parquet output: `pip install pandas pyarrow`
    * For `.env` file support: `pip install python-dotenv`
    * For document loading: Install your `ADLoader` dependency (e.g., `pip install ADLoader`). The current code assumes `ADLoader` is available in your Python environment.

### Configuration (API Key)

You must set your Google API key. The application looks for the `GOOGLE_API_KEY` environment variable.

You can set it directly in your terminal session:
```bash
export GOOGLE_API_KEY='YOUR_API_KEY'
```
Replace `'YOUR_API_KEY'` with your actual key.

Alternatively, if you installed `python-dotenv`, you can create a file named `.env` in the project's root directory with the following content:
```dotenv
GOOGLE_API_KEY='YOUR_API_KEY'
``` 
The script will automatically load this file.

**Important:** Do not commit your `.env` file to version control. Add `.env` to your `.gitignore`.

## 5. Usage

The script is run as a module from the project root directory:

```bash 
python -m synthetic_data_generator.main [OPTIONS]
```

Use `python -m synthetic_data_generator.main --help` to see a full list of arguments and their defaults.

### Basic Command Structure

The command requires specifying an input source (`--query` or `--documents`) and a data format (`--format` or `--custom_format_file`).

```bash
python -m synthetic_data_generator.main <INPUT_SOURCE_OPTION> <FORMAT_OPTION> [OTHER_OPTIONS]
```

### Input Source

Choose exactly one: 

* `--query <TEXT>`: A natural language query to guide data generation.
* `--documents <PATH>`: Path to a document file or directory to use as context. Requires the `ADLoader` dependency.

### Data Format
 
Choose exactly one:

*   `--format {pair-class, pair-score, pair, triplet, qa}`: Use a predefined logical format.
    *   `pair-class`: Textual entailment pairs (`premise`, `hypothesis`, `label` 0/1/2).
    *   `pair-score`: Semantic similarity pairs (`sentence1`, `sentence2`, `score` 0.0-1.0).
    *   `pair`: Anchor-positive pairs (`anchor`, `positive`).
    *   `triplet`: Anchor-positive-negative triplets (`anchor`, `positive`, `negative`).
    *   `qa`: Context-question-answer triplets (`context`, `question`, `answer`).
*   `--custom_format_file <PATH>`: Path to a JSON file defining a custom data format structure. See [Custom Format Specification](#custom-format-specification) for details.

### Generation Parameters
 
* `--samples <N>`: Target number of *unique* samples to generate. Defaults to 50.
* `--no-refine`: Disable automatic query refinement using the LLM (only applicable with `--query`). By default, refinement is enabled.

### Model Parameters
 
*   `--model <NAME>`: Google Gemini model name (e.g., `gemini-2.0-flash`, `gemini-1.5-pro-latest`). Defaults to `gemini-2.0-flash`.
*   `--temperature <T>`: Generation temperature (0.0-2.0). Defaults to 0.4.
*   `--top_p <P>`: Nucleus sampling threshold (>0.0-1.0). Defaults to 0.95.
*   `--top_k <K>`: Sample from top K tokens. Defaults to None (disabled).

### Output Parameters

*   `--output_prefix <PREFIX>`: Prefix for output filenames. Defaults to `synthetic_data`. Output files will be named like `<PREFIX>_<format>_<source>.<ext>`.
*   `--output_format {jsonl, csv, parquet, all} [FMT ...]`: Output file format(s). Can specify multiple formats. Use `all` for jsonl, csv, and parquet (if available). Defaults to `all`.
*   `--incremental_save`: Save unique samples incrementally to JSONL as they are generated. This is recommended for large datasets to reduce memory usage. Requires `--output_prefix`. If enabled, JSONL is saved during generation, and other formats (CSV, Parquet) are saved at the end by loading the generated JSONL file.

### Advanced Parameters

*   `--batch_size <B>`: Number of samples aimed for per API call. The actual number requested may vary slightly to optimize the generation loop. Defaults to 25.
*   `--max_retries <R>`: Maximum retry attempts per batch for API, parsing, or validation failures. Defaults to 5.
*   `--log_level {DEBUG, INFO, WARNING, ERROR, CRITICAL}`: Set console logging level. Defaults to `INFO`.

### Examples
 
**Generate 50 QA pairs about Python programming (default settings):**

```bash
python -m synthetic_data_generator.main \
    --query "Generate questions and answers about Python programming." \
    --format qa
```
*(Output files: `synthetic_data_qa_from_query.jsonl`, `synthetic_data_qa_from_query.csv`, `synthetic_data_qa_from_query.parquet`)* 

**Generate 20 textual entailment pairs (`pair-class`) from a query:**

```bash
python -m synthetic_data_generator.main \
    --query "Generate pairs of sentences where one entails the other." \
    --format pair-class \
    --samples 20
```

**Generate 100 samples using a custom format from a query:**

```bash
python -m synthetic_data_generator.main \
    --query "Generate descriptions for characters in a fantasy setting." \
    --custom_format_file path/to/my_format.json \
    --samples 100
```
*(See [Custom Format Specification](#custom-format-specification) for `my_format.json` structure)* 

**Generate 50 QA pairs from documents in a directory:**

```bash
# Requires ADLoader dependency
python -m synthetic_data_generator.main \
    --documents path/to/my/docs \
    --format qa \
    --samples 50
```

**Generate 10 samples, save only as JSONL and CSV, with a custom prefix:**

```bash
python -m synthetic_data_generator.main \
    --query "Generate simple facts." \
    --format pair \
    --samples 10 \
    --output_prefix output/simple_facts \
    --output_format jsonl csv
```
*(Output files: `output/simple_facts_pair_from_query.jsonl`, `output/simple_facts_pair_from_query.csv`)* 

**Generate 5000 samples incrementally to JSONL, then save as Parquet:**

```bash
python -m synthetic_data_generator.main \
    --query "Generate a large dataset of product reviews." \
    --format pair-score \
    --samples 5000 \
    --output_prefix large_reviews \
    --incremental_save \
    --output_format jsonl parquet
```
*(`large_reviews_pair-score_from_query.jsonl` is written incrementally. After generation, this file is loaded to create `large_reviews_pair-score_from_query.parquet`)* 

**Generate with DEBUG logging:**

```bash
python -m synthetic_data_generator.main \
    --query "Test logging." \
    --format qa \
    --samples 5 \
    --log_level DEBUG
```

## 6. Custom Format Specification

A custom format is defined by a JSON file. The structure must be a JSON object with the following keys: 

* `name` (string, optional): A user-friendly name for the format. Defaults to the filename without extension.
* `description` (string, optional): A brief description of the format's purpose. Defaults to a generic description based on the name.
*   `fields` (object, required): A dictionary where keys are the desired field names in the output data, and values are objects defining each field.

Each field definition object must contain:

*   `type` (string, required): The expected data type. Must be one of: `"string"`, `"integer"`, `"float"`, `"boolean"`.
*   `description` (string, required): A description of the field's content and purpose. This is crucial for guiding the LLM.
*   `required` (boolean, optional): Whether the field is required. Defaults to `true`. If `false`, the LLM may omit the field or provide `null`.
*   `example` (any, optional): An example value for the field. This helps the LLM understand the desired content and format.

**Example `my_format.json`:**

```json
{
  "name": "character_description",
  "description": "Describes a fictional character with name, occupation, and a brief bio.",
  "fields": {
    "character_name": {
      "type": "string",
      "description": "The full name of the fictional character.",
      "required": true,
      "example": "Elara Meadowlight"
    },
    "occupation": {
      "type": "string",
      "description": "The character's profession or role in their world.",
      "required": true,
      "example": "Starship Pilot"
    },
    "bio": {
      "type": "string",
      "description": "A brief biographical summary, including key traits or history.",
      "required": true,
      "example": "A skilled pilot known for navigating treacherous asteroid fields and a mysterious past."
    },
    "age": {
      "type": "integer",
      "description": "The character's age in years.",
      "required": false,
      "example": 35
    },
    "is_protagonist": {
      "type": "boolean",
      "description": "True if the character is a main protagonist, false otherwise.",
      "required": false,
      "example": true
    }
  }
}
```
 
## 7. Document Loading

The `--documents` option requires an external dependency, `ADLoader` (AutoDocumentLoader), which is assumed to be available in your Python environment. This class is expected to take a path (file or directory) in its constructor and have a `load()` method that returns a list of objects, where each object has a `page_content` attribute containing the text to be used as context.

If `ADLoader` is not installed or fails to import, the `--documents` option will be disabled, and attempting to use it will result in a `LoaderError`.

## 8. Testing

The project includes a test suite using `pytest`. 

1.  Install testing dependencies (if not already included in your `requirements.txt`):
    ```bash
    pip install pytest pytest-mock
    ```
2.  Run the tests from the project root directory:
    ```bash
    pytest
    ```
The tests are organized into `unit/` and `integration/` directories within the `tests/` folder. 

## 9. Contributing

Contributions are welcome! Please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-feature-name`).
3.  Make your changes.
4.  Write tests for your changes.
5.  Ensure tests pass (`pytest`).
6.  Ensure code style is consistent (consider using `black` and `isort`).
7.  Commit your changes (`git commit -m 'Add your feature'`).
8.  Push to your fork (`git push origin feature/your-feature-name`).
9.  Create a Pull Request to the main repository.

## 10. Versioning

The project version is defined in `synthetic_data_generator/__init__.py`. You can check the version using the `--version` flag:

```bash
python -m synthetic_data_generator.main --version
``` 

## 11. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (Note: You should add a LICENSE file if you haven't already).

## 12. Contact

If you have any questions or issues, please open an issue on the GitHub repository or contact etetteh via GitHub.

## 13. Acknowledgements

* [Google Gemini](https://ai.google.dev/models/gemini) for the powerful language models.
* [LangChain](https://www.langchain.com/) for the flexible framework for developing LLM applications.
* [tqdm](https://github.com/tqdm/tqdm) for the beautiful progress bars.
* [Pandas](https://pandas.pydata.org/) and [PyArrow](https://arrow.apache.org/docs/python/) for Parquet support.
* [python-dotenv](https://github.com/theskumar/python-dotenv) for environment variable management.
* [ADLoader](https://github.com/your-adloader-repo) (Placeholder) for document loading capabilities.

---
© 2025 etetteh