"""
Concrete implementation of DataFormatHandler for predefined formats.
"""
import logging
from typing import List, Dict, Any, Optional, Literal, Callable, Tuple

from .base import DataFormatHandler # Import abstract base class
from .. import exceptions # Import custom exceptions
from .. import config # Import config constants and types

logger = logging.getLogger(__name__)

class PredefinedFormatHandler(DataFormatHandler):
    """
    Handles specifics for predefined data formats (e.g., 'qa', 'triplet').

    Provides format-specific descriptions, examples, validation rules, and
    guidance prompts by extending the common base methods.
    """
    # Store details centrally for each predefined format
    _PREDEFINED_DETAILS = {
        # --- Updated Guidance incorporating MATRIX-inspired ideas ---
        "pair-class": {
            "desc": "Generate premise-hypothesis pairs representing textual entailment relationships (0=entailment, 1=neutral, 2=contradiction).",
            "example": '`[{"premise": "...", "hypothesis": "...", "label": 0|1|2}]`',
            "fields": {"premise": (str, None), "hypothesis": (str, None), "label": (int, lambda x: x in [0, 1, 2])},
            "guidance": """**Guidance for Generating High-Quality Textual Entailment Pairs:**
                - **Labels:** 0=Entailment (Hypothesis MUST be true if Premise is true), 1=Neutral (Hypothesis MIGHT be true), 2=Contradiction (Hypothesis CANNOT be true if Premise is true).
                - **Quality:** Premise should be clear, specific. Hypothesis related, natural. Relationship unambiguous.
                - **Variety:** CRITICAL: Aim for balanced labels (0, 1, 2). Cover diverse topics (science, news, tech). Vary complexity (facts, causality, logic). Vary structure (conditionals, comparisons). Generate examples that describe simple scenes or interactions.
                - **Avoid:** Ambiguous premises, trivial contradictions, requiring unstated assumptions for entailment.
                - **Verification:** Double-check label correctness and balance across the batch."""
        },
        "pair-score": {
            "desc": "Generate sentence pairs with semantic similarity scores (0.0-1.0).",
            "example": '`[{"sentence1": "...", "sentence2": "...", "score": 0.0-1.0}]`',
            "fields": {"sentence1": (str, None), "sentence2": (str, None), "score": (float, lambda x: 0.0 <= x <= 1.0)},
            "guidance": """**Guidance for Semantic Similarity Pairs:**
                - **Scores:** Represent the full spectrum (0.0 = unrelated, 1.0 = identical meaning).
                - **Quality:** Focus on semantic meaning, not just keyword overlap. Pairs should be natural sentences.
                - **Variety:** Cover diverse domains and topics. Include paraphrases, related concepts, and unrelated pairs. Vary sentence length and structure. Consider pairs describing related actions or scenes."""
        },
        "pair": {
            "desc": "Generate pairs where 'positive' sentence is semantically close or entailed by 'anchor' sentence.",
            "example": '`[{"anchor": "...", "positive": "..."}]`',
            "fields": {"anchor": (str, None), "positive": (str, None)},
            "guidance": """**Guidance for Anchor-Positive Pairs:**
                - **Relationship:** 'Positive' must share core meaning or be directly inferable from 'anchor'.
                - **Quality:** Sentences should be natural and the relationship clear.
                - **Variety:** Include paraphrases, inferences, elaborations, cause-effect. Describe related events or compositions of concepts from the anchor. Avoid trivial identity pairs."""
        },
        "triplet": {
            "desc": "Generate triplets: anchor/positive are related, anchor/negative are unrelated or contradictory.",
            "example": '`[{"anchor": "...", "positive": "...", "negative": "..."}]`',
            "fields": {"anchor": (str, None), "positive": (str, None), "negative": (str, None)},
            "guidance": """**Guidance for Triplet Generation:**
                - **Relationships:** Anchor-Positive clearly related (paraphrase, inference). Anchor-Negative clearly distinct (different topic, contradiction).
                - **Quality:** 'Negative' should be plausible on its own but semantically distant from the anchor. Avoid easily distinguishable negatives.
                - **Variety:** Cover diverse topics. Generate challenging negatives. Compose scenarios where positive relates to anchor, but negative describes a different (yet plausible) scenario."""
        },
        "qa": {
            "desc": "Generate context-question-answer triplets. The answer must be derivable *only* from the context.",
            "example": '`[{"context": "...", "question": "...", "answer": "..."}]`',
            "fields": {"context": (str, None), "question": (str, None), "answer": (str, None)},
            "guidance": """**Guidance for Question Answering Pairs:**
                - **Grounding:** Question MUST be answerable solely from the provided context. Answer must be accurate, concise, and directly extracted or inferred from the context.
                - **Quality:** Questions should be natural and require understanding, not just keyword matching. Answers should be minimal correct spans or syntheses from the text.
                - **Variety:** Vary question types (what, who, why, how, multi-hop). Vary context length and domain. If generating from query, create plausible context first, potentially describing a scene or situation."""
        }
    }

    def __init__(self, format_name: config.PredefinedDataFormat):
        """
        Initializes the handler for a specific predefined format.

        Args:
            format_name: The name of the predefined format (e.g., "qa").

        Raises:
            exceptions.ConfigurationError: If the format_name is not recognized.
        """
        if format_name not in self._PREDEFINED_DETAILS:
            raise exceptions.ConfigurationError(f"Unknown predefined format name: {format_name}")
        self._format_name = format_name
        self._details = self._PREDEFINED_DETAILS[format_name]

    def get_format_name(self) -> str:
        """Returns the name of this predefined format."""
        return self._format_name

    def get_description(self) -> str:
        """Returns the description of this predefined format."""
        return self._details["desc"]

    def get_example_structure_string(self) -> str:
        """Returns the example JSON structure string for this format."""
        return self._details["example"]

    def get_field_names(self) -> Optional[List[str]]:
        """Returns the defined field names for this predefined format."""
        return list(self._details["fields"].keys())

    def build_system_prompt(self) -> str:
        """Builds the system prompt including format-specific guidance."""
        base = self._get_common_system_prompt_base()
        guidance = self._details["guidance"]
        # Append format-specific guidance to the common base prompt
        return f"{base}\n\n**Format Specific Guidance ({self.get_format_name()}):**\n{guidance}\n"

    def build_query_prompt(self, query: str, num_samples: int) -> str:
        """Builds the query-based user prompt (uses common implementation)."""
        return self._get_common_query_prompt(query, num_samples)

    def build_document_prompt(self, documents: List[str], num_samples: int) -> str:
        """Builds the document-based user prompt (uses common implementation)."""
        return self._get_common_document_prompt(documents, num_samples)

    def validate_item(self, item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
        """Validates a single item against this predefined format's specification."""
        if not isinstance(item, dict): raise exceptions.ValidationError(f"Item {item_index} is not a dictionary (got {type(item)}).")
        item_keys = set(item.keys())
        format_spec = self._details["fields"]
        required_keys = set(format_spec.keys())

        # Check for missing required keys
        missing_keys = required_keys - item_keys
        if missing_keys:
            raise exceptions.ValidationError(f"Item {item_index} (format '{self._format_name}') missing required field(s): {missing_keys}.")

        # Check for unexpected extra keys (optional, issue warning)
        extra_keys = item_keys - required_keys
        if extra_keys:
            logger.warning(f"Item {item_index} (format '{self._format_name}') has unexpected extra field(s): {extra_keys}. Ignoring them.")

        validated_item = {} # Build validated item to return potentially coerced values
        for field, (expected_type, validation_func) in format_spec.items():
            if field not in item:
                 # This case should be caught by the missing_keys check earlier, but safeguard
                 raise exceptions.ValidationError(f"Internal Error: Field '{field}' suddenly missing in item {item_index} during validation.")

            value = item[field]

            # Handle potential type coercion (int -> float)
            if expected_type == float and isinstance(value, int):
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    raise exceptions.ValidationError(f"Item {item_index}, field '{field}': Could not coerce integer '{item[field]}' to float.")

            # Validate type
            if not isinstance(value, expected_type):
                raise exceptions.ValidationError(f"Item {item_index}, field '{field}': Expected type {expected_type.__name__}, got {type(value).__name__}. Value: '{str(value)[:50]}'.")

            # Validate non-empty string (warning only)
            if expected_type == str and not value.strip():
                logger.warning(f"Item {item_index}, field '{field}' is an empty or whitespace-only string.")

            # Apply custom validation lambda function if present
            if validation_func and not validation_func(value):
                raise exceptions.ValidationError(f"Item {item_index}, field '{field}': Value '{str(value)[:50]}' failed constraint check ({self._format_name}).")

            validated_item[field] = value # Store potentially coerced value

        # Return dictionary containing only the validated fields according to the spec
        # This also implicitly handles ignoring extra keys
        return {k: validated_item[k] for k in format_spec.keys()}
