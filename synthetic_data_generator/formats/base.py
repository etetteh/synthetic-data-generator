"""
Abstract base class for data format handlers.

Defines the interface for different data generation formats.
"""
import abc
from typing import List, Dict, Union, Optional, Any, Callable, Tuple

# Re-export LangChain message types needed by handlers
from langchain_core.messages import SystemMessage, HumanMessage

class DataFormatHandler(abc.ABC):
    """
    Abstract base class defining the interface for handling data format specifics.

    This class uses the Strategy pattern. Concrete subclasses implement methods
    for building prompts, validating data, and providing metadata specific to
    a particular output format (predefined or custom).
    """

    @abc.abstractmethod
    def get_format_name(self) -> str:
        """Return the user-friendly name of the data format."""
        pass

    @abc.abstractmethod
    def get_description(self) -> str:
        """Return a concise description of the data format's purpose."""
        pass

    @abc.abstractmethod
    def build_system_prompt(self) -> str:
        """
        Build the system prompt for the LLM.

        This prompt typically includes general instructions for data generation
        and specific requirements related to the format (structure, fields, etc.).

        Returns:
            The complete system prompt string.
        """
        pass

    @abc.abstractmethod
    def build_query_prompt(self, query: str, num_samples: int) -> str:
        """
        Build the user prompt for query-based generation.

        Args:
            query: The user's input query (potentially refined).
            num_samples: The number of samples requested for this batch.

        Returns:
            The complete user prompt string for query-based generation.
        """
        pass

    @abc.abstractmethod
    def build_document_prompt(self, documents: List[str], num_samples: int) -> str:
        """
        Build the user prompt for document-based generation.

        Args:
            documents: A list of document text chunks to use as context.
            num_samples: The number of samples requested for this batch.

        Returns:
            The complete user prompt string for document-based generation.
        """
        pass

    @abc.abstractmethod
    def get_example_structure_string(self) -> str:
        """
        Get a string representation of the expected JSON structure for prompts.

        This helps the LLM understand the desired output format.

        Returns:
            A string illustrating the JSON structure (e.g., `[{"key": "value"}]`).
        """
        pass

    @abc.abstractmethod
    def get_field_names(self) -> Optional[List[str]]:
        """
        Get the ordered list of field names defined for this format.

        Used primarily for writing CSV headers. Returns None if field names
        are not applicable or cannot be determined reliably.

        Returns:
            A list of field name strings, or None.
        """
        pass

    @abc.abstractmethod
    def validate_item(self, item: Dict[str, Any], item_index: int) -> Dict[str, Any]:
        """
        Validate a single generated data item against the format's rules.

        Checks for required fields, correct data types, and potentially other
        constraints defined by the format. May perform type coercion (e.g., int to float).

        Args:
            item: The dictionary representing the generated data sample.
            item_index: The index of the item within its generation batch (for logging).

        Returns:
            The validated (and potentially type-coerced) dictionary item.

        Raises:
            ValidationError: If the item fails validation checks.
        """
        pass

    # --- Common Prompt Building Logic (Can be moved to a mixin or helper) ---
    # Keeping these here for now as they are tightly coupled to prompt building
    # but could be refactored if more common prompt logic emerges.

    def _get_matrix_inspired_guidance(self) -> str:
        """Provides general guidance inspired by multimodal/compositional concepts."""
        return """
            **MATRIX-Inspired Generation Principles (Apply where relevant):**
            - **Compositionality:** Generate complex scenarios by combining multiple entities, actions, or concepts. Describe their interactions clearly.
            - **Multimodal Grounding (Textual):** If applicable, generate text that implies or describes associated non-textual elements (e.g., "The image shows...", "The accompanying audio features...", "In the video scene..."). Ensure descriptions are consistent.
            - **Perspective Shifting:** Consider generating examples from different viewpoints or roles within a scenario.
            - **Temporal Dynamics:** If relevant, describe sequences of events or changes over time.
            - **Instruction Following:** Ensure the generated data strictly follows any specific constraints or instructions mentioned in the user query or document context.
        """

    def _get_common_system_prompt_base(self) -> str:
        """Returns the common boilerplate part of the system prompt, including MATRIX guidance."""
        # Incorporate MATRIX-inspired guidance into the base system prompt
        return f"""
            You are an advanced synthetic data generator specializing in creating diverse, realistic, and high-quality training examples for machine learning. Your objective is to produce data that strictly conforms to the specified JSON format '{self.get_format_name()}' while ensuring the data is logically sound, topically relevant, contextually accurate, and compositionally rich where appropriate.

            **Format Description:** {self.get_description()}

            **Required JSON Structure per Sample (must be in a list):**
            {self.get_example_structure_string()}

            **Core Principles for Generation:**
            1. **Diversity:** Generate examples covering a wide range of topics, scenarios, complexities, writing styles, and linguistic variations relevant to the request. Ensure thorough coverage across different domains, perspectives, and contexts.
            2. **Specificity:** Create concrete, detailed examples rather than generic ones. When asked for examples about specific domains (e.g., "recent news"), ensure outputs are actually about that domain.
            3. **Accuracy & Consistency:** Ensure factual correctness (if applicable) and logical consistency within each example. The relationships between data elements must be valid and realistic.
            4. **Format Compliance:** Adhere STRICTLY to the requested JSON structure and field definitions. Ensure required fields are present and types match. Output ONLY the JSON data structure (a list of JSON objects). Do not include any extra text, explanations, or markdown formatting around the JSON.
            5. **Uniqueness:** Each generated example must be distinct in both content and structure from others in the batch.
            6. **Internal Reasoning:** Before providing the final output, internally perform a detailed chain-of-thought analysis to ensure accuracy, diversity, and strict adherence to the format. DO NOT include any of your internal reasoning in the final output.

            {self._get_matrix_inspired_guidance()}

            **Reasoning Process (Internal Monologue - Do Not Output):**
            - *Understand:* Fully grasp the format requirements ({self.get_format_name()}), field definitions, and logical relationships. Consider compositional and descriptive needs.
            - *Contextualize:* Analyze the user's request (query/document) carefully. Identify key entities, actions, relationships, and constraints.
            - *Diversify & Compose:* Brainstorm varied scenarios. Combine elements logically to create rich, non-repetitive examples. If relevant, describe implied multimodal aspects textually.
            - *Validate:* Verify logical consistency, factual accuracy (if applicable), and adherence to the request domain.
            - *Check Format & Uniqueness:* Ensure all format rules are met and the example is unique.

            Now, generate samples based on the user's input while strictly adhering to these principles.
        """

    def _get_common_query_prompt(self, query: str, num_samples: int) -> str:
        """Returns the common boilerplate part of the query-based user prompt."""
        # Add instruction reinforcing MATRIX ideas like specificity and compositionality
        return f"""
            Based on the theme or request in the query below, generate {num_samples} diverse and completely unique synthetic data samples that precisely match the request domain.

            Query: "{query}"

            Target Format Name: {self.get_format_name()}
            Required JSON Structure Per Sample (must be in a list): {self.get_example_structure_string()}

            **Instructions:**
            - Focus specifically on the domain mentioned in the query (e.g., if about 'financial reports', generate data related to finance).
            - Create examples with varied complexity and compositionality; combine multiple relevant concepts or entities where appropriate.
            - Maintain logical consistency within each example based on the format '{self.get_format_name()}'.
            - Ensure a balanced distribution of different categories/classes if applicable to the format.
            - Provide specific, concrete examples rather than vague or generic ones.
            - Each example should be distinct and focus on different aspects of the requested domain or scenario.
            - Internally perform a chain-of-thought analysis to verify that each sample meets the quality, diversity, and format requirements. DO NOT output any part of this internal reasoning.
            - Adhere strictly to the JSON structure shown above. Include all required fields and respect data types.
            - Output ONLY the raw JSON list `[...]` containing the generated samples. Do not add any other text or explanations.
        """

    def _get_common_document_prompt(self, documents: List[str], num_samples: int) -> str:
        """Returns the common boilerplate part of the document-based user prompt."""
        if not documents:
            raise ValueError("Cannot build document prompt with an empty list of documents.")
        docs_text = "\n\n---\n\n".join(doc.strip() for doc in documents if doc and doc.strip())
        if not docs_text:
             raise ValueError("Cannot build document prompt with only empty document strings after stripping.")

        # Add instruction reinforcing MATRIX ideas like grounding and extracting interactions
        return f"""
            You are provided with the following document text(s). Generate {num_samples} diverse and completely unique synthetic data samples based *faithfully* on the information contained within these documents, conforming to the '{self.get_format_name()}' format.

            Document Text(s):
            --- START DOCUMENT TEXT ---
            {docs_text}
            --- END DOCUMENT TEXT ---

            Target Format Name: {self.get_format_name()}
            Required JSON Structure Per Sample (must be in a list): {self.get_example_structure_string()}

            **Instructions:**
            - Use ONLY information explicitly stated or strongly implied in the provided document text(s). Do not invent external information.
            - Extract or synthesize different aspects, relationships, or interactions mentioned in the documents to create varied but faithful examples.
            - Maintain logical consistency within each example according to the format '{self.get_format_name()}' and the document context.
            - Create examples of varying complexity based on the document information.
            - Ensure a balanced distribution of different categories/classes if applicable to the format, based on document content.
            - Preserve factual accuracy derived from the documents.
            - Each example should highlight different information, perspectives, or relationships found in the documents.
            - Adhere strictly to the JSON structure shown above. Include all required fields and respect data types.
            - Output ONLY the raw JSON list `[...]` containing the generated samples. Do not add any other text or explanations.
        """
