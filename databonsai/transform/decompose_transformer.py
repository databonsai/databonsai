from typing import Dict, List
from pydantic import field_validator
from databonsai.transform.base_transformer import BaseTransformer


class DecomposeTransformer(BaseTransformer):
    """
    This class extends the BaseTransformer class and overrides the transform method to
    decompose the input data into a list of dictionaries based on a provided output schema.

    Attributes:
        output_schema (Dict[str, str]): A dictionary representing the schema of the output dictionaries.

    Raises:
        ValueError: If the output schema dictionary is empty, or if the transformed data
                    does not match the expected format or schema.

    """

    output_schema: Dict[str, str]

    @field_validator("output_schema")
    def validate_schema(cls, v):
        """
        Validates the output schema.

        Args:
            v (Dict[str, str]): The output schema to be validated.

        Raises:
            ValueError: If the output schema dictionary is empty.

        Returns:
            Dict[str, str]: The validated output schema.
        """
        if not v:
            raise ValueError("Schema dictionary cannot be empty.")
        return v

    def transform(self, input_data: str, max_tokens=1000) -> List[Dict[str, str]]:
        """
        Transforms the input data into a list of dictionaries using the specified LLM provider.

        Args:
            input_data (str): The text data to be transformed.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 1000.

        Returns:
            List[Dict[str, str]]: The transformed data as a list of dictionaries.

        Raises:
            ValueError: If the transformed data does not match the expected format or schema.
        """
        system_message = f"""
        Use the following prompt to transform the input data:
        Input Data: {self.prompt}

        The transformed data should be a list of dictionaries, where each dictionary has the following schema:
        {self.output_schema}

        Reply with a JSON-formatted list of dictionaries. Do not make any conversation.
        """

        # Call the LLM provider to perform the transformation
        response = self.llm_provider.generate(system_message, input_data)

        try:
            transformed_data = eval(response)
        except (SyntaxError, NameError, TypeError, ZeroDivisionError):
            raise ValueError("Invalid format in the transformed data.")

        # Validate the transformed data
        if not isinstance(transformed_data, list):
            raise ValueError("Transformed data must be a list.")

        for item in transformed_data:
            if not isinstance(item, dict):
                raise ValueError(
                    "Each item in the transformed data must be a dictionary."
                )

            if set(item.keys()) != set(self.output_schema.keys()):
                raise ValueError(
                    "The keys in the transformed data do not match the schema."
                )

        return transformed_data
