from typing import Dict, List, Optional
from pydantic import field_validator, model_validator, computed_field
from databonsai.transform.base_transformer import BaseTransformer


class DecomposeTransformer(BaseTransformer):
    """
    This class extends the BaseTransformer class and overrides the transform method to decompose the input data into a list of dictionaries based on a provided output schema.

    Attributes:
        output_schema (Dict[str, str]): A dictionary representing the schema of the output dictionaries.
        examples (Optional[List[Dict[str, str]]]): A list of example inputs and their corresponding decomposed outputs.

    Raises:
        ValueError: If the output schema dictionary is empty, or if the transformed data does not match the expected format or schema.
    """

    output_schema: Dict[str, str]
    examples: Optional[List[Dict[str, str]]] = []

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

    @field_validator("examples")
    def validate_examples(cls, v):
        """
        Validates the examples list.

        Args:
            v (List[Dict[str, str]]): The examples list to be validated.

        Raises:
            ValueError: If the examples list is not a list of dictionaries or if any dictionary is missing the "example" or "response" key.

        Returns:
            List[Dict[str, str]]: The validated examples list.
        """
        if not isinstance(v, list):
            raise ValueError("Examples must be a list of dictionaries.")
        for example in v:
            if not isinstance(example, dict):
                raise ValueError("Each example must be a dictionary.")
            if "example" not in example or "response" not in example:
                raise ValueError(
                    "Each example dictionary must have 'example' and 'response' keys."
                )
        return v

    @model_validator(mode="after")
    def validate_examples_responses(self):
        """
        Validates that the "response" values in the examples are valid JSON-formatted lists of dictionaries that match the output schema.
        """
        if self.examples:
            for example in self.examples:
                response = example.get("response")
                try:
                    response_data = eval(response)
                except (SyntaxError, NameError, TypeError, ZeroDivisionError):
                    raise ValueError(
                        f"Invalid format in the example response: {response}"
                    )
                if not isinstance(response_data, list):
                    raise ValueError(
                        f"Example response must be a JSON-formatted list: {response}"
                    )
                for item in response_data:
                    if not isinstance(item, dict):
                        raise ValueError(
                            f"Each item in the example response must be a dictionary: {response}"
                        )
                    if set(item.keys()) != set(self.output_schema.keys()):
                        raise ValueError(
                            f"The keys in the example response do not match the output schema: {response}"
                        )
        return self

    @computed_field
    @property
    def system_message(self) -> str:
        system_message = f"""
        Use the following prompt to transform the input data:
        Input Data: {self.prompt}
        The transformed data should be a list of dictionaries, where each dictionary has the following schema:
        {self.output_schema}
        Reply with a JSON-formatted list of dictionaries. Do not make any conversation.
        """

        # Add in few-shot examples
        if self.examples:
            for example in self.examples:
                system_message += (
                    f"\nEXAMPLE: {example['example']}  RESPONSE: {example['response']}"
                )

        return system_message

    # @computed_field
    # @property
    # def system_message_batch(self) -> str:
    #     system_message = f"""
    #     Use the following prompt to transform each input data:
    #     Input Data: {self.prompt}
    #     The transformed data should be a list of dictionaries, where each dictionary has the following schema:
    #     {self.output_schema}
    #     Reply with a JSON-formatted list of dictionaries for each input, separated by ||. Do not make any conversation.
    #     Example: Content 1: <content>, Content 2: <content> \n Response: <JSON-formatted list of dictionaries 1>||<JSON-formatted list of dictionaries 2>
    #     """

    #     # Add in few-shot examples
    #     if self.examples:
    #         system_message += "\nExample: "
    #         for idx, example in enumerate(self.examples):
    #             system_message += f"Content {str(idx+1)}: {example['example']}, "
    #         system_message += f"\nResponse: "
    #         for example in self.examples:
    #             system_message += f"{example['response']}||"

    #     return system_message

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
        # Call the LLM provider to perform the transformation
        response = self.llm_provider.generate(
            self.system_message, input_data, max_tokens=max_tokens
        )

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

    # def transform_batch(self, input_data: List[str], max_tokens=1000) -> List[List[Dict[str, str]]]:
    #     """
    #     Transforms a batch of input data into lists of dictionaries using the specified LLM provider.

    #     Args:
    #         input_data (List[str]): A list of text data to be transformed.
    #         max_tokens (int, optional): The maximum number of tokens to generate in each response. Defaults to 1000.

    #     Returns:
    #         List[List[Dict[str, str]]]: A list of transformed data, where each element is a list of dictionaries corresponding to the respective input data.

    #     Raises:
    #         ValueError: If the transformed data does not match the expected format or schema.
    #     """
    #     # Call the LLM provider to perform the batch transformation
    #     response = self.llm_provider.generate_batch(
    #         self.system_message_batch, input_data, max_tokens=max_tokens
    #     )

    #     # Split the response into individual transformed data
    #     transformed_data_list = response.split("||")

    #     # Strip any leading/trailing whitespace from each transformed data
    #     transformed_data_list = [data.strip() for data in transformed_data_list]

    #     if len(transformed_data_list) != len(input_data):
    #         raise ValueError(
    #             f"Length of output list ({len(transformed_data_list)}) does not match the length of input list ({len(input_data)})."
    #         )

    #     # Evaluate and validate each transformed data
    #     result = []
    #     for data in transformed_data_list:
    #         try:
    #             transformed_data = eval(data)
    #         except (SyntaxError, NameError, TypeError, ZeroDivisionError):
    #             raise ValueError(f"Invalid format in the transformed data: {data}")

    #         # Validate the transformed data
    #         if not isinstance(transformed_data, list):
    #             raise ValueError(f"Transformed data must be a list: {data}")
    #         for item in transformed_data:
    #             if not isinstance(item, dict):
    #                 raise ValueError(
    #                     f"Each item in the transformed data must be a dictionary: {data}"
    #                 )
    #             if set(item.keys()) != set(self.output_schema.keys()):
    #                 raise ValueError(
    #                     f"The keys in the transformed data do not match the schema: {data}"
    #                 )

    #         result.append(transformed_data)

    #     return result
