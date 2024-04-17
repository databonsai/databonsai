from typing import List, Optional, Dict
from pydantic import BaseModel, field_validator, model_validator, computed_field
from databonsai.llm_providers import LLMProvider


class BaseTransformer(BaseModel):
    """
    A base class for transforming input data using a specified LLM provider.

    Attributes:
        prompt (str): The prompt used to guide the transformation process.
        llm_provider (LLMProvider): An instance of an LLM provider to be used for transformation.
        examples (Optional[List[Dict[str, str]]]): A list of example inputs and their corresponding transformed outputs.

    """

    prompt: str
    llm_provider: LLMProvider
    examples: Optional[List[Dict[str, str]]] = []

    class Config:
        arbitrary_types_allowed = True

    @field_validator("prompt")
    def validate_prompt(cls, v):
        """
        Validates the prompt.

        Args:
            v (str): The prompt to be validated.

        Raises:
            ValueError: If the prompt is empty.

        Returns:
            str: The validated prompt.
        """
        if not v:
            raise ValueError("Prompt cannot be empty.")
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

    @computed_field
    @property
    def system_message(self) -> str:
        system_message = f"""
        Use the following prompt to transform the input data:
        Prompt: {self.prompt}
        """

        # Add in fewshot examples
        if self.examples:
            for example in self.examples:
                system_message += (
                    f"\nEXAMPLE: {example['example']}  RESPONSE: {example['response']}"
                )

        return system_message

    @computed_field
    @property
    def system_message_batch(self) -> str:
        system_message = f"""
        Use the following prompt to transform each input data:
        Prompt: {self.prompt}
        Respond with the transformed data for each input, separated by ||. Do not make any other conversation.
        Example: Content 1: <content>, Content 2: <content> \n Response: <transformed data 1>||<transformed data 2>
        """

        # Add in fewshot examples
        if self.examples:
            system_message += "\nExample: "
            for idx, example in enumerate(self.examples):
                system_message += f"Content {str(idx+1)}: {example['example']}, "
            system_message += f"\nResponse: "
            for example in self.examples:
                system_message += f"{example['response']}||"

        return system_message

    def transform(self, input_data: str, max_tokens=1000) -> str:
        """
        Transforms the input data using the specified LLM provider.

        Args:
            input_data (str): The text data to be transformed.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 1000.

        Returns:
            str: The transformed data.
        """
        # Call the LLM provider to perform the transformation
        response = self.llm_provider.generate(
            self.system_message, input_data, max_tokens=max_tokens
        )
        transformed_data = response.strip()
        return transformed_data

    def transform_batch(self, input_data: List[str], max_tokens=1000) -> List[str]:
        """
        Transforms a batch of input data using the specified LLM provider.

        Args:
            input_data (List[str]): A list of text data to be transformed.
            max_tokens (int, optional): The maximum number of tokens to generate in each response. Defaults to 1000.

        Returns:
            List[str]: A list of transformed data, where each element corresponds to the transformed version of the respective input data.
        """
        # Call the LLM provider to perform the batch transformation
        response = self.llm_provider.generate_batch(
            self.system_message_batch, input_data, max_tokens=max_tokens
        )

        # Split the response into individual transformed data
        transformed_data_list = response.split("||")

        # Strip any leading/trailing whitespace from each transformed data
        transformed_data_list = [data.strip() for data in transformed_data_list]

        if len(transformed_data_list) != len(input_data):
            raise ValueError(
                f"Length of output list ({len(transformed_data_list)}) does not match the length of input list ({len(input_data)})."
            )

        return transformed_data_list
