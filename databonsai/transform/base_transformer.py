from typing import List
from pydantic import BaseModel, validator
from databonsai.llm_providers import LLMProvider


class BaseTransformer(BaseModel):
    """
    A base class for transforming input data using a specified LLM provider.

    Attributes:
        prompt (str): The prompt used to guide the transformation process.
        llm_provider (LLMProvider): An instance of an LLM provider to be used for transformation.

    """

    prompt: str
    llm_provider: LLMProvider

    class Config:
        arbitrary_types_allowed = True

    @validator("prompt")
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

    def transform(self, input_data: str, max_tokens=1000) -> str:
        """
        Transforms the input data using the specified LLM provider.

        Args:
            input_data (str): The text data to be transformed.
            max_tokens (int, optional): The maximum number of tokens to generate in the response. Defaults to 1000.

        Returns:
            str: The transformed data.
        """
        system_message = f"""
        Use the following prompt to transform the input data:
        Prompt: {self.prompt}
        """

        # Call the LLM provider to perform the transformation
        response = self.llm_provider.generate(system_message, input_data)
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
        system_message = f"""
        Use the following prompt to transform each input data:
        Prompt: {self.prompt}
        
        Respond with the transformed data for each input, separated by ##. Do not make any other conversation.
        Example:
        <transformed data 1>##<transformed data 2>##<transformed data 3>
        """

        # Call the LLM provider to perform the batch transformation
        response = self.llm_provider.generate_batch(
            system_message, input_data, max_tokens=max_tokens
        )
        # print(response)
        # Split the response into individual transformed data
        transformed_data_list = response.split("##")

        # Strip any leading/trailing whitespace from each transformed data
        transformed_data_list = [data.strip() for data in transformed_data_list]
        if len(transformed_data_list) != len(input_data):
            raise ValueError(
                f"Length of output list ({len(transformed_data_list)}) does not match the length of input list ({len(input_data)})."
            )
        return transformed_data_list
