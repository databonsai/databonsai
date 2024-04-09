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
