from typing import Dict, List
from pydantic import BaseModel, validator
from databonsai.llm_providers import OpenAIProvider, LLMProvider


class BaseCategorizer(BaseModel):
    """
    A base class for categorizing input data using a specified LLM provider.

    Attributes:
        categories (Dict[str, str]): A dictionary mapping category names to their descriptions.
        llm_provider (LLMProvider): An instance of an LLM provider to be used for categorization.

    """

    categories: Dict[str, str]
    llm_provider: LLMProvider

    class Config:
        arbitrary_types_allowed = True

    @validator("categories")
    def validate_categories(cls, v):
        """
        Validates the categories dictionary.

        Args:
            v (Dict[str, str]): The categories dictionary to be validated.

        Raises:
            ValueError: If the categories dictionary is empty or has less than two key-value pairs.

        Returns:
            Dict[str, str]: The validated categories dictionary.
        """
        if not v:
            raise ValueError("Categories dictionary cannot be empty.")
        if len(v) < 2:
            raise ValueError(
                "Categories dictionary must have more than one key-value pair."
            )
        return v

    def categorize(self, input_data: str) -> str:
        """
        Categorizes the input data using the specified LLM provider.

        Args:
            input_data (str): The text data to be categorized.

        Returns:
            str: The predicted category for the input data.

        Raises:
            ValueError: If the predicted category is not one of the provided categories.
        """
        system_message = f"""
        Each category is formatted as <category>: <description of data that fits the category>
        {str(self.categories)}
        Classify the given text snippet into one of the following categories:
        {str(list(self.categories.keys()))}
        Only reply with the category. Do not make any other conversation.
        """

        # Call the LLM provider to get the predicted category
        response = self.llm_provider.generate(system_message, input_data)
        predicted_category = response.strip()

        # Validate that the predicted category is one of the provided categories
        if predicted_category not in self.categories:
            raise ValueError(
                f"Predicted category '{predicted_category}' is not one of the provided categories."
            )

        return predicted_category

    def categorize_batch(self, input_data: List[str]) -> List[str]:
        """
        Categorizes a batch of input data using the specified LLM provider. For less advanced LLMs, call this method on batches of 3-5 inputs (depending on the length of the input data).

        Args:
            input_data (List[str]): A list of text data to be categorized.

        Returns:
            List[str]: A list of predicted categories for the input data.

        Raises:
            ValueError: If the predicted categories are not a subset of the provided categories.
        """
        system_message = f"""
        Each category is formatted as <category>: <description of data that fits the category>
        {str(self.categories)}
        Classify each given text snippet into one of the following categories:
        {str(list(self.categories.keys()))}. Reply with a list of categories, separated by ||, one for each snippet.
        Only reply with the categories. Do not make any other conversation.
        """
        # Call the LLM provider to get the predicted category
        response = self.llm_provider.generate_batch(system_message, input_data)
        predicted_categories = [category.strip() for category in response.split("||")]
        if len(predicted_categories) != len(input_data):
            raise ValueError(
                f"Number of predicted categories ({len(predicted_categories)}) does not match the number of input data ({len(input_data)})."
            )
        for predicted_category in predicted_categories:
            if predicted_category not in self.categories:
                raise ValueError(
                    f"Predicted category '{predicted_category}' is not one of the provided categories."
                )
        return predicted_categories
