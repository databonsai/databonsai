from typing import List
from databonsai.categorize.base_categorizer import BaseCategorizer


class MultiCategorizer(BaseCategorizer):
    """
    A class for categorizing input data into multiple categories using a specified LLM provider.

    This class extends the BaseCategorizer class and overrides the categorize method to allow
    for multiple category predictions.
    """

    class Config:
        arbitrary_types_allowed = True

    def categorize(self, input_data: str) -> List[str]:
        """
        Categorizes the input data into multiple categories using the specified LLM provider.

        Args:
            input_data (str): The text data to be categorized.

        Returns:
            List[str]: A list of predicted categories for the input data.

        Raises:
            ValueError: If the predicted categories are not a subset of the provided categories.
        """
        categories = self.categories
        system_message = f"""
        Each category is formatted as <category>: <description of data that fits the category>
        {str(categories)}
        Classify the given text snippet into one or more of the following categories:
        {str(categories.keys())}
        Reply with a comma-separated list of categories. Do not make any other conversation.
        """

        # Call the LLM provider to get the predicted categories
        response = self.llm_provider.generate(system_message, input_data)
        predicted_categories = [category.strip() for category in response.split(",")]

        # Validate that the predicted categories are a subset of the provided categories
        if not set(predicted_categories).issubset(categories.keys()):
            raise ValueError(
                f"Predicted categories {predicted_categories} are not a subset of the provided categories."
            )

        return predicted_categories
