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
        {str(list(categories.keys()))}
        Reply with a list of categories separated by || for each text snippet. Do not make any other conversation.
        """

        # Call the LLM provider to get the predicted categories
        response = self.llm_provider.generate(system_message, input_data)
        predicted_categories = [category.strip() for category in response.split("||")]

        # Validate that the predicted categories are a subset of the provided categories
        if not set(predicted_categories).issubset(categories.keys()):
            raise ValueError(
                f"Predicted categories {predicted_categories} are not a subset of the provided categories."
            )

        return predicted_categories

    def categorize_batch(self, input_data: List[str]) -> List[str]:
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
        {str(list(categories.keys()))}
        Reply with a list of || separated categories for each snippet. Separate snippets with ##. Example: <snippet 1 category>||<snippet 1 category>##<snippet 2 category>. Do not make any other conversation.
        """

        # Call the LLM provider to get the predicted categories

        response = self.llm_provider.generate_batch(system_message, input_data)

        # Split the response into category sets for each input data
        category_sets = response.split("##")

        if len(category_sets) != len(input_data):
            raise ValueError(
                f"Number of predicted category sets ({len(category_sets)}) does not match the number of input data ({len(input_data)})."
            )

        predicted_categories_list = []
        for category_set in category_sets:
            predicted_categories = [
                category.strip() for category in category_set.split("||")
            ]

            for predicted_category in predicted_categories:
                if predicted_category not in self.categories:
                    raise ValueError(
                        f"Predicted category '{predicted_category}' is not one of the provided categories."
                    )

            predicted_categories_str = ",".join(predicted_categories)
            predicted_categories_list.append(predicted_categories_str)

        return predicted_categories_list
