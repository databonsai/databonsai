from typing import List
from databonsai.categorize.base_categorizer import BaseCategorizer
from pydantic import model_validator, computed_field


class MultiCategorizer(BaseCategorizer):
    """
    A class for categorizing input data into multiple categories using a specified LLM provider.

    This class extends the BaseCategorizer class and overrides the categorize method to allow
    for multiple category predictions.
    """

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def validate_examples_responses(self):
        """
        Validates that the "response" values in the examples are within the categories keys. If there are multiple categories, they should be separated by commas.
        """

        if self.examples:
            category_keys = set(self.categories.keys())
            for example in self.examples:
                response = example.get("response")
                response_categories = response.split(",")
                for response_category in response_categories:
                    if response_category not in category_keys:
                        raise ValueError(
                            f"Example response '{response}' is not one of the provided categories, {str(list(self.categories.keys()))}."
                        )

        return self

    @computed_field
    @property
    def system_message(self) -> str:
        system_message = f"""
        Each category is formatted as <category>: <description of data that fits the category>
        {str(self.categories)}
        Classify the given text snippet into one or more of the following categories:
        {str(list(self.categories.keys()))}
        Reply with a list of categories separated by || for each text snippet. Do not make any other conversation.
        """

        # Add in fewshot examples
        if self.examples:
            for example in self.examples:
                system_message += f"\nEXAMPLE: {example['example']}  RESPONSE: {example['response'].replace(',', '||')}"
        return system_message

    @computed_field
    @property
    def system_message_batch(self) -> str:
        categories = self.categories
        system_message = f"""
        Each category is formatted as <category>: <description of data that fits the category>
        {str(categories)}
        Classify the given text snippet into one or more of the following categories:
        {str(list(categories.keys()))}
        Reply with a list of || separated categories for each content snippet. Separate them with ##. Example: Content 1: <content>, Content 2: <content> \n Response: <category>||<category>##<category>. Do not make any other conversation. Do not mention Content in your response.
        """

        # Add in fewshot examples
        if self.examples:
            system_message += "\nExample: "
            for idx, example in enumerate(self.examples):
                system_message += f"Content {str(idx+1)}: {example['example']}, "
            system_message += f"\nResponse: "
            for example in self.examples:
                system_message += f"{example['response'].replace(',', '||')}##"
        return system_message

    def categorize(self, input_data: str) -> str:
        """
        Categorizes the input data into multiple categories using the specified LLM provider.

        Args:
            input_data (str): The text data to be categorized.

        Returns:
            str: A string of categories for the input data, separated by commas.

        Raises:
            ValueError: If the predicted categories are not a subset of the provided categories.
        """

        # Call the LLM provider to get the predicted categories
        response = self.llm_provider.generate(self.system_message, input_data)
        predicted_categories = [category.strip() for category in response.split("||")]

        return ",".join(self.validate_predicted_categories(predicted_categories))

    def categorize_batch(self, input_data: List[str]) -> List[str]:
        """
        Categorizes the input data into multiple categories using the specified LLM provider.

        Args:
            input_data (str): The text data to be categorized.

        Returns:
            List[str]: A list of predicted categories for the input data. If there are multiple categories, they will be separated by commas.

        Raises:
            ValueError: If the predicted categories are not a subset of the provided categories.
        """

        # Call the LLM provider to get the predicted categories
        response = self.llm_provider.generate_batch(
            self.system_message_batch, input_data
        )
        print(response)
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
            predicted_categories_str = ",".join(
                self.validate_predicted_categories(predicted_categories)
            )
            predicted_categories_list.append(predicted_categories_str)

        return predicted_categories_list
