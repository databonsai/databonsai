from typing import List, Dict
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
        categories_with_numbers = "\n".join(
            [f"{i}: {desc}" for i, desc in enumerate(self.categories.values())]
        )
        system_message = f"""
        Each category is formatted as <number>: <description of data that fits the category>
        {categories_with_numbers}
        Classify the given text snippet into one or more of the following categories:
        {str(list(range(len(self.categories))))}
        Do not use any other categories.
        Assign multiple categories to one content snippet by separating the categories with ||. Do not make any other conversation.
        """

        # Add in fewshot examples
        if self.examples:
            for example in self.examples:
                response_numbers = [
                    str(self.inverse_category_mapping[category.strip()])
                    for category in example["response"].split(",")
                ]
                system_message += f"\nEXAMPLE: {example['example']}  RESPONSE: {'||'.join(response_numbers)}"
        return system_message

    @computed_field
    @property
    def system_message_batch(self) -> str:
        categories_with_numbers = "\n".join(
            [f"{i}: {desc}" for i, desc in enumerate(self.categories.values())]
        )
        system_message = f"""
        Each category is formatted as <number>: <description of data that fits the category>
        {categories_with_numbers}
        Classify the given text snippet into one or more of the following categories:
        {str(list(range(len(self.categories))))}
        Do not use any other categories.
        Assign multiple categories to one content snippet by separating the categories with ||. Differentiate between each content snippet using ##. EXAMPLE: <content1>##<content2> \n RESPONSE: <category number of content1>||<category number of content1>##<category number of content2> Do not make any other conversation.
        """

        # Add in fewshot examples
        if self.examples:
            system_message += "\nEXAMPLE: "
            system_message += (
                f"{'##'.join([example['example'] for example in self.examples])}"
            )
            response_numbers_list = []
            for example in self.examples:
                response_numbers = [
                    str(self.inverse_category_mapping[category.strip()])
                    for category in example["response"].split(",")
                ]
                response_numbers_list.append("||".join(response_numbers))
            system_message += f"\nRESPONSE: {'##'.join(response_numbers_list)}"
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

        # Call the LLM provider to get the predicted category numbers
        response = self.llm_provider.generate(self.system_message, input_data)
        predicted_category_numbers = [
            int(category.strip()) for category in response.split("||")
        ]

        # Convert the category numbers back to category keys
        predicted_categories = [
            self.category_mapping[number] for number in predicted_category_numbers
        ]
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
        if len(input_data) == 1:
            return [self.categorize(next(iter(input_data)))]

        input_data_prompt = "##".join(input_data)
        # Call the LLM provider to get the predicted category numbers
        response = self.llm_provider.generate(
            self.system_message_batch, input_data_prompt
        )
        # Split the response into category number sets for each input data
        category_number_sets = response.split("##")

        if len(category_number_sets) != len(input_data):
            raise ValueError(
                f"Number of predicted category sets ({len(category_number_sets)}) does not match the number of input data ({len(input_data)})."
            )

        predicted_categories_list = []
        for category_number_set in category_number_sets:
            predicted_category_numbers = [
                int(category.strip()) for category in category_number_set.split("||")
            ]
            predicted_categories = [
                self.category_mapping[number] for number in predicted_category_numbers
            ]
            predicted_categories_str = ",".join(
                self.validate_predicted_categories(predicted_categories)
            )
            predicted_categories_list.append(predicted_categories_str)

        return predicted_categories_list
