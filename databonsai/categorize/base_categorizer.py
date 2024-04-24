from typing import Dict, List, Optional
from pydantic import BaseModel, field_validator, model_validator, computed_field
from databonsai.llm_providers import OpenAIProvider, LLMProvider
from databonsai.utils.logs import logger


class BaseCategorizer(BaseModel):
    """
    A base class for categorizing input data using a specified LLM provider.

    Attributes:
        categories (Dict[str, str]): A dictionary mapping category names to their descriptions.
        llm_provider (LLMProvider): An instance of an LLM provider to be used for categorization.

    """

    categories: Dict[str, str]
    llm_provider: LLMProvider
    examples: Optional[List[Dict[str, str]]] = []
    strict: bool = True

    class Config:
        arbitrary_types_allowed = True

    @field_validator("categories")
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
        Validates that the "response" values in the examples are within the categories keys.
        """

        if self.examples:
            category_keys = set(self.categories.keys())
            for example in self.examples:
                response = example.get("response")
                if response not in category_keys:
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
        Classify the given text snippet into one of the following categories:
        {str(list(self.categories.keys()))}
        Do not use any other categories.
        Only reply with the category. Do not make any other conversation.
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
        Each category is formatted as <category>: <description of data that fits the category>
        {str(self.categories)}
        Classify each given text snippet into one of the following categories:
        {str(list(self.categories.keys()))}.
         Do not use any other categories. If there are multiple snippets, separate each category with ||. 
        EXAMPLE: <Text about category 1>  RESPONSE: <Category 1> 
        EXAMPLE: Content 1: <Text about category 1>, Content 2: <Text about category 2> RESPONSE: <Category 1> || <Category 2>
        Choose one category for each text snippet.
        Only reply with the categories. Do not make any other conversation.
        """

        # Add in fewshot examples
        if self.examples:
            system_message += "\n EXAMPLE:"
            for idx, example in enumerate(self.examples):
                system_message += f"Content {idx+1}: {example['example']}, "
            system_message += f"\n RESPONSE: {'||'.join([example['response'] for example in self.examples])}"

        return system_message

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
        # Call the LLM provider to get the predicted category
        response = self.llm_provider.generate(self.system_message, input_data)
        predicted_category = response.strip()

        # Validate that the predicted category is one of the provided categories
        if predicted_category not in self.categories:
            if self.strict:
                raise ValueError(
                    f"Predicted category '{predicted_category}' is not one of the provided categories. Use 'strict=False' when instantiating the categorizer to allow categories not in the categories dict."
                )
            else:
                logger.warning(
                    f"Predicted category '{predicted_category}' is not one of the provided categories. Use 'strict=True' when instantiating the categorizer to raise an error."
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
        # If there is only one input, call the categorize method
        if len(input_data) == 1:
            return self.validate_predicted_categories([self.categorize(input_data[0])])
        # Call the LLM provider to get the predicted category
        response = self.llm_provider.generate_batch(
            self.system_message_batch, input_data
        )
        predicted_categories = [category.strip() for category in response.split("||")]
        if len(predicted_categories) != len(input_data):
            raise ValueError(
                f"Number of predicted categories ({len(predicted_categories)}) does not match the number of input data ({len(input_data)})."
            )
        return self.validate_predicted_categories(predicted_categories)

    def validate_predicted_categories(
        self, predicted_categories: List[str]
    ) -> List[str]:
        # Filter out empty strings from the predicted categories
        filtered_categories = [
            category for category in predicted_categories if category
        ]

        # Validate each category in the filtered list
        for predicted_category in filtered_categories:
            if predicted_category not in self.categories:
                if self.strict:
                    raise ValueError(
                        f"Predicted category '{predicted_category}' is not one of the provided categories. Use 'strict=False' when instantiating the categorizer to allow categories not in the categories dict."
                    )
                else:
                    # Warn the user if the predicted category is not one of the provided categories
                    logger.warning(
                        f"Predicted category '{predicted_category}' is not one of the provided categories. Use 'strict=True' when instantiating the categorizer to raise an error."
                    )
        return filtered_categories
