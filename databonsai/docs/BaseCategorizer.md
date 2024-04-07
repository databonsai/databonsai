# BaseCategorizer

The `BaseCategorizer` class provides functionality for categorizing input data
utilizing a specified LLM provider. This class serves as a base for implementing
categorization tasks where inputs are classified into predefined categories.

## Features

-   **Custom Categories**: Define your own categories for data classification.
-   **Input Validation**: Ensures the integrity of categories and input data for
    reliable categorization.

## Attributes

-   `categories`: A dictionary (`Dict[str, str]`) mapping category names to
    their descriptions. This structure allows for a clear definition of possible
    categories for classification.
-   `llm_provider`: An instance of `LLMProvider`. This is the model that
    performs the actual categorization based on the input data.

categorize Categorizes the input data using the specified LLM provider.

## Methods

### categorize

Categorizes the input data using the specified LLM provider.

#### Arguments

-   `input_data (str)`: The text data to be categorized.

#### Returns

-   `str`: The predicted category for the input data.

#### Raises

-   `ValueError`: If the predicted category is not among the provided
    categories.

## Usage

Setup the LLM provider and categories (as a dictionary)

```python
from databonsai.categorize import MultiCategorizer, BaseCategorizer
from databonsai.llm_providers import OpenAIProvider, AnthropicProvider

provider = OpenAIProvider()  # Or AnthropicProvider()
categories = {
    "Weather": "Insights and remarks about weather conditions.",
    "Sports": "Observations and comments on sports events.",
    "Celebrities": "Celebrity sightings and gossip",
    "Others": "Comments do not fit into any of the above categories",
    "Anomaly": "Data that does not look like comments or natural language",
}
```

Categorize your data:

```python
categorizer = BaseCategorizer(
    categories=categories,
    llm_provider=provider,
)
category = categorizer.categorize("It's been raining outside all day")
print(category)
```

Output:

```python
Weather
```
