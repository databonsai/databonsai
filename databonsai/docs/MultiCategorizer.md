# MultiCategorizer

The `MultiCategorizer` class is an extension of the `BaseCategorizer` class,
providing functionality for categorizing input data into multiple categories
using a specified LLM provider. This class overrides the `categorize` method to
enable the prediction of multiple categories for a given input.

## Features

-   **Multiple Category Prediction**: Categorize input data into one or more
    predefined categories.
-   **Subset Validation**: Ensures that the predicted categories are a subset of
    the provided categories.

## Methods

### `categorize`

Categorizes the input data into multiple categories using the specified LLM
provider.

#### Arguments

-   `input_data (str)`: The text data to be categorized.

#### Returns

-   `List[str]`: A list of predicted categories for the input data.

#### Raises

-   `ValueError`: If the predicted categories are not a subset of the provided
    categories.

## Usage

Setup the LLM provider and categories (as a dictionary):

```python
from databonsai.categorize import MultiCategorizer
from databonsai.llm_providers import OpenAIProvider, AnthropicProvider

provider = OpenAIProvider()  # Or AnthropicProvider()
categories = {
   "Weather": "Insights and remarks about weather conditions.",
   "Sports": "Observations and comments on sports events.",
   "Celebrities": "Celebrity sightings and gossip",
   "Others": "Comments do not fit into any of the above categories",
   "Anomaly": "Data that does not look like comments or natural language",
}


tagger = MultiCategorizer(
    categories=categories,
    llm_provider=provider,
)

tags = tagger.categorize(
    "It's been raining outside all day, and I saw Elon Musk. 13rewfdsacw10289u(#!*@)"  # Data has anomalies
)
print(tags)
```

Output:

```python
['Weather', 'Celebrities', 'Anomaly']
```
