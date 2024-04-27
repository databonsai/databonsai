# MultiCategorizer

The `MultiCategorizer` class is an extension of the `BaseCategorizer` class,
providing functionality for categorizing input data into multiple categories
using a specified LLM provider. This class overrides the `categorize` and
`categorize_batch` methods to enable the prediction of multiple categories for a
given input.

## Features

-   **Multiple Category Prediction**: Categorize input data into one or more
    predefined categories.
-   **Subset Validation**: Ensures that the predicted categories are a subset of
    the provided categories.
-   **Batch Categorization**: Categorizes multiple inputs simultaneously for
    token savings.

## Attributes

-   `categories` (Dict[str, str]): A dictionary mapping category names to their
    descriptions. This structure allows for a clear definition of possible
    categories for classification.
-   `llm_provider` (LLMProvider): An instance of an LLM provider to be used for
    categorization.
-   `examples` (Optional[List[Dict[str, str]]]): A list of example inputs and
    their corresponding categories to improve categorization accuracy. If there
    are multiple categories for an example, they should be separated by commas.
-   `strict` (bool): If True, raises an error when the predicted category is not
    one of the provided categories.

## Computed Fields

-   `system_message` (str): A system message used for single input
    categorization based on the provided categories and examples.
-   `system_message_batch` (str): A system message used for batch input
    categorization based on the provided categories and examples.
-   `category_mapping` (Dict[int, str]): Mapping of category index to category
    name
-   `inverse_category_mapping` (Dict[str, int]): Mapping of category name to
    index

## Methods

### `categorize`

Categorizes the input data into multiple categories using the specified LLM
provider.

#### Arguments

-   `input_data` (str): The text data to be categorized.

#### Returns

-   `str`: A string of categories for the input data, separated by commas.

#### Raises

-   `ValueError`: If the predicted categories are not a subset of the provided
    categories.

### `categorize_batch`

Categorizes a batch of input data into multiple categories using the specified
LLM provider. For less advanced LLMs, call this method on batches of 3-5 inputs
(depending on the length of the input data).

#### Arguments

-   `input_data` (List[str]): A list of text data to be categorized.

#### Returns

-   `List[str]`: A list of predicted categories for each input data. If there
    are multiple categories for an input, they will be separated by commas.

#### Raises

-   `ValueError`: If the predicted categories are not a subset of the provided
    categories or if the number of predicted category sets does not match the
    number of input data.

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
    examples=[
        {
            "example": "Big stormy skies over city causes league football game to be cancelled",
            "response": "Weather,Sports",
        },
        {
            "example": "Elon musk likes to play golf",
            "response": "Sports,Celebrities",
        },
    # strict=False, # Default true, set to False to allow for categories not in the provided
    ],
)
categories = tagger.categorize(
    "It's been raining outside all day, and I saw Elon Musk. 13rewfdsacw10289u(#!*@)"
)
print(categories)
```

Output:

```python
['Weather', 'Celebrities', 'Anomaly']
```

Categorize a list of data:

```python
mixed_headlines = [
    "Storm Delays Government Budget Meeting, Weather and Politics Clash",
    "Olympic Star's Controversial Tweets Ignite Political Debate, Sports Meets Politics",
    "Local Football Hero Opens New Gym, Sports and Business Combine",
    "Tech CEO's Groundbreaking Climate Initiative, Technology and Environment at Forefront",
    "Celebrity Chef Fights for Food Security Legislation, Culinary Meets Politics",
    "Hollywood Biopic of Legendary Athlete Set to Premiere, Blending Sports and Cinema",
    "Massive Flooding Disrupts Local Elections, Intersection of Weather and Politics",
    "Tech Billionaire Invests in Sports Teams, Merging Business with Athletics",
    "Pop Star's Concert Raises Funds for Disaster Relief, Combining Music with Charity",
    "Film Festival Highlights Environmental Documentaries, Merging Cinema and Green Activism",
]
categories = tagger.categorize_batch(mixed_headlines)
```

Output:

```python
['Weather,Politics', 'Sports,Politics', 'Sports,Others', 'Tech', 'Politics,Celebrities', 'Sports,Celebrities', 'Weather,Politics', 'Tech,Sports', 'Celebrities,Others', 'Celebrities,Others']
```

Categorize a long list of data, or a dataframe column (with batching):

```python
from databonsai.utils import apply_to_column_batch, apply_to_column

categories = []
success_idx = apply_to_column_batch(
    input_column=mixed_headlines,
    output_column=categories,
    function=tagger.categorize_batch,
    batch_size=3,
    start_idx=0
)
```

Without batching:

```python
categories = []
success_idx = apply_to_column(
    input_column=mixed_headlines,
    output_column=categories,
    function=tagger.categorize
)
```
