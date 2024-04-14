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

### categorize_batch

Categorizes all of the input data using the specified LLM provider. Do not give
too large a list as input. Instead, use `apply_to_column_batch` to handle large
datasets.

#### Arguments

-   `input_data List[str]`: List of text data to be categorized

#### Returns

-   `List[str]`: The predicted categories for the input data.

#### Raises

-   `ValueError`: If the predicted category is not among the provided
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
success_idx = apply_to_column(
    input_column=mixed_headlines,
    output_column=categories,
    function=tagger.categorize
)
```
