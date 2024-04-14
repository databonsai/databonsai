# databonsai <img width="64" height="64" src="https://img.icons8.com/external-justicon-flat-justicon/64/external-bonsai-tree-justicon-flat-justicon.png" alt="external-bonsai-tree-justicon-flat-justicon"/>

[![PyPI version](https://badge.fury.io/py/databonsai.svg)](https://badge.fury.io/py/databonsai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/databonsai.svg)](https://pypi.org/project/databonsai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Clean &amp; curate your data with LLMs

databonsai is a Python library that uses LLMs to perform data cleaning tasks.

## Features

-   Suite of tools for data processing using LLMs including categorization,
    transformation, and decomposition
-   Validation of LLM outputs
-   Batch processing for token savings
-   Retry logic with exponential backoff for handling rate limits and transient
    errors

## Installation

```bash
pip install databonsai
```

Store your API keys on an .env file in the root of your project, or specify it
as an argument when initializing the provider.

```bash
OPENAI_API_KEY=xxx # if you use OpenAiProvider
ANTHROPIC_API_KEY=xxx # If you use AnthropicProvider
```

## Quickstart

### Categorization

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

### Dataframes & Lists (Save tokens with batching!)

If you have a pandas dataframe or list, use `apply_to_column_batch` for some
handy features:

-   batching saves tokens by not resending the schema each time
-   progress bar
-   returns the last successful index so you can resume from there, in case of
    any error (llm_provider already implements exponential backoff, but just in
    case)
-   modifies your output list in place, so you don't lose any progress

Use the method as such:

```python
success_idx = apply_to_column_batch(input_column, output_column, function, batch_size, start_idx)
```

Parameters:

-   `input_column`: The name of the column from which data will be read.
-   `output_column`: The name of the column to which data will be written.
-   `function`: The function to apply to each batch of data.
-   `batch_size`: The number of rows in each batch.
-   `start_idx`: The starting index from which to begin processing.

Returns:

-   `success_idx`: The index of the last successful row processed.

(Continued from the previous code example)

```python
from databonsai.utils import apply_to_column_batch, apply_to_column
import pandas as pd

headlines = [
    "Massive Blizzard Hits the Northeast, Thousands Without Power",
    "Local High School Basketball Team Wins State Championship After Dramatic Final",
    "Celebrated Actor Launches New Environmental Awareness Campaign",
    "President Announces Comprehensive Plan to Combat Cybersecurity Threats",
    "Tech Giant Unveils Revolutionary Quantum Computer",
    "Tropical Storm Alina Strengthens to Hurricane as It Approaches the Coast",
    "Olympic Gold Medalist Announces Retirement, Plans Coaching Career",
    "Film Industry Legends Team Up for Blockbuster Biopic",
    "Government Proposes Sweeping Reforms in Public Health Sector",
    "Startup Develops App That Predicts Traffic Patterns Using AI",
]
df = pd.DataFrame(headlines, columns=["Headline"])
df["Category"] = None # Initialize it if it doesn't exist, as we modify it in place
success_idx = apply_to_column_batch( df["Headline"], df["Category"], categorizer.categorize_batch, batch_size=3, start_idx=0)
```

By default, exponential backoff is used to handle rate limiting. This is handled
in the LLM providers and can be configured.

If it fails midway (even after exponential backoff), you can resume from the
last successful index + 1.

```python
success_idx = apply_to_column_batch( df["Headline"], df["Category"], categorizer.categorize_batch, batch_size=10, start_idx=success_idx+1)
```

This also works for regular python lists.

Note that the better the LLM model, the greater the batch_size you can use
(depending on the length of your inputs). If you're getting errors, reduce the
batch_size, or use a better LLM model.

To use it without batching:

```python
success_idx = apply_to_column( df["Headline"], df["Category"], categorizer.categorize)
```

### View token usage

Token usage is recorded for each provider. Use these to estimate your costs!

```
print(provder.input_tokens)
print(provder.output_tokens)
```

## [Docs](./docs/)

### Tools (Check out the docs for usage examples and details)

-   [BaseCategorizer](./docs/BaseCategorizer.md) - categorize data into a
    category
-   [MultiCategorizer](./docs/MultiCategorizer.md) - categorize data into
    multiple categories
-   [BaseTransformer](./docs/BaseTransformer.md) - transform data with a prompt
-   [DecomposeTransformer](./docs/DecomposeTransformer.md) - decompose data into
    a structured format based on a schema

### LLM Providers

-   [OpenAIProvider](./docs/OpenAIProvider.md) - OpenAI
-   [AnthropicProvider](./docs/AnthropicProvider.md) - Anthropic
-   CustomProvider (TBD)

### Examples (TBD)

-   [Examples](./databonsai/examples/) (TBD)

### Acknowledgements

Bonsai icon from icons8 https://icons8.com/icon/74uBtdDr5yFq/bonsai
