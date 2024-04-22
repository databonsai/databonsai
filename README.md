# databonsai <img width="64" height="64" src="https://img.icons8.com/external-justicon-flat-justicon/64/external-bonsai-tree-justicon-flat-justicon.png" alt="external-bonsai-tree-justicon-flat-justicon"/>

[![PyPI version](https://badge.fury.io/py/databonsai.svg)](https://badge.fury.io/py/databonsai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/databonsai.svg)](https://pypi.org/project/databonsai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Clean &amp; curate your data with LLMs

databonsai is a Python library that uses LLMs to perform data cleaning tasks.

## Features

-   Suite of tools for data processing using LLMs including categorization,
    transformation, and extraction
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

provider = OpenAIProvider()  # Or AnthropicProvider(). Works best with gpt-4-turbo or any claude model
categories = {
    "Weather": "Insights and remarks about weather conditions.",
    "Sports": "Observations and comments on sports events.",
    "Politics": "Political events related to governments, nations, or geopolitical issues.",
    "Celebrities": "Celebrity sightings and gossip",
    "Others": "Comments do not fit into any of the above categories",
    "Anomaly": "Data that does not look like comments or natural language",
}
few_shot_examples = [
        {"example": "Big stormy skies over city", "response": "Weather"},
        {"example": "The team won the championship", "response": "Sports"},
        {"example": "I saw a famous rapper at the mall", "response": "Celebrities"},
    ]
```

Categorize your data:

```python
categorizer = BaseCategorizer(
    categories=categories,
    llm_provider=provider,
    examples = few_shot_examples

)
category = categorizer.categorize("It's been raining outside all day")
print(category)
```

Output:

```python
Weather
```

Use categorize_batch to categorize a batch. This saves tokens as it only sends
the schema and few shot examples once! (Works best for better models. Ideally,
use at least 3 few shot examples.)

```python
categories = categorizer.categorize_batch([
    "Massive Blizzard Hits the Northeast, Thousands Without Power",
    "Local High School Basketball Team Wins State Championship After Dramatic Final",
    "Celebrated Actor Launches New Environmental Awareness Campaign",
])
print(categories)
```

Output:

```python
['Weather', 'Sports', 'Celebrities']
```

### AutoBatch for Larger datasets

If you have a pandas dataframe or list, use `apply_to_column_autobatch`

-   Batching data for LLM api calls saves tokens by not sending the prompt for
    every row. However, too large a batch size / complex tasks can lead to
    errors. Naturally, the better the LLM model, the larger the batch size you
    can use.

-   This batching is handled adaptively (i.e., it will increase the batch size
    if the response is valid and reduce it if it's not, with a decay factor)

Other features:

-   Progress bar
-   Returns the last successful index so you can resume from there, in case it
    exceeds max_retries
-   Modifies your output list in place, so you don't lose any progress

Retry Logic:

-   LLM providers have retry logic built in for API related errors. This can be
    configured in the provider.
-   The retry logic in the apply_to_column_autobatch is for handling invalid
    responses (e.g. unexpected category, different number of outputs, etc.)

```python
from databonsai.utils import apply_to_column_batch, apply_to_column, apply_to_column_autobatch
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
success_idx = apply_to_column_autobatch( df["Headline"], df["Category"], categorizer.categorize_batch, batch_size=3, start_idx=0)
```

There are many more options available for autobatch, such as setting a
max_retries, decay factor, and more. Check [Utils](./docs/Utils.md) for more
details

If it fails midway (even after exponential backoff), you can resume from the
last successful index + 1.

```python
success_idx = apply_to_column_autobatch( df["Headline"], df["Category"], categorizer.categorize_batch, batch_size=10, start_idx=success_idx+1)
```

This also works for regular python lists.

Note that the better the LLM model, the greater the batch_size you can use
(depending on the length of your inputs). If you're getting errors, reduce the
batch_size, or use a better LLM model.

To use it with batching, but with a fixed batch size:

```python
success_idx = apply_to_column_batch( df["Headline"], df["Category"], categorizer.categorize_batch, batch_size=3, start_idx=0)
```

To use it without batching:

```python
success_idx = apply_to_column( df["Headline"], df["Category"], categorizer.categorize)
```

### View System Prompt

```python
print(categorizer.system_message)
print(categorizer.system_message_batch)
```

### View token usage

Token usage is recorded for OpenAI and Anthropic. Use these to estimate your
costs!

```python
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
-   [ExtractTransformer](./docs/ExtractTransformer.md) - Extract data into a
    structured format based on a schema
-   .. more coming soon!

### LLM Providers

-   [OpenAIProvider](./docs/OpenAIProvider.md) - OpenAI
-   [AnthropicProvider](./docs/AnthropicProvider.md) - Anthropic
-   [OllamaProvider](./docs/OllamaProvider.md) - Ollama
-   .. more coming soon!

### Examples

-   [Examples](./databonsai/examples/) (TBD)

### Acknowledgements

Bonsai icon from icons8 https://icons8.com/icon/74uBtdDr5yFq/bonsai
