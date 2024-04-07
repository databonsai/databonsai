# databonsai <img width="64" height="64" src="https://img.icons8.com/external-justicon-flat-justicon/64/external-bonsai-tree-justicon-flat-justicon.png" alt="external-bonsai-tree-justicon-flat-justicon"/>

[![PyPI version](https://badge.fury.io/py/databonsai.svg)](https://badge.fury.io/py/databonsai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/databonsai.svg)](https://pypi.org/project/databonsai/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Clean &amp; curate your data with LLMs

databonsai is a Python library that uses LLMs to perform data cleaning tasks.

## Features

-   Categorization / Transformation of data using LLMs
-   Decomposition of data into structured formats using LLMs
-   Validation of LLM outputs
-   Retry logic with exponential backoff for handling rate limits and transient
    errors

## Installation

```bash
pip install databonsai
```

Store your API keys on an .env file in the root of your project.

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

Multiple categories can also be returned. This is useful for tagging data!

```python
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

### Transformation

Prepare the transformer:

```python
pii_remover = BaseTransformer(
    prompt="Replace any Personal Identity Identifiers (PII) in the given text with <type of PII>. PII includes any information that can be used to identify an individual, such as names, addresses, phone numbers, email addresses, social security numbers, etc.",
    llm_provider=provider,
)
```

Run the transformation:

```python
print(
    pii_remover.transform(
        "John Doe, residing at 1234 Maple Street, Anytown, CA, 90210, recently contacted customer support to report an issue. He provided his phone number, (555) 123-4567, and email address, johndoe@email.com, for follow-up communication."
    )
)
```

Output:

```python
<Name>, residing at <Address>, <City>, <State>, <ZIP code>, recently contacted customer support to report an issue. They provided their phone number, <Phone number>, and email address, <Email address>, for follow-up communication.
```

### Dataframes

If you have a pandas dataframe, you can simple run the following.

```python
df['category'] = df['content'].apply(categorizer.categorize)
```

```python
df['transformed'] = df['content'].apply(language_check.transform)
```

The library uses exponential backoff to handle rate limiting. Go to
docs/llm_providers.md for more information on how to handle rate limiting.

### Decomposition

Prepare a decompose transformer with a prompt and output schema.

```python
output_schema = {
    "question": "generated question about given information",
    "answer": "answer to the question, only using information from the given data",
}

qna = DecomposeTransformer(
    prompt="Your goal is to create a set of questions and answers to help a person memorise every single detail of a document.",
    output_schema=output_schema,
    llm_provider=provider,
)
```

Here's the text we want to decompose:

```python
text = """ Sky-gazers across North America are in for a treat on April 8 when a total solar eclipse will pass over Mexico, the United States and Canada.

The event will be visible to millions — including 32 million people in the US alone — who live along the route the moon’s shadow will travel during the eclipse, known as the path of totality. For those in the areas experiencing totality, the moon will appear to completely cover the sun. Those along the very center line of the path will see an eclipse that lasts between 3½ and 4 minutes, according to NASA.

The next total solar eclipse won’t be visible across the contiguous United States again until August 2044. (It’s been nearly seven years since the “Great American Eclipse” of 2017.) And an annular eclipse won’t appear across this part of the world again until 2046."""
```

Decompose the text:

```python
print(qna.transform(text))
```

Output:

```python
[
    {
        "question": "When will the total solar eclipse pass over Mexico, the United States, and Canada?",
        "answer": "The total solar eclipse will pass over Mexico, the United States, and Canada on April 8.",
    },
    {
        "question": "What is the path of totality?",
        "answer": "The path of totality is the route the moon's shadow will travel during the eclipse where the moon will appear to completely cover the sun.",
    },
    {
        "question": "How long will the eclipse last for those along the very center line of the path of totality?",
        "answer": "For those along the very center line of the path of totality, the eclipse will last between 3½ and 4 minutes.",
    },
    {
        "question": "When will the next total solar eclipse be visible across the contiguous United States?",
        "answer": "The next total solar eclipse visible across the contiguous United States will be in August 2044.",
    },
    {
        "question": "When will an annular eclipse next appear across the contiguous United States?",
        "answer": "An annular eclipse won't appear across the contiguous United States again until 2046.",
    },
]
```

### View token usage

Token usage is recorded for each provider. Use these to estimate your costs!

```
print(provder.input_tokens)
print(provder.output_tokens)
```

### Read More:

-   [Documentation](./databonsai/docs/)
-   [Examples](./databonsai/examples/) (TBD)

### Acknowledgements

Bonsai icon from icons8 https://icons8.com/icon/74uBtdDr5yFq/bonsai
