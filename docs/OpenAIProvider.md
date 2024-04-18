# OpenAIProvider

The `OpenAIProvider` class is a provider class that interacts with OpenAI's API
for generating text completions. It supports exponential backoff retries (from
tenacity's library) to handle temporary failures, which is particularly useful
when dealing with large datasets.

## Initialization

The `__init__` method initializes the `OpenAIProvider` with an API key and retry
parameters.

### Parameters

-   `api_key (str)`: OpenAI API key.
-   `multiplier (int)`: The multiplier for the exponential backoff in retries
    (default: 1).
-   `min_wait (int)`: The minimum wait time between retries (default: 1).
-   `max_wait (int)`: The maximum wait time between retries (default: 60).
-   `max_tries (int)`: The maximum number of attempts before giving up (default:
    10).
-   `model (str)`: The default model to use for text generation (default:
    "gpt-4-turbo").
-   `temperature (float)`: The temperature parameter for text generation
    (default: 0).

## Methods

### `generate`

The `generate` method generates a text completion using OpenAI's API, given a
system prompt and a user prompt. It is decorated with retry logic to handle
temporary failures.

#### Parameters

-   `system_prompt (str)`: The system prompt to provide context or instructions
    for the generation.
-   `user_prompt (str)`: The user's prompt, based on which the text completion
    is generated.
-   `max_tokens (int)`: The maximum number of tokens to generate in the response
    (default: 1000).
-   `json (bool)`: Whether to use OpenAI's JSON response format (default:
    False).

#### Returns

-   `str`: The generated text completion.

### `generate_batch`

The `generate_batch` method generates a text completion using OpenAI's API, with
a given system prompt and a list of user prompts. It is decorated with retry
logic to handle temporary failures.

#### Parameters

-   `system_prompt (str)`: The system prompt to provide context or instructions
    for the generation.
-   `user_prompts (List[str])`: The list of user prompts, based on which the
    text completion is generated.
-   `max_tokens (int)`: The maximum number of tokens to generate in the response
    (default: 1000).
-   `json (bool)`: Whether to use OpenAI's JSON response format (default:
    False).

#### Returns

-   `str`: The generated text completion.

## Retry Decorator

The `retry_with_exponential_backoff` decorator is used to apply retry logic with
exponential backoff to instance methods. It captures the `self` context to
access instance attributes for retry configuration.

## Usage

If your OPENAI_API_KEY is defined in .env:

```python
from databonsai.llm_providers import OpenAIProvider

provider = OpenAIProvider()

```

Or, provide the api key as an argument:

```python
provider = OpenAIProvider(api_key="your_openai_api_key")
```

Other parameters, for example:

```python
provider = OpenAIProvider(model="gpt-4-turbo", max_tries=5, max_wait=120)
```
