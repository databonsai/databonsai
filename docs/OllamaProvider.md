# OllamaProvider

The `OllamaProvider` class is a provider class that interacts with Ollama's API
for generating text completions. Note that tokens are not counted for Ollama,
and there is no retry logic (since it's not needed).

## Initialization

The `__init__` method initializes the `OllamaProvider` with an optional Ollama
client or host, and retry parameters.

### Parameters

-   `model (str)`: The default model to use for text generation (default:
    "llama3").
-   `temperature (float)`: The temperature parameter for text generation
    (default: 0).
-   `host (str)`: The host URL for the Ollama API (optional).

## Methods

### `generate`

The `generate` method generates a text completion using Ollama's API, with a
given system prompt and a user prompt.

#### Parameters

-   `system_prompt (str)`: The system prompt to provide context or instructions
    for the generation.
-   `user_prompt (str)`: The user's prompt, based on which the text completion
    is generated.
-   `max_tokens (int)`: The maximum number of tokens to generate in the response
    (default: 1000).

#### Returns

-   `str`: The generated text completion.

### `generate_batch`

The `generate_batch` method generates a text completion using Ollama's API, with
a given system prompt and a list of user prompts.

#### Parameters

-   `system_prompt (str)`: The system prompt to provide context or instructions
    for the generation.
-   `user_prompts (List[str])`: The list of user prompts, based on which the
    text completion is generated.
-   `max_tokens (int)`: The maximum number of tokens to generate in the response
    (default: 1000).

#### Returns

-   `str`: The generated text completion.

## Usage

If you have a host URL for the Ollama API:

```python
from databonsai.llm_providers import OllamaProvider

provider = OllamaProvider(model="llama3")
```

or, provide a host

```python
provider = OllamaProvider(host="http://localhost:11434", model="llama3")
```

This uses ollama's python library under the hood.
