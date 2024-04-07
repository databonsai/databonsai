# BaseTransformer

The `BaseTransformer` class is a base class for transforming input data using a
specified LLM provider. It provides a foundation for implementing data
transformation tasks using language models.

## Attributes

-   `prompt (str)`: The prompt used to guide the transformation process. It
    provides instructions or context for the LLM provider to perform the desired
    transformation.
-   `llm_provider (LLMProvider)`: An instance of an LLM provider to be used for
    transformation. The LLM provider is responsible for generating the
    transformed data based on the input data and the provided prompt.

## Methods

### `transform`

The `transform` method transforms the input data using the specified LLM
provider.

#### Arguments

-   `input_data (str)`: The text data to be transformed.
-   `max_tokens (int, optional)`: The maximum number of tokens to generate in
    the response. Defaults to 1000.

#### Returns

-   `str`: The transformed data.

## Validation

The `BaseTransformer` class includes a validator for the `prompt` attribute:

-   `validate_prompt(cls, v)`: Validates the prompt to ensure it is not empty.
    If the prompt is empty, a `ValueError` is raised.

## Usage

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
