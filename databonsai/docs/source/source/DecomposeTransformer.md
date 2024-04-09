# DecomposeTransformer

The `DecomposeTransformer` class extends the `BaseTransformer` class and
overrides the `transform` method to decompose the input data into a list of
dictionaries based on a provided output schema. It allows for transforming input
data into a structured format according to a specified schema.

## Attributes

-   `output_schema (Dict[str, str])`: A dictionary representing the schema of
    the output dictionaries. It defines the expected keys and their
    corresponding value types in the transformed data.

## Methods

### `transform`

The `transform` method transforms the input data into a list of dictionaries
using the specified LLM provider.

#### Arguments

-   `input_data (str)`: The text data to be transformed.
-   `max_tokens (int, optional)`: The maximum number of tokens to generate in
    the response. Defaults to 1000.

#### Returns

-   `List[Dict[str, str]]`: The transformed data as a list of dictionaries.

#### Raises

-   `ValueError`: If the transformed data does not match the expected format or
    schema.

## Validation

The `DecomposeTransformer` class includes validators for the `output_schema`
attribute:

-   `validate_schema(cls, v)`: Validates the output schema to ensure it is not
    empty. If the schema dictionary is empty, a `ValueError` is raised.

## Usage

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
