# ExtractTransformer

The `ExtractTransformer` class extends the `BaseTransformer` class and overrides
the `transform` method to extract a given schema from the input data into a list
of dictionaries. It allows for transforming input data into a structured format
according to a specified schema.

## Features

-   **Custom Output Schema**: Define your own output schema to structure the
    transformed data.
-   **Input Validation**: Ensures the integrity of the output schema, examples,
    and input data for reliable transformation.
-   **Few-Shot Learning**: Supports providing example inputs and responses to
    improve transformation accuracy.

## Attributes

-   `output_schema` (Dict[str, str]): A dictionary representing the schema of
    the output dictionaries. It defines the expected keys and their
    corresponding value types in the transformed data.
-   `examples` (Optional[List[Dict[str, str]]]): A list of example inputs and
    their corresponding extracted outputs.

## Computed Fields

-   `system_message` (str): A system message used for single input
    transformation based on the provided prompt, output schema, and examples.

## Methods

### `transform`

Transforms the input data into a list of dictionaries using the specified LLM
provider.

#### Arguments

-   `input_data` (str): The text data to be transformed.
-   `max_tokens` (int, optional): The maximum number of tokens to generate in
    the response. Defaults to 1000.

#### Returns

-   `List[Dict[str, str]]`: The transformed data as a list of dictionaries.

#### Raises

-   `ValueError`: If the transformed data does not match the expected format or
    schema.

## Usage

Prepare a Extract transformer with a prompt and output schema:

```python
from databonsai.llm_providers import OpenAIProvider
from databonsai.transform import ExtractTransformer

output_schema = {
    "question": "generated question about given information",
    "answer": "answer to the question, only using information from the given data",
}

qna = ExtractTransformer(
    prompt="Your goal is to create a set of questions and answers to help a person memorise every single detail of a document.",
    output_schema=output_schema,
    llm_provider=OpenAIProvider(),
    examples=[
        {
            "example": "Bananas are naturally radioactive due to their potassium content. They contain potassium-40, a radioactive isotope of potassium, which contributes to a tiny amount of radiation in every banana.",
            "response": str(
                [
                    {
                        "question": "Why are bananas naturally radioactive?",
                        "answer": "Bananas are naturally radioactive due to their potassium content.",
                    },
                    {
                        "question": "What is the radioactive isotope of potassium in bananas?",
                        "answer": "The radioactive isotope of potassium in bananas is potassium-40.",
                    },
                ]
            ),
        }
    ],
)
```

Here's the text we want to extract questions and answers from:

```python
text = """ Sky-gazers across North America are in for a treat on April 8 when a total solar eclipse will pass over Mexico, the United States and Canada.

The event will be visible to millions — including 32 million people in the US alone — who live along the route the moon’s shadow will travel during the eclipse, known as the path of totality. For those in the areas experiencing totality, the moon will appear to completely cover the sun. Those along the very center line of the path will see an eclipse that lasts between 3½ and 4 minutes, according to NASA.

The next total solar eclipse won’t be visible across the contiguous United States again until August 2044. (It’s been nearly seven years since the “Great American Eclipse” of 2017.) And an annular eclipse won’t appear across this part of the world again until 2046."""

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

Batching is not supported for ExtractTransformer yet.
