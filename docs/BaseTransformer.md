# BaseTransformer

The `BaseTransformer` class is a base class for transforming input data using a
specified LLM provider. It provides a foundation for implementing data
transformation tasks using language models.

## Features

-   **Transformation Prompts**: Define your own prompts to guide the
    transformation process.
-   **Input Validation**: Ensures the integrity of prompts, examples, and input
    data for reliable transformation.
-   **Few-Shot Learning**: Supports providing example inputs and responses to
    improve transformation accuracy.
-   **Batch Transformation**: Transforms multiple inputs simultaneously for
    token savings.

## Attributes

-   `prompt` (str): The prompt used to guide the transformation process. It
    provides instructions or context for the LLM provider to perform the desired
    transformation.
-   `llm_provider` (LLMProvider): An instance of an LLM provider to be used for
    transformation. The LLM provider is responsible for generating the
    transformed data based on the input data and the provided prompt.
-   `examples` (Optional[List[Dict[str, str]]]): A list of example inputs and
    their corresponding transformed outputs to improve transformation accuracy.

## Computed Fields

-   `system_message` (str): A system message used for single input
    transformation based on the provided prompt and examples.
-   `system_message_batch` (str): A system message used for batch input
    transformation based on the provided prompt and examples.

## Methods

### `transform`

Transforms the input data using the specified LLM provider.

#### Arguments

-   `input_data` (str): The text data to be transformed.
-   `max_tokens` (int, optional): The maximum number of tokens to generate in
    the response. Defaults to 1000.

#### Returns

-   `str`: The transformed data.

### `transform_batch`

Transforms a batch of input data using the specified LLM provider.

#### Arguments

-   `input_data` (List[str]): A list of text data to be transformed.
-   `max_tokens` (int, optional): The maximum number of tokens to generate in
    each response. Defaults to 1000.

#### Returns

-   `List[str]`: A list of transformed data, where each element corresponds to
    the transformed version of the respective input data.

#### Raises

-   `ValueError`: If the length of the output list does not match the length of
    the input list.

## Usage

Prepare the transformer:

```python
from databonsai.llm_providers import OpenAIProvider
from databonsai.transform import BaseTransformer

pii_remover = BaseTransformer(
    prompt="Replace any Personal Identity Identifiers (PII) in the given text with <type of PII>. PII includes any information that can be used to identify an individual, such as names, addresses, phone numbers, email addresses, social security numbers, etc.",
    llm_provider=AnthropicProvider(),
    examples=[
        {
            "example": "My name is John Doe and my phone number is (555) 123-4567.",
            "response": "My name is <Name> and my phone number is <Phone number>.",
        },
        {
            "example": "My email address is johndoe@gmail.com.",
            "response": "My email address is <Email address>.",
        },
    ],
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

Transforma a list of data: (use apply_to_column_batch for large datasets)

```python
pii_texts = [
    "Just confirmed the reservation for Amanda Clark, Passport No. B2345678, traveling to Tokyo on May 15, 2024.",
    "Received payment from Michael Thompson, Credit Card ending in 4547, Billing Address: 45 Westview Lane, Springfield, IL.",
    "Application received from Julia Martinez, DOB 03/19/1994, SSN 210-98-7654, applying for the marketing position.",
    "Lease agreement finalized for Henry Wilson, Tenant ID: WILH12345, property at 89 Riverside Drive, Brooklyn, NY.",
    "Registration details for Lucy Davis, Student ID 20231004, enrolled in Advanced Chemistry, Fall semester.",
    "David Lee called about his insurance claim, Policy #9988776655, regarding the accident on April 5th.",
    "Booking confirmation for Sarah H. Richards, Flight AC202, Seat 14C, Frequent Flyer #GH5554321, departing June 12, 2024.",
    "Kevin Brown's gym membership has been renewed, Member ID: 654321, Phone: (555) 987-6543, Email: kbrown@example.com.",
    "Prescription ready for Emma Thomas, Health ID 567890123, prescribed by Dr. Susan Hill on April 10th, 2024.",
    "Alice Johnson requested a copy of her employment contract, Employee No. 112233, hired on August 1st, 2023.",
]

cleaned_texts = pii_remover.transform_batch(pii_texts)
print(cleaned_texts)
```

Output:

```python
['Just confirmed the reservation for <type of PII>, Passport No. <type of PII>, traveling to Tokyo on May 15, 2024.', 'Received payment from <type of PII>, Credit Card ending in <type of PII>, Billing Address: 45 Westview Lane, Springfield, IL.', 'Application received from <type of PII>, DOB 03/19/1994, SSN <type of PII>, applying for the marketing position.', 'Lease agreement finalized for <type of PII>, Tenant ID: WILH12345, property at 89 Riverside Drive, Brooklyn, NY.', 'Registration details for <type of PII>, Student ID 20231004, enrolled in Advanced Chemistry, Fall semester.', '<type of PII> called about his insurance claim, Policy #9988776655, regarding the accident on April 5th.', 'Booking confirmation for <type of PII>, Flight AC202, Seat 14C, Frequent Flyer #GH5554321, departing June 12, 2024.', "Kevin Brown's gym membership has been renewed, Member ID: 654321, Phone: (555) 987-6543, Email: kbrown@example.com.", 'Prescription ready for <type of PII>, Health ID 567890123, prescribed by Dr. Susan Hill on April 10th, 2024.', 'Alice Johnson requested a copy of her employment contract, Employee No. 112233, hired on August 1st, 2023.']
```

Transform a long list of data or a dataframe column (with batching):

```python
from databonsai.utils import apply_to_column_batch, apply_to_column

cleaned_texts = []
success_idx = apply_to_column_batch(
    pii_texts, cleaned_texts, pii_remover.transform_batch, 4, 0
)
```

Without batching:

```
cleaned_texts = []
success_idx = apply_to_column(
    pii_texts, cleaned_texts, pii_remover.transform
)
```
