## Util Methods

### `apply_to_column`

Applies a function to each value in a column of a DataFrame or a normal Python
list, starting from a specified index.

#### Arguments

-   `input_column` (List): The column of the DataFrame or a normal Python list
    to which the function will be applied.
-   `output_column` (List): A list where the processed values will be stored.
    The function will mutate this list in-place.
-   `func` (Callable): The function to apply to each value in the column. It
    should take a single value as input and return a single value.
-   `start_idx` (int, optional): The index from which to start applying the
    function. Default is 0.

#### Returns

-   `int`: The index of the last successfully processed value.

#### Raises

-   `ValueError`: If the input or output column conditions are not met or if the
    starting index is out of bounds.

### `apply_to_column_batch`

Applies a function to batches of values in a column of a DataFrame or a normal
Python list, starting from a specified index.

#### Arguments

-   `input_column` (List): The column of the DataFrame or a normal Python list
    to which the function will be applied.
-   `output_column` (List): A list where the processed values will be stored.
    The function will mutate this list in-place.
-   `batch_func` (Callable): The batch function to apply to each batch of values
    in the column. It should take a list of values as input and return a list of
    processed values.
-   `batch_size` (int, optional): The size of each batch. Default is 5.
-   `start_idx` (int, optional): The index from which to start applying the
    function. Default is 0.

#### Returns

-   `int`: The index of the last successfully processed batch.

#### Raises

-   `ValueError`: If the input or output column conditions are not met or if the
    starting index or batch sizes are out of bounds.

### `apply_to_column_autobatch`

Applies a function to the input column using adaptive batch processing, starting
from a specified index and adjusting batch sizes based on success or failure.

#### Arguments

-   `input_column` (List): The input column to be processed.
-   `output_column` (List): The list where the processed results will be stored.
-   `batch_func` (Callable): The batch function used for processing.
-   `max_retries` (int): The maximum number of retries for failed batches.
-   `max_batch_size` (int): The maximum allowed batch size.
-   `batch_size` (int): The initial batch size.
-   `ramp_factor` (float): The factor by which the batch size is increased after
    a successful batch.
-   `ramp_factor_decay` (float): The decay rate for the ramp factor after each
    successful batch.
-   `reduce_factor` (float): The factor by which the batch size is reduced after
    a failed batch.
-   `reduce_factor_decay` (float): The decay rate for the reduce factor after
    each failed batch.
-   `start_idx` (int): The index from which to start processing the input
    column.

#### Returns

-   `int`: The index of the last successfully processed item in the input
    column.

#### Raises

-   `ValueError`: If the input or output column conditions are not met or if
    processing fails despite retries.

## Usage:

### AutoBatch for Larger datasets

If you have a pandas dataframe or list, use `apply_to_column_autobatch`

-   Batching data for LLM api calls saves tokens by not sending the prompt for
    every row. However, too large a batch size / complex tasks can lead to
    errors. Naturally, the better the LLM model, the larger the batch size you
    can use.

-   This batching is handled adaptively (i.e., it will increase the batch size
    if the response is valid and reduce it if it's not, with a decay factor)

Other features:

-   progress bar
-   returns the last successful index so you can resume from there, in case it
    exceeds max_retries
-   modifies your output list in place, so you don't lose any progress

Retry Logic:

-   LLM providers have retry logic built in for API related errors. This can be
    configured in the provider.
-   The retry logic in the apply_to_column_autobatch is for handling invalid
    responses (e.g. unexpected category, different number of outputs, etc.)

```python
from databonsai.utils import (
    apply_to_column_batch,
    apply_to_column,
    apply_to_column_autobatch,
)
import pandas as pd

from databonsai.categorize import MultiCategorizer, BaseCategorizer
from databonsai.llm_providers import OpenAIProvider, AnthropicProvider

provider = OpenAIProvider()
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
categorizer = BaseCategorizer(
    categories=categories, llm_provider=provider, examples=few_shot_examples
)
category = categorizer.categorize("It's been raining outside all day")
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
df["Category"] = None  # Initialize it if it doesn't exist, as we modify it in place
success_idx = apply_to_column_autobatch(
    df["Headline"],
    df["Category"],
    categorizer.categorize_batch,
    batch_size=3,
    start_idx=0,
)
```

There are many more options available for autobatch, such as setting a
max_retries, decay factor, and more. Check the docs for more details.

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
