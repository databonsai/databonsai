import pytest
from databonsai.categorize import BaseCategorizer, MultiCategorizer
from databonsai.llm_providers import OpenAIProvider, AnthropicProvider
from databonsai.utils import apply_to_column, apply_to_column_batch
import pandas as pd


@pytest.fixture
def sample_categories():
    return {
        "Weather": "Insights and remarks about weather conditions.",
        "Sports": "Observations and comments on sports events.",
        "Celebrities": "Celebrity sightings and gossip",
        "Others": "Comments do not fit into any of the above categories",
        "Anomaly": "Data that does not look like comments or natural language",
    }


@pytest.fixture(
    params=[
        OpenAIProvider(model="gpt-4-turbo", max_tries=1),
        AnthropicProvider(max_tries=1),
    ]
)
def sample_provider(request):
    return request.param


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "text": [
                "Massive Blizzard Hits the Northeast, Thousands Without Power",
                "Local High School Basketball Team Wins State Championship After Dramatic Final",
                "Celebrated Actor Launches New Environmental Awareness Campaign",
                "Startup Develops App That Predicts Traffic Patterns Using AI",
                "asdfoinasedf'awesdf",
            ]
        }
    )


@pytest.fixture
def sample_list():
    return [
        "Massive Blizzard Hits the Northeast, Thousands Without Power",
        "Local High School Basketball Team Wins State Championship After Dramatic Final",
        "Celebrated Actor Launches New Environmental Awareness Campaign",
        "Startup Develops App That Predicts Traffic Patterns Using AI",
        "asdfoinasedf'awesdf",
    ]


def test_base_categorizer(sample_categories, sample_provider):
    """
    Test the BaseCategorizer class.
    """
    categorizer = BaseCategorizer(
        categories=sample_categories,
        llm_provider=sample_provider,
        examples=[
            {"example": "Big stormy skies over city", "response": "Weather"},
            {"example": "The team won the championship", "response": "Sports"},
            {"example": "I saw a famous rapper at the mall", "response": "Celebrities"},
        ],
    )

    assert categorizer.categorize("It's raining heavily today.") == "Weather"
    assert categorizer.categorize("The football match was exciting!") == "Sports"
    assert categorizer.categorize("I saw Emma Watson at the mall.") == "Celebrities"
    assert categorizer.categorize("This is a random comment.") == "Others"
    assert categorizer.categorize("1234567890!@#$%^&*()") == "Anomaly"


def test_base_categorizer_batch(sample_categories, sample_provider, sample_list):
    """
    Test the BaseCategorizer class with a batch of examples.
    """
    categorizer = BaseCategorizer(
        categories=sample_categories,
        llm_provider=sample_provider,
        examples=[
            {"example": "Big stormy skies over city", "response": "Weather"},
            {"example": "The team won the championship", "response": "Sports"},
            {"example": "I saw a famous rapper at the mall", "response": "Celebrities"},
        ],
    )
    print(categorizer.system_message_batch)
    assert categorizer.categorize_batch(sample_list) == [
        "Weather",
        "Sports",
        "Celebrities",
        "Others",
        "Anomaly",
    ]


def test_apply_to_column(sample_categories, sample_provider, sample_dataframe):
    """
    Test the apply_to_column function.
    """
    categorizer = BaseCategorizer(
        categories=sample_categories,
        llm_provider=sample_provider,
        examples=[
            {"example": "Big stormy skies over city", "response": "Weather"},
            {"example": "The team won the championship", "response": "Sports"},
            {"example": "I saw a famous rapper at the mall", "response": "Celebrities"},
        ],
    )

    df = sample_dataframe.copy()
    df["category"] = None

    success_idx = apply_to_column(df["text"], df["category"], categorizer.categorize)

    assert success_idx == 5
    assert df["category"].tolist() == [
        "Weather",
        "Sports",
        "Celebrities",
        "Others",
        "Anomaly",
    ]


def test_apply_to_column_batch(sample_categories, sample_provider, sample_dataframe):
    """
    Test the apply_to_column_batch function.
    """
    categorizer = BaseCategorizer(
        categories=sample_categories,
        llm_provider=sample_provider,
        examples=[
            {"example": "Big stormy skies over city", "response": "Weather"},
            {"example": "The team won the championship", "response": "Sports"},
            {"example": "I saw a famous rapper at the mall", "response": "Celebrities"},
        ],
    )

    df = sample_dataframe.copy()
    df["category"] = None

    success_idx = apply_to_column_batch(
        df["text"], df["category"], categorizer.categorize_batch, batch_size=2
    )

    assert success_idx == 5
    assert df["category"].tolist() == [
        "Weather",
        "Sports",
        "Celebrities",
        "Others",
        "Anomaly",
    ]


def test_apply_to_column_batch_start_idx(
    sample_categories, sample_provider, sample_dataframe
):
    """
    Test the apply_to_column_batch function with a start index.
    """
    categorizer = BaseCategorizer(
        categories=sample_categories,
        llm_provider=sample_provider,
        examples=[
            {"example": "Big stormy skies over city", "response": "Weather"},
            {"example": "The team won the championship", "response": "Sports"},
            {"example": "I saw a famous rapper at the mall", "response": "Celebrities"},
        ],
    )

    df = sample_dataframe.copy()
    df["category"] = None

    success_idx = apply_to_column_batch(
        df["text"],
        df["category"],
        categorizer.categorize_batch,
        batch_size=2,
        start_idx=1,
    )

    assert success_idx == 5
    assert df["category"].tolist() == [
        None,
        "Sports",
        "Celebrities",
        "Others",
        "Anomaly",
    ]


def test_multi_categorizer(sample_categories, sample_provider):
    """
    Test the MultiCategorizer class.
    """
    categorizer = MultiCategorizer(
        categories=sample_categories,
        llm_provider=sample_provider,
        examples=[
            {
                "example": "Big stormy skies over city causes league football game to be cancelled",
                "response": "Weather,Sports",
            },
            {
                "example": "Elon musk likes to play golf",
                "response": "Sports,Celebrities",
            },
        ],
    )

    assert set(
        categorizer.categorize("It's raining and I saw Emma Watson.").split(",")
    ) == {
        "Weather",
        "Celebrities",
    }
    assert set(
        categorizer.categorize("The football match was exciting and it's sunny!").split(
            ","
        )
    ) == {"Sports", "Weather"}


def test_multi_categorizer_batch(sample_categories, sample_provider):
    """
    Test the MultiCategorizer class with a batch of examples.
    """
    categorizer = MultiCategorizer(
        categories=sample_categories,
        llm_provider=sample_provider,
        examples=[
            {
                "example": "Big stormy skies over city causes league football game to be cancelled",
                "response": "Weather,Sports",
            },
            {
                "example": "Elon musk likes to play golf",
                "response": "Sports,Celebrities",
            },
        ],
    )

    examples = [
        "Thunderstorms cause major delays in baseball tournament",
        "Famous actor spotted at local charity basketball game",
        "Heavy rainfall leads to postponement of soccer match",
    ]

    expected_output = [
        ["Weather", "Sports"],
        ["Celebrities", "Sports"],
        ["Weather", "Sports"],
    ]

    actual_output = categorizer.categorize_batch(examples)

    # Convert the actual output to sets for order-insensitive comparison
    actual_output_sets = [set(categories.split(",")) for categories in actual_output]

    # Convert the expected output to sets for order-insensitive comparison
    expected_output_sets = [set(categories) for categories in expected_output]

    assert len(actual_output_sets) == len(expected_output_sets)
    for actual_set, expected_set in zip(actual_output_sets, expected_output_sets):
        assert actual_set == expected_set
