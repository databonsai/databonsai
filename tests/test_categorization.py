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


@pytest.fixture
def sample_provider():
    return OpenAIProvider(model="gpt-3.5-turbo")  # or AnthropicProvider()


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


def test_apply_to_column(sample_categories, sample_provider, sample_dataframe):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
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
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
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
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
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


def test_base_categorizer(sample_categories, sample_provider):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    assert categorizer.categorize("It's raining heavily today.") == "Weather"
    assert categorizer.categorize("The football match was exciting!") == "Sports"
    assert categorizer.categorize("I saw Emma Watson at the mall.") == "Celebrities"
    assert categorizer.categorize("This is a random comment.") == "Others"
    assert categorizer.categorize("1234567890!@#$%^&*()") == "Anomaly"


def test_base_categorizer_batch(sample_categories, sample_provider, sample_list):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    assert categorizer.categorize_batch(sample_list) == [
        "Weather",
        "Sports",
        "Celebrities",
        "Others",
        "Anomaly",
    ]


def test_multi_categorizer(sample_categories, sample_provider):
    categorizer = MultiCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    assert set(categorizer.categorize("It's raining and I saw Emma Watson.")) == {
        "Weather",
        "Celebrities",
    }
    assert set(
        categorizer.categorize("The football match was exciting and it's sunny!")
    ) == {"Sports", "Weather"}
