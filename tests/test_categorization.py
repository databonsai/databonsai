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
    return OpenAIProvider()  # or AnthropicProvider()


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "text": [
                "It's raining heavily today.",
                "The football match was exciting!",
                "I saw Emma Watson at the mall.",
                "Random text 123456",
            ]
        }
    )


def test_apply_to_column(sample_categories, sample_provider, sample_dataframe):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    df = sample_dataframe.copy()
    df["category"] = None

    success_idx = apply_to_column(df["text"], df["category"], categorizer.categorize)

    assert success_idx == 4
    assert df["category"].tolist() == ["Weather", "Sports", "Celebrities", "Others"]


def test_apply_to_column_batch(sample_categories, sample_provider, sample_dataframe):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    df = sample_dataframe.copy()
    df["category"] = None

    success_idx = apply_to_column_batch(
        df["text"], df["category"], categorizer.categorize_batch, batch_size=2
    )

    assert success_idx == 4
    assert df["category"].tolist() == ["Weather", "Sports", "Celebrities", "Others"]


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

    assert success_idx == 4
    assert df["category"].tolist() == [None, "Sports", "Celebrities", "Others"]


def test_apply_to_column_batch_interrupted(
    sample_categories, sample_provider, sample_dataframe, monkeypatch
):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    df = sample_dataframe.copy()
    df["category"] = None

    def interrupt_categorization(texts):
        if len(texts) > 1:
            raise Exception("Simulated interruption")
        return [categorizer.categorize(text) for text in texts]

    monkeypatch.setattr(categorizer, "categorize_batch", interrupt_categorization)

    with pytest.raises(Exception):
        success_idx = apply_to_column_batch(
            df["text"], df["category"], categorizer.categorize_batch, batch_size=2
        )

    assert success_idx == 2
    assert df["category"].tolist() == ["Weather", "Sports", None, None]

    success_idx = apply_to_column_batch(
        df["text"],
        df["category"],
        categorizer.categorize_batch,
        batch_size=1,
        start_idx=success_idx,
    )

    assert success_idx == 4
    assert df["category"].tolist() == ["Weather", "Sports", "Celebrities", "Others"]


def test_base_categorizer(sample_categories, sample_provider):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    assert categorizer.categorize("It's raining heavily today.") == "Weather"
    assert categorizer.categorize("The football match was exciting!") == "Sports"
    assert categorizer.categorize("I saw Emma Watson at the mall.") == "Celebrities"
    assert categorizer.categorize("This is a random comment.") == "Others"
    assert categorizer.categorize("1234567890!@#$%^&*()") == "Anomaly"


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
    assert set(categorizer.categorize("Random text 123456")) == {"Others", "Anomaly"}


def test_batch_processing(sample_categories, sample_provider):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    df = pd.DataFrame(
        {
            "text": [
                "It's raining heavily today.",
                "The football match was exciting!",
                "I saw Emma Watson at the mall.",
            ]
        }
    )
    df["category"] = None

    success_idx = apply_to_column_batch(
        df["text"], df["category"], categorizer.categorize_batch, batch_size=2
    )

    assert success_idx == 3
    assert df["category"].tolist() == ["Weather", "Sports", "Celebrities"]


def test_error_handling(sample_categories, sample_provider):
    categorizer = BaseCategorizer(
        categories=sample_categories, llm_provider=sample_provider
    )

    assert categorizer.categorize(None) == "Others"
    assert categorizer.categorize("") == "Others"
    assert categorizer.categorize("A" * 10000) == "Others"  # Assuming a very long text
