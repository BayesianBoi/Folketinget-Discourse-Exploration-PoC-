import pandas as pd

from dk_politics_topics.analysis.sentiment import LexiconSentiment, aggregate_sentiment_by_area, resolve_provider
from dk_politics_topics.config import SentimentConfig


def test_lexicon_sentiment_scoring():
    cfg = SentimentConfig(lexicon_positive=["god"], lexicon_negative=["dårlig"])
    provider = LexiconSentiment(cfg)
    assert provider.score("det er god politik") > 0
    assert provider.score("det er dårlig politik") < 0


def test_aggregate_sentiment_by_area():
    cfg = SentimentConfig(lexicon_positive=["god"], lexicon_negative=["dårlig"])
    provider = resolve_provider(cfg)
    filtered = pd.DataFrame({"doc_id": ["1", "2"], "topic_id": [0, 0], "weight": [0.6, 0.7], "area": ["climate", "climate"]})
    metadata = pd.DataFrame(
        {
            "doc_id": ["1", "2"],
            "party": ["A", "B"],
            "time_bin": ["2020", "2020"],
            "text": ["god sag", "dårlig sag"],
        }
    )
    agg = aggregate_sentiment_by_area(filtered, metadata, provider, time_bin_col="time_bin")
    assert set(agg["party"]) == {"A", "B"}
