import pandas as pd

from dk_politics_topics.analysis.sentiment import aggregate_sentiment_by_area, resolve_provider
from dk_politics_topics.config import SentimentConfig


def test_transformers_sentiment_scoring(monkeypatch):
    import transformers.pipelines

    class DummyPipeline:
        def __call__(self, texts, batch_size=32):
            outputs = []
            for text in texts:
                t = (text or "").lower()
                if "god" in t:
                    outputs.append(
                        [
                            {"label": "POSITIVE", "score": 0.9},
                            {"label": "NEGATIVE", "score": 0.1},
                        ]
                    )
                else:
                    outputs.append(
                        [
                            {"label": "POSITIVE", "score": 0.2},
                            {"label": "NEGATIVE", "score": 0.8},
                        ]
                    )
            return outputs

    def dummy_pipeline(*args, **kwargs):
        return DummyPipeline()

    monkeypatch.setattr(transformers.pipelines, "pipeline", dummy_pipeline)

    cfg = SentimentConfig(approach="huggingface", huggingface_model="dummy", device="cpu")
    provider = resolve_provider(cfg)

    scores = provider.score_batch(["det er god politik", "det er dårlig politik"])
    assert scores[0] > 0
    assert scores[1] < 0


def test_aggregate_sentiment_by_area(monkeypatch):
    import transformers.pipelines

    class DummyPipeline:
        def __call__(self, texts, batch_size=32):
            outputs = []
            for text in texts:
                t = (text or "").lower()
                if "god" in t:
                    outputs.append(
                        [
                            {"label": "POSITIVE", "score": 0.9},
                            {"label": "NEGATIVE", "score": 0.1},
                        ]
                    )
                else:
                    outputs.append(
                        [
                            {"label": "POSITIVE", "score": 0.2},
                            {"label": "NEGATIVE", "score": 0.8},
                        ]
                    )
            return outputs

    def dummy_pipeline(*args, **kwargs):
        return DummyPipeline()

    monkeypatch.setattr(transformers.pipelines, "pipeline", dummy_pipeline)

    cfg = SentimentConfig(approach="huggingface", huggingface_model="dummy", device="cpu")
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
    means = dict(zip(agg["party"], agg["mean_sentiment"]))
    assert means["A"] > 0
    assert means["B"] < 0
