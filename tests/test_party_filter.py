import pandas as pd

from dk_politics_topics.preprocess.clean import filter_parties_by_min_share


def test_filter_parties_by_min_share_drops_small_parties():
    df = pd.DataFrame(
        {
            "party": ["A"] * 98 + ["B"] * 1 + ["C"] * 1,
            "text": ["x"] * 100,
        }
    )
    out = filter_parties_by_min_share(df, party_col="party", min_share=0.02)
    assert set(out["party"].unique()) == {"A"}

