import pandas as pd

from dk_politics_topics.io.schemas import CorpusSchema, validate_corpus_df


def test_validate_corpus_df_handles_short_texts_and_duplicates():
    df = pd.DataFrame(
        {
            "doc_id": ["a", "a", "b"],
            "date": ["2020-01-01", "2020-01-02", "2021-03-04"],
            "party": ["S", "S", "V"],
            "text": ["lang tekst om noget vigtigt", "kort", "en anden lang tekst"],
        }
    )
    validated, issues = validate_corpus_df(df, CorpusSchema())
    assert len(validated) == 2
    assert validated["doc_id"].nunique() == 2
    assert any("duplikerede" in issue for issue in issues)
