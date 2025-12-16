import pandas as pd

from dk_politics_topics.preprocess.time_binning import add_time_bins


def test_add_time_bins_year_and_quarter():
    df = pd.DataFrame({"date": ["2020-01-05", "2021-07-10"]})
    yearly = add_time_bins(df, freq="year", date_col="date")
    assert set(yearly["time_bin"]) == {"2020", "2021"}

    quarterly = add_time_bins(df, freq="quarter", date_col="date")
    assert set(quarterly["time_bin"]) == {"2020Q1", "2021Q3"}
