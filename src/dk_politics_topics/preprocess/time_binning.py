import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def add_time_bins(df: pd.DataFrame, freq: str = "year", date_col: str = "date", label_col: str = "time_bin") -> pd.DataFrame:
    """Assign time bins (year or quarter)."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    if freq.lower() in {"year", "y"}:
        period = df[date_col].dt.to_period("Y")
        df[label_col] = period.astype(str)
    elif freq.lower() in {"quarter", "q"}:
        period = df[date_col].dt.to_period("Q")
        df[label_col] = period.astype(str)
    else:
        raise ValueError("freq must be 'year' or 'quarter'")

    logger.info("Assigned %s time bins", freq)
    return df
