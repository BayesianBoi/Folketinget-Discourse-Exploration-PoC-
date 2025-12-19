from __future__ import annotations

from pathlib import Path
from typing import Optional

import plotly.express as px
import pandas as pd

from ..utils.logging import get_logger

logger = get_logger(__name__)


def save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))
    logger.info("Saved plot to %s", path)
    return path


def plot_prevalence_over_time(
    df: pd.DataFrame,
    output_path: Path,
    party: Optional[str] = None,
    time_bin_col: str = "year",
) -> Path:
    data = df[df["party"] == party] if party else df
    fig = px.line(
        data,
        x=time_bin_col,
        y="mean_weight",
        color="topic_id",
        title=f"Topic prevalence over time{f' - {party}' if party else ''}",
    )
    return save_fig(fig, output_path)


def plot_party_comparison(
    df: pd.DataFrame,
    output_path: Path,
    time_bin_col: str = "year",
) -> Path:
    fig = px.bar(
        df,
        x="party",
        y="mean_weight",
        color="topic_id",
        barmode="group",
        facet_col=time_bin_col,
        title="Topic comparison across parties",
    )
    return save_fig(fig, output_path)


def plot_controversial_sentiment(
    df: pd.DataFrame,
    output_path: Path,
    time_bin_col: str = "year",
) -> Path:
    fig = px.line(
        df,
        x=time_bin_col,
        y="mean_sentiment",
        color="party",
        facet_row="area",
        title="Sentiment over time for controversial topics",
    )
    return save_fig(fig, output_path)
