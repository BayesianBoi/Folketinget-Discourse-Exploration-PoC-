"""
data_exploration.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def explore_data(df):
    """Exploration of the dataset"""
    print("DATA EXPLORATION")
    
    # just basic info such as parties, dates, and number of speeches
    print(f"\nTotal speeches: {len(df):,}")
    print(f"Date range: {df["date"].min()} to {df["date"].max()}")
    print(f"Unique parties: {df["party"].nunique()}")
    
    # speeches per year
    print("\nSpeeches by year:")
    year_counts = df["year"].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,}")
    
    # how are the speeches dsiributed for the participants
    print("\nTop 10 parties:")
    party_counts = df["party"].value_counts().head(10)
    for party, count in party_counts.items():
        pct = (count / len(df)) * 100
        print(f"  {party:8s}: {count:6,} ({pct:5.1f}%)")
    
    # chair speeches (which should be removed. they contain absolutely no info other than pleasentries)
    chair_count = (df["chair"] == True).sum()
    print(f"\nChair speeches: {chair_count:,} ({chair_count/len(df)*100:.1f}%)")
    
    # Speech length
    print("\nSpeech length statistics:")
    print(f"  Mean: {df["text_length"].mean():.0f} chars")
    print(f"  Median: {df["text_length"].median():.0f} chars")
    print(f"  <100 chars: {(df["text_length"] < 100).sum():,} ({(df["text_length"] < 100).sum()/len(df)*100:.1f}%)")
    
    return year_counts, party_counts


# plot for speeches over the years
def plot_temporal_distribution(df, output_path="/work/Exam/out/exploration/temporal_distribution.png"):
    """Plot speeches over time."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # Speeches by year
    year_counts = df["year"].value_counts().sort_index()
    year_counts.plot(kind="bar", ax=axes[0], color="steelblue", edgecolor="black")
    axes[0].set_title("Speeches per Year", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Number of Speeches")
    axes[0].tick_params(axis="x", rotation=45)
    
    # Time series
    speeches_per_month = df.groupby(df["date"].dt.to_period("M")).size()
    speeches_per_month.index = speeches_per_month.index.to_timestamp()
    speeches_per_month.plot(ax=axes[1], linewidth=2, color="darkblue")
    axes[1].set_title("Speeches Over Time (Monthly)", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Number of Speeches")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

# plot for speeches per party
def plot_party_distribution(df, output_path="/work/Exam/out/exploration/party_distribution.png"):
    """Plot party distribution."""
    party_counts = df["party"].value_counts().head(15)
    
    plt.figure(figsize=(12, 6))
    party_counts.plot(kind="barh", color="coral", edgecolor="black")
    plt.title("Top 15 Parties by Speech Count", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Speeches")
    plt.gca().invert_yaxis()
    
    # Add counts
    for i, (party, count) in enumerate(party_counts.items()):
        pct = (count / len(df)) * 100
        plt.text(count + 500, i, f"{count:,} ({pct:.1f}%)", va="center")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()