"""
data_loader.py - Load and filter the folketinget speech data
"""
import pandas as pd
import pyreadr # the speehces are saved in rds

def load_data(filepath):
    """LOADS THE SPEECHES"""
    print("LOADING DATA")
    
    data_dict = pyreadr.read_r(filepath)
    # converting to df
    df = list(data_dict.values())[0]
    
    print(f"\nLoaded {len(df):,} speeches")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def filter_data(df, start_year=2019, end_year=2022, min_length=100):
    """FILTERS THE SPEECHES TO SPECIFIC YEARS, ETC"""
    print("FILTERING DATA")
    
    # Adding extra columns
    df["date"] = pd.to_datetime(df["date"]) # date
    df["year"] = df["date"].dt.year # year from the date
    df["text_length"] = df["text"].fillna("").str.len() # length of the speech
    
    # Apply filters
    df_filtered = df[
        (df["chair"] == False) &           # getting rid of procedural speeches. its just "on to you mr minister"
        (df["party"] != "-") &             # removing the no parties
        (df["text"].notna()) &             # removing the entries without any text
        (df["text_length"] > min_length) & # filtering the extremely short speeches
        (df["year"] >= start_year) &       # time range oof the speech
        (df["year"] <= end_year)
    ].copy()
    
    print(f"\nOriginal: {len(df):,} speeches")
    print(f"Filtered: {len(df_filtered):,} speeches ({len(df_filtered)/len(df)*100:.1f}%)")
    print(f"Time period: {start_year}-{end_year}")
    print(f"Min length: {min_length} characters")
    print(f"Unique parties: {df_filtered["party"].nunique()}")
    
    return df_filtered

def sample_speeches(df, sample_size=30000, random_state=69):
    """SAMPLES 30000 of the speeches"""
    print("SAMPLING THE SPEECHES")
    
    print(f"\nTarget sample: {sample_size:,} speeches")
    
    # random sample based on the sample size
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=random_state)
    
    print(f"Sample created: {len(df_sample):,} speeches")
    print(f"\nParty distribution in sample:")
    for party, count in df_sample["party"].value_counts().items():
        pct = (count / len(df_sample)) * 100
        print(f"  {party:8s}: {count:5,} ({pct:5.2f}%)")
    
    return df_sample