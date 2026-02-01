import pandas as pd
from src.features import add_features

def test_add_features():
    sample_df = pd.DataFrame({
        "budget": [10_000_000, 20_000_000],
        "revenue": [15_000_000, 10_000_000],
        "runtime": [100, 140],
        "popularity": [5.0, 8.0],
        "vote_average": [6.5, 7.2]
    })

    df = add_features(sample_df)

    # Feature existence
    assert "roi" in df.columns
    assert "success" in df.columns
    assert "log_budget" in df.columns

    # Success must be binary
    assert set(df["success"].unique()).issubset({0, 1})