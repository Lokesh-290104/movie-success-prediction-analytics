import pandas as pd
from src.data_loader import load_and_clean_data


def test_load_and_clean_data_basic():
    # Create a tiny fake dataset
    data = {
        "budget": [1000000, 0],
        "revenue": [5000000, 100000],
        "popularity": [10, 5],
        "runtime": [120, 90],
        "vote_average": [7.0, 6.0],
        "title": ["Test Movie 1", "Test Movie 2"],
        "genres": ["[{'id': 18, 'name': 'Drama'}]", "[]"]
    }

    df = pd.DataFrame(data)
    test_csv = "tests/temp_movies.csv"
    df.to_csv(test_csv, index=False)

    cleaned_df = load_and_clean_data(test_csv)

    # Budget and revenue must be > 0
    assert (cleaned_df["budget"] > 0).all()
    assert (cleaned_df["revenue"] > 0).all()

    # Required columns exist
    for col in ["budget", "revenue", "popularity", "runtime", "vote_average", "main_genre"]:
        assert col in cleaned_df.columns