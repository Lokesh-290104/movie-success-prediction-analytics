import pandas as pd
import numpy as np
import ast

def load_and_clean_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    # Keep only required columns (defensive)
    df = df[
        [
            "budget",
            "revenue",
            "popularity",
            "runtime",
            "vote_average",
            "title",
            "genres",
        ]
    ]

    # ---- FORCE NUMERIC TYPES (CRITICAL) ----
    for col in ["budget", "revenue", "popularity", "runtime", "vote_average"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- DROP BROKEN ROWS ----
    df = df.dropna(subset=["budget", "revenue", "runtime", "popularity", "vote_average"])

    # ---- REMOVE IMPOSSIBLE MOVIES ----
    df = df[
        (df["budget"] > 0) &
        (df["revenue"] > 0) &
        (df["runtime"].between(40, 240))
    ]

    # ---- EXTRACT MAIN GENRE SAFELY ----
    def extract_main_genre(genres):
        try:
            parsed = ast.literal_eval(genres)
            if isinstance(parsed, list) and len(parsed) > 0:
                return parsed[0]["name"]
        except:
            pass
        return "Unknown"

    df["main_genre"] = df["genres"].apply(extract_main_genre)

    return df