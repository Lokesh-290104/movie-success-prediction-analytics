import pandas as pd
import ast

def load_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[
        "budget", "revenue", "popularity",
        "runtime", "vote_average",
        "title", "genres"
    ]]

    df = df[(df["budget"] > 0) & (df["revenue"] > 0)]
    df.dropna(inplace=True)

    df["success"] = (df["revenue"] > df["budget"]).astype(int)

    df["main_genre"] = df["genres"].apply(
        lambda x: ast.literal_eval(x)[0]["name"] if x != "[]" else "Unknown"
    )

    return df