import numpy as np
import pandas as pd

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ---- PROFIT & ROI ----
    df["profit"] = df["revenue"] - df["budget"]
    df["roi"] = df["profit"] / df["budget"]

    # ---- SUCCESS DEFINITION (REALISTIC) ----
    # Success = at least 20% ROI
    df["success"] = (df["roi"] >= 0.2).astype(int)

    # ---- LOG FEATURES ----
    df["log_budget"] = np.log1p(df["budget"])
    df["log_revenue"] = np.log1p(df["revenue"])

    # ---- RUNTIME BUCKETS ----
    df["runtime_bucket"] = pd.cut(
        df["runtime"],
        bins=[0, 90, 120, 180, 300],
        labels=["short", "medium", "long", "very_long"]
    )

    return df