import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def test_model_prediction_shape():
    X = pd.DataFrame({
        "log_budget": np.log1p([10_000_000, 20_000_000]),
        "popularity": [5.0, 8.0],
        "runtime": [100, 140],
        "vote_average": [6.5, 7.2]
    })
    y = [1, 0]

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    probs = model.predict_proba(X)

    # Probabilities must sum to 1
    assert probs.shape == (2, 2)
    assert np.allclose(probs.sum(axis=1), 1)