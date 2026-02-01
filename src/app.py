import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from scipy.stats import ttest_ind, chi2_contingency

from data_loader import load_and_clean_data
from features import add_features

# ---------------- Page Config ----------------
st.set_page_config(page_title="🎬 MovieIQ Dashboard", layout="wide")

with st.sidebar.expander("ℹ️ About this app", expanded=True):
    st.markdown("""
    **MovieIQ** estimates a movie’s **revenue potential** using historical data.

    ⚠️ The ML model is trained on the **full dataset**.  
    Filters only affect **visualization**, not training.

    ✔ Clean data pipeline  
    ✔ Realistic success definition (ROI-based)  
    ✔ No data leakage  
    ✔ Probability-based prediction  
    ✔ Explainable AI (feature contribution)
    """)

st.title("🎬 MovieIQ – Movie Revenue Potential Dashboard")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your movies CSV file", type=["csv"])

if uploaded_file:
    # -------- LOAD & FEATURE ENGINEERING --------
    df = load_and_clean_data(uploaded_file)
    df = add_features(df)

    # -------- KPIs --------
    col1, col2, col3 = st.columns(3)
    col1.metric("🎞️ Total Movies", len(df))
    col2.metric("📈 Success Rate (ROI ≥ 20%)", f"{df['success'].mean() * 100:.1f}%")
    col3.metric("🎬 Unique Genres", df["main_genre"].nunique())

    # -------- SIDEBAR FILTERS (VISUAL ONLY) --------
    st.sidebar.header("🔍 Filter Options")

    selected_genres = st.sidebar.multiselect(
        "Select Genre(s)",
        options=sorted(df["main_genre"].unique()),
        default=df["main_genre"].unique()
    )

    min_votes = st.sidebar.slider(
        "Minimum Vote Average",
        0.0, 10.0, 3.0
    )

    if min_votes > 5.0:
        st.sidebar.warning(
            "⚠️ High vote filter applied. "
            "Low-rated movies are hidden from charts."
        )

    view_df = df[
        (df["main_genre"].isin(selected_genres)) &
        (df["vote_average"] >= min_votes)
    ]

    # -------- DATASET PREVIEW --------
    st.subheader("🎯 Dataset Overview (Filtered)")
    st.write(view_df.head())

    # -------- VISUALIZATION --------
    st.subheader("💸 Budget vs Revenue")
    fig1, ax1 = plt.subplots()
    sns.scatterplot(
        data=view_df,
        x="budget",
        y="revenue",
        hue="success",
        palette="coolwarm",
        ax=ax1
    )
    st.pyplot(fig1)

    # -------- MACHINE LEARNING --------
    st.subheader("🤖 Machine Learning Model")

    FEATURES = ["log_budget", "popularity", "runtime", "vote_average"]
    X = df[FEATURES]
    y = df["success"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    st.markdown(f"**Model Accuracy:** {model.score(X_test, y_test):.2%}")
    st.text(classification_report(y_test, model.predict(X_test)))

    # -------- PREDICTION --------
    st.subheader("🎬 Predict Movie Revenue Potential")

    with st.form("prediction_form"):
        input_budget = st.number_input("Budget (USD)", 1_000, 500_000_000, 10_000_000)
        input_popularity = st.slider("Popularity", 0.0, 100.0, 10.0)
        input_runtime = st.slider("Runtime (minutes)", 40, 240, 120)
        input_vote_average = st.slider("Vote Average", 0.0, 10.0, 6.5)
        submit = st.form_submit_button("Predict")

    if submit:
        input_df = pd.DataFrame({
            "log_budget": [np.log1p(input_budget)],
            "popularity": [input_popularity],
            "runtime": [input_runtime],
            "vote_average": [input_vote_average],
        })

        success_proba = model.predict_proba(input_df)[0][1]

        st.subheader("📊 Prediction Confidence")
        st.progress(success_proba)

        if success_proba >= 0.65:
            st.success(f"🌟 High Revenue Potential ({success_proba:.2%})")
        else:
            st.error(f"🚨 Low Revenue Potential ({success_proba:.2%})")

        # -------- EXPLAINABLE AI --------
        st.subheader("🔍 Why this prediction?")

        base_proba = success_proba
        impacts = {}

        for col in FEATURES:
            temp = input_df.copy()
            temp[col] = df[col].mean()  # neutralize feature
            new_proba = model.predict_proba(temp)[0][1]
            impacts[col] = base_proba - new_proba

        impact_df = (
            pd.DataFrame.from_dict(impacts, orient="index", columns=["Impact"])
            .sort_values("Impact", ascending=False)
        )

        impact_df["Effect"] = impact_df["Impact"].apply(
            lambda x: "Positive" if x > 0 else "Negative"
        )

        st.dataframe(
            impact_df.style.format({"Impact": "{:+.3f}"})
        )

        st.caption(
            "Feature impact shows how much each input pushed the prediction "
            "toward success or failure."
        )

else:
    st.info("👇 Upload a CSV file to get started.")