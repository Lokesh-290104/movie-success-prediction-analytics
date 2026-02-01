# MovieIQ — Predictive Analytics on Film Success

MovieIQ is an end-to-end data analysis and dashboard project that investigates drivers of commercial success in films and provides interpretable, probabilistic predictions based on historical data.

The work emphasizes robust data cleaning, feature engineering (ROI-based), statistical analysis, explainable machine learning, and automated tests.

## Problem Statement

Film production carries high financial risk. Raw revenue is not sufficient to judge success because budgets vary widely.

Project objectives:

- Identify factors associated with commercial success
- Define a business-meaningful success metric (ROI)
- Build an interpretable model to support decisions
- Expose findings and prediction rationale via a Streamlit dashboard

## Dataset

Source: Kaggle — TMDB Movies Dataset

Approximate size: 45,000 records

Key fields used: `budget`, `revenue`, `popularity`, `runtime`, `vote_average`, `genres`, `title`

## Data Cleaning & Preparation

Main steps applied to produce an analysis-ready table:

- Coerce numeric fields stored as strings to numeric types
- Remove entries with zero or missing `budget` or `revenue`
- Filter out unrealistic runtimes and malformed records
- Parse nested `genres` into a usable format

These steps reduce bias from invalid records and enable reliable feature computation.

## Feature Engineering

Business-driven success metric: Return on Investment (ROI)

$$\text{ROI} = \frac{\text{Revenue} - \text{Budget}}{\text{Budget}}$$

Labeling rule: a movie is marked successful when ROI ≥ 20%.

Additional features: log-transformed `budget`, runtime buckets, profit, and derived categorical flags.

## Exploratory & Statistical Analysis

Included analyses and visualizations:

- Budget vs revenue scatter and density plots
- Genre-wise success and ROI summaries
- Statistical tests:
	- T-test: `vote_average` distribution by success label
	- Chi-square: runtime bucket association with success

Key empirical takeaway: ratings alone provide limited predictive power; budget and popularity are stronger correlates of success.

## Machine Learning (Decision Support)

- Model: Random Forest Classifier used to estimate probability of meeting the ROI threshold.
- Role: decision-support — the model supplements analysis and highlights patterns, but does not replace business judgment.
- Output: calibrated probability of success with per-feature contributions exposed to the user.

Model training follows standard pipeline practices (train/validate split, feature transformation, simple hyperparameter selection). Results are used for interpretation and scenario exploration rather than automated actions.

## Explainable AI

For individual predictions the dashboard provides:

- Feature-level contributions and direction of influence
- Global feature importance and partial dependence plots

This transparency supports operational decisions and interview discussions about model behavior.

## Testing & Validation

Unit tests are implemented with `pytest` and cover:

- Data cleaning functions
- Feature engineering logic
- Model pipeline basic integrity

Run tests locally with:

```bash
pytest tests/
```

## Project Structure

Top-level layout (key files):

```
LICENSE
README.md
requirements.txt
assets/              # images and static assets used by the dashboard
data/                # raw and example datasets (e.g. movies_metadata.csv)
src/
	__init__.py
	app.py             # Streamlit app entrypoint
	data_loader.py
	features.py
tests/
	temp_movies.csv
	test_data_loader.py
	test_features.py
	test_model.py
```

## Demo / Run Instructions

Local execution (recommended for evaluation and interview demos):

1. Create and activate a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Launch the dashboard

```bash
streamlit run src/app.py
```

Deployment: a hosted demo may be available; if so, a stable link will be added here. Do not assume a live deployment without confirmation.

## Notes for Interviewers

- The repository demonstrates end-to-end workflow: data ingestion, cleaning, feature engineering, statistical analysis, a model-as-aid approach, and automated tests.
- Discussion topics suitable for interviews: ROI threshold choice, dataset biases, feature selection, model explainability techniques, and test coverage.

## Author

**Lokesh Maheshwari**

Final Year IT Student | Aspiring Data Analyst

LinkedIn: https://www.linkedin.com/in/lokesh-maheshwari-8a4b83278/
GitHub: https://github.com/Lokesh-290104

## License

This project is licensed under the MIT License.

