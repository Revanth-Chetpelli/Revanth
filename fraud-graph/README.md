# Fraud and Money-Laundering Detection with Graph Machine Learning

A local-first investigation tool that scores simulated mobile-money transactions, models links between sender and receiver accounts, explains risk signals, and presents cases for human review.

## Project status

This repository contains a local-first prototype built step by step with **PaySim**, a synthetic mobile-money transaction dataset. It is simulated data and does not represent a real financial institution or real people.

## Repository layout

```text
fraud-graph/
├── data/
│   ├── raw/                 # Downloaded source data; never commit the dataset
│   └── processed/           # Cleaned and feature-ready data; never commit generated data
├── docs/                    # Architecture, data dictionary, model card, demo guide
├── models/                  # Locally trained model files; excluded from Git
├── notebooks/               # Step-by-step data science workflow
├── reports/                 # Evaluation outputs and figures
├── src/                     # Reusable pipeline, API, and dashboard code
├── tests/                   # Automated checks
├── .gitignore
├── requirements.txt
└── README.md
```

## Tools

- Python and pandas for preparing and exploring data.
- Jupyter notebooks inside Visual Studio Code for understandable, repeatable analysis.
- scikit-learn and LightGBM for tabular model baselines.
- NetworkX for account and transaction graph analysis.
- FastAPI and Streamlit for a local scoring service and analyst interface.
- Git and GitHub for version tracking and submission.

An optional GraphSAGE model can be considered after the data pipeline and graph-feature baseline are validated. It is not required for the first working version.

## Setup

Open this folder in Visual Studio Code. Install the **Python** and **Jupyter** extensions, then open a terminal in the project folder and run:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name fraud-graph --display-name "Python (fraud-graph)"
```

Open notebooks in order: `01_data_audit.ipynb` (inspect the dataset), `02_graph_basics.ipynb` (see accounts and transfers), `03_temporal_graph_features.ipynb` (small feature prototype), `04_temporal_split.ipynb` (fair time-based evaluation periods), `05_full_data_features.ipynb` (full-data historical features), `06_model_baselines.ipynb` (model comparison and graph-feature ablation), and `07_case_explainability.ipynb` (rank cases and inspect model contributions and earlier account neighborhoods). Select the project `.venv` kernel and run cells from top to bottom. Put the downloaded PaySim CSV in `data/raw/`. The raw dataset, generated Parquet files, and fitted models are intentionally excluded from Git; the small evaluation CSVs can be included.

## Run the local API and dashboard

After notebooks 05–07 have completed and created the local model and report files, open **two VS Code terminals** in the project folder.

Terminal 1 starts the scoring API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
```

Terminal 2 starts the analyst screen:

```powershell
.\.venv\Scripts\python.exe -m streamlit run src/dashboard.py
```

Open the Streamlit address printed in Terminal 2 (usually `http://localhost:8501`). Keep both terminals running while using the app. The API documentation is at `http://127.0.0.1:8000/docs`.

The dashboard can score an example transaction, show the highest-risk synthetic cases, inspect a model explanation and the account's earlier transaction neighborhood, and compare model results. It uses local files and makes no external service calls. A score is a review signal, not proof of wrongdoing.

Run the automated API checks from a terminal in the project folder:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Planned workflow

1. Audit the dataset, labels, class imbalance, time coverage, and leakage risks.
2. Build chronological train, validation, and test splits.
3. Compare Logistic Regression and Random Forest baselines with an imbalance-aware boosted tree.
4. Build a directed account graph and calculate transaction-network features.
5. Compare tabular-only, tabular-plus-graph-features, and (if practical) GraphSAGE results.
6. Evaluate precision, recall, F1, PR-AUC, confusion matrices, Precision@K, and Recall@K; select alert thresholds on validation data.
7. Explain transaction and entity risk signals, then expose scoring and investigation endpoints through FastAPI and a Streamlit analyst dashboard.
8. Add tests, a data dictionary, model card, architecture, demo guide, and reproducible setup instructions.

## Data and responsible use

Follow the dataset provider's terms and attribution. Never commit private or sensitive transaction data, secrets, local environments, or trained model binaries. Scores are investigation signals for human review; they are not proof of fraud or money laundering.

## Project documents

- [Architecture overview](docs/architecture.md)
- [Data dictionary](docs/data_dictionary.md)
- [Model card and evaluation](docs/model_card.md)
- [Demo guide and mentor questions](docs/demo_guide.md)
