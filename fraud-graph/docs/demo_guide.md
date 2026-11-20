# Demo guide

## Before the demonstration

1. Open the project folder in VS Code.
2. Confirm that `data/raw/` contains the PaySim CSV and that notebooks 05, 06, and 07 have completed. Their processed data, model, and report files are local and excluded from Git.
3. Start the API in one terminal:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn src.api:app --reload
   ```

4. Start the dashboard in a second terminal:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run src/dashboard.py
   ```

5. Open the local Streamlit URL, usually `http://localhost:8501`. The API's interactive docs are at `http://127.0.0.1:8000/docs`.

## Suggested walkthrough

1. **Explain the goal:** “This project ranks transactions from the simulated PaySim dataset for human review. It is a prototype, not a real fraud verdict.”
2. **Score a transaction:** show the sample input, score, validation-selected cutoff, and top feature contributions. Explain that a contribution is not itself a percentage or proof.
3. **Review the queue:** show the risk-ranked cases and clarify that the checkmark means the score passed the review threshold.
4. **Inspect a case:** choose rank 1 and load its model explanation and earlier account neighborhood. Emphasize that the neighborhood contains only activity before the case step.
5. **Compare models:** show PR-AUC, precision, recall, F1, and Precision/Recall@K. Mention that historical graph features did not improve PR-AUC in this test, though they give account context.
6. **Close with limitations:** all labels and records are simulated; the unusually strong metrics do not demonstrate real-world performance.

## Screenshots to capture

Capture these views using Windows Snipping Tool or `Win+Shift+S`:

- Dashboard title and the KPI cards.
- A scored transaction and its explanation.
- The ranked review queue.
- A case detail with the earlier account neighborhood.
- The model comparison table.

Save screenshots in a local `docs/screenshots/` folder if you plan to include them in the final GitHub submission. Inspect each screenshot for local paths, usernames, or other details you do not want to share.

## Technical questions to practice

**Why split by time?** Fraud detection predicts future transactions. A random split could mix later patterns into training and make the evaluation too optimistic.

**What makes a feature “past-only”?** For a transaction at step `t`, the account activity query uses `step < t`. It cannot count the transaction itself or any later transaction.

**Why PR-AUC?** Fraud is rare, so precision-recall summarizes the tradeoff on the minority class more usefully than accuracy alone.

**What does the threshold mean?** It is a cutoff chosen on validation data for this experiment. A score over it becomes a candidate for human review, not a confirmed fraud event.

**Did graph features improve the model?** Not by PR-AUC here: LightGBM with only transaction details scored 0.9335, while adding historical account features scored 0.9282. The historical features still provide useful context for investigation.

**What are the main tools?** Python, pandas, DuckDB, scikit-learn, LightGBM, NetworkX, FastAPI, Streamlit, Jupyter notebooks, and Git.
