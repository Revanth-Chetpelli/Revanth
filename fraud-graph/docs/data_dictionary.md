# Data dictionary

## Dataset

PaySim is a **simulated** mobile-money transaction dataset. The original file is kept locally at `data/raw/PS_20174392719_1491204439457_log.csv` and is intentionally not committed to Git. The project reads the data and writes derived Parquet files under `data/processed/`.

## Source columns

| Column | Meaning in this project | Model use |
|---|---|---|
| `step` | Synthetic time step for the transaction. | Used for chronological split and as a model feature. |
| `type` | Transaction category such as `TRANSFER` or `CASH_OUT`. | Encoded as category indicator columns. |
| `amount` | Transaction amount in the dataset's units. | Model feature. |
| `nameOrig` | Synthetic sender/account identifier. | Used to join prior activity; excluded as a raw model feature. |
| `oldbalanceOrg` | Sender balance before the transaction. | Model feature. |
| `newbalanceOrig` | Sender balance after the transaction. | Excluded from model inputs to avoid relying on a post-transaction value. |
| `nameDest` | Synthetic receiver/account identifier. | Used to join prior activity; excluded as a raw model feature. |
| `oldbalanceDest` | Receiver balance before the transaction. | Model feature. |
| `newbalanceDest` | Receiver balance after the transaction. | Excluded from model inputs to avoid relying on a post-transaction value. |
| `isFraud` | Dataset-provided synthetic fraud label. | Target for training and offline evaluation only; never sent to the scoring endpoint. |
| `isFlaggedFraud` | Dataset-provided system flag. | Audited as a baseline/reference; excluded from model inputs. |

The audit found 6,362,620 transactions and 8,213 records labeled fraud (about 0.129% overall). There were no missing values in the audited columns. Fraud labels occur in the `TRANSFER` and `CASH_OUT` categories in this dataset. The built-in flag marked only 16 of the 8,213 labeled fraud rows, so it is not an adequate detector by itself.

## Historical account features

Each transaction gets features calculated from records with `step` strictly less than its own step:

| Feature | Meaning |
|---|---|
| `prior_outgoing_count` | Number of earlier transactions sent by this sender account. |
| `prior_outgoing_amount` | Sum of amounts in those earlier outgoing transactions. |
| `prior_incoming_count` | Number of earlier transactions received by this receiver account. |
| `prior_incoming_amount` | Sum of amounts in those earlier incoming transactions. |

The model uses `step`, `type`, `amount`, `oldbalanceOrg`, `oldbalanceDest`, and the four historical features. Category encoding creates columns such as `type_CASH_OUT`. A value of `1` means the transaction has that category; `0` means it does not.

## Saved output files

| Path | Contents |
|---|---|
| `data/processed/paysim_base.parquet` | Selected source columns for local processing and past-only account lookups. |
| `data/processed/paysim_graph_features.parquet` | Transaction rows, labels for offline work, time split, and historical graph features. |
| `reports/model_comparison.csv` | Validation-selected thresholds and held-out test metrics. |
| `reports/investigation_queue.csv` | Ranked cases and scores; does not include the fraud target. |
| `reports/top_case_explanations.csv` | Local feature contributions for selected cases. |
| `reports/top_case_neighborhoods.csv` | Earlier transactions connected to selected case accounts. |
