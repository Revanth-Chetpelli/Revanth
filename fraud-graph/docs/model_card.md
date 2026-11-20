# Model card: PaySim transaction risk ranking

## Summary

| Item | Description |
|---|---|
| Model | LightGBM binary classifier with transaction and past account-activity features. |
| Intended use | Rank synthetic PaySim transactions for a learning demonstration and analyst-review workflow. |
| Not intended for | Real payment decisions, account freezes, accusations, or unsupervised enforcement. |
| Data | PaySim synthetic mobile-money transactions. |
| Target | Dataset label `isFraud`, used only during training and offline evaluation. |
| Selected threshold | Approximately `0.99497`, selected on validation data to maximize F1. |

## Data and evaluation design

Transactions are split chronologically by `step`, rather than randomly:

| Period | Steps | Rows | Labeled fraud | Fraud rate |
|---|---:|---:|---:|---:|
| Train | 1–520 | 6,082,007 | 5,781 | 0.0951% |
| Validation | 521–631 | 191,147 | 1,180 | 0.6173% |
| Test | 632–743 | 89,466 | 1,252 | 1.3994% |

The changing label rate is a substantial time-distribution shift and can affect threshold behavior. The training run retains all fraud rows and samples a portion of non-fraud rows for speed; validation and test metrics use their complete periods at natural class balance.

Account-history counts and sums include only earlier time steps. Raw account IDs, `isFraud`, `isFlaggedFraud`, and post-transaction balances are excluded from model inputs.

## Held-out test results

| Model | PR-AUC | Precision | Recall | F1 | Precision@1000 | Recall@1000 |
|---|---:|---:|---:|---:|---:|---:|
| LightGBM, transaction details only | 0.9335 | 0.9521 | 0.7628 | 0.8470 | 0.9520 | 0.7604 |
| LightGBM + historical account features | 0.9282 | 0.9557 | 0.7588 | 0.8459 | 0.9530 | 0.7612 |
| Random Forest + historical account features | 0.8855 | 0.9117 | 0.7500 | 0.8230 | 0.9230 | 0.7372 |
| Logistic Regression + historical account features | 0.5684 | 0.6210 | 0.3874 | 0.4771 | 0.5480 | 0.4377 |

The graph-history features did **not** improve PR-AUC in this experiment compared with transaction details alone. This is an important finding, not a reason to claim they improved the detector. They may still be useful for giving reviewers account context.

## Limitations and risks

- PaySim is simulated. Very strong test scores do not prove that this model works on real-world fraud.
- Fraud prevalence rises from 0.095% in train to 1.399% in test, so the periods are not identically distributed.
- The model's score is not a calibrated real-world probability. Contributions describe model behavior, not causal evidence.
- A validation-selected threshold can behave differently after deployment or under a different fraud rate.
- Account activity features currently summarize prior sender and receiver activity; richer graph structures and network motifs are not fully modeled.
- The queue is intended for human review. The project cannot establish intent, culpability, or money laundering.

## Responsible use

Use the score to prioritize investigation in the synthetic demo only. Review transaction details and relevant context; do not treat the label or score as proof. Any real-world use would require representative data, independent validation, calibration, privacy and fairness review, operational monitoring, and human governance.
