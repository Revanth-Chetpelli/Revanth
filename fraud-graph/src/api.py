"""Local FastAPI service for model scoring and case investigation.

All account activity used for a score is restricted to earlier PaySim steps.
The API returns a review signal, never a finding of confirmed fraud.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import duckdb
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "fraud_risk_model.joblib"
BASE_PATH = ROOT / "data" / "processed" / "paysim_base.parquet"
QUEUE_PATH = ROOT / "reports" / "investigation_queue.csv"
EXPLANATION_PATH = ROOT / "reports" / "top_case_explanations.csv"
NEIGHBORHOOD_PATH = ROOT / "reports" / "top_case_neighborhoods.csv"
METRICS_PATH = ROOT / "reports" / "model_comparison.csv"

app = FastAPI(
    title="PaySim fraud risk review API",
    description="Scores synthetic PaySim transactions for analyst review.",
    version="1.0.0",
)


class ScoreRequest(BaseModel):
    step: int = Field(ge=1, le=743, description="PaySim time step")
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]
    amount: float = Field(ge=0, le=10_000_000)
    oldbalanceOrg: float = Field(ge=0, le=100_000_000)
    oldbalanceDest: float = Field(ge=0, le=100_000_000)
    nameOrig: str = Field(min_length=1, max_length=32)
    nameDest: str = Field(min_length=1, max_length=32)


@lru_cache(maxsize=1)
def load_artifact() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def _db() -> duckdb.DuckDBPyConnection:
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Processed PaySim file not found: {BASE_PATH}")
    return duckdb.connect(database=":memory:")


def _history_features(payload: ScoreRequest) -> dict[str, float]:
    """Summarize only prior account activity; label columns are never queried."""
    connection = _db()
    try:
        result = connection.execute(
            """SELECT
                 count(*) FILTER (WHERE nameOrig = ?) AS out_count,
                 coalesce(sum(amount) FILTER (WHERE nameOrig = ?), 0) AS out_amount,
                 count(*) FILTER (WHERE nameDest = ?) AS in_count,
                 coalesce(sum(amount) FILTER (WHERE nameDest = ?), 0) AS in_amount
               FROM read_parquet(?)
               WHERE step < ? AND (nameOrig = ? OR nameDest = ?)""",
            [payload.nameOrig, payload.nameOrig, payload.nameDest, payload.nameDest,
             str(BASE_PATH), payload.step, payload.nameOrig, payload.nameDest],
        ).fetchone()
    finally:
        connection.close()
    return {
        "prior_outgoing_count": float(result[0]),
        "prior_outgoing_amount": float(result[1]),
        "prior_incoming_count": float(result[2]),
        "prior_incoming_amount": float(result[3]),
    }


def _feature_frame(payload: ScoreRequest, history: dict[str, float], artifact: dict) -> pd.DataFrame:
    row = {
        "step": payload.step,
        "type": payload.type,
        "amount": payload.amount,
        "oldbalanceOrg": payload.oldbalanceOrg,
        "oldbalanceDest": payload.oldbalanceDest,
        **history,
    }
    frame = pd.get_dummies(pd.DataFrame([row]), columns=["type"], dtype=float)
    return frame.reindex(columns=artifact["encoded_columns"], fill_value=0)


@app.get("/health")
def health() -> dict:
    missing = [str(path) for path in (MODEL_PATH, BASE_PATH) if not path.exists()]
    return {"status": "ok" if not missing else "needs_setup", "missing_files": missing}


@app.post("/score")
def score(payload: ScoreRequest) -> dict:
    try:
        artifact = load_artifact()
        history = _history_features(payload)
        frame = _feature_frame(payload, history, artifact)
        probability = float(artifact["model"].predict_proba(frame)[0, 1])
        threshold = float(artifact["threshold"])
        contributions = artifact["model"].booster_.predict(frame, pred_contrib=True)[0]
        explanation = sorted(
            [
                {"feature": name, "contribution": float(value),
                 "feature_value": float(frame.iloc[0][name]),
                 "direction": "raises score" if value > 0 else "lowers score"}
                for name, value in zip(frame.columns, contributions[:-1])
            ],
            key=lambda item: abs(item["contribution"]), reverse=True,
        )[:5]
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "risk_score": probability,
        "alert_candidate": probability >= threshold,
        "threshold": threshold,
        "interpretation": "Candidate for human review; not a confirmed fraud finding.",
        "history_features": history,
        "top_contributors": explanation,
    }


@app.get("/cases")
def cases(limit: int = Query(default=25, ge=1, le=200)) -> dict:
    if not QUEUE_PATH.exists():
        raise HTTPException(status_code=503, detail="Run notebook 07 to create the investigation queue.")
    data = pd.read_csv(QUEUE_PATH).head(limit)
    return {"count": len(data), "cases": data.to_dict(orient="records")}


@app.get("/cases/{risk_rank}")
def case_detail(risk_rank: int) -> dict:
    if risk_rank < 1:
        raise HTTPException(status_code=422, detail="risk_rank must be at least 1")
    if not QUEUE_PATH.exists():
        raise HTTPException(status_code=503, detail="Run notebook 07 to create the investigation queue.")
    queue = pd.read_csv(QUEUE_PATH)
    selected = queue.loc[queue["risk_rank"] == risk_rank]
    if selected.empty:
        raise HTTPException(status_code=404, detail="No case with that risk rank.")
    result = selected.iloc[0].to_dict()
    for path, key in ((EXPLANATION_PATH, "explanations"), (NEIGHBORHOOD_PATH, "neighborhood")):
        if path.exists():
            frame = pd.read_csv(path)
            frame = frame.loc[frame["risk_rank" if key == "explanations" else "case_risk_rank"] == risk_rank]
            result[key] = frame.to_dict(orient="records")
        else:
            result[key] = []
    return result


@app.get("/metrics")
def metrics() -> dict:
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Run notebook 06 to create model metrics.")
    return {"models": pd.read_csv(METRICS_PATH).to_dict(orient="records")}


@app.get("/network/{account_id}")
def account_network(
    account_id: str,
    before_step: int = Query(ge=1, le=743),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict:
    """Return earlier synthetic transactions touching an account, without labels."""
    try:
        connection = _db()
        try:
            data = connection.execute(
                """SELECT step, type, amount, nameOrig, nameDest
                   FROM read_parquet(?)
                   WHERE step < ? AND (nameOrig = ? OR nameDest = ?)
                   ORDER BY step DESC LIMIT ?""",
                [str(BASE_PATH), before_step, account_id, account_id, limit],
            ).df()
        finally:
            connection.close()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"account_id": account_id, "before_step": before_step,
            "count": len(data), "transactions": data.to_dict(orient="records")}
