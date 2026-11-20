"""Analyst view for the local PaySim fraud review API."""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("FRAUD_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Fraud risk review", page_icon=":material/security:", layout="wide")
st.title("PaySim fraud risk review")
st.caption("Synthetic transaction investigation workspace. Scores are review signals, not proof of fraud.")


@st.cache_data(ttl=30)
def get_json(path: str, params: dict | None = None) -> dict:
    response = requests.get(f"{API_URL}{path}", params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def api_error(exc: Exception) -> None:
    st.error(f"Could not reach the local scoring service at {API_URL}. Start the API first. Details: {exc}", title="API is unavailable")


try:
    health = get_json("/health")
    if health["status"] != "ok":
        st.warning("Project data or the trained model is missing. Run notebooks 05–07 first.", title="Setup is incomplete")
        st.stop()
except requests.RequestException as exc:
    api_error(exc)
    st.stop()

try:
    metrics_data = pd.DataFrame(get_json("/metrics")["models"])
    queue = pd.DataFrame(get_json("/cases", {"limit": 50})["cases"])
except (requests.RequestException, KeyError, ValueError) as exc:
    api_error(exc)
    st.stop()

graph_row = metrics_data.loc[metrics_data["model"] == "LightGBM + graph features"]
if not graph_row.empty:
    with st.container(horizontal=True):
        st.metric("Test PR-AUC", f"{graph_row.iloc[0]['test_pr_auc']:.3f}", border=True)
        st.metric("Test recall", f"{graph_row.iloc[0]['test_recall']:.1%}", border=True)
        st.metric("Test precision", f"{graph_row.iloc[0]['test_precision']:.1%}", border=True)
        st.metric("Review queue", f"{len(queue):,} shown", border=True)

score_tab, queue_tab, model_tab = st.tabs(["Score a transaction", "Review queue", "Model comparison"])

with score_tab:
    st.subheader("Try a transaction")
    st.caption("Enter account IDs and transaction details. Historical activity uses earlier steps only.")
    with st.form("score_transaction"):
        with st.container(horizontal=True):
            step = st.number_input("Time step", min_value=1, max_value=743, value=700, step=1)
            tx_type = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"])
            amount = st.number_input("Amount", min_value=0.0, max_value=10_000_000.0, value=100_000.0, step=1000.0)
        with st.container(horizontal=True):
            oldbalance_org = st.number_input("Sender balance before", min_value=0.0, value=200_000.0, step=1000.0)
            oldbalance_dest = st.number_input("Receiver balance before", min_value=0.0, value=0.0, step=1000.0)
            sender = st.text_input("Sender account ID", value="C1000000000")
            receiver = st.text_input("Receiver account ID", value="C2000000000")
        submitted = st.form_submit_button("Calculate risk score", type="primary")
    if submitted:
        try:
            response = requests.post(f"{API_URL}/score", json={
                "step": int(step), "type": tx_type, "amount": amount,
                "oldbalanceOrg": oldbalance_org, "oldbalanceDest": oldbalance_dest,
                "nameOrig": sender, "nameDest": receiver,
            }, timeout=60)
            response.raise_for_status()
            result = response.json()
            st.metric("Model risk score", f"{result['risk_score']:.1%}")
            if result["alert_candidate"]:
                st.warning("This transaction crossed the validation-selected review threshold.", title="Review candidate")
            else:
                st.success("This transaction is below the validation-selected review threshold.")
            st.caption(result["interpretation"])
            st.subheader("Largest score contributors")
            st.caption("Feature value is what the model received. Contribution shows how it shifted the model's internal score; it is not a percentage. A category with value 0 can still receive a contribution because the model considers combinations of features.")
            explanation = pd.DataFrame(result["top_contributors"])
            if not explanation.empty:
                explanation = explanation[["feature", "feature_value", "contribution", "direction"]]
                explanation = explanation.rename(columns={
                    "feature": "Model feature",
                    "feature_value": "Value used",
                    "contribution": "Score contribution",
                    "direction": "Effect",
                })
            st.dataframe(explanation, hide_index=True,
                         alt="Features that most raised or lowered the transaction score")
            st.subheader("Earlier account activity")
            st.json(result["history_features"])
        except requests.RequestException as exc:
            st.error(f"Scoring failed: {exc}", title="Could not score transaction")

with queue_tab:
    st.subheader("Highest risk candidates")
    if queue.empty:
        st.info("The review queue is empty.")
    else:
        visible = queue[[column for column in ["risk_rank", "risk_score", "alert_candidate", "step", "type", "amount", "nameOrig", "nameDest"] if column in queue]].copy()
        st.dataframe(visible, hide_index=True, alt="Highest risk synthetic PaySim transactions for human review")
        selected_rank = st.number_input("Case rank to inspect", min_value=1, max_value=max(1, int(queue["risk_rank"].max())), value=1, step=1)
        if st.button("Load case details", key="load_case"):
            try:
                details = get_json(f"/cases/{int(selected_rank)}")
                st.write({k: details.get(k) for k in ["risk_rank", "risk_score", "step", "type", "amount", "nameOrig", "nameDest"]})
                st.subheader("Model explanation")
                st.dataframe(pd.DataFrame(details.get("explanations", [])), hide_index=True,
                             alt="Local model contributions for selected case")
                st.subheader("Earlier account neighborhood")
                st.dataframe(pd.DataFrame(details.get("neighborhood", [])), hide_index=True,
                             alt="Earlier transactions connected to the selected synthetic account")
            except (requests.RequestException, KeyError, ValueError) as exc:
                st.error(f"Could not load the case: {exc}")

with model_tab:
    st.subheader("Held-out test results")
    st.caption("The test period is later in time than training. PR-AUC is useful when fraud labels are rare.")
    st.dataframe(metrics_data, hide_index=True, alt="Held-out test comparison of fraud risk models")
    st.info("PaySim is simulated data. These scores do not establish real-world fraud detection performance.")
