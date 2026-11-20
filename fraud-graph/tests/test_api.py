from src.api import ScoreRequest, account_network, cases, case_detail, health, score


def test_health_reports_required_local_artifacts():
    assert health()["status"] == "ok"


def test_score_returns_review_signal_and_explanation():
    result = score(
        ScoreRequest(
            step=700,
            type="TRANSFER",
            amount=100_000,
            oldbalanceOrg=120_000,
            oldbalanceDest=0,
            nameOrig="C_API_TEST_SENDER",
            nameDest="C_API_TEST_RECEIVER",
        ),
    )
    assert 0 <= result["risk_score"] <= 1
    assert isinstance(result["alert_candidate"], bool)
    assert result["interpretation"].startswith("Candidate for human review")
    assert result["top_contributors"]
    assert "prior_outgoing_count" in result["history_features"]


def test_case_list_does_not_include_actual_fraud_label():
    result = cases(limit=3)
    assert result["count"] == 3
    assert all("isFraud" not in case for case in result["cases"])


def test_case_detail_has_local_explanation_and_neighborhood():
    detail = case_detail(1)
    assert detail["risk_rank"] == 1
    assert "isFraud" not in detail
    assert isinstance(detail["explanations"], list)
    assert isinstance(detail["neighborhood"], list)


def test_network_endpoint_returns_prior_transactions_without_labels():
    result = account_network("C1618899924", before_step=743, limit=5)
    assert result["count"] <= 5
    assert all("isFraud" not in row for row in result["transactions"])
