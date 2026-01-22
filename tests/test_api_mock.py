from datetime import datetime, timezone

from fastapi.testclient import TestClient

from serving.api import app


def _sample_request():
    now = datetime.now(timezone.utc).isoformat()
    return {
        "incident_id": "TEST-1",
        "title": "Latency spike",
        "severity": "sev2",
        "timestamp": now,
        "alert_text": "p99 latency above threshold",
        "logs_text": "ERROR timeout to database",
        "metrics_snapshot": [
            {
                "name": "http_request_latency_seconds_p99",
                "labels": {"service": "demo", "region": "us-east-1"},
                "value": 2.1,
                "timestamp": now,
            }
        ],
        "environment": {
            "service": "demo",
            "cluster": "prod",
            "region": "us-east-1",
            "deploy_version": "v1",
        },
    }


def test_triage_endpoint_returns_structure():
    client = TestClient(app)
    resp = client.post("/v1/triage", json=_sample_request())
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["checklist"]) >= 3
    assert "postmortem" in data
    assert isinstance(data["tool_calls"], list)
    if data["tool_calls"]:
        call = data["tool_calls"][0]
        assert call["tool_name"] == "promql_query"
        assert "query" in call["arguments"]
