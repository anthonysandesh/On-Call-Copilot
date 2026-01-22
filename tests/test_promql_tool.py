import pytest

from tools.promql_tool import PromQLTool
from tools.tool_schemas import PromQLQuery
from tools.validators import PromQLValidationError, validate_promql_syntax


def test_promql_validation_passes():
    assert validate_promql_syntax("rate(http_requests_total[5m])")


def test_promql_validation_rejects_bad_query():
    with pytest.raises(PromQLValidationError):
        validate_promql_syntax("rate(http_requests_total[5m]")


def test_promql_tool_mock_returns_series():
    tool = PromQLTool(mode="mock")
    call = PromQLQuery(
        query="rate(http_requests_total[5m])",
        start="2024-01-01T00:00:00Z",
        end="2024-01-01T00:05:00Z",
        step_seconds=60,
    )
    result = tool.run(call)
    assert result.series
    assert result.tool_name == "promql_query"
