"""
API endpoint with what the model APIs have cost since the process started (V13).

GET /usage summarizes the usage recorder of the API embedding encoder and the
API reranker (operations "embedding" and "rerank") and the LLM client's
recorder: calls, failed calls, tokens, latency and the cost each provider
reported, by operation. It is read-only and off by default
(USAGE_ENDPOINT_ENABLED=true serves it; otherwise it answers 404), so a
deployment does not publish its spending unless asked to.
"""

from fastapi import APIRouter, HTTPException, status

from app.api.v1.dependencies import get_usage_settings, usage_summary

router = APIRouter()


@router.get("/usage")
def usage() -> dict:
    """
    Calls, tokens, latency and reported cost by operation since the process started.

    Returns:
        {"since": ISO time the recorder was created (None before the first
        API component was built), "api": {operation: row, "total": row},
        "llm": {operation: row, "total": row} or None without an LLM client}.
        A row has calls, failed_calls, input_tokens, output_tokens,
        total_latency_s, mean_latency_s, median_latency_s and
        reported_cost_usd (see UsageRecorder.summary).

    Raises:
        404: The endpoint is disabled (USAGE_ENDPOINT_ENABLED is not true)
    """
    if not get_usage_settings().enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    return usage_summary()
