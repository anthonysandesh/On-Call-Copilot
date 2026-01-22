# Web Tier Latency Runbook

## High latency after deploy
- Compare the deploy timestamp to the first spike in p99 latency.
- Roll back to the previous version if latency increases and error rates climb.
- Inspect ingress and application pods for restarts or throttling.

## Elevated 5xx errors
- Check recent config changes in the gateway or service mesh.
- Validate upstream dependencies are healthy; trace errors back to the failing dependency.
- If only one region is impacted, drain traffic and redeploy with safe config.

## Cache saturation
- Verify cache hit ratio and eviction rates.
- Increase cache shard count temporarily and invalidate stale keys linked to the incident.
