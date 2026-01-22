# Database Error Runbook

## Connection spikes
- Check connection counts versus max_connections; rotate credentials if stale clients are stuck.
- Restart only the affected application pods; avoid restarting the primary database during peak traffic.
- Use connection pooling metrics to identify runaway clients.

## Slow queries
- Identify queries with longest duration using pg_stat_statements or the database profiler.
- Add temporary indices for hot paths and verify query plans after changes.
- If replication lag is high, stop long-running analytical queries and fail traffic over to a healthy replica.

## Disk or storage saturation
- Clear down temporary tables and vacuum bloated tables.
- Expand storage by 10-20% and schedule follow-up capacity planning.
- Alert the platform team if IOPS are capped for more than 5 minutes.
