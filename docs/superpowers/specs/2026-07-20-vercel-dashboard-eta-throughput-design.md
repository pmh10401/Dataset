# Vercel Dashboard ETA and Throughput Design

## Goal

Show useful remaining time, recent processing speed, elapsed time, and remaining
dataset count on the Vercel dashboard without changing or polling the experiment
process directly.

## Data Flow

1. The local dashboard merges the canonical queue record with the matching live
   heartbeat so `started_at` is preserved.
2. The exporter observes the sanitized local snapshot every 10 seconds.
3. A process-local tracker retains at most two minutes of `(time, done)` samples
   for the active experiment.
4. The exporter calculates recent throughput only when time and completed count
   both increase.
5. Supabase receives the enriched, sanitized snapshot. No labels, logs, paths,
   process IDs, or experiment inputs are added.

## Calculation Rules

- `average speed = datasets_done / elapsed_minutes`
- `recent speed = delta_done / delta_minutes` over the available rolling window,
  capped at two minutes
- `remaining count = max(0, expected_datasets - datasets_done)`
- `ETA = remaining count / recent speed` when recent speed is available
- otherwise `ETA = remaining count / average speed`
- if neither speed is valid, ETA remains unavailable and the UI displays
  `계산 중`
- samples reset when the active experiment changes, progress decreases, or the
  experiment is no longer running

## UI

The existing four-metric band remains stable. The first two metrics display:

- Remaining time: formatted ETA plus remaining dataset count
- Processing speed: recent speed when available, otherwise average speed
- A small basis label: `최근 2분`, `최근 표본`, or `전체 평균`

Elapsed time remains in System Status. Values that cannot yet be calculated are
shown as `계산 중`, never as a misleading zero.

## Safety

- The exporter reads the existing local API only.
- No additional database reads or model calls are introduced.
- TEST labels and anomaly positions remain excluded from the snapshot.
- The tracker is in memory; restart only causes a short warm-up period.
- Supabase schema and RLS policies do not change.

## Verification

- Unit-test tracker reset, rolling speed, average fallback, ETA, and no-progress
  behavior.
- Unit-test snapshot sanitization after enrichment.
- Run Next.js tests and production build.
- Restart only the local dashboard and exporter, not the experiment.
- Verify local API, Supabase latest row, and Vercel-visible snapshot values.

