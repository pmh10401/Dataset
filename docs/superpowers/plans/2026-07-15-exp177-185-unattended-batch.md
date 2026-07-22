# Exp177-185 Unattended Batch Implementation Plan

- [ ] Implement deterministic feature, score, conformal, and fusion helpers.
- [ ] Test label-free prediction records, lane exclusivity, fixed grids, and
      deterministic device selection.
- [ ] Implement stage commands with unique outputs and frozen hashes.
- [ ] Implement a dedicated sequential runner with retries, dependencies,
      per-stage timeouts, a five-hour checkpoint, and a twelve-hour deadline.
- [ ] Expose the dedicated queue state to the read-only dashboard.
- [ ] Run unit tests and a ten-dataset smoke batch.
- [ ] Register Exp177-185 and start one detached runner with seven workers.
- [ ] Verify dashboard visibility once and do not start an agent monitor.
