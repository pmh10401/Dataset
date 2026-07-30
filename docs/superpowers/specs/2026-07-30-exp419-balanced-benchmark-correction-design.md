# Exp419 Balanced Benchmark Correction Design

## 1. Decision and purpose

Exp419 remains the experiment that validates deterministic CPU process
parallelism for the unchanged Exp417 scientific computation. The first private
benchmark attempt exposed two pre-official development defects:

1. non-lexical cost-ordered shard inventories were passed directly to the
   lexical `Exp417CheckpointStore` universe contract; and
2. the preregistered 15-dataset timing subset was dominated by one dataset and
   could not represent the parallelism of the frozen 1,117-dataset population.

The checkpoint-universe defect is fixed and reviewed in commits `f54ac4d` and
`b706393`. This design corrects only the second defect before any valid timing
benchmark or official Exp419 output exists.

The objective is not to make an easier benchmark. It is to preserve the
original workload anchors while adding enough frozen TRAIN-only work that a
measured `2.0x` parallel speedup is mathematically possible with conservative
headroom.

## 2. Evidence for the correction

Under database SHA-256
`8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`,
the original 15-dataset subset has total proxy cost `388,627,551`.
`StarLightCurves_normal_3` alone contributes `353,424,384`, or `90.94%`.
Largest-first assignment therefore has an ideal proxy speedup ceiling of only
`1.0996059372x` for 2, 4, or 6 dataset workers.

The complete frozen population has total proxy cost `3,996,170,205`.
Its largest dataset contributes `8.84%`. Largest-first assignment produces:

- 4-worker loads between `999,042,458` and `999,042,646`;
- 6-worker loads between `666,028,190` and `666,028,501`.

The original subset therefore measures a single-dataset bottleneck rather than
the throughput structure of the official population.

The terminated attempt is diagnostic-only evidence:

- wrapper status: `143`;
- no benchmark JSON was published;
- no worker trial was qualified;
- no official checkpoint or output root was opened.

It must never be accepted, resumed, merged, or used as official evidence.

## 3. Frozen balanced-subset algorithm

The correction retains every original quantile anchor and adds datasets through
one deterministic TRAIN-only rule.

1. Compute the existing per-dataset cost and stratum from frozen TRAIN metadata.
2. Select the original anchors at quantiles `(0, 0.25, 0.50, 0.75, 1.0)` in
   each non-empty stratum, exactly as Exp419 already specifies.
3. Sort all remaining datasets by `(-cost, dataset)`.
4. Define a plan's proxy speedup headroom as:

   `sum(selected costs) / max(shard total cost)`.

5. Add the next remaining dataset until both the 4-worker and 6-worker
   largest-first plans have proxy speedup headroom of at least `2.5x`.
6. Return the selected dataset names in canonical lexical order.

The `2.5x` proxy threshold is a preregistered feasibility margin, not a
replacement for the measured `2.0x` gate. It gives process startup, database
reads, checkpoint writes, and load-proxy error room while retaining the actual
wall-clock requirement.

For the frozen inputs, this rule selects 19 datasets: the original 15 plus:

- `StarLightCurves_normal_2`, cost `171,196,416`;
- `HandOutlines_normal_1`, cost `153,600,300`;
- `StarLightCurves_normal_1`, cost `88,169,472`;
- `HandOutlines_normal_0`, cost `86,893,884`.

The corrected total is `888,487,623`; the largest dataset is `39.78%`.
The frozen proxy evidence is:

- 2-worker loads: `444,237,111`, `444,250,512`; headroom `1.9999698346x`;
- 4-worker loads: `353,424,384`, `178,469,604`, `178,343,436`,
  `178,250,199`; headroom `2.5139397937x`;
- 6-worker loads: `353,424,384`, `171,196,416`, `153,600,300`,
  `88,169,472`, `86,893,884`, `35,203,167`; headroom `2.5139397937x`.

The 2-worker candidate remains mandatory benchmark evidence but is not expected
to reach a measured `2.0x` after overhead. Qualification requires at least one
of the unchanged 4-worker or 6-worker candidates to pass all gates.

## 4. Scientific and operational invariants

The correction does not change:

- the exact 1,117-dataset official population;
- Exp417 preprocessing, architecture, horizons, seeds, scoring, calibration,
  qualification, lane exclusion, workload gates, or candidate identity;
- the `(1, 2, 4, 6)` worker candidates or `spawn` start method;
- one PyTorch intra-op and inter-op thread per process;
- the CPU-only backend;
- the `27,487,790,694`-byte aggregate RSS limit;
- exact one-worker versus parallel scientific parity;
- the measured `2.0x` wall-clock qualification gate;
- TEST-label isolation, review-only policy, or local-only artifact policy;
- the exact fresh approval phrase required after benchmark qualification.

MLX and MPS remain excluded. This correction changes only orchestration
metadata selection and cannot express a scientific compute stage in MLX
without changing the frozen Exp417 execution boundary. It therefore fails the
project's MLX eligibility condition before any device benchmark is warranted.

## 5. Evidence binding and failure behavior

The benchmark JSON must additionally bind:

- original anchor dataset names;
- deterministic added dataset names;
- proxy headroom threshold `2.5`;
- total selected cost;
- exact 2-, 4-, and 6-worker proxy shard loads and headroom values;
- the frozen context, input, Git, source-manifest, and selected-subset digests.

The official verifier must recompute the balanced selection and proof from the
current frozen TRAIN metadata. It rejects:

- the old 15-dataset subset;
- an easier or hand-picked replacement subset;
- missing, extra, duplicate, reordered, or cost-mutated records;
- altered threshold, loads, headroom, or digests;
- any benchmark produced from the terminated diagnostic root.

The invalid root remains untouched at:

`/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark`

The corrected no-clobber benchmark root is:

`/Users/mac/Documents/Dataset/exp419_deterministic_cpu_parallel_benchmark_balanced`

Any test, parity, finite-output, fallback, memory, source, input, Git, subset,
or measured-speed failure remains fail-closed and forbids the official run.

## 6. Code and document changes

The implementation changes are limited to:

- `contracts.py`: immutable `BenchmarkSelectionProof` with
  `anchor_datasets`, `added_datasets`, `selected_datasets`,
  `proxy_headroom_gate`, `total_cost`, exact 2/4/6-worker `shard_loads`,
  exact 2/4/6-worker `proxy_headrooms`, and canonical `sha256`;
- `planner.py`:
  `select_balanced_benchmark_datasets(costs) -> BenchmarkSelectionProof`;
- benchmark CLI: render the proof into the benchmark JSON;
- official CLI: recompute and verify the proof before opening run roots;
- reporting: bind the proof into official runtime evidence;
- tests: deterministic selection, adversarial verifier, and no-regression
  coverage;
- Exp419 design, preregistration, runbook, and implementation plan;
- `SOURCE_MANIFEST.json` regeneration after reviewed source changes.

Worker execution, coordinator scheduling, Exp417 scientific modules, and
official artifact names remain unchanged.

## 7. Test and execution sequence

Implementation follows test-first development:

1. add failing planner tests for retaining all 15 anchors, deterministic
   heaviest-first additions, exact `2.5x` termination, and the frozen 19-name
   result;
2. add failing CLI/verifier tests that reject the old 15-name subset and every
   mutated proof field;
3. implement the minimal selection/proof changes;
4. regenerate and verify the exact frozen source manifest;
5. run focused tests, full Exp419 tests, full Exp417 regressions, syntax,
   label-isolation, input-integrity, and clean-tree checks;
6. run the corrected private benchmark once with no automatic monitor;
7. independently verify all four trials, exact parity, finite output, no
   fallback, memory, actual speedup, utilization, and every digest;
8. only after qualification, request the exact fresh official approval phrase;
9. run and audit the official 1,117-dataset bundle.

The prior invalid benchmark is not a timing observation used to tune the new
selection. The selection rule is derived solely from the preregistered TRAIN
cost proxy and the mathematical feasibility of the unchanged measured gate.
