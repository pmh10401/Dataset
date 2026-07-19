# Exp226-235 Artifact Manifest

The machine-readable inventory is [MANIFEST.csv](MANIFEST.csv). It records 47
frozen audit, prediction, result, shadow, and repository summary artifacts with
SHA-256, byte size, code commit, metric scope, and limitations.

## Reproduction Identity

- Code commit: `78c53c4bb06c5931109612fd47185f0367814789`
- Branch: `experiment/exp177-185-batch`
- Database SHA-256: `8458e6787031afba104a2c93b6b578dacf183f9b85c5252938ec8e092673ef3c`
- Full source manifest SHA-256: `bf73796c046a6102fce53d3b6fc606afa1ed0a5161ebb6e92d94698ab56a68a0`
- Canonical source manifest SHA-256: `36810c14a5cca10d9758aa55144bcd785e1bae4d12278039cac83af2b96901ce`
- Seed: `20260719`
- External artifact root: `/Users/minho/Documents/Dataset/exp226_235_hard_alert_quality`

## Commands

```bash
python3 -m unittest discover \
  -s experiments/exp226_235_hard_alert_quality/tests -p 'test_*.py'

python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline \
  --stage all --experiments 226-235 --workers 7 --reset

# Resumed after explicitly supporting the pre-existing n_train < 5 abstention contract.
python3 -m experiments.exp226_235_hard_alert_quality.run_pipeline \
  --stage all --experiments 226-235 --workers 7 --resume

python3 -m experiments.exp226_235_hard_alert_quality.independent_reaggregate \
  --output-root /Users/minho/Documents/Dataset/exp226_235_hard_alert_quality \
  --experiments 226-235
```

## Scope

- All performance results are retrospective counterfactual evaluations.
- Hard alert metrics are autonomous.
- Review and research metrics are human-assisted or research-only and remain separate.
- Exp234 is research-only; Exp235 is a label-free prospective shadow input format.
- Exp230 is blocked because frozen repeated-seed/kernel evidence is unavailable.
- No TEST metric selected an operating alpha, threshold, margin, weight, cluster count, or FDR level.
- This run does not establish end-to-end strict TRAIN-only provenance or prospective deployment performance.

Volatile runtime heartbeat and resumable checkpoint files are intentionally not
included in the frozen 47-file release inventory.
