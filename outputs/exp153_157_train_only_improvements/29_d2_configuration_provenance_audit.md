# D2 configuration provenance audit

## Conclusion

**End-to-end strict TRAIN-only configuration provenance is not established.**

Exp153-157 recompute features, OOF scores, thresholds, candidate fusion, confirmation, and risk selection without TEST labels in the prediction phase. However, the frozen source configurations were inherited from experiments whose result summaries contain TEST-label metrics, and the chosen configurations align with the top observed mean-F1 configurations in those sweeps. This is sufficient to block an end-to-end strict TRAIN-only claim.

This finding was frozen before Exp153-157 full TEST evaluation and will not be used to change a source, alpha, threshold, fusion variant, confirmation variant, or risk target in the current experiments.

## Source-by-source evidence

| Source | Frozen configuration | Historical evidence | D2 finding |
|---|---|---|---|
| ROCKET Exp40 | `rocket_256_knn3_local_gap` | The Exp40 summary row with `count_cap_3pct` has mean F1 `0.5813338488097736`, the highest row shown in the frozen summary. | Historical TEST performance could have influenced configuration choice. |
| Exp55 | `train_global_minmax_clip_spectrogram_32_pca32_knn3` | The source is explicitly named `exp55_best`; its `count_cap_3pct` row has mean F1 `0.4035650105298579`, the leading Exp55 summary result. | TEST-informed selection is strongly indicated. |
| Exp56 | `glcm_rp_32_pca32_knn3` | The source is explicitly named `exp56_best`; its `count_cap_2pct` row has mean F1 `0.36475090609214944`, the leading Exp56 summary result. | TEST-informed selection is strongly indicated. |
| Exp84 | `aeon_mrh_mr1024_hk4_g32_prune1024_stable_tail_local_gap_knn3` | Its `count_cap_3pct` row has mean F1 `0.5760153083986965`, above the alternative Exp84 configurations in the frozen summary. | No prior preregistration proving TEST-independent selection was found. |

## Code lineage

1. `experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py:86-113` fixes the four source configurations.
2. `run_experiment_60_62_rocket_imaging_selector_variants.py:128-165` binds Exp55/56 to named prior result configurations.
3. `run_model_hard_research_experiments.py:1260-1297` defines the Exp84 sweep and the currently inherited configuration.
4. Exp153-157's bridge imports those frozen configurations through Exp152, then performs the new policy calculations without changing them.

## What remains valid

- The Exp153-157 prediction policy can be audited as family-independent and TEST-length independent.
- TRAIN OOF scores can be used to audit calibration and sensitivity variants.
- TEST labels remain evaluation-only after frozen prediction hashes.
- Current results remain useful as retrospective counterfactual ablations.

## Claims that remain prohibited

- The full development history was end-to-end strict TRAIN-only.
- The four source configurations were selected without observing historical TEST performance.
- The results are prospective or deployment validated.

## Frozen evidence hashes

- Exp40 summary: `cc4abef6aa17a7935354ffa077951d8d033ed2f327722eddbc717171f14a31da`
- Exp55 summary: `b1e5a0ab2cc30ca64231173f6fae991e6b4078f9f60bba7dd1bd57671ed7897d`
- Exp56 summary: `720db8f6b0b37a92f29e9e41f781589717f7e361c4af263ccf9ef6c651927f09`
- Exp84 summary: `c8c8f15e66baed031018d86db8433d888cb5d3038e0f0030a73707d16a294013`
- Exp151/152 source definition: `667449980f5860796604c2b5639277c9d096fac1c4bc534346f4752521e83516`
- Imaging selector definition: `4ab38ef8a0ad152ccfc18c1297c40aca31ba79ac29348b9f5da029682507b2ed`
- Model-hard experiment definition: `f9f06973009259590b4afd165dc275cff9b834a628ccb17e248d015d55680194`
