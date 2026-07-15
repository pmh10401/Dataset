# Exp163 Manifest

- branch: `exp153-157-train-only-improvements`
- commit: `7e5a252ae7ed4b124380ed09d39a7ffe3fa8a389`
- generated: `2026-07-14T14:41:31.422686+00:00`
- experiment: `experiment_163_alternative_source_tail_agreement_review`
- scope: retrospective counterfactual
- metric boundary: Hard autonomous; Standard/Priority and combined human-assisted
- limitation: candidate-empty gate uses frozen TEST-score outcome; not end-to-end strict TRAIN-only or prospective

| file | sha256 | stage |
|---|---|---|
| `experiments/exp163/__init__.py` | `0c688cb789a3dc4e240aec90c70880d1016728e7583aaa471ed8cc5afb51db55` | code |
| `experiments/exp163/alternative_tail_core.py` | `e18c3c08c6530b5b0088a4813e716f841042000a840c4c452771cb27df532472` | code |
| `experiments/exp163/exp163_io.py` | `ba85d9157dc1122eee7ff09dd2333a0c12e4e5fca534bd2127f493f25d3acf54` | code |
| `experiments/exp163/independent_reaggregate_exp163.py` | `b21cf32ecf1ca863c897dbdc6f5e5a0bf33c47f27d093d410dd3df6eca621d8e` | code |
| `experiments/exp163/run_experiment_163_alternative_source_tail_agreement_review.py` | `9a8c2ba717d5931d1b8fdc0e1fc3b90e117e331d668cad59649dc3fe7fb8e705` | code |
| `experiments/exp163/test_exp163_alternative_tail_review.py` | `beb8abf2804fe51bcca3dada3e0f65f736558ffdc7a91f14169a5ad2c520892d` | code |
| `experiments/exp163/test_exp163_structure.py` | `ca3d297e4e09d5428fce1da07c8f151c70e39eb1b9eb7b5305685f34648ab2b9` | code |
| `docs/EXP163_PREREGISTRATION_KO.md` | `894405ee9d176d1324fae10f5baaf73ebdfb23b0ca003bfa4ecd0c06cc0d6322` | documentation |
| `docs/EXP163_RUNBOOK_KO.md` | `072bc9edcfc20d361ffaf4b7aeff58cea1169311eaf47e45cc63cf03434b617c` | documentation |
| `docs/superpowers/specs/2026-07-14-exp163-alternative-source-tail-agreement-review-design.md` | `e89fc7c91b0d04fd8eda4f7208badee472e3388834a2879acd4450894d54ee91` | documentation |
| `docs/superpowers/plans/2026-07-14-exp163-alternative-source-tail-agreement-review.md` | `ad65c8e15f72af2241b9349bac687a79672c51ecfef2aed61e174b357dca6c74` | documentation |
| `outputs/antigravity_collaboration/EXP163_GEMINI_IMPLEMENTATION_PROMPT_KO.md` | `a0efbebffd54e6bf48ef1b86ec19be7e00b4fa7cb626c2f02d488df752590e67` | documentation |
| `outputs/antigravity_collaboration/exp163_implementation_agy.log` | `b308ecc72f3652cae0a2e8b4e9ff799e90b51ec82bacbd4930fd4189ea4ea6cf` | documentation |
| `outputs/exp163_alternative_source_tail_agreement_review/00_prediction_config.json` | `65ebb132a1ea391dc9b609343bbad964a74c663d827228791985f19e297952e1` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/01_input_audit.csv` | `bb2437ee557ec7f0d3dd8f21dc372dda1a90afaf743e50d980563856785365dd` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/01_input_audit_summary.json` | `cc3ada7ccf1aa8da03d6dbe54a05ad2f0b7114bedd181eaf675312e4df811a42` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/02_exp163_predictions.jsonl` | `54fb3f1a94a268428b00835c7433d2ef0b033c8880a254c96c59d49e1477c655` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/03_prediction_manifest.json` | `aac177e2dddb89ae13b368315d528d53af958237facf8f844b475331aadbea81` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/04_exp163_results.csv` | `1f3710f273856fd949d41980037d4f7853e957bcd76f7db902b8ed80ee335ece` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/05_exp163_summary.csv` | `6b72fafbf3abe964b9fe300821ba6d80eabfe7b85cb5dfa38b02029a38d04fc8` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/06_exp163_summary.md` | `50b068cee6af6735d164fea49f0faddffd52c2b9ae9e3a36c58d730086278021` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/07_candidate_level_diff.csv` | `51a37ce392248ae2ae19b045bc8e2c1d86561aabb9839f057e6b191e361858b3` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/08_workload_summary.csv` | `182c6b419e78557465700ac5179b7eaa24e460d8cdd59ea7ec72e4a63fd2de87` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/09_source_agreement_diagnostics.csv` | `6723a4fb22efd660a8994d892754a97e01fba38378304cf4bb1e3e6b0204c126` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/09_threshold_resolution.csv` | `3604ae2b6a42dcaae2654e6e9aa3fa48443a1babce43596d2f6b297bf59e0b52` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/10_full_prediction_log.txt` | `17127fd9b1a980b0bd313e0f4c48bebdf80a878b76a37fc39cf8d327e1f3b064` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/11_independent_prediction_validation_log.txt` | `a2170c5b79ba4b62543b590abf084f8a06281b0e3c66e90ea6b526e8400e8862` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/11_independent_validation.json` | `069c24e4b555fd81704fa080e8699e919fad2f1b348c0d70209827c37547f09f` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/12_full_evaluation_log.txt` | `2642fb0749187b8ac682533b5f4113aa7afe8d454e0b2736a285562f97812b15` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/13_independent_full_validation_log.txt` | `069c24e4b555fd81704fa080e8699e919fad2f1b348c0d70209827c37547f09f` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/14_invariance_test_log.txt` | `62641989830f0deb7bf9dc4854639e3acad6c2fa64e2b89650ff3b62a7238877` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/15_invariance_test_report.md` | `c44828905a3388dba023e90d93406d675902bd3dae9df737e52888f4b688c087` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/16_presentation_claims.md` | `0605573e80253ca02e202e9f65cc58e9bd34ba206665888b135dc1bca3d68e00` | artifact |
| `outputs/exp163_alternative_source_tail_agreement_review/VERIFICATION_REPORT.md` | `3b7a25915ece66b6b2b2909695777dc3d5e72704a9c4923a89ff8417acb35224` | artifact |
