import sys
import sqlite3
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import serve_rank_dashboard as dash


def test_summarize_by_config_includes_operational_metrics():
    rows = [
        {
            "dataset_name": "A",
            "config_name": "rocket",
            "threshold_method": "count_cap_2pct",
            "f1": "0.5",
            "auc_roc": "0.8",
            "auc_pr": "0.4",
            "predicted_count": "3",
            "tp": "2",
            "fp": "1",
            "fn": "1",
            "train_exceed_rate": "0.005",
        },
        {
            "dataset_name": "B",
            "config_name": "rocket",
            "threshold_method": "count_cap_2pct",
            "f1": "1.0",
            "auc_roc": "1.0",
            "auc_pr": "0.9",
            "predicted_count": "1",
            "tp": "1",
            "fp": "0",
            "fn": "0",
            "train_exceed_rate": "0.010",
        },
    ]
    summary = dash.summarize_by_config(rows)
    best = summary[0]
    assert best["strategy"] == "rocket::count_cap_2pct"
    assert best["mean_fp"] == 0.5
    assert best["mean_tp"] == 1.5
    assert round(best["alert_precision"], 4) == 0.75
    assert best["mean_train_exceed_rate"] == 0.0075
    assert best["operational_tier"] == "운영 기본 후보"


def test_parse_log_error_summary_groups_reasons_and_families():
    text = "\n".join(
        [
            "2026-07-07 01:00:00,000 [ERROR] Error evaluating dataset GestureMidAirD1_normal_1: setting an array element with a sequence.",
            "2026-07-07 01:00:01,000 [ERROR] Error evaluating dataset MelbournePedestrian_normal_1: Input contains NaN.",
            "2026-07-07 01:00:02,000 [ERROR] Error evaluating Yoga_normal_1: out of memory",
        ]
    )
    errors = dash.parse_error_events(text.splitlines())
    summary = dash.error_summary(errors)
    assert summary["total_errors"] == 3
    assert summary["by_reason"]["variable_length_inhomogeneous_shape"] == 1
    assert summary["by_reason"]["nan_or_inf"] == 1
    assert summary["by_reason"]["memory"] == 1
    assert summary["by_family"]["GestureMidAirD1"] == 1
    assert summary["by_family"]["Yoga"] == 1


def test_coverage_snapshot_reports_done_and_error_gap():
    rows = [{"dataset_name": "A"}, {"dataset_name": "B"}, {"dataset_name": "B"}]
    errors = [{"dataset_name": "C", "family": "C", "reason": "nan_or_inf"}]
    coverage = dash.coverage_snapshot(rows, expected=5, errors=errors)
    assert coverage["datasets_done"] == 2
    assert coverage["error_datasets"] == 1
    assert coverage["remaining_or_missing"] == 2
    assert coverage["coverage_percent"] == 40.0


def test_recent_from_rows_uses_latest_unique_dataset_as_fallback():
    rows = [
        {"dataset_name": "A", "sequence_length": "100", "train_count": "20"},
        {"dataset_name": "A", "sequence_length": "100", "train_count": "20"},
        {"dataset_name": "B", "sequence_length": "80", "train_score_count": "15"},
    ]
    recent = dash.recent_from_rows(rows, limit=2)
    assert [item["name"] for item in recent] == ["A", "B"]
    assert recent[-1]["length"] == 80
    assert recent[-1]["train_size"] == 15


def test_recent_from_rows_accepts_exp165_dataset_and_n_train_columns():
    recent = dash.recent_from_rows([{"dataset": "SmallTrain", "n_train": "8"}])

    assert recent == [{"time": "csv", "name": "SmallTrain", "length": "-", "train_size": 8}]


def test_family_trouble_spots_rank_problem_families_for_strategy():
    rows = [
        {"dataset_name": "A_normal_1", "config_name": "rocket", "threshold_method": "cap2", "f1": "0", "tp": "0", "fp": "2", "fn": "1", "predicted_count": "2"},
        {"dataset_name": "A_normal_2", "config_name": "rocket", "threshold_method": "cap2", "f1": "0.5", "tp": "1", "fp": "1", "fn": "1", "predicted_count": "2"},
        {"dataset_name": "B_normal_1", "config_name": "rocket", "threshold_method": "cap2", "f1": "1", "tp": "1", "fp": "0", "fn": "0", "predicted_count": "1"},
        {"dataset_name": "A_normal_1", "config_name": "rocket", "threshold_method": "cap3", "f1": "1", "tp": "1", "fp": "0", "fn": "0", "predicted_count": "1"},
    ]
    spots = dash.family_trouble_spots(rows, "rocket::cap2", limit=2)
    assert spots[0]["family"] == "A"
    assert spots[0]["datasets"] == 2
    assert spots[0]["zero_f1_count"] == 1
    assert spots[0]["total_fp"] == 3
    assert spots[0]["alert_precision"] == 1 / 4


def test_throughput_snapshot_uses_recent_progress_events():
    lines = [
        "2026-07-07 08:00:00,000 [INFO] Progress: [  10/100] rows=90 | best=x meanF1=0.1",
        "2026-07-07 08:05:00,000 [INFO] Progress: [  30/100] rows=270 | best=x meanF1=0.2",
    ]
    events = dash.parse_progress_events(lines)
    snapshot = dash.throughput_snapshot(events, done=30, expected=100, elapsed=600)
    assert snapshot["recent_datasets_per_minute"] == 4.0
    assert snapshot["average_datasets_per_minute"] == 3.0
    assert round(snapshot["recent_eta_seconds"]) == 1050
    assert snapshot["last_progress"]["done"] == 30


def test_experiment_compare_includes_running_and_delta_to_best():
    completed = [
        {
            "id": "done1",
            "label": "Done 1",
            "best_strategy": "a",
            "mean_f1": 0.5,
            "mean_fp": 1.0,
            "history_status": "completed_valid",
            "comparison_eligible": True,
        },
        {
            "id": "done2",
            "label": "Done 2",
            "best_strategy": "b",
            "mean_f1": 0.7,
            "mean_fp": 2.0,
            "history_status": "completed_valid",
            "comparison_eligible": True,
        },
    ]
    active_summary = [{"strategy": "active", "mean_f1": 0.6, "mean_fp": 0.5}]
    rows = dash.experiment_compare_snapshot("active_exp", "Active Exp", active_summary, completed, is_running=True)
    assert rows[0]["status"] == "running"
    assert rows[0]["id"] == "active_exp"
    assert rows[0]["delta_vs_best_f1"] == -0.1
    assert rows[1]["id"] == "done2"
    assert rows[1]["delta_vs_best_f1"] == 0.0


def test_invalid_history_is_visible_but_excluded_from_compare():
    history = [
        {"id": "valid", "history_status": "completed_valid", "comparison_eligible": True, "mean_f1": 0.6},
        {"id": "invalid", "history_status": "invalid", "comparison_eligible": False, "mean_f1": 0.9},
        {"id": "research", "history_status": "completed_research", "comparison_eligible": True, "mean_f1": 0.8},
        {"id": "no_metric", "history_status": "completed_valid", "comparison_eligible": True, "mean_f1": None},
    ]

    rows = dash.experiment_compare_snapshot("", "", [], history, is_running=False)

    assert [row["id"] for row in rows] == ["valid"]


def test_running_experiment_is_visible_without_metrics():
    rows = dash.experiment_compare_snapshot(
        "experiment_153_157_train_only_improvements",
        "Exp153-157",
        [],
        [],
        is_running=True,
    )

    assert rows == [
        {
            "id": "experiment_153_157_train_only_improvements",
            "label": "Exp153-157",
            "status": "running",
            "best_strategy": None,
            "mean_f1": None,
            "median_f1": None,
            "mean_fp": None,
            "alert_precision": None,
            "mean_train_exceed_rate": None,
            "operational_tier": None,
            "operational_excluded": None,
            "delta_vs_best_f1": None,
        }
    ]


def test_experiment_compare_allows_only_running_active_summary_outside_history():
    completed = [
        {
            "id": "valid",
            "label": "Valid",
            "mean_f1": 0.6,
            "history_status": "completed_valid",
            "comparison_eligible": True,
        }
    ]
    for history_status in ("invalid", "completed_research"):
        rows = dash.experiment_compare_snapshot(
            f"{history_status}_active",
            "Non-running active",
            [{"strategy": "active", "mean_f1": 0.9, "history_status": history_status}],
            completed,
            is_running=False,
        )
        assert [row["id"] for row in rows] == ["valid"]

    running = dash.experiment_compare_snapshot(
        "experiment_153_157_train_only_improvements",
        "Exp153-157",
        [],
        completed,
        is_running=True,
    )
    assert running[0]["id"] == "experiment_153_157_train_only_improvements"
    assert running[0]["status"] == "running"
    assert running[0]["mean_f1"] is None


def test_metric_glossary_has_plain_korean_examples():
    glossary = dash.metric_glossary_snapshot()
    by_id = {item["id"]: item for item in glossary}
    assert "mean_fp" in by_id
    assert "alert_precision" in by_id
    assert "train_exceed" in by_id
    assert "정상" in by_id["mean_fp"]["description"]
    assert "예:" in by_id["mean_fp"]["example"]
    assert by_id["alert_precision"]["direction"]
    assert by_id["train_exceed"]["group"] == "운영 지표"


def test_metric_help_ui_is_collapsible_and_click_targeted():
    html = dash.INDEX_HTML
    assert '<details class="panel metric-help-panel"' in html
    assert "<summary" in html
    assert "open>" not in html.split('<details class="panel metric-help-panel"', 1)[1].split("</details>", 1)[0]
    assert 'id="metricHelpPopover"' in html
    assert "showMetricHelp" in html
    assert "hideMetricHelp" in html


def test_resource_metric_uses_compact_font_class():
    html = dash.INDEX_HTML
    assert 'class="value resource-value" id="cpu"' in html
    assert ".resource-value" in html
    assert "font-size: 20px" in html


def test_experiment_compare_does_not_hide_lower_ranked_completed_runs():
    html = dash.INDEX_HTML
    compare_block = html.split("const compareRows = document.getElementById('compareRows');", 1)[1].split("setText('compareInfo'", 1)[0]
    assert ".slice(0, 10)" not in compare_block


def test_dashboard_registers_new_queued_experiments():
    assert "experiment_43_explanation_space_transforms" in dash.EXPERIMENTS
    assert "experiment_44_classical_embedding_baselines" in dash.EXPERIMENTS
    assert dash.EXPERIMENTS["experiment_43_explanation_space_transforms"]["detail_csv"].name == "experiment_43_explanation_space_transforms_results.csv"
    assert dash.EXPERIMENTS["experiment_44_classical_embedding_baselines"]["summary_csv"].name == "experiment_44_classical_embedding_baselines_summary.csv"
    assert "experiment_153_157_train_only_improvements" in dash.EXPERIMENTS


def test_process_tree_snapshot_aggregates_child_processes():
    ps_output = "\n".join(
        [
            "100 1 10.0 1.0 python",
            "101 100 40.0 2.0 python",
            "102 101 50.0 3.0 python",
            "200 1 90.0 4.0 other",
        ]
    )
    snapshot = dash.process_tree_snapshot(100, ps_output=ps_output)
    assert snapshot["pid"] == 100
    assert snapshot["process_count"] == 3
    assert snapshot["cpu"] == 100.0
    assert snapshot["mem"] == 6.0
    assert snapshot["child_pids"] == [101, 102]


def test_resource_sampling_does_not_raise_when_process_creation_is_temporarily_unavailable():
    original = dash.subprocess.check_output

    def unavailable(*args, **kwargs):
        raise BlockingIOError(35, "Resource temporarily unavailable")

    dash.subprocess.check_output = unavailable
    try:
        assert dash.process_tree_snapshot(100) is None
        assert dash.system_cpu_snapshot()["available"] is False
    finally:
        dash.subprocess.check_output = original


def test_system_cpu_snapshot_parses_top_output():
    top_output = "CPU usage: 12.34% user, 5.66% sys, 82.00% idle"
    snapshot = dash.system_cpu_snapshot(top_output=top_output)
    assert snapshot["cpu_percent"] == 18.0
    assert snapshot["cpu_user_percent"] == 12.34
    assert snapshot["cpu_system_percent"] == 5.66
    assert snapshot["source"] == "top"


def test_gpu_snapshot_handles_unavailable_sampler():
    snapshot = dash.gpu_snapshot(nvidia_output="", powermetrics_output="", errors=["powermetrics requires sudo"])
    assert snapshot["available"] is False
    assert "requires sudo" in snapshot["note"]


def test_completed_snapshot_exposes_operational_columns():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        detail = tmp_path / "detail.csv"
        summary = tmp_path / "summary.csv"
        detail.write_text("dataset_name,config_name,threshold_method,f1,tp,fp,fn,predicted_count,train_exceed_rate\nA,c,m,1,1,0,0,1,0.0\n")
        summary.write_text(
            "config_name,threshold_method,mean_f1,median_f1,mean_fp,mean_tp,alert_precision,mean_train_exceed_rate,mean_predicted_count,num_datasets\n"
            "c,m,1.0,1.0,0.0,1.0,1.0,0.0,1.0,1\n"
        )
        original = dash.EXPERIMENTS.copy()
        try:
            dash.EXPERIMENTS.clear()
            dash.EXPERIMENTS["x"] = {
                "label": "X",
                "detail_csv": detail,
                "summary_csv": summary,
                "stdout_log": tmp_path / "out.log",
            }
            snapshot = dash.completed_result_snapshot({"completed": ["x"], "history": [{"id": "x", "finished_at": "now"}]})
        finally:
            dash.EXPERIMENTS.clear()
            dash.EXPERIMENTS.update(original)
        row = {item["id"]: item for item in snapshot}["x"]
        assert row["mean_fp"] == 0.0
        assert row["alert_precision"] == 1.0
        assert row["mean_train_exceed_rate"] == 0.0


def test_unified_experiment_items_include_external_planned_and_blocked_states():
    state = {"queue": ["queued_exp"], "completed": ["done_exp"], "running": None}
    external = {"id": "live_exp", "label": "Live", "status": "running", "expected_datasets": 10}
    plan = [
        {"id": "planned_exp", "label": "Planned", "status": "planned"},
        {"id": "blocked_exp", "label": "Blocked", "status": "blocked", "blocked_reason": "missing grain"},
    ]

    items = dash.unified_experiment_items(state, external, plan)
    by_id = {item["id"]: item for item in items}

    assert by_id["live_exp"]["status"] == "running"
    assert by_id["queued_exp"]["status"] == "queued"
    assert by_id["planned_exp"]["status"] == "planned"
    assert by_id["blocked_exp"]["blocked_reason"] == "missing grain"


def test_dashboard_detects_active_exp151_152_from_process_and_checkpoint():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        checkpoint = output_dir / "prediction_checkpoint_v2.jsonl"
        checkpoint.write_text("{}\n{}\n{}\n", encoding="utf-8")
        ps_output = (
            "123 1 12.0 2.0 /usr/bin/python3 "
            "experiments/exp151_152/run_experiment_151_152_virtual_run_conformal_policy_v2.py "
            "--phase predict --workers 7 --output-dir "
            f"{output_dir}\n"
        )

        live = dash.detect_exp151_152_run(ps_output=ps_output)

        assert live["status"] == "running"
        assert live["id"] == "experiment_151_152_virtual_run_conformal_policy_v2"
        assert live["pid"] == 123
        assert live["datasets_done"] == 3
        assert live["rows"] == 24
        assert live["expected_datasets"] == 1117


def test_dashboard_detects_active_exp153_157_cache_from_process_and_cache_files():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        cache_dir = output_dir / "cache"
        cache_dir.mkdir()
        for index in range(3):
            (cache_dir / f"dataset_{index}.npz").touch()
        ps_output = (
            "321 1 80.0 2.0 /usr/bin/python3 "
            "experiments/exp153_157/run_experiment_153_157_pipeline.py "
            "--stage cache --workers 7 --output-dir "
            f"{output_dir}\n"
        )

        live = dash.detect_exp153_157_run(ps_output=ps_output)

        assert live["status"] == "running"
        assert live["id"] == "experiment_153_157_train_only_improvements"
        assert live["pid"] == 321
        assert live["phase"] == "cache"
        assert live["datasets_done"] == 3
        assert live["expected_datasets"] == 1117
        assert abs(live["progress_percent"] - 3 / 1117 * 100) < 1e-9


def test_unified_experiment_items_running_external_overrides_same_queued_id():
    exp_id = "experiment_153_157_train_only_improvements"
    state = {"queue": [exp_id], "completed": [], "running": None}
    external = {
        "id": exp_id,
        "label": "Exp153-157",
        "status": "running",
        "progress_percent": 12.5,
    }

    items = dash.unified_experiment_items(state, external, [])
    by_id = {item["id"]: item for item in items}

    assert by_id[exp_id]["status"] == "running"
    assert by_id[exp_id]["progress_percent"] == 12.5


def test_unified_experiment_items_preserve_completed_history_order():
    items = dash.unified_experiment_items({"completed": ["z_old", "a_new"]}, {}, [])
    completed = [item["id"] for item in items if item["status"] == "completed"]

    assert completed == ["z_old", "a_new"]


def test_unified_experiment_items_mark_bundle_complete_after_external_completion():
    plan = [{"id": "audit", "status": "planned", "runner_id": "bundle"}]
    items = dash.unified_experiment_items({}, {"id": "bundle", "status": "completed"}, plan)

    assert items[0]["status"] == "completed"


def test_dataset_context_snapshot_explains_benchmark_grain():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "sample.db"
        conn = sqlite3.connect(db_path)
        conn.executescript(
            """
            CREATE TABLE datasets (
                id INTEGER PRIMARY KEY, name TEXT, train_normal_count INTEGER,
                test_normal_count INTEGER, test_anomaly_count INTEGER, test_total_count INTEGER
            );
            CREATE TABLE instances (
                id INTEGER PRIMARY KEY, dataset_id INTEGER, split TEXT,
                instance_index INTEGER, label TEXT, values_blob BLOB, labels_blob BLOB
            );
            INSERT INTO datasets VALUES (1, 'Demo', 3, 9, 1, 10);
            """
        )
        conn.commit()
        conn.close()

        context = dash.dataset_context_snapshot(db_path)

        assert context["dataset_count"] == 1
        assert context["train_normal_instances"] == 3
        assert context["test_anomaly_instances"] == 1
        assert context["candidate_index_meaning"] == "TEST time-series instance index"
        assert context["production_run_mapping_verified"] is False


def test_gpt_handoff_contains_queue_metrics_and_dataset_caveat():
    status = {
        "active_experiment": "exp_demo",
        "progress_percent": 25.0,
        "unified_experiments": [{"id": "next", "status": "planned"}],
        "experiment_history": [
            {
                "id": "exp_old",
                "label": "Exp old",
                "history_status": "completed_research",
                "mean_f1": 0.5,
                "mean_fp": 1.0,
                "alert_precision": 0.8,
                "mean_combined_f1": 0.6,
                "review_candidate_count": 3,
                "status_reason": "retrospective only",
            }
        ],
        "runtime_health": {"heartbeat": {}},
    }
    context = {
        "dataset_count": 1119,
        "evaluation_dataset_count": 1117,
        "candidate_index_meaning": "TEST time-series instance index",
        "production_run_mapping_verified": False,
    }

    handoff = dash.gpt_handoff_snapshot(status, context)
    markdown = dash.gpt_handoff_markdown(handoff)

    assert handoff["schema_version"] == "1.0"
    assert handoff["current_status"]["active_experiment"] == "exp_demo"
    assert handoff["experiment_history"][0]["id"] == "exp_old"
    assert "## Experiment history" in markdown
    assert "Exp old | completed_research | 0.5 | 1.0 | 0.8 | 0.6 | 3 | retrospective only" in markdown
    assert "TEST time-series instance index" in markdown
    assert "production run mapping is not verified" in markdown


def test_gpt_handoff_markdown_limits_completed_experiments_but_json_keeps_them():
    items = [{"id": f"done_{index}", "status": "completed"} for index in range(15)]
    handoff = dash.gpt_handoff_snapshot(
        {"active_experiment": "x", "progress_percent": 0, "unified_experiments": items},
        {"candidate_index_meaning": "TEST time-series instance index", "production_run_mapping_verified": False},
    )

    markdown = dash.gpt_handoff_markdown(handoff)

    assert len(handoff["experiment_work_items"]) == 15
    assert markdown.count(": completed") == 10
    assert "5 additional completed experiments omitted" in markdown


def test_dashboard_has_gpt_export_and_dataset_context_controls():
    html = dash.INDEX_HTML
    assert 'id="copyGptHandoff"' in html
    assert 'id="downloadGptJson"' in html
    assert 'id="datasetContext"' in html


def test_dashboard_server_supports_clean_restart():
    assert dash.ReusableThreadingHTTPServer.allow_reuse_address is True
    assert dash.ReusableThreadingHTTPServer.daemon_threads is True


def test_history_snapshot_uses_summary_without_loading_regular_detail_csv():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = root / "summary.csv"
        detail = root / "large_detail.csv"
        summary.write_text("mean_f1,mean_fp,mean_tp\n0.5,1,2\n", encoding="utf-8")
        detail.write_text("dataset_name,f1\nA,0.5\n", encoding="utf-8")
        original_experiments = dash.EXPERIMENTS.copy()
        original_historical = dash.HISTORICAL_EXPERIMENT_META.copy()
        original_reader = dash.read_csv_rows
        calls = []
        try:
            dash.EXPERIMENTS.clear()
            dash.HISTORICAL_EXPERIMENT_META.clear()
            dash.EXPERIMENTS["x"] = {
                "label": "X",
                "summary_csv": summary,
                "detail_csv": detail,
                "stdout_log": root / "run.log",
            }

            def recording_reader(path):
                calls.append(Path(path))
                return original_reader(path)

            dash.read_csv_rows = recording_reader
            dash.completed_result_snapshot({"completed": ["x"], "history": []})
        finally:
            dash.read_csv_rows = original_reader
            dash.EXPERIMENTS.clear()
            dash.EXPERIMENTS.update(original_experiments)
            dash.HISTORICAL_EXPERIMENT_META.clear()
            dash.HISTORICAL_EXPERIMENT_META.update(original_historical)

        assert summary in calls
        assert detail not in calls


def test_large_csv_is_not_retained_in_dashboard_cache():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "large.csv"
        path.write_text("dataset_name\n" + "A\n" * 20, encoding="utf-8")
        original_limit = dash.MAX_CACHED_CSV_BYTES
        try:
            dash.MAX_CACHED_CSV_BYTES = 10
            dash.read_csv_rows(path)
            assert str(path) not in dash.CSV_CACHE
        finally:
            dash.MAX_CACHED_CSV_BYTES = original_limit
            dash.CSV_CACHE.pop(str(path), None)


def test_history_ui_has_status_filter_and_separate_review_metrics():
    html = dash.INDEX_HTML

    assert '<h2>Experiment History</h2>' in html
    assert 'id="historyStatusFilter"' in html
    assert 'history_status' in html
    assert 'metric_availability_reason' in html
    assert 'status_reason' in html
    assert 'mean_combined_f1' in html
    assert 'review_candidate_count' in html
    assert 'historyMetricText' in html
    assert "historyMetricText(variant, 'mean_f1', fmt, item)" in html
    assert 'history-variants' in html
    assert 'history-artifacts' in html
    assert 'renderHistoryArtifacts' in html
    assert 'item.artifact_links' in html
    assert 'table-layout: fixed' in html
    assert 'Completed Results' not in html
    for label in ("유효 완료", "연구 완료", "무효", "차단", "미실행", "실행 중", "대기"):
        assert label in html


def test_history_metric_text_rejects_unavailable_values_before_number_conversion():
    html = dash.INDEX_HTML
    helper = html.split("function historyMetricText", 1)[1].split("function renderHistoryVariants", 1)[0]

    assert "const value = item[field];" in helper
    assert "value === null || value === undefined || value === ''" in helper
    assert "return historyReason(reasonItem);" in helper
    assert "Number.isFinite(Number(value))" in helper
    assert helper.index("value === null || value === undefined || value === ''") < helper.index("Number.isFinite(Number(value))")
    assert "historyMetricText(item, 'mean_f1', fmt)" in html
    assert "historyMetricText(variant, 'mean_f1', fmt, item)" in html


def test_historical_catalog_covers_exp138_through_exp168():
    for number in range(138, 169):
        assert any(exp_id.startswith(f"experiment_{number}") for exp_id in dash.HISTORICAL_EXPERIMENT_META)


def test_normalize_runtime_state_hides_dead_running_pid_and_stale_queue_item():
    state = {
        "completed": [],
        "queue": ["experiment_153_157_train_only_improvements"],
        "running": {
            "id": "experiment_153_157_train_only_improvements",
            "pid": 99999998,
            "child_pid": 99999999,
        },
    }

    normalized = dash.normalize_runtime_state_for_display(state, pid_checker=lambda _pid: False)

    assert normalized["running"] is None
    assert normalized["queue"] == []


def test_exp164_is_audit_only_and_exp165_is_registered_for_live_queue():
    exp164 = dash.HISTORICAL_EXPERIMENT_META["experiment_164_train_normal_agreement_risk_audit"]
    exp165 = dash.HISTORICAL_EXPERIMENT_META["experiment_165_small_sample_calibration"]

    assert exp164["comparison_eligible"] is False
    assert exp164["scope"] == "retrospective_train_normal_calibration_audit"
    assert exp165["comparison_eligible"] is False
    assert "experiment_165_small_sample_calibration" in dash.EXPERIMENTS
    assert dash.EXPERIMENTS["experiment_165_small_sample_calibration"]["history_status"] == "completed_research"
    assert dash.EXPERIMENTS["experiment_165_small_sample_calibration"]["comparison_eligible"] is False


def test_exp167_168_are_registered_as_non_operational_research_results():
    exp167_id = "experiment_167_score_calibration_parity_audit"
    exp168_id = "experiment_168_fold_matched_parity_evaluation"

    for exp_id in (exp167_id, exp168_id):
        assert exp_id in dash.EXPERIMENTS
        assert exp_id in dash.HISTORICAL_EXPERIMENT_META
        assert dash.EXPERIMENTS[exp_id]["history_status"] == "completed_research"
        assert dash.EXPERIMENTS[exp_id]["comparison_eligible"] is False

    exp167 = dash.EXPERIMENTS[exp167_id]
    assert exp167["scope"] == "label_free_score_calibration_parity_audit"
    assert exp167["expected_datasets"] == 316

    exp168 = dash.EXPERIMENTS[exp168_id]
    assert exp168["scope"] == "retrospective_human_assisted_review_sensitivity"
    assert exp168["expected_datasets"] == 316
    assert exp168["selected_alpha"] is None
    assert exp168["autonomous_metric_scope"] == "not_evaluated"


def test_experiment_catalog_includes_exp33_reconstruction_artifacts():
    info = dash.EXPERIMENTS["experiment_33_evalset_reconstruction"]

    assert info["history_status"] == "completed_research"
    assert info["comparison_eligible"] is False
    assert info["detail_csv"].name == "experiment_33_evalset_reconstruction_results.csv"
    assert info["summary_csv"].name == "experiment_33_evalset_reconstruction_summary.csv"
    assert info["stdout_log"].name == "experiment_33_evalset_reconstruction.log"
    links = dash.history_artifact_links("experiment_33_evalset_reconstruction", info)
    assert {link["kind"] for link in links} == {"detail", "summary", "log"}


def test_history_artifact_path_rejects_unknown_ids_and_paths_outside_allowed_roots():
    assert dash.history_artifact_path("missing", "summary") is None
    assert dash.history_artifact_path("experiment_33_evalset_reconstruction", "unknown") is None

    experiment_id = "experiment_999_external_artifact"
    original = dash.EXPERIMENTS.get(experiment_id)
    dash.EXPERIMENTS[experiment_id] = {"summary_csv": Path("/tmp/not-allowed.csv")}
    try:
        assert dash.history_artifact_path(experiment_id, "summary") is None
    finally:
        if original is None:
            dash.EXPERIMENTS.pop(experiment_id, None)
        else:
            dash.EXPERIMENTS[experiment_id] = original


def test_historical_catalog_records_have_explicit_status_contracts():
    meta = dash.HISTORICAL_EXPERIMENT_META
    required = {"label", "history_status", "comparison_eligible", "scope", "status_reason"}
    for info in meta.values():
        assert required <= set(info)
        for key in ("detail_csv", "summary_csv", "stdout_log"):
            if key in info:
                path_text = str(info[key]).lower()
                assert all(marker not in path_text for marker in ("smoke", "probe", "check"))

    assert meta["experiment_149_virtual_run_conformal_policy"]["history_status"] == "invalid"
    assert meta["experiment_150_virtual_run_conformal_policy"]["history_status"] == "invalid"
    assert meta["experiment_145_train_normal_count_budget"]["history_status"] == "blocked"
    assert meta["experiment_148_policy_level_train_only_train_budget"]["history_status"] == "blocked"
    assert meta["experiment_151_virtual_run_conformal_policy_v2"]["selected_alpha"] is None
    assert meta["experiment_152_virtual_run_conformal_policy_v2"]["selected_alpha"] is None


def test_experiment_history_status_reconciles_state_before_artifacts():
    running = dash.experiment_history_status(
        "historical",
        {"history_status": "completed_research", "detail_csv": "/missing.csv"},
        {"running": {"id": "historical"}},
    )
    assert running == {"history_status": "running", "status_reason": "active process"}

    queued = dash.experiment_history_status(
        "historical",
        {"history_status": "completed_research", "detail_csv": "/missing.csv"},
        {"queue": ["historical"]},
    )
    assert queued == {"history_status": "queued", "status_reason": "sequential queue"}

    invalid = dash.experiment_history_status(
        "historical",
        {"history_status": "invalid", "status_reason": "known defect"},
        {},
    )
    assert invalid == {"history_status": "invalid", "status_reason": "known defect"}


def test_experiment_history_status_uses_artifact_presence_for_research_rows():
    with tempfile.TemporaryDirectory() as tmp:
        detail = Path(tmp) / "full_results.csv"
        detail.write_text("dataset_name\nA\n", encoding="utf-8")
        complete = dash.experiment_history_status(
            "historical",
            {"history_status": "completed_research", "status_reason": "audit", "detail_csv": detail},
            {},
        )
        missing = dash.experiment_history_status(
            "historical",
            {"history_status": "completed_research", "status_reason": "audit", "detail_csv": detail.with_name("missing.csv")},
            {},
        )

    assert complete == {"history_status": "completed_research", "status_reason": "audit"}
    assert missing == {"history_status": "not_run", "status_reason": "result artifact missing"}


def test_hard_schema_never_uses_combined_f1_as_autonomous_f1():
    metrics = dash.normalize_historical_metrics(
        {
            "mean_hard_f1": "0.60",
            "mean_combined_f1": "0.82",
            "mean_hard_fp": "0.30",
            "hard_alert_precision": "0.84",
            "standard_review_total_candidates": "639",
        },
        [],
    )

    assert metrics["mean_f1"] == 0.60
    assert metrics["mean_combined_f1"] == 0.82
    assert metrics["mean_fp"] == 0.30
    assert metrics["alert_precision"] == 0.84
    assert metrics["review_candidate_count"] == 639
    assert metrics["standard_review_candidate_count"] == 639
    assert metrics["priority_review_candidate_count"] is None
    assert metrics["total_review_candidate_count"] == 639


def test_audit_schema_reports_reason_instead_of_fabricated_f1():
    metrics = dash.normalize_historical_metrics({"binding_dataset_count": "12"}, [])

    assert metrics["mean_f1"] is None
    assert metrics["metric_availability_reason"] == "audit only"


def test_alpha_variants_remain_ordered_and_selected_alpha_stays_none():
    metrics = dash.normalize_historical_metrics(
        {"mean_hard_f1": "0.20", "mean_combined_f1": "0.90"},
        [
            {"alpha": "0.005", "mean_hard_f1": "0.10", "mean_combined_f1": "0.30"},
            {"alpha": "0.01", "mean_hard_f1": "0.20", "mean_combined_f1": "0.60"},
            {"alpha": "0.02", "mean_hard_f1": "0.30", "mean_combined_f1": "0.90"},
        ],
    )

    assert [variant["alpha"] for variant in metrics["variants"]] == [0.005, 0.01, 0.02]
    assert [variant["mean_f1"] for variant in metrics["variants"]] == [0.10, 0.20, 0.30]
    assert metrics["selected_alpha"] is None


def test_hard_detail_rows_supply_mean_fields_and_micro_precision():
    metrics = dash.normalize_historical_metrics(
        {},
        [
            {"hard_tp": "2", "hard_fp": "1", "hard_fn": "1", "hard_f1": "0.67", "standard_review_count": "5"},
            {"hard_tp": "1", "hard_fp": "0", "hard_fn": "0", "hard_f1": "1.00", "standard_review_count": "3"},
        ],
    )

    assert metrics["mean_f1"] == 0.835
    assert metrics["mean_fp"] == 0.5
    assert metrics["mean_tp"] == 3
    assert metrics["mean_fn"] == 1
    assert metrics["alert_precision"] == 0.75
    assert metrics["review_candidate_count"] == 8


def test_exp149_150_stay_invalid_and_exp151_152_stay_completed_research():
    meta = dash.HISTORICAL_EXPERIMENT_META

    assert meta["experiment_149_virtual_run_conformal_policy"]["history_status"] == "invalid"
    assert meta["experiment_150_virtual_run_conformal_policy"]["history_status"] == "invalid"
    assert meta["experiment_151_virtual_run_conformal_policy_v2"]["history_status"] == "completed_research"
    assert meta["experiment_152_virtual_run_conformal_policy_v2"]["history_status"] == "completed_research"
    assert meta["experiment_151_virtual_run_conformal_policy_v2"]["selected_alpha"] is None
    assert meta["experiment_152_virtual_run_conformal_policy_v2"]["selected_alpha"] is None
    assert meta["experiment_149_virtual_run_conformal_policy"]["selected_alpha"] is None
    assert meta["experiment_150_virtual_run_conformal_policy"]["selected_alpha"] is None


def test_snapshot_filters_shared_alpha_rows_and_keeps_unselected_metrics_empty():
    experiment_ids = (
        "experiment_151_virtual_run_conformal_policy_v2",
        "experiment_152_virtual_run_conformal_policy_v2",
    )
    header = "experiment_id,alpha,mean_hard_f1,mean_combined_f1,hard_tp,hard_fp,standard_review_candidates,priority_review_candidates\n"
    rows = []
    for experiment_id, offset in zip(experiment_ids, (0, 10)):
        for index, alpha in enumerate(("0.005", "0.01", "0.02", "0.05")):
            rows.append(
                f"{experiment_id},{alpha},{0.10 + offset + index},{0.20 + offset + index},"
                f"{10 + offset + index},{1 + index},{20 + offset + index},{30 + offset + index}\n"
            )

    with tempfile.TemporaryDirectory() as tmp:
        shared = Path(tmp) / "shared_alpha_summary.csv"
        shared.write_text(header + "".join(rows), encoding="utf-8")
        originals = {experiment_id: dash.HISTORICAL_EXPERIMENT_META[experiment_id] for experiment_id in experiment_ids}
        try:
            for experiment_id in experiment_ids:
                dash.HISTORICAL_EXPERIMENT_META[experiment_id] = {
                    **originals[experiment_id],
                    "summary_csv": shared,
                    "alpha_sensitivity_csv": shared,
                    "detail_csv": shared,
                }
            snapshot = dash.completed_result_snapshot({"completed": list(experiment_ids), "history": []})
        finally:
            for experiment_id, info in originals.items():
                dash.HISTORICAL_EXPERIMENT_META[experiment_id] = info

    by_id = {row["id"]: row for row in snapshot}
    for experiment_id in experiment_ids:
        row = by_id[experiment_id]
        assert [variant["alpha"] for variant in row["variants"]] == [0.005, 0.01, 0.02, 0.05]
        assert len({variant["alpha"] for variant in row["variants"]}) == 4
        assert row["mean_f1"] is None
        assert row["mean_fp"] is None
        assert row["mean_combined_f1"] is None
        assert row["selected_alpha"] is None
        assert row["best_strategy"] == "4 alpha variants (no selection)"
        assert row["standard_review_candidate_count"] is None
        assert row["priority_review_candidate_count"] is None
        assert row["total_review_candidate_count"] is None
        assert row["variants"][0]["standard_review_candidate_count"] is not None
        assert row["variants"][0]["priority_review_candidate_count"] is not None


def test_not_run_snapshot_suppresses_stale_summary_metrics():
    experiment_id = "experiment_999_not_run_stale_summary"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        summary = tmp_path / "stale_summary.csv"
        summary.write_text("config_name,mean_f1,mean_fp\nstale,0.99,12\n", encoding="utf-8")
        original = dash.EXPERIMENTS.get(experiment_id)
        dash.EXPERIMENTS[experiment_id] = {
            "label": "stale not-run experiment",
            "detail_csv": tmp_path / "missing_results.csv",
            "summary_csv": summary,
            "stdout_log": tmp_path / "missing.log",
        }
        try:
            row = next(item for item in dash.completed_result_snapshot({}) if item["id"] == experiment_id)
        finally:
            if original is None:
                dash.EXPERIMENTS.pop(experiment_id, None)
            else:
                dash.EXPERIMENTS[experiment_id] = original

    assert row["history_status"] == "not_run"
    assert row["mean_f1"] is None
    assert row["mean_fp"] is None
    assert row["alert_precision"] is None
    assert row["metric_availability_reason"] == "metric not recorded"


def test_queued_historical_catalog_row_suppresses_stale_detail_metrics():
    experiment_id = "experiment_141_family_neutral_full_coverage"
    row = next(
        item
        for item in dash.completed_result_snapshot({"queue": [experiment_id]})
        if item["id"] == experiment_id
    )

    assert row["history_status"] == "queued"
    assert row["mean_f1"] is None
    assert row["mean_fp"] is None
    assert row["alert_precision"] is None
    assert row["mean_combined_f1"] is None
    assert row["review_candidate_count"] is None


def test_malformed_historical_csv_isolated_and_reported():
    experiment_id = "experiment_999_malformed_artifact"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        detail = tmp_path / "malformed_results.csv"
        detail.write_bytes(b"dataset_name,mean_f1\nA,\xff\n")
        original = dash.EXPERIMENTS.get(experiment_id)
        dash.EXPERIMENTS[experiment_id] = {
            "label": "malformed artifact experiment",
            "detail_csv": detail,
            "summary_csv": detail,
            "stdout_log": tmp_path / "missing.log",
        }
        try:
            row = next(item for item in dash.completed_result_snapshot({}) if item["id"] == experiment_id)
        finally:
            if original is None:
                dash.EXPERIMENTS.pop(experiment_id, None)
            else:
                dash.EXPERIMENTS[experiment_id] = original
            dash.CSV_READ_ERRORS.pop(str(detail), None)
            dash.CSV_CACHE.pop(str(detail), None)

    assert row["history_status"] == "invalid"
    assert row["comparison_eligible"] is False
    assert row["mean_f1"] is None
    assert "artifact unreadable" in row["status_reason"]


def test_catalog_audit_only_snapshot_reports_audit_reason():
    snapshot = dash.completed_result_snapshot({"completed": [], "queue": [], "history": []})
    by_id = {row["id"]: row for row in snapshot}

    assert by_id["experiment_143_test_length_budget_binding_audit"]["metric_availability_reason"] == "audit only"
    assert by_id["experiment_143_test_length_budget_binding_audit"]["mean_f1"] is None
    assert by_id["experiment_145_train_normal_count_budget"]["history_status"] == "blocked"
    assert by_id["experiment_149_virtual_run_conformal_policy"]["history_status"] == "invalid"
    assert by_id["experiment_149_virtual_run_conformal_policy"]["mean_f1"] is None
    assert by_id["experiment_149_virtual_run_conformal_policy"]["mean_fp"] is None
    assert len(by_id["experiment_149_virtual_run_conformal_policy"]["variants"]) == 4


def test_completed_history_order_precedes_naturally_sorted_catalog_entries():
    snapshot = dash.completed_result_snapshot(
        {
            "completed": [
                "experiment_152_virtual_run_conformal_policy_v2",
                "experiment_151_virtual_run_conformal_policy_v2",
            ],
            "queue": [],
            "history": [],
            "running": {"id": "experiment_153_157_train_only_improvements"},
        }
    )

    assert [row["id"] for row in snapshot[:2]] == [
        "experiment_152_virtual_run_conformal_policy_v2",
        "experiment_151_virtual_run_conformal_policy_v2",
    ]
    assert next(row for row in snapshot if row["id"] == "experiment_153_157_train_only_improvements")["history_status"] == "running"


def test_build_status_marks_detected_exp153_157_as_running_history():
    originals = {
        "read_state": dash.read_state,
        "read_external_live_run": dash.read_external_live_run,
        "detect_exp151_152_run": dash.detect_exp151_152_run,
        "detect_exp153_157_run": dash.detect_exp153_157_run,
        "read_heartbeat": dash.read_heartbeat,
        "process_tree_snapshot": dash.process_tree_snapshot,
        "system_resource_snapshot": dash.system_resource_snapshot,
        "dataset_context_snapshot": dash.dataset_context_snapshot,
        "unified_experiment_items": dash.unified_experiment_items,
        "experiment_queue_snapshot": dash.experiment_queue_snapshot,
    }
    try:
        dash.read_state = lambda: {"completed": [], "queue": [], "history": [], "running": {}}
        dash.read_external_live_run = lambda: {}
        dash.detect_exp151_152_run = lambda: {}
        dash.detect_exp153_157_run = lambda: {
            "id": "experiment_153_157_train_only_improvements",
            "label": "Exp153-157",
            "status": "running",
            "pid": 321,
        }
        dash.read_heartbeat = lambda: {}
        dash.process_tree_snapshot = lambda pid: {"pid": pid}
        dash.system_resource_snapshot = lambda: {}
        dash.dataset_context_snapshot = lambda: {}
        dash.unified_experiment_items = lambda *args: []
        dash.experiment_queue_snapshot = lambda *args, **kwargs: []

        status = dash.build_status()
    finally:
        for name, value in originals.items():
            setattr(dash, name, value)

    by_id = {row["id"]: row for row in status["experiment_history"]}
    assert by_id["experiment_153_157_train_only_improvements"]["history_status"] == "running"
    assert "experiment_153_157_train_only_improvements" not in {row["id"] for row in status["completed_results"]}


if __name__ == "__main__":
    test_summarize_by_config_includes_operational_metrics()
    test_parse_log_error_summary_groups_reasons_and_families()
    test_coverage_snapshot_reports_done_and_error_gap()
    test_recent_from_rows_uses_latest_unique_dataset_as_fallback()
    test_recent_from_rows_accepts_exp165_dataset_and_n_train_columns()
    test_family_trouble_spots_rank_problem_families_for_strategy()
    test_throughput_snapshot_uses_recent_progress_events()
    test_experiment_compare_includes_running_and_delta_to_best()
    test_invalid_history_is_visible_but_excluded_from_compare()
    test_running_experiment_is_visible_without_metrics()
    test_experiment_compare_allows_only_running_active_summary_outside_history()
    test_metric_glossary_has_plain_korean_examples()
    test_metric_help_ui_is_collapsible_and_click_targeted()
    test_resource_metric_uses_compact_font_class()
    test_experiment_compare_does_not_hide_lower_ranked_completed_runs()
    test_dashboard_registers_new_queued_experiments()
    test_dashboard_server_supports_clean_restart()
    test_history_snapshot_uses_summary_without_loading_regular_detail_csv()
    test_large_csv_is_not_retained_in_dashboard_cache()
    test_process_tree_snapshot_aggregates_child_processes()
    test_system_cpu_snapshot_parses_top_output()
    test_gpu_snapshot_handles_unavailable_sampler()
    test_completed_snapshot_exposes_operational_columns()
    test_history_ui_has_status_filter_and_separate_review_metrics()
    test_history_metric_text_rejects_unavailable_values_before_number_conversion()
    test_historical_catalog_covers_exp138_through_exp168()
    test_normalize_runtime_state_hides_dead_running_pid_and_stale_queue_item()
    test_exp164_is_audit_only_and_exp165_is_registered_for_live_queue()
    test_exp167_168_are_registered_as_non_operational_research_results()
    test_experiment_catalog_includes_exp33_reconstruction_artifacts()
    test_history_artifact_path_rejects_unknown_ids_and_paths_outside_allowed_roots()
    test_historical_catalog_records_have_explicit_status_contracts()
    test_experiment_history_status_reconciles_state_before_artifacts()
    test_experiment_history_status_uses_artifact_presence_for_research_rows()
    test_hard_schema_never_uses_combined_f1_as_autonomous_f1()
    test_audit_schema_reports_reason_instead_of_fabricated_f1()
    test_alpha_variants_remain_ordered_and_selected_alpha_stays_none()
    test_hard_detail_rows_supply_mean_fields_and_micro_precision()
    test_exp149_150_stay_invalid_and_exp151_152_stay_completed_research()
    test_snapshot_filters_shared_alpha_rows_and_keeps_unselected_metrics_empty()
    test_not_run_snapshot_suppresses_stale_summary_metrics()
    test_queued_historical_catalog_row_suppresses_stale_detail_metrics()
    test_malformed_historical_csv_isolated_and_reported()
    test_catalog_audit_only_snapshot_reports_audit_reason()
    test_completed_history_order_precedes_naturally_sorted_catalog_entries()
    test_build_status_marks_detected_exp153_157_as_running_history()
    print("rank dashboard operational tests passed")
