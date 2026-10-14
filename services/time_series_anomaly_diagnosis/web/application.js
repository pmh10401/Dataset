// 시계열 이상 진단 서비스 웹 작업 공간

let catalogEntries = [];
let selectedDataset = null;
let currentRunRecords = [];
let activeRunRecord = null;
let pollTimeoutId = null;

// 한글 상태 및 가용성 매핑
const AVAILABILITY_LABELS = {
  standard: "분석 가능",
  recovery_numba: "분석 가능",
  insufficient_training_data: "학습 데이터 부족 (5개 미만)",
  missing_verified_route: "검증된 경로 없음",
};

const ROUTE_LABELS = {
  standard: "표준 분석",
  recovery_numba: "복구 전처리",
};

const STATUS_LABELS = {
  queued: "대기 중",
  running: "실행 중",
  completed: "완료",
  failed: "실패",
  interrupted: "중단됨",
};

function showAlert(message, retryCallback) {
  const alertBox = document.getElementById("global-alert");
  if (!alertBox) return;
  alertBox.replaceChildren();

  const msgSpan = document.createElement("span");
  msgSpan.textContent = message;
  alertBox.append(msgSpan);

  if (retryCallback) {
    const retryBtn = document.createElement("button");
    retryBtn.type = "button";
    retryBtn.className = "alert-retry-btn";
    retryBtn.textContent = "다시 시도";
    retryBtn.addEventListener("click", () => {
      clearAlert();
      retryCallback();
    });
    alertBox.append(retryBtn);
  }

  alertBox.hidden = false;
}

function clearAlert() {
  const alertBox = document.getElementById("global-alert");
  if (alertBox) {
    alertBox.hidden = true;
    alertBox.replaceChildren();
  }
}

function updateConnectionState(connected, teamUrl = "") {
  const el = document.getElementById("connection-state");
  if (!el) return;
  const wrapper = el.closest(".connection");
  if (connected) {
    el.textContent = "분석 서버 연결됨";
    if (teamUrl) el.title = `팀 접속 주소: ${teamUrl}`;
    wrapper?.classList.add("connected");
    wrapper?.classList.remove("disconnected");
  } else {
    el.textContent = "분석 서버 연결 끊김";
    wrapper?.classList.add("disconnected");
    wrapper?.classList.remove("connected");
  }
}

async function loadServiceStatus() {
  try {
    const status = await requestJson("/api/status", {}, () => loadServiceStatus());
    updateConnectionState(status.status === "ready", status.team_url || "");
  } catch (err) {
    // Alert handled by requestJson
  }
}

async function requestJson(path, options = {}, retryCallback = null) {
  let response;
  try {
    response = await fetch(path, options);
  } catch (err) {
    updateConnectionState(false);
    showAlert(`네트워크 오류: ${err.message}`, retryCallback);
    throw err;
  }

  updateConnectionState(true);
  let payload;
  try {
    payload = await response.json();
  } catch (err) {
    showAlert(`요청 실패: ${err.message}`, retryCallback);
    throw err;
  }
  if (!response.ok) {
    const errorMsg = payload.error || `요청 실패 (${response.status})`;
    showAlert(`요청 실패: ${errorMsg}`, retryCallback);
    throw new Error(errorMsg);
  }
  return payload;
}

function switchView(viewId) {
  document.querySelectorAll(".view").forEach((view) => {
    view.hidden = view.id !== viewId;
  });
  const target = document.getElementById(viewId);
  if (target) {
    target.hidden = false;
    target.querySelector("h1")?.focus();
  }
  document.querySelectorAll(".nav-button").forEach((btn) => {
    const active = btn.dataset.view === viewId;
    btn.classList.toggle("active", active);
    if (active) btn.setAttribute("aria-current", "page");
    else btn.removeAttribute("aria-current");
  });
}

// 1. 데이터셋 카탈로그 불러오기
async function loadDatasetCatalog() {
  const catalogList = document.getElementById("dataset-list");
  if (catalogList) {
    catalogList.innerHTML = "<div class=\"empty-state\">데이터셋 목록을 불러오는 중...</div>";
  }

  try {
    const data = await requestJson("/api/datasets", {}, () => loadDatasetCatalog());
    catalogEntries = Array.isArray(data) ? data : [];
    renderDatasetCatalog();
  } catch (err) {
    if (catalogList) {
      catalogList.innerHTML = "<div class=\"empty-state unavailable\">목록을 불러오지 못했습니다.</div>";
    }
    const resultCount = document.getElementById("result-count");
    if (resultCount) resultCount.textContent = "불러오기 실패";
  }
}

function filterCatalog(entries, query, availableOnly) {
  query = query.trim().toLowerCase();
  return entries.filter((item) => item.dataset_name.toLowerCase().includes(query)
    && (!availableOnly || item.analysis_available));
}

function renderDatasetCatalog() {
  const catalogList = document.getElementById("dataset-list");
  if (!catalogList) return;
  catalogList.replaceChildren();

  const query = (document.getElementById("dataset-search")?.value || "").trim().toLowerCase();
  const filtered = filterCatalog(catalogEntries, query,
    document.getElementById("available-only")?.checked ?? true);
  const resultCount = document.getElementById("result-count");
  const totalCount = document.getElementById("total-count");
  if (resultCount) resultCount.textContent = `${filtered.length}개 표시`;
  if (totalCount) totalCount.textContent = `전체 ${catalogEntries.length.toLocaleString("ko-KR")}개 · 분석 가능 ${catalogEntries.filter(item => item.analysis_available).length.toLocaleString("ko-KR")}개`;

  if (filtered.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = query ? "일치하는 데이터셋이 없습니다." : "등록된 데이터셋이 없습니다.";
    catalogList.append(empty);
    return;
  }

  filtered.forEach((item) => {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "dataset-row";
    if (selectedDataset && selectedDataset.dataset_name === item.dataset_name) {
      row.classList.add("selected");
    }

    const infoDiv = document.createElement("div");
    infoDiv.className = "dataset-main-info";

    const nameText = document.createElement("span");
    nameText.className = "dataset-name-text";
    nameText.textContent = item.dataset_name;

    const stats = document.createElement("span");
    stats.className = "dataset-stats";
    stats.textContent = `학습 ${item.train_count.toLocaleString("ko-KR")} · 평가 ${item.test_count.toLocaleString("ko-KR")} · 길이 ${item.series_length.toLocaleString("ko-KR")}`;

    infoDiv.append(nameText, stats);

    const tag = document.createElement("span");
    tag.className = `dataset-status-tag ${item.analysis_available ? "" : "unavailable"}`;
    const reasonText = AVAILABILITY_LABELS[item.availability] || item.availability;
    tag.textContent = reasonText;

    row.append(infoDiv, tag);

    if (item.analysis_available) {
      row.setAttribute("role", "option");
      row.setAttribute("aria-selected", selectedDataset?.dataset_name === item.dataset_name ? "true" : "false");
      row.addEventListener("click", () => selectDataset(item));
      row.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
          e.preventDefault();
          selectDataset(item);
        }
      });
    } else {
      row.disabled = true;
      row.setAttribute("aria-disabled", "true");
    }

    catalogList.append(row);
  });
}

function selectDataset(item) {
  selectedDataset = item;
  const startBtn = document.getElementById("start-analysis");
  document.getElementById("selection-heading").textContent = item.dataset_name;
  document.getElementById("selection-help").textContent = "분석 조건을 확인했습니다. 준비가 되면 분석을 시작하세요.";
  document.getElementById("fact-train").textContent = `${item.train_count.toLocaleString("ko-KR")}개`;
  document.getElementById("fact-eval").textContent = `${item.test_count.toLocaleString("ko-KR")}개`;
  document.getElementById("fact-length").textContent = item.series_length.toLocaleString("ko-KR");
  document.getElementById("fact-route").textContent = ROUTE_LABELS[item.availability] || item.method_name || "자동 지정";
  document.getElementById("analysis-notice").textContent = "";
  if (startBtn) startBtn.disabled = false;
  document.querySelectorAll(".dataset-row").forEach((row) => {
    const nameEl = row.querySelector(".dataset-name-text");
    const isThis = nameEl && nameEl.textContent === item.dataset_name;
    row.classList.toggle("selected", isThis);
    if (row.hasAttribute("role")) {
      row.setAttribute("aria-selected", isThis ? "true" : "false");
    }
  });
}

// 2. 분석 요청 제출
async function submitAnalysis() {
  if (!selectedDataset || !selectedDataset.analysis_available) {
    showAlert("실행 가능한 데이터셋을 먼저 선택해주세요.");
    return;
  }

  const startBtn = document.getElementById("start-analysis");
  if (startBtn) startBtn.disabled = true;

  try {
    await requestJson("/api/analysis-runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_name: selectedDataset.dataset_name }),
    });
    const notice = document.getElementById("analysis-notice");
    if (notice) notice.textContent = `${selectedDataset.dataset_name} 분석 요청을 안전하게 저장했습니다.`;
    await refreshAnalysisRuns();
    switchView("queue-view");
  } catch (err) {
    // requestJson already displays the error.
  } finally {
    if (startBtn) startBtn.disabled = !selectedDataset?.analysis_available;
  }
}

// 3. 작업 대기열 및 기록 갱신
async function refreshAnalysisRuns() {
  if (pollTimeoutId) {
    clearTimeout(pollTimeoutId);
    pollTimeoutId = null;
  }

  try {
    const runs = await requestJson("/api/analysis-runs", {}, () => refreshAnalysisRuns());
    currentRunRecords = Array.isArray(runs) ? runs : [];
    renderQueueList();
    renderHistoryList();

    const hasPending = currentRunRecords.some((r) => r.status === "queued" || r.status === "running");
    if (hasPending) {
      pollTimeoutId = setTimeout(refreshAnalysisRuns, 2000);
    }
  } catch (err) {
    for (const id of ["queue-list", "history-list"]) {
      const container = document.getElementById(id);
      if (container) container.innerHTML = "<p class=\"empty-state unavailable\">분석 기록을 불러오지 못했습니다.</p>";
    }
  }
}

function renderQueueList() {
  const container = document.getElementById("queue-list");
  if (!container) return;
  container.replaceChildren();

  const activeRuns = currentRunRecords.filter(
    (r) => r.status === "queued" || r.status === "running"
  );
  const summary = document.getElementById("queue-summary");
  if (summary) summary.textContent = activeRuns.length
    ? `${activeRuns.length}개 진행 중 · 첫 분석은 계산 준비로 시간이 더 걸릴 수 있습니다.`
    : "진행 중인 분석이 없습니다. 완료된 결과는 아래에서 열 수 있습니다.";
  const completed = currentRunRecords.find(run => run.status === "completed");
  const latest = document.getElementById("latest-result");
  if (latest) {
    latest.hidden = !completed;
    latest.textContent = completed ? `${completed.dataset_name} 결과 열기 →` : "";
    latest.onclick = completed ? () => openAnalysisRun(completed.run_id) : null;
  }

  if (activeRuns.length === 0) {
    const p = document.createElement("p");
    p.className = "empty-state";
    p.textContent = "현재 대기 중이거나 실행 중인 작업이 없습니다.";
    container.append(p);
    return;
  }

  activeRuns.forEach((run, idx) => {
    const card = document.createElement("div");
    card.className = "plain-row";

    const info = document.createElement("div");
    info.className = "plain-info";

    const title = document.createElement("strong");
    title.textContent = run.dataset_name;

    const desc = document.createElement("span");
    desc.textContent = `요청 ${run.requested_at || "-"} · ${run.method_name || "자동 지정"}`;

    info.append(title, desc);

    const badge = document.createElement("div");
    badge.className = `plain-status ${run.status}`;
    badge.textContent = run.status === "queued" ? `대기 ${idx + 1}번째` : STATUS_LABELS[run.status] || run.status;

    card.append(info, badge);
    container.append(card);
  });
}

function renderHistoryList() {
  const container = document.getElementById("history-list");
  if (!container) return;
  container.replaceChildren();

  if (currentRunRecords.length === 0) {
    const p = document.createElement("p");
    p.className = "empty-state";
    p.textContent = "분석 실행 기록이 없습니다.";
    container.append(p);
    return;
  }

  currentRunRecords.forEach((run) => {
    const card = document.createElement(run.status === "completed" ? "button" : "div");
    card.className = `plain-row ${run.status}`;
    if (run.status === "completed") card.type = "button";

    const info = document.createElement("div");
    info.className = "plain-info";

    const title = document.createElement("strong");
    title.textContent = run.dataset_name;

    const desc = document.createElement("span");
    desc.textContent = run.status === "failed"
      ? `실패 사유: ${run.error_message || "알 수 없는 오류"}`
      : `${run.method_name || "자동 지정"} · 요청 ${run.requested_at || "-"}${run.completed_at ? " · 완료 " + run.completed_at : ""}`;

    info.append(title, desc);

    const badge = document.createElement("div");
    badge.className = `plain-status ${run.status}`;
    badge.textContent = run.status === "completed" ? "결과 보기" : STATUS_LABELS[run.status] || run.status;

    card.append(info, badge);

    if (run.status === "completed") {
      card.setAttribute("aria-label", `${run.dataset_name} 완료 결과 열기`);
      card.addEventListener("click", () => openAnalysisRun(run.run_id));
    }

    container.append(card);
  });
}

// 4. 분석 결과 상세 열기
async function openAnalysisRun(runId) {
  try {
    const record = await requestJson(`/api/analysis-runs/${runId}`, {}, () => openAnalysisRun(runId));
    activeRunRecord = record;
    renderAnalysisResult(record);
    switchView("result-workspace");
  } catch (err) {
    // Alert handled by requestJson
  }
}

function renderAnalysisResult(record) {
  const resultTitle = document.getElementById("result-title");
  const resultSub = document.getElementById("result-subtitle");
  const resultMeta = document.getElementById("result-meta");

  if (resultTitle) resultTitle.textContent = `${record.dataset_name} 진단 결과`;
  if (resultSub) {
    resultSub.textContent = `상태: ${STATUS_LABELS[record.status] || record.status} · 방식: ${record.method_name || "-"}`;
  }

  if (resultMeta) {
    resultMeta.replaceChildren();
    const badge = document.createElement("span");
    badge.className = `plain-status ${record.status}`;
    badge.textContent = STATUS_LABELS[record.status] || record.status;
    resultMeta.append(badge);
  }

  const result = record.result;
  clearActiveSeries();
  if (!result) {
    clearCandidateLists();
    renderDetectionSummary(null, record.requires_reanalysis);
    return;
  }

  renderDetectionSummary(result, false);

  // 4-1. 자동 확정 이상치 렌더링
  renderAutomaticDetections(result);

  // 4-2. 표본 수 제한으로 자동 확정한 항목의 경고 렌더링
  renderResolutionLimitedWarning(result);

  // 4-3. 다중 근거 검토 후보 렌더링
  renderMultipleEvidenceReview(result.multiple_evidence_review);

  // 4-4. 모델 평가 지표 렌더링
  renderModelEvaluation(result.metrics);

  // 4-5. 첫 번째 시계열 자동 조회 (자동 확정, 점수 순)
  const firstIndex =
    result.automatic_anomaly_indices && result.automatic_anomaly_indices.length > 0
      ? result.automatic_anomaly_indices[0]
      : result.rows && result.rows.length > 0
      ? result.rows[0].series_index
      : null;

  if (firstIndex !== null) {
    openSeries(firstIndex);
  }
}

function clearActiveSeries() {
  const activeLabel = document.getElementById("active-series-label");
  if (activeLabel) activeLabel.textContent = "확인할 시계열이 없습니다.";

  const canvas = document.getElementById("series-chart");
  if (canvas) {
    const context = canvas.getContext("2d");
    if (context) context.clearRect(0, 0, canvas.width, canvas.height);
  }
}

function clearCandidateLists() {
  ["automatic-detections", "resolution-limited-warning", "multiple-evidence-review", "model-evaluation"].forEach((id) => {
    const el = document.getElementById(id);
    if (el) el.replaceChildren();
  });
}

function renderMultipleEvidenceReview(review) {
  const container = document.getElementById("multiple-evidence-review");
  if (!container) return;
  container.replaceChildren();

  if (!review || !review.evidence_available) {
    const empty = document.createElement("span");
    empty.className = "muted-text";
    empty.textContent = "이 데이터셋에는 다중 근거 검토 자료가 없습니다.";
    container.append(empty);
    return;
  }

  const candidates = review.candidates || [];
  if (candidates.length === 0) {
    const empty = document.createElement("span");
    empty.className = "muted-text";
    empty.textContent = "다중 근거 검토 후보가 없습니다.";
    container.append(empty);
    return;
  }

  candidates.forEach((candidate) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-btn";
    button.dataset.seriesIndex = candidate.series_index;
    button.textContent = `#${candidate.series_index} · ${candidate.priority} · 근거 ${candidate.support_count}개`;
    button.addEventListener("click", () => openSeries(candidate.series_index));
    container.append(button);
  });
}

function renderDetectionSummary(result, requiresReanalysis) {
  const container = document.getElementById("detection-summary");
  if (!container) return;
  container.className = "detection-summary";
  if (requiresReanalysis) {
    container.classList.add("warning");
    container.textContent = "이 기록은 이전 결과 형식입니다. 현재 데이터베이스 입력으로 다시 분석해 주세요.";
    return;
  }
  if (!result) {
    container.textContent = "완료된 분석 결과가 없습니다.";
    return;
  }
  const automaticCount = (result.automatic_anomaly_indices || []).length;
  const warningCount = (result.resolution_limited_warning_indices || []).length;
  container.classList.add(warningCount > 0 ? "warning" : "success");
  container.textContent = `자동 확정 ${automaticCount}개 · 표본 수 제한 승격 ${warningCount}개`;
  const calibration = result.calibration;
  if (warningCount > 0 && calibration) {
    container.textContent += ` — 정상 기준 ${calibration.reference_count}개에서는 최소 p ${calibration.minimum_p_value.toFixed(4)}가 일반 기준 p ≤ ${calibration.automatic_threshold.toFixed(4)}보다 큽니다. 가능한 최소 p에 도달해 자동 확정했지만 통계적 확신은 더 낮습니다.`;
  }
}

function renderResolutionLimitedWarning(result) {
  const container = document.getElementById("resolution-limited-warning");
  if (!container) return;
  container.replaceChildren();

  const indices = new Set(result.resolution_limited_warning_indices || []);
  const rows = (result.rows || []).filter((row) => indices.has(row.series_index));
  if (rows.length === 0) {
    const empty = document.createElement("span");
    empty.className = "muted-text";
    empty.textContent = "표본 수 제한으로 확정된 이상치가 없습니다.";
    container.append(empty);
    return;
  }

  rows.forEach((row) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "candidate-btn";
    button.dataset.seriesIndex = row.series_index;
    button.textContent = `#${row.series_index} · p ${row.p_value.toFixed(4)} · 가능한 최소 p에서 자동 확정`;
    button.addEventListener("click", () => openSeries(row.series_index));
    container.append(button);
  });
}

function renderAutomaticDetections(result) {
  const container = document.getElementById("automatic-detections");
  if (!container) return;
  container.replaceChildren();

  const automaticIndices = new Set(result.automatic_anomaly_indices || []);
  const warningIndices = new Set(result.resolution_limited_warning_indices || []);
  const automaticRows = (result.rows || []).filter((r) => automaticIndices.has(r.series_index));

  if (automaticRows.length === 0) {
    const empty = document.createElement("span");
    empty.className = "muted-text";
    empty.textContent = "자동 확정된 이상 시계열이 없습니다.";
    container.append(empty);
    return;
  }

  automaticRows.forEach((row) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "candidate-btn";
    btn.dataset.seriesIndex = row.series_index;
    btn.textContent = warningIndices.has(row.series_index)
      ? `#${row.series_index} · p ${row.p_value.toFixed(4)} · 표본 수 제한으로 자동 확정`
      : `#${row.series_index} (점수: ${typeof row.score === "number" ? row.score.toFixed(4) : row.score})`;
    btn.addEventListener("click", () => openSeries(row.series_index));
    container.append(btn);
  });
}

function renderModelEvaluation(metrics) {
  const container = document.getElementById("model-evaluation");
  if (!container) return;
  container.replaceChildren();

  if (!metrics) {
    const empty = document.createElement("span");
    empty.className = "muted-text";
    empty.textContent = "모델 평가 지표가 없습니다.";
    container.append(empty);
    return;
  }

  const items = [
    { label: "ROC AUC", val: typeof metrics.auc === "number" ? metrics.auc.toFixed(4) : "-" },
    { label: "오탐 (FP)", val: metrics.false_positives ?? "-" },
    { label: "미탐 (FN)", val: metrics.false_negatives ?? "-" },
    { label: "정탐 (TP)", val: metrics.true_positives ?? "-" },
  ];

  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "metric-item";

    const lbl = document.createElement("div");
    lbl.className = "metric-label";
    lbl.textContent = item.label;

    const val = document.createElement("div");
    val.className = "metric-val";
    val.textContent = item.val;

    div.append(lbl, val);
    container.append(div);
  });
}

async function openSeries(seriesIndex) {
  if (!activeRunRecord || !activeRunRecord.run_id) return;
  const requestedRunId = activeRunRecord.run_id;

  const activeLabel = document.getElementById("active-series-label");
  if (activeLabel) {
    activeLabel.textContent = `시계열 #${seriesIndex} 로딩 중...`;
  }

  document.querySelectorAll(".candidate-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.seriesIndex === String(seriesIndex));
  });

  try {
    const data = await requestJson(
      `/api/analysis-runs/${requestedRunId}/series/${seriesIndex}`,
      {},
      () => activeRunRecord && activeRunRecord.run_id === requestedRunId && openSeries(seriesIndex)
    );
    if (!activeRunRecord || activeRunRecord.run_id !== requestedRunId) return;
    if (activeLabel) {
      activeLabel.textContent = `현재 선택: 시계열 #${seriesIndex}`;
    }
    const canvas = document.getElementById("series-chart");
    if (canvas) {
      drawSeriesChart(canvas, data);
    }
  } catch (err) {
    if (!activeRunRecord || activeRunRecord.run_id !== requestedRunId) return;
    if (activeLabel) {
      activeLabel.textContent = `시계열 #${seriesIndex} 불러오기 실패`;
    }
  }
}

// 5. Canvas 시계열 그래프 렌더링
function chartPoint(canvas, index, length, value, minimum, maximum) {
  const padX = 20;
  const padY = 20;
  const plotWidth = canvas.width - padX * 2;
  const plotHeight = canvas.height - padY * 2;
  const x = padX + (index * plotWidth) / Math.max(length - 1, 1);
  const y = canvas.height - padY - ((value - minimum) * plotHeight) / Math.max(maximum - minimum, Number.EPSILON);
  return [x, y];
}

function drawLine(context, values, minimum, maximum, color, lineWidth = 2) {
  if (!values || values.length === 0) return;
  context.beginPath();
  values.forEach((val, idx) => {
    const [x, y] = chartPoint(context.canvas, idx, values.length, val, minimum, maximum);
    if (idx === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = color;
  context.lineWidth = lineWidth;
  context.stroke();
}

function drawReferenceBand(context, lower, upper, minimum, maximum) {
  if (!lower || !upper || lower.length === 0 || upper.length === 0) return;
  context.beginPath();
  upper.forEach((val, idx) => {
    const [x, y] = chartPoint(context.canvas, idx, upper.length, val, minimum, maximum);
    if (idx === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  for (let idx = lower.length - 1; idx >= 0; idx -= 1) {
    const [x, y] = chartPoint(context.canvas, idx, lower.length, lower[idx], minimum, maximum);
    context.lineTo(x, y);
  }
  context.closePath();
  context.fillStyle = "rgba(83, 116, 201, 0.18)";
  context.fill();
}

function drawSeriesChart(canvas, series) {
  requestAnimationFrame(() => {
    const context = canvas.getContext("2d");
    const curves = [series.train_q10, series.train_q90, series.train_median, series.test_values].filter(Array.isArray);
    const flatValues = curves.flat();

    context.clearRect(0, 0, canvas.width, canvas.height);

    if (flatValues.length === 0) {
      context.fillStyle = "#5d687a";
      context.font = "14px sans-serif";
      context.textAlign = "center";
      context.fillText("표시할 시계열 데이터가 없습니다.", canvas.width / 2, canvas.height / 2);
      return;
    }

    let minimum = Math.min(...flatValues);
    let maximum = Math.max(...flatValues);
    if (minimum === maximum) {
      minimum -= 1.0;
      maximum += 1.0;
    }

    // 축 보조선
    context.strokeStyle = "#e6ecf2";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(20, 20);
    context.lineTo(20, canvas.height - 20);
    context.lineTo(canvas.width - 20, canvas.height - 20);
    context.stroke();

    // 1. 정상 기준 범위 (q10–q90 밴드)
    drawReferenceBand(context, series.train_q10, series.train_q90, minimum, maximum);

    // 2. 학습 중앙값 (파란색 선)
    drawLine(context, series.train_median, minimum, maximum, "#2457d6", 2);

    // 3. 선택한 평가 시계열 (붉은색 선)
    drawLine(context, series.test_values, minimum, maximum, "#d13f3f", 2);
  });
}

// 초기화 이벤트 리스너 등록
document.addEventListener("DOMContentLoaded", () => {
  // 탐색 버튼 바인딩
  document.querySelectorAll(".nav-button").forEach((btn) => {
    btn.addEventListener("click", () => {
      const viewId = btn.dataset.view;
      switchView(viewId);
      if (viewId === "queue-view" || viewId === "history-view") {
        refreshAnalysisRuns();
      }
    });
  });

  // 데이터셋 검색 필터
  const searchInput = document.getElementById("dataset-search");
  if (searchInput) {
    searchInput.addEventListener("input", renderDatasetCatalog);
  }
  document.getElementById("available-only")?.addEventListener("change", renderDatasetCatalog);
  document.getElementById("clear-search")?.addEventListener("click", () => {
    searchInput.value = "";
    renderDatasetCatalog();
    searchInput.focus();
  });
  document.getElementById("back-to-history")?.addEventListener("click", () => {
    switchView("history-view");
    refreshAnalysisRuns();
  });

  // 분석 시작 버튼
  const startBtn = document.getElementById("start-analysis");
  if (startBtn) {
    startBtn.addEventListener("click", submitAnalysis);
  }

  // 초기 카탈로그 및 대기열 로드
  loadServiceStatus();
  loadDatasetCatalog();
  refreshAnalysisRuns();
});
