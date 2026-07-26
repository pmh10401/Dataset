"use client";

import Link from "next/link";
import { ArrowLeft, CheckCircle2, Clock3, Info, Save, ShieldCheck } from "lucide-react";
import { FormEvent, useCallback, useEffect, useState } from "react";

import {
  ALPHA_PRESETS,
  type AlphaSetting,
  type AlphaSettingEvent,
  type AlphaValue,
} from "@/src/lib/alpha-settings";

type SettingsResponse = {
  setting: AlphaSetting | null;
  history: AlphaSettingEvent[];
  canWrite: boolean;
  automaticHardAlertsUnchanged: boolean;
};

function alphaLabel(value: AlphaValue | null | undefined) {
  return value == null ? "미적용" : value.toFixed(3);
}

function statusLabel(status: AlphaSetting["status"] | undefined) {
  if (status === "applied") return "적용 완료";
  if (status === "failed") return "적용 실패";
  if (status === "pending") return "적용 대기";
  return "설정 없음";
}

function formatDate(value: string | null | undefined) {
  return value ? new Date(value).toLocaleString("ko-KR") : "-";
}

export default function AlphaSettings() {
  const [data, setData] = useState<SettingsResponse | null>(null);
  const [selected, setSelected] = useState<AlphaValue>(0.005);
  const [reason, setReason] = useState("");
  const [acknowledged, setAcknowledged] = useState(false);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    const response = await fetch("/api/settings/alpha", { cache: "no-store" });
    if (!response.ok) {
      setMessage("설정 정보를 불러오지 못했습니다.");
      setLoading(false);
      return;
    }
    const next = (await response.json()) as SettingsResponse;
    setData(next);
    setSelected(next.setting?.desired_alpha ?? 0.005);
    setLoading(false);
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSaving(true);
    setMessage("");
    const response = await fetch("/api/settings/alpha", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ alpha: selected, reason, acknowledged }),
    });
    const body = await response.json();
    if (!response.ok) {
      setMessage(body.error === "write permission required"
        ? "이 계정에는 설정 변경 권한이 없습니다."
        : "요청을 저장하지 못했습니다. 입력 내용을 확인해 주세요.");
      setSaving(false);
      return;
    }
    setMessage("변경 요청을 저장했습니다. 탐지기가 확인할 때까지 현재 적용값은 유지됩니다.");
    setReason("");
    setAcknowledged(false);
    await load();
    setSaving(false);
  }

  const setting = data?.setting;
  const selectedPreset = ALPHA_PRESETS.find((preset) => preset.value === selected) ?? ALPHA_PRESETS[0];
  const canSubmit = Boolean(data?.canWrite && acknowledged && reason.trim().length >= 5 && !saving);

  return (
    <main className="settings-shell">
      <header className="settings-header">
        <Link className="icon-button" href="/" title="대시보드로 돌아가기"><ArrowLeft size={18} /></Link>
        <div>
          <span className="eyebrow">OPERATION SETTINGS</span>
          <h1>Alpha 신뢰도 설정</h1>
          <p>기존 자동 경보 가운데 어떤 항목을 더 강한 근거로 표시할지 정합니다.</p>
        </div>
      </header>

      <section className="safety-notice">
        <ShieldCheck size={22} />
        <div>
          <strong>자동 경보는 바뀌지 않습니다</strong>
          <p>Alpha는 자동 경보를 추가하거나 제거하지 않고, 이미 생성된 경보의 신뢰도 표시만 조정합니다.</p>
        </div>
      </section>

      <section className="settings-status-band" aria-label="현재 설정 상태">
        <div><span>현재 적용값</span><strong>{alphaLabel(setting?.effective_alpha)}</strong></div>
        <div><span>요청값</span><strong>{alphaLabel(setting?.desired_alpha)}</strong></div>
        <div><span>상태</span><strong className={`setting-status ${setting?.status ?? "none"}`}>{statusLabel(setting?.status)}</strong></div>
        <div><span>마지막 요청</span><strong>{formatDate(setting?.requested_at)}</strong></div>
      </section>

      {loading ? <section className="settings-loading">설정 정보를 확인하고 있습니다.</section> : (
        <div className="settings-layout">
          <form className="settings-form" onSubmit={submit}>
            <div className="settings-section-heading">
              <div><span>1</span><h2>강한 신뢰 표시 범위</h2></div>
              <p>두 값은 과거 결과를 설명하기 위한 고정 선택지이며, 실제 운영 성능을 보장하지 않습니다.</p>
            </div>

            <div className="preset-grid" role="radiogroup" aria-label="Alpha 선택">
              {ALPHA_PRESETS.map((preset) => (
                <label className={`preset-option ${selected === preset.value ? "selected" : ""}`} key={preset.key}>
                  <input
                    type="radio"
                    name="alpha"
                    value={preset.value}
                    checked={selected === preset.value}
                    onChange={() => setSelected(preset.value)}
                  />
                  <span className="preset-name">{preset.label}</span>
                  <strong>{preset.value.toFixed(3)}</strong>
                  <small>{preset.description}</small>
                </label>
              ))}
            </div>

            <div className="impact-strip">
              <Info size={18} />
              <div>
                <strong>과거 평가 참고</strong>
                <span>강한 신뢰 {selectedPreset.strongAlertCount}건 · 실제 이상 {selectedPreset.strongTruePositiveCount}건 · 오탐 {selectedPreset.strongFalsePositiveCount}건 · 정밀도 {(selectedPreset.retrospectivePrecision * 100).toFixed(1)}%</span>
              </div>
            </div>

            <div className="settings-section-heading">
              <div><span>2</span><h2>변경 사유와 확인</h2></div>
              <p>운영 이력에 남을 이유를 5자 이상 입력합니다.</p>
            </div>
            <label className="reason-field">
              <span>변경 사유</span>
              <textarea
                value={reason}
                onChange={(event) => setReason(event.target.value)}
                maxLength={500}
                placeholder="예: 오탐 억제를 우선해 엄격 설정으로 변경"
                disabled={!data?.canWrite}
              />
              <small>{reason.length} / 500</small>
            </label>
            <label className="acknowledgement">
              <input
                type="checkbox"
                checked={acknowledged}
                onChange={(event) => setAcknowledged(event.target.checked)}
                disabled={!data?.canWrite}
              />
              <span>자동 경보 수는 변하지 않으며, 요청값과 현재 적용값이 다를 수 있다는 변경 내용을 확인했습니다.</span>
            </label>

            {!data?.canWrite && <p className="form-error">현재 계정은 설정을 조회할 수 있지만 변경 권한은 없습니다.</p>}
            {message && <p className="settings-message">{message}</p>}
            <button className="primary-action" type="submit" disabled={!canSubmit}>
              {saving ? <Clock3 size={18} /> : <Save size={18} />}
              {saving ? "요청 저장 중" : "변경 요청"}
            </button>
          </form>

          <aside className="settings-history">
            <div className="section-title"><h3>최근 변경 이력</h3><span>{data?.history.length ?? 0}건</span></div>
            <div className="history-list">
              {data?.history.length ? data.history.map((event) => (
                <article key={event.id}>
                  <CheckCircle2 size={16} />
                  <div><strong>Alpha {event.requested_alpha.toFixed(3)}</strong><p>{event.reason}</p><span>{formatDate(event.created_at)}</span></div>
                </article>
              )) : <p className="history-empty">아직 기록된 변경 요청이 없습니다.</p>}
            </div>
          </aside>
        </div>
      )}
    </main>
  );
}
