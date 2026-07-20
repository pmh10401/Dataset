export type HistoryStatus = {
  running: boolean;
  active_experiment: string | null;
  datasets_done: number;
  error_count: number;
};

export function shouldWriteHistory(
  previous: HistoryStatus | null,
  current: HistoryStatus,
  secondsSincePrevious: number,
) {
  if (!previous) return true;
  if (previous.running !== current.running) return true;
  if (previous.active_experiment !== current.active_experiment) return true;
  if (previous.datasets_done !== current.datasets_done) return true;
  if (previous.error_count !== current.error_count) return true;
  return secondsSincePrevious >= 60;
}
