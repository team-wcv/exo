/** Which Cat mode run result the dashboard should still show. */
export interface CatModeRun {
  active: boolean;
  state: "idle" | "running" | "stopped" | "finished" | "error";
  instance_id?: string | null;
  started_at?: number | null;
}

/** Stable identity of one run; a new start gets a new key. */
export function catModeRunKey(run: CatModeRun): string | null {
  if (run.instance_id == null || run.started_at == null) return null;
  return `${run.instance_id}@${run.started_at}`;
}

/**
 * A finished run's result (including its error) is only worth showing while
 * the operator has not dismissed it and the instance it ran on still exists.
 * The API keeps the last result in memory until restart, so without this a
 * failure on a long-gone instance stays on screen across reloads.
 */
export function showCatModeResult(
  run: CatModeRun,
  dismissedKey: string | null,
  liveInstanceIds: readonly string[] | null,
): boolean {
  if (run.active || run.state === "idle" || run.state === "running")
    return false;
  const key = catModeRunKey(run);
  if (key !== null && key === dismissedKey) return false;
  if (
    liveInstanceIds !== null &&
    run.instance_id != null &&
    !liveInstanceIds.includes(run.instance_id)
  )
    return false;
  return true;
}
