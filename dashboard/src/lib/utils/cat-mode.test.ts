import assert from "node:assert/strict";
import test from "node:test";
import {
  catModeRunKey,
  showCatModeResult,
  type CatModeRun,
} from "./cat-mode.ts";

const failed: CatModeRun = {
  active: false,
  state: "error",
  instance_id: "d4b29634",
  started_at: 1794965667.01728,
};

test("shows a failed run while its instance is live", () => {
  assert.equal(showCatModeResult(failed, null, ["d4b29634"]), true);
});

test("hides a failed run once its instance is gone or replaced", () => {
  assert.equal(showCatModeResult(failed, null, ["9f0b1414"]), false);
  assert.equal(showCatModeResult(failed, null, []), false);
});

test("hides a dismissed run but shows the next one", () => {
  const dismissed = catModeRunKey(failed);
  assert.equal(showCatModeResult(failed, dismissed, ["d4b29634"]), false);
  const next = { ...failed, started_at: failed.started_at! + 60 };
  assert.equal(showCatModeResult(next, dismissed, ["d4b29634"]), true);
});

test("never shows a result for an idle or running controller", () => {
  assert.equal(
    showCatModeResult({ ...failed, state: "idle" }, null, null),
    false,
  );
  assert.equal(
    showCatModeResult(
      { ...failed, active: true, state: "running" },
      null,
      null,
    ),
    false,
  );
});

test("keeps showing when the instance list is not loaded yet", () => {
  assert.equal(showCatModeResult(failed, null, null), true);
});
