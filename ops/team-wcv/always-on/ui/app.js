"use strict";
const byId = (id) => document.getElementById(id);
let config,
  history = [],
  controller = null,
  online = false,
  selectedModel = null;
const apiBase = new URL("v1/", window.location.href);
const histories = new Map();

function message(role, content = "") {
  byId("conversation").querySelector(".empty")?.remove();
  const box = document.createElement("article");
  box.className = `message ${role}`;
  const label = document.createElement("p");
  label.className = "role";
  label.textContent = role;
  const thinking = document.createElement("details");
  thinking.className = "reasoning";
  thinking.hidden = true;
  const summary = document.createElement("summary");
  summary.textContent = "Reasoning";
  const thoughts = document.createElement("pre");
  thinking.append(summary, thoughts);
  const text = document.createElement("pre");
  text.className = "content";
  text.textContent = content;
  box.append(label, thinking, text);
  byId("conversation").append(box);
  return { box, text, thinking, thoughts };
}

function scroll() {
  const el = byId("conversation");
  el.scrollTop = el.scrollHeight;
}
function historyKey() {
  return selectedModel.id === config.model_id
    ? `local-inference-${config.id}`
    : `local-inference-${config.id}-${selectedModel.id}`;
}

function restoreHistory() {
  const key = historyKey();
  if (histories.has(key)) {
    history = histories.get(key);
  } else {
    history = [];
    try {
      const saved = JSON.parse(localStorage.getItem(key) || "[]");
      if (Array.isArray(saved))
        history = saved.filter(
          (m) =>
            m &&
            ["user", "assistant"].includes(m.role) &&
            typeof m.content === "string",
        );
    } catch {
      history = [];
    }
    histories.set(key, history);
  }
  byId("conversation").replaceChildren();
  for (const item of history) message(item.role, item.content);
}

function save() {
  histories.set(historyKey(), history);
  try {
    localStorage.setItem(historyKey(), JSON.stringify(history));
    byId("history-note")?.remove();
  } catch {
    // Long conversations can exceed browser storage while inference still works.
    if (!byId("history-note")) {
      const note = document.createElement("p");
      note.id = "history-note";
      note.className = "error";
      note.setAttribute("role", "status");
      note.textContent =
        "This conversation is available in this tab, but could not be saved for reopening.";
      byId("conversation").before(note);
    }
  }
}

async function health() {
  const modelId = selectedModel.id;
  let available = false;
  try {
    const [response, readiness] = await Promise.all([
      fetch(new URL("models", apiBase), {
        signal: AbortSignal.timeout(6000),
        cache: "no-store",
      }),
      fetch(new URL(config.health_path, window.location.href), {
        signal: AbortSignal.timeout(6000),
        cache: "no-store",
      }),
    ]);
    const result = await response.json();
    const ready = await readiness.json().catch(() => ({}));
    available =
      response.ok &&
      readiness.ok &&
      (result.data || []).some((m) => m.id === modelId) &&
      (!config.require_loaded_model || ready.engine_pool?.loaded_count > 0);
  } catch {
    available = false;
  }
  if (modelId !== selectedModel.id) return;
  online = available;
  byId("status").textContent = online
    ? config.load_on_demand
      ? "Available · loads on request"
      : "Ready"
    : "Starting or unavailable";
  byId("status").className = online ? "ready" : "";
  byId("send").disabled = !online || controller !== null;
}

byId("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = byId("prompt").value.trim();
  if (!prompt || controller || !online) return;
  history.push({ role: "user", content: prompt });
  message("user", prompt);
  save();
  byId("prompt").value = "";
  const answer = message("assistant");
  controller = new AbortController();
  byId("send").disabled = true;
  byId("clear").disabled = true;
  byId("model-picker").disabled = true;
  byId("cancel").hidden = false;
  const started = performance.now();
  let first = null,
    content = "",
    thoughts = "",
    usage = null,
    finish = null;
  const timer = setInterval(() => {
    byId("metrics").textContent =
      `${content || thoughts ? "Generating" : "Processing prompt"} · ${((performance.now() - started) / 1000).toFixed(1)}s`;
  }, 250);
  try {
    const response = await fetch(new URL("chat/completions", apiBase), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      signal: controller.signal,
      body: JSON.stringify({
        model: selectedModel.id,
        messages: history,
        stream: true,
        stream_options: { include_usage: true },
        max_tokens: 8192,
      }),
    });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(
        detail.error?.message ||
          detail.detail ||
          `Service returned HTTP ${response.status}`,
      );
    }
    if (!response.body)
      throw new Error("The service returned no response stream.");
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    function consume(line) {
      if (!line.startsWith("data:")) return;
      const raw = line.slice(5).trim();
      if (!raw || raw === "[DONE]") return;
      const chunk = JSON.parse(raw);
      if (chunk.error)
        throw new Error(chunk.error.message || "Generation failed");
      usage = chunk.usage || usage;
      for (const choice of chunk.choices || []) {
        finish = choice.finish_reason || finish;
        const delta = choice.delta || {};
        const reason = delta.reasoning_content || delta.reasoning || "";
        if ((delta.content || reason) && first === null)
          first = (performance.now() - started) / 1000;
        content += delta.content || "";
        thoughts += reason;
      }
      answer.text.textContent = content;
      answer.thinking.hidden = !thoughts;
      answer.thoughts.textContent = thoughts;
      scroll();
    }
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) consume(line.replace(/\r$/, ""));
    }
    buffer += decoder.decode();
    if (buffer.trim()) consume(buffer.trim());
    if (!content && !thoughts)
      throw new Error("The service finished without returning text.");
    if (finish === "length")
      answer.text.textContent += "\n\n[Output limit reached. Ask to continue.]";
  } catch (error) {
    const note =
      error.name === "AbortError"
        ? "Canceled."
        : `Request failed: ${error.message}`;
    const warning = document.createElement("p");
    warning.className = "error";
    warning.textContent = note;
    answer.box.append(warning);
  } finally {
    clearInterval(timer);
    if (content) {
      history.push({
        role: "assistant",
        content,
        ...(thoughts ? { reasoning_content: thoughts } : {}),
      });
      save();
    }
    byId("metrics").textContent =
      `${((performance.now() - started) / 1000).toFixed(1)}s total${first === null ? "" : ` · First update ${first.toFixed(1)}s`}${usage?.completion_tokens ? ` · ${usage.completion_tokens} generated tokens` : ""}`;
    controller = null;
    byId("cancel").hidden = true;
    byId("clear").disabled = false;
    byId("model-picker").disabled = false;
    await health();
    byId("prompt").focus();
    scroll();
  }
});
byId("cancel").addEventListener("click", () => controller?.abort());
byId("clear").addEventListener("click", () => {
  history = [];
  save();
  byId("conversation").replaceChildren();
  byId("metrics").textContent = "";
  byId("prompt").focus();
});
byId("prompt").addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
    event.preventDefault();
    byId("composer").requestSubmit();
  }
});

(async () => {
  const response = await fetch("service.json", { cache: "no-store" });
  if (!response.ok) throw new Error("Service configuration unavailable");
  config = await response.json();
  byId("title").textContent = config.title;
  document.title = config.title;
  byId("description").textContent = config.description;
  const models = config.models || [
    { id: config.model_id, label: config.model_label },
  ];
  selectedModel = models.find((m) => m.id === config.model_id) || models[0];
  const picker = byId("model-picker");
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.label;
    picker.append(option);
  }
  picker.value = selectedModel.id;
  picker.hidden = models.length < 2;
  byId("model").hidden = models.length > 1;
  byId("model").textContent = selectedModel.label;
  picker.addEventListener("change", async () => {
    if (controller) {
      picker.value = selectedModel.id;
      return;
    }
    save();
    selectedModel = models.find((m) => m.id === picker.value) || selectedModel;
    online = false;
    byId("send").disabled = true;
    byId("model-id").textContent = selectedModel.id;
    byId("metrics").textContent = "";
    restoreHistory();
    await health();
  });
  byId("base-url").textContent = apiBase.href;
  byId("model-id").textContent = selectedModel.id;
  const local = location.hostname.endsWith(".localhost");
  for (const link of config.links) {
    const a = document.createElement("a");
    a.textContent = link.label;
    a.href = local ? link.local : link.tailnet;
    byId("navigation").append(a);
  }
  restoreHistory();
  await health();
  setInterval(health, 15000);
})().catch((error) => {
  byId("status").textContent = "Configuration unavailable";
  byId("description").textContent = error.message;
});
