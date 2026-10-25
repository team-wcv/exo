"use strict";
const byId = (id) => document.getElementById(id);
let config,
  history = [],
  controller = null,
  online = false;
const apiBase = new URL("v1/", window.location.href);

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
function save() {
  localStorage.setItem(`local-inference-${config.id}`, JSON.stringify(history));
}

async function health() {
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
    online =
      response.ok &&
      readiness.ok &&
      (result.data || []).some((m) => m.id === config.model_id) &&
      (!config.require_loaded_model || ready.engine_pool?.loaded_count > 0);
  } catch {
    online = false;
  }
  byId("status").textContent = online ? "Ready" : "Starting or unavailable";
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
        model: config.model_id,
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
  byId("model").textContent = config.model_label;
  byId("base-url").textContent = apiBase.href;
  byId("model-id").textContent = config.model_id;
  const local = location.hostname.endsWith(".localhost");
  for (const link of config.links) {
    const a = document.createElement("a");
    a.textContent = link.label;
    a.href = local ? link.local : link.tailnet;
    byId("navigation").append(a);
  }
  try {
    const saved = JSON.parse(
      localStorage.getItem(`local-inference-${config.id}`) || "[]",
    );
    if (Array.isArray(saved))
      history = saved.filter(
        (m) =>
          ["user", "assistant"].includes(m.role) &&
          typeof m.content === "string",
      );
  } catch {
    history = [];
  }
  for (const item of history) message(item.role, item.content);
  await health();
  setInterval(health, 15000);
})().catch((error) => {
  byId("status").textContent = "Configuration unavailable";
  byId("description").textContent = error.message;
});
