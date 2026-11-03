<script lang="ts">
  import { onMount } from "svelte";

  let { modelId }: { modelId: string | null } = $props();
  type Status = {
    enabled: boolean;
    active: boolean;
    state: "idle" | "running" | "stopped" | "finished" | "error";
    phase: "waiting" | "generating" | "stopping" | null;
    model_id: string | null;
    stop_at: number | null;
    completed_requests: number;
    error: string | null;
  };
  let status = $state<Status | null>(null);
  let pending = $state(false);
  let connected = $state(false);
  let error = $state("");
  let now = $state(Date.now());
  let disposed = false;
  let polling = false;
  let revision = 0;
  const seconds = $derived(
    Math.max(0, Math.ceil(((status?.stop_at ?? 0) * 1000 - now) / 1000)),
  );
  const countdown = $derived(
    `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`,
  );
  // Relative paths preserve /twins/ when the dashboard is mounted there.
  const endpoint = "./v1/cat-mode";

  async function refresh() {
    if (polling || pending) return;
    polling = true;
    const requestRevision = revision;
    try {
      const response = await fetch(endpoint, {
        cache: "no-store",
        signal: AbortSignal.timeout(5000),
      });
      if (disposed || requestRevision !== revision) return;
      if (response.status === 404) {
        connected = false;
        if (!status?.enabled) status = null;
        return;
      }
      if (!response.ok) throw new Error("Cat mode controller unavailable");
      const next: Status = await response.json();
      if (disposed || requestRevision !== revision) return;
      status = next;
      connected = true;
    } catch {
      if (!disposed && requestRevision === revision) connected = false;
    } finally {
      polling = false;
    }
  }

  async function toggle() {
    if (pending || !connected) return;
    pending = true;
    revision += 1;
    error = "";
    try {
      const response = await fetch(endpoint, {
        method: status?.active ? "DELETE" : "POST",
        headers: { "Content-Type": "application/json" },
        body: status?.active
          ? undefined
          : JSON.stringify({ model_id: modelId }),
        signal: AbortSignal.timeout(10000),
      });
      const result = await response.json();
      if (!response.ok)
        throw new Error(
          result.error?.message ?? result.detail ?? "Cat mode request failed",
        );
      if (!disposed) status = result;
    } catch (cause) {
      if (!disposed)
        error =
          cause instanceof Error ? cause.message : "Cat mode request failed";
    } finally {
      pending = false;
      if (!disposed) void refresh();
    }
  }

  onMount(() => {
    void refresh();
    const poll = setInterval(() => void refresh(), 2000);
    const tick = setInterval(() => (now = Date.now()), 1000);
    return () => {
      disposed = true;
      clearInterval(poll);
      clearInterval(tick);
    };
  });
</script>

{#if status?.enabled}
  <div class="cat-mode" aria-label="Cat mode controls">
    <div class="details" aria-live="polite">
      {#if !connected}
        <span>Cat mode connection lost — status unconfirmed</span>
      {:else if status.active}
        <span
          >Cat mode · {countdown} · {status.completed_requests} requests · {status.phase ===
          "generating"
            ? "Generating"
            : status.phase === "stopping"
              ? "Stopping…"
              : "Waiting for model"}</span
        >
      {:else}
        <span
          >{status.state === "finished"
            ? "Cat mode finished"
            : status.state === "stopped"
              ? "Cat mode stopped"
              : status.state === "error"
                ? "Cat mode stopped after an error"
                : "Warm the loaded model for 10 minutes"}</span
        >
      {/if}
      <small title={status.active ? (status.model_id ?? "") : (modelId ?? "")}
        >{status.active
          ? status.model_id
          : (modelId ?? "Load a text model to enable Cat mode")}</small
      >
      {#if error || status.error}<span class="error"
          >{error || status.error}</span
        >{/if}
    </div>
    <button
      class:active={status.active}
      disabled={pending ||
        !connected ||
        status.phase === "stopping" ||
        (!status.active && !modelId)}
      onclick={toggle}
      aria-label={status.active ? "Stop cat mode" : "Start cat mode"}
      title="Repeated short requests, one at a time. Stops after 10 minutes; continues if you close this tab."
      >{pending
        ? "Please wait…"
        : status.active
          ? "Stop cat mode"
          : "🐈 Cat mode"}</button
    >
  </div>
{/if}

<style>
  .cat-mode {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 0.75rem;
    padding: 0.6rem 1.5rem;
    font-family: monospace;
    font-size: 0.8rem;
  }
  .details {
    display: flex;
    flex-direction: column;
    min-width: 0;
    max-width: 100%;
    color: #aaa;
  }
  small {
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    max-width: 32rem;
  }
  button {
    color: #facc15;
    border: 1px solid #72601d;
    border-radius: 4px;
    padding: 0.5rem 0.85rem;
    background: #19170d;
    cursor: pointer;
    white-space: nowrap;
  }
  button.active {
    border-color: #facc15;
  }
  button:disabled {
    opacity: 0.5;
    cursor: default;
  }
  button:focus-visible {
    outline: 2px solid #facc15;
    outline-offset: 3px;
  }
  .error {
    color: #f88;
    max-width: 32rem;
    overflow-wrap: anywhere;
  }
  @media (max-width: 600px) {
    .cat-mode {
      padding: 0.5rem 1rem;
      justify-content: space-between;
    }
    .details {
      width: 100%;
    }
  }
</style>
