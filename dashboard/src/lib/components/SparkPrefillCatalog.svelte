<script lang="ts">
  import { onMount } from "svelte";
  import { prefillFleetState, featureFlags } from "$lib/stores/app.svelte";
  import {
    assessSparkRoute,
    fleetNode,
    gib,
    type PrefillModel,
    type RouteAssessment,
  } from "$lib/utils/spark-prefill";
  import { getNodesWithModelDownloaded } from "$lib/utils/downloads";

  let {
    models,
    onSelect,
    canSelect,
  }: {
    models: PrefillModel[];
    onSelect: (id: string) => void;
    canSelect: (id: string) => boolean;
  } = $props();
  let view = $state<"all" | "verified" | "candidates">("all");
  let now = $state(Date.now());
  onMount(() => {
    const timer = setInterval(() => (now = Date.now()), 5000);
    return () => clearInterval(timer);
  });
  const fleet = $derived(prefillFleetState());
  const rows = $derived.by(() => {
    return models
      .map((model) => ({
        model,
        routes: ["twins", "studio"].map((route) =>
          assessSparkRoute(
            model,
            route as "twins" | "studio",
            fleet,
            featureFlags().disaggregation,
            now,
          ),
        ),
      }))
      .filter(
        ({ routes }) =>
          view === "all" ||
          (view === "verified"
            ? routes.some((r) => r.verified)
            : routes.some((r) => r.capacityCandidate)),
      )
      .sort(
        (a, b) =>
          Number(b.routes.some((r) => r.verified)) -
            Number(a.routes.some((r) => r.verified)) ||
          Number(b.routes.some((r) => r.capacityCandidate)) -
            Number(a.routes.some((r) => r.capacityCandidate)) ||
          (b.model.storage_size_megabytes ?? 0) -
            (a.model.storage_size_megabytes ?? 0),
      );
  });
  function tone(route: RouteAssessment): string {
    if (route.status === "ready") return "text-green-300";
    if (route.status === "loading" || route.status === "candidate")
      return "text-amber-300";
    return "text-white/50";
  }
  function downloadedOn(modelId: string): string {
    const ids = getNodesWithModelDownloaded(fleet.downloads ?? {}, modelId);
    const names = ids.map(
      (id) => fleet.nodeIdentities?.[id]?.friendlyName ?? id.slice(0, 8),
    );
    return names.length
      ? `Downloads recorded on ${names.join(", ")}`
      : "Checkpoint downloads required";
  }
  const sparkId = $derived(fleetNode(fleet, "spark"));
  const sparkMemory = $derived(
    sparkId ? fleet.nodeMemory?.[sparkId] : undefined,
  );
</script>

<section
  class="p-3 space-y-3 font-sans"
  aria-label="Spark prefill model compatibility"
>
  <div
    class="rounded border border-exo-yellow/25 bg-exo-yellow/5 p-3 space-y-2"
  >
    <h3 class="text-sm text-exo-yellow font-semibold">
      Spark prefill → Mac decode
    </h3>
    <p class="text-xs text-white/70 leading-relaxed">
      <strong class="text-white/90">Twins</strong> means smbp + smbpt over RDMA. Studio
      decodes on its own. Spark needs a full copy of the same model variant.
    </p>
    {#if sparkMemory?.ramTotal}
      <p class="text-xs text-white/60">
        Spark: {gib(sparkMemory.ramTotal.inBytes)} total · {gib(
          sparkMemory.ramAvailable?.inBytes ?? 0,
        )} available
      </p>
    {/if}
    <p class="text-xs text-white/60 leading-relaxed">
      <strong class="text-green-300">Verified</strong> = tested exact route.
      <strong class="text-amber-300">Memory candidate</strong> = capacity estimate;
      runtime compatibility is untested.
    </p>
    <details class="text-xs text-white/50">
      <summary class="cursor-pointer hover:text-white/80"
        >Memory and performance notes</summary
      >
      <p class="mt-2 leading-relaxed">
        Estimates reserve at least 8 GiB or 10% of each device's RAM. Actual
        needs depend on context, cache settings and model layout. Download
        records may include shards; Spark needs the complete checkpoint.
      </p>
      <p class="mt-2 leading-relaxed">
        Remote prefill is used for long uncached prompts; short or cached chat
        can decode without calling Spark. It is not a guarantee of lower
        latency.
      </p>
    </details>
  </div>

  <div class="flex flex-wrap gap-2" aria-label="Spark route filters">
    {#each [{ value: "all", label: "All models" }, { value: "verified", label: "Verified routes" }, { value: "candidates", label: "Memory candidates" }] as filter}
      <button
        type="button"
        aria-pressed={view === filter.value}
        class="rounded border px-2.5 py-1.5 text-xs {view === filter.value
          ? 'border-exo-yellow/50 bg-exo-yellow/10 text-exo-yellow'
          : 'border-white/15 text-white/60 hover:text-white'}"
        onclick={() => (view = filter.value as typeof view)}
        >{filter.label}</button
      >
    {/each}
    <span class="text-xs text-white/40 self-center">{rows.length} variants</span
    >
  </div>

  <p class="text-xs text-white/50">
    Selecting a model chooses the checkpoint only. A split route must be
    configured and validated before relying on Spark prefill.
  </p>

  {#each rows as { model, routes } (model.id)}
    <article
      class="rounded border border-white/10 bg-black/20 p-3 space-y-3"
      data-spark-model={model.id}
    >
      <div class="flex items-start justify-between gap-3">
        <div class="min-w-0">
          <h4 class="text-sm text-white/90 font-medium">
            {model.base_model || model.name || model.id}
          </h4>
          <p class="text-xs text-white/60 mt-1">
            {model.quantization || "Unspecified quantization"} · {model.storage_size_megabytes
              ? gib(model.storage_size_megabytes * 1024 ** 2)
              : "Unknown size"}
          </p>
          <p class="text-[11px] text-white/40 break-all mt-1">{model.id}</p>
        </div>
        <button
          type="button"
          class="shrink-0 rounded border border-exo-yellow/30 px-2.5 py-1.5 text-xs text-exo-yellow hover:bg-exo-yellow/10 disabled:opacity-35 disabled:cursor-not-allowed"
          disabled={!canSelect(model.id)}
          onclick={() => onSelect(model.id)}>Select model</button
        >
      </div>
      <div class="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {#each routes as route}
          <div
            class="rounded bg-white/[0.025] border border-white/10 p-2.5"
            data-decode-route={route.route}
          >
            <div class="flex flex-wrap justify-between gap-1 items-center">
              <h5 class="text-xs font-semibold text-white/85">
                Spark → {route.route === "twins" ? "Twins · RDMA" : "Studio"}
              </h5>
              {#if route.verified}<span
                  class="rounded bg-green-400/10 px-1.5 py-0.5 text-[10px] text-green-300"
                  title={route.verification}>Verified</span
                >
              {:else}<span class="text-[10px] text-white/40">Not verified</span
                >{/if}
            </div>
            <p class="text-xs font-medium mt-2 {tone(route)}">{route.label}</p>
            <p class="text-[11px] text-white/55 mt-1 leading-relaxed">
              {route.detail}
            </p>
            {#if route.verification}<p
                class="text-[10px] text-green-300/60 mt-2"
              >
                {route.verification}
              </p>{/if}
          </div>
        {/each}
      </div>
      <p class="text-[11px] text-white/45">{downloadedOn(model.id)}</p>
    </article>
  {:else}
    <p class="text-sm text-white/60 py-8 text-center">
      No variants match these route and model filters.
    </p>
  {/each}
</section>
