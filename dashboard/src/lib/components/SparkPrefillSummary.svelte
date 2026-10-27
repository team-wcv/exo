<script lang="ts">
  import { onMount } from "svelte";
  import { prefillFleetState, featureFlags } from "$lib/stores/app.svelte";
  import {
    assessSparkRoute,
    hasSparkFleet,
    type PrefillModel,
  } from "$lib/utils/spark-prefill";
  let { model, onCompare }: { model?: PrefillModel; onCompare: () => void } =
    $props();
  let now = $state(Date.now());
  onMount(() => {
    const timer = setInterval(() => (now = Date.now()), 5000);
    return () => clearInterval(timer);
  });
  const fleet = $derived(prefillFleetState());
  const routes = $derived(
    model
      ? ["twins", "studio"].map((r) =>
          assessSparkRoute(
            model,
            r as "twins" | "studio",
            fleet,
            featureFlags().disaggregation,
            now,
          ),
        )
      : [],
  );
</script>

{#if hasSparkFleet(fleet)}
  <div
    class="rounded border border-exo-yellow/20 bg-exo-yellow/5 p-3 mb-3"
    aria-label="Selected model Spark prefill routes"
  >
    <div class="text-xs font-mono text-exo-yellow mb-2">
      SPARK PREFILL → MAC DECODE
    </div>
    {#each routes as route}
      <div class="flex justify-between gap-2 text-xs py-1" title={route.detail}>
        <span class="text-white/75"
          >{route.route === "twins" ? "Twins · RDMA" : "Studio"}</span
        >
        <span
          class={route.status === "ready" ? "text-green-300" : "text-white/55"}
          >{route.label}</span
        >
      </div>
    {/each}
    <button
      type="button"
      class="mt-2 text-xs text-exo-yellow underline underline-offset-4 hover:text-white"
      onclick={onCompare}>Compare Spark prefill models</button
    >
  </div>
{/if}
