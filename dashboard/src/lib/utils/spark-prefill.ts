/** Fleet guidance: capacity estimates are deliberately separate from verification. */
export interface PrefillModel {
  id: string;
  name?: string;
  base_model?: string;
  quantization?: string;
  storage_size_megabytes?: number;
  supports_tensor?: boolean;
  tasks?: string[];
}

interface FleetInstance {
  shardAssignments?: {
    modelId?: string;
    shards?: [string, string, unknown][];
    nodeToRunner?: Record<string, string>;
    runnerToShard?: Record<string, unknown>;
  };
}

export interface PrefillFleetState {
  reachable?: boolean;
  lastSeen?: Record<string, string>;
  nodeIdentities?: Record<string, { friendlyName?: string }>;
  nodeMemory?: Record<
    string,
    {
      ramTotal?: { inBytes: number };
      ramAvailable?: { inBytes: number };
    }
  >;
  nodeBackends?: Record<string, string[]>;
  nodeRdmaCtl?: Record<string, { enabled?: boolean }>;
  topology?: {
    nodes: string[];
    connections?: Record<
      string,
      Record<
        string,
        {
          sourceRdmaIface?: string;
          sinkRdmaIface?: string;
          sinkMultiaddr?: { ip_address?: string };
        }[]
      >
    >;
  };
  instances?: Record<
    string,
    {
      MlxRingInstance?: FleetInstance;
      MlxJacclInstance?: FleetInstance;
      VllmInstance?: FleetInstance;
    }
  >;
  runners?: Record<string, unknown>;
  prefillServerPorts?: Record<string, number>;
  downloads?: Record<string, unknown[]>;
  instanceLinks?: Record<
    string,
    { prefillInstances: string[]; decodeInstances: string[] }
  >;
}

export type DecodeRoute = "twins" | "studio";
export interface RouteAssessment {
  route: DecodeRoute;
  verified: boolean;
  verification?: string;
  status: "ready" | "loading" | "candidate" | "blocked" | "offline";
  label: string;
  detail: string;
  capacityCandidate: boolean;
}

// Exact variant and route evidence from the fleet acceptance run, not a family-wide claim.
const VERIFIED: Record<string, Partial<Record<DecodeRoute, string>>> = {
  "pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit": {
    twins:
      "Verified 23 Oct 2026 on this fleet using the reference recurrence and mixed 4/8-bit checkpoint. Both Mac decode ranks imported Spark's cache and returned readable text. Experimental CUDA recurrence is not verified.",
  },
};

const GIB = 1024 ** 3;
export const routeName = (route: DecodeRoute) =>
  route === "twins" ? "Twins · RDMA" : "Studio";
export const gib = (bytes: number) => `${(bytes / GIB).toFixed(1)} GiB`;

function live(state: PrefillFleetState, id: string, now: number): boolean {
  const age = now - Date.parse(state.lastSeen?.[id] ?? "");
  return (
    Number.isFinite(age) &&
    age >= -5000 &&
    age < 45000 &&
    !!state.topology?.nodes.includes(id)
  );
}

export function fleetNode(
  state: PrefillFleetState,
  role: string,
  now = Date.now(),
): string | undefined {
  const matches = Object.entries(state.nodeIdentities ?? {})
    .filter(
      ([, identity]) =>
        identity.friendlyName?.toLowerCase().replace(/\s*\(.*$/, "") ===
        `wc-${role}`,
    )
    .map(([id]) => id);
  const online = matches.filter((id) => live(state, id, now));
  return online.length === 1
    ? online[0]
    : online.length === 0
      ? matches[0]
      : undefined;
}

export function hasSparkFleet(state: PrefillFleetState): boolean {
  return !!fleetNode(state, "spark");
}

function allocations(instance: FleetInstance): [string, string][] {
  const assignments = instance.shardAssignments;
  return (
    assignments?.shards?.map(([node, runner]) => [node, runner]) ??
    Object.entries(assignments?.nodeToRunner ?? {})
  );
}

type LayoutStatus = "supported" | "unsupported" | "unknown";

function record(value: unknown): Record<string, unknown> | undefined {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : undefined;
}

function shardMetadata(instance: FleetInstance): unknown[] {
  const assignments = instance.shardAssignments;
  return (
    assignments?.shards?.map(([, , shard]) => shard) ??
    allocations(instance).map(
      ([, runner]) => assignments?.runnerToShard?.[runner],
    )
  );
}

function layerCount(instance: FleetInstance): unknown {
  const tagged = record(shardMetadata(instance)[0]);
  return (
    record(tagged?.TensorShardMetadata) ?? record(tagged?.PipelineShardMetadata)
  )?.nLayers;
}

function layoutStatus(
  instance: FleetInstance,
  worldSize: number,
  tensor: boolean,
): LayoutStatus {
  const metadata = shardMetadata(instance);
  if (metadata.length !== worldSize) return "unknown";
  const ranks = new Set<number>();
  const layerCounts = new Set<number>();
  for (const raw of metadata) {
    const tagged = record(raw);
    const tensorShard = record(tagged?.TensorShardMetadata);
    const pipelineShard = record(tagged?.PipelineShardMetadata);
    if (tensor && pipelineShard) return "unsupported";
    const shard = tensorShard ?? pipelineShard;
    if (!shard) return "unknown";
    const { startLayer, endLayer, nLayers, deviceRank } = shard;
    if (
      typeof nLayers !== "number" ||
      !Number.isInteger(nLayers) ||
      nLayers <= 0 ||
      typeof startLayer !== "number" ||
      typeof endLayer !== "number" ||
      typeof deviceRank !== "number" ||
      !Number.isInteger(deviceRank) ||
      typeof shard.worldSize !== "number"
    )
      return "unknown";
    // The prefill wire carries every layer; the decoder cannot ingest a layer subset.
    if (
      startLayer !== 0 ||
      endLayer !== nLayers ||
      shard.worldSize !== worldSize ||
      deviceRank < 0 ||
      deviceRank >= worldSize
    )
      return "unsupported";
    const cardLayers = record(shard.modelCard)?.nLayers;
    if (typeof cardLayers === "number" && cardLayers !== nLayers)
      return "unsupported";
    ranks.add(deviceRank);
    layerCounts.add(nLayers);
  }
  return ranks.size === worldSize && layerCounts.size === 1
    ? "supported"
    : "unsupported";
}

function residentCopy(
  state: PrefillFleetState,
  model: string,
  nodes: string[],
  kind: "MlxRingInstance" | "MlxJacclInstance",
  prefill: boolean,
): boolean {
  return Object.values(state.instances ?? {}).some((wrapper) => {
    const instance = wrapper[kind];
    if (instance?.shardAssignments?.modelId !== model) return false;
    if (
      layoutStatus(instance, nodes.length, nodes.length === 2) !== "supported"
    )
      return false;
    const shards = allocations(instance);
    if (
      shards.length !== nodes.length ||
      !nodes.every((id) => shards.some(([node]) => node === id))
    )
      return false;
    return shards.every(([, runner]) => {
      const status = state.runners?.[runner];
      if (
        typeof status !== "object" ||
        status === null ||
        !("RunnerReady" in status || "RunnerRunning" in status)
      )
        return false;
      const port =
        state.prefillServerPorts?.[runner] ??
        ("RunnerReady" in status
          ? (status.RunnerReady as { prefillServerPort?: number })
              .prefillServerPort
          : undefined);
      return !prefill || (typeof port === "number" && port > 0);
    });
  });
}

function linkedRoute(
  state: PrefillFleetState,
  model: string,
  spark: string,
  decode: string[],
): "ready" | "loading" | "unsupported" | "unknown" | null {
  let preparing = false;
  let invalidLayout: "unsupported" | "unknown" | null = null;
  for (const link of Object.values(state.instanceLinks ?? {})) {
    for (const prefillId of link.prefillInstances) {
      const prefill = state.instances?.[prefillId]?.MlxRingInstance;
      if (!prefill || prefill.shardAssignments?.modelId !== model) continue;
      const sources = allocations(prefill);
      if (sources.length !== 1 || sources[0][0] !== spark) continue;
      for (const decodeId of link.decodeInstances) {
        const wrapper = state.instances?.[decodeId];
        const instance =
          decode.length === 2
            ? wrapper?.MlxJacclInstance
            : wrapper?.MlxRingInstance;
        if (!instance || instance.shardAssignments?.modelId !== model) continue;
        const targets = allocations(instance);
        if (
          targets.length !== decode.length ||
          !decode.every((id) => targets.some(([node]) => node === id))
        )
          continue;
        const layouts = [
          layoutStatus(prefill, 1, false),
          layoutStatus(instance, decode.length, decode.length === 2),
        ];
        if (layouts.some((layout) => layout !== "supported")) {
          if (layouts.includes("unsupported")) invalidLayout = "unsupported";
          else invalidLayout ??= "unknown";
          continue;
        }
        if (layerCount(prefill) !== layerCount(instance)) {
          invalidLayout = "unsupported";
          continue;
        }
        const runnerStates = [...sources, ...targets].map(
          ([, id]) => state.runners?.[id],
        );
        const ready = runnerStates.every(
          (r) =>
            typeof r === "object" &&
            r !== null &&
            ("RunnerReady" in r || "RunnerRunning" in r),
        );
        const sourceState = state.runners?.[sources[0][1]];
        const sourcePort =
          state.prefillServerPorts?.[sources[0][1]] ??
          (typeof sourceState === "object" &&
          sourceState !== null &&
          "RunnerReady" in sourceState
            ? (sourceState.RunnerReady as { prefillServerPort?: number })
                .prefillServerPort
            : undefined);
        if (ready && typeof sourcePort === "number" && sourcePort > 0)
          return "ready";
        preparing = true;
      }
    }
  }
  return preparing ? "loading" : invalidLayout;
}

export function assessSparkRoute(
  model: PrefillModel,
  route: DecodeRoute,
  state: PrefillFleetState,
  enabled: boolean,
  now = Date.now(),
): RouteAssessment {
  const verification = VERIFIED[model.id]?.[route];
  const result = (
    status: RouteAssessment["status"],
    label: string,
    detail: string,
    capacityCandidate = false,
  ): RouteAssessment => ({
    route,
    verified: !!verification,
    verification,
    status,
    label,
    detail,
    capacityCandidate,
  });
  if (state.reachable === false)
    return result(
      "offline",
      "Cluster unavailable",
      "Live cluster status is unavailable. Reconnect before using this route.",
    );
  if (!enabled)
    return result(
      "blocked",
      "Not enabled",
      "Remote prefill is disabled on this cluster.",
    );
  if (!model.tasks?.includes("TextGeneration"))
    return result(
      "blocked",
      "Text models only",
      "This route requires a text-generation model.",
    );
  // These caches are explicitly rejected by the deployed MLX wire adapter.
  if (/deepseek[- ]v4/i.test(`${model.id} ${model.base_model ?? ""}`))
    return result(
      "blocked",
      "Unsupported cache",
      "This fleet runtime cannot transfer DeepSeek V4's cache.",
    );
  if (route === "twins" && !model.supports_tensor)
    return result(
      "blocked",
      "No tensor support listed",
      "Two-Mac decode requires model tensor-sharding support.",
    );

  const spark = fleetNode(state, "spark", now);
  const ids =
    route === "twins"
      ? [fleetNode(state, "smbp", now), fleetNode(state, "smbpt", now)]
      : [fleetNode(state, "studio", now)];
  if (!spark || !live(state, spark, now))
    return result(
      "offline",
      "Spark offline",
      "Reconnect Spark to assess or use this route.",
    );
  if (ids.some((id) => !id || !live(state, id, now)))
    return result(
      "offline",
      `${routeName(route)} unavailable`,
      "Every decode device must be connected with fresh cluster state.",
    );
  const decode = ids as string[];
  if (
    !state.nodeBackends?.[spark]?.includes("MlxCuda") ||
    decode.some((id) => !state.nodeBackends?.[id]?.includes("MlxMetal"))
  )
    return result(
      "blocked",
      "Backend unavailable",
      "Spark needs MLX CUDA and the decode Macs need MLX Metal.",
    );
  if (route === "twins") {
    const rdma = (from: string, to: string) =>
      state.topology?.connections?.[from]?.[to]?.some(
        (e) => e.sourceRdmaIface && e.sinkRdmaIface,
      );
    if (
      !decode.every((id) => state.nodeRdmaCtl?.[id]?.enabled) ||
      !rdma(decode[0], decode[1]) ||
      !rdma(decode[1], decode[0])
    )
      return result(
        "blocked",
        "RDMA link unavailable",
        "Connect and enable Thunderbolt/RDMA between both MacBooks.",
      );
  }
  if (
    !decode.every((id) =>
      state.topology?.connections?.[id]?.[spark]?.some(
        (e) => e.sinkMultiaddr?.ip_address,
      ),
    )
  )
    return result(
      "blocked",
      "No prefill network path",
      "The decode devices must be able to reach Spark over the inference LAN.",
    );
  const linked = linkedRoute(state, model.id, spark, decode);
  if (linked === "unsupported")
    return result(
      "blocked",
      "Unsupported linked layout",
      "Spark must prefill every layer. Twins require full-layer tensor shards; Studio requires a full-model single-rank decoder. Recreate the incompatible placement before using this route.",
    );
  if (linked === "unknown")
    return result(
      "blocked",
      "Shard layout unavailable",
      "The linked instances do not report enough shard metadata to confirm this route. Readiness cannot be established from the communication backend alone.",
    );
  if (linked === "ready")
    return result(
      "ready",
      "Route ready",
      "A matching Spark prefill instance is linked to this decoder and its runners are ready.",
      true,
    );
  if (linked === "loading")
    return result(
      "loading",
      "Route preparing",
      "The matching route exists, but one or more runners are not ready.",
      true,
    );

  const bytes = (model.storage_size_megabytes ?? 0) * 1024 ** 2;
  const nodes = [spark, ...decode];
  const totals = nodes.map(
    (id) => state.nodeMemory?.[id]?.ramTotal?.inBytes ?? 0,
  );
  if (bytes <= 0 || totals.some((n) => n <= 0))
    return result(
      "blocked",
      "Memory estimate unavailable",
      "Model size and device memory are needed to estimate this route.",
    );
  const budgets = totals.map((total) => total - Math.max(8 * GIB, total * 0.1));
  if (bytes > budgets[0])
    return result(
      "blocked",
      "Too large for Spark",
      `Spark needs the full ${gib(bytes)} checkpoint. Estimated weight budget: ${gib(budgets[0])}.`,
    );
  const perDecoder = bytes / decode.length;
  if (decode.some((_, i) => perDecoder > budgets[i + 1]))
    return result(
      "blocked",
      "Too large for decoder",
      `Estimated weights per decode device: ${gib(perDecoder)}. Additional memory is reserved for execution.`,
    );
  const available = nodes.map(
    (id) => state.nodeMemory?.[id]?.ramAvailable?.inBytes ?? 0,
  );
  const reuseSpark = residentCopy(
    state,
    model.id,
    [spark],
    "MlxRingInstance",
    true,
  );
  const reuseDecode = residentCopy(
    state,
    model.id,
    decode,
    route === "twins" ? "MlxJacclInstance" : "MlxRingInstance",
    false,
  );
  const requirements = [
    reuseSpark ? 0 : bytes,
    ...decode.map(() => (reuseDecode ? 0 : perDecoder)),
  ].map((weights, i) => weights + totals[i] - budgets[i]);
  const fitsFree = requirements.every((needed, i) => available[i] >= needed);
  const note =
    route === "twins"
      ? "Two-way weight split is an estimate; model layouts and cache support still require validation."
      : "Cache support and this exact checkpoint still require validation on Studio.";
  return result(
    "candidate",
    fitsFree ? "Memory candidate" : "Free memory needed",
    `${fitsFree ? "Fits the current memory estimate." : "Fits total capacity, but other allocations must be freed first."} ${reuseSpark ? "Spark's loaded copy can be reused. " : ""}${verification ? "Previously verified; create or restore this route to use it." : note}`,
    true,
  );
}
