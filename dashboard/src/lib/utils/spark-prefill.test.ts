import assert from "node:assert/strict";
import test from "node:test";
import {
  assessSparkRoute,
  type PrefillFleetState,
  type PrefillModel,
} from "./spark-prefill.ts";

const now = Date.parse("2026-10-23T12:00:00Z");
const GIB = 1024 ** 3;
const model: PrefillModel = {
  id: "pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit",
  tasks: ["TextGeneration"],
  storage_size_megabytes: 99 * 1024,
  supports_tensor: true,
};

function shard(
  kind: "TensorShardMetadata" | "PipelineShardMetadata",
  rank: number,
  worldSize: number,
  start = 0,
  end = 48,
  layers = 48,
) {
  return {
    [kind]: {
      deviceRank: rank,
      worldSize,
      startLayer: start,
      endLayer: end,
      nLayers: layers,
      modelCard: { nLayers: layers },
    },
  };
}

function fleet(): PrefillFleetState {
  const ids = ["spark", "a", "b", "studio"];
  return {
    reachable: true,
    lastSeen: Object.fromEntries(
      ids.map((id) => [id, new Date(now).toISOString()]),
    ),
    nodeIdentities: {
      spark: { friendlyName: "wc-spark" },
      a: { friendlyName: "wc-smbp" },
      b: { friendlyName: "wc-smbpt" },
      studio: { friendlyName: "wc-studio (123)" },
    },
    nodeMemory: Object.fromEntries(
      ids.map((id) => [
        id,
        {
          ramTotal: { inBytes: (id === "spark" ? 121.7 : 128) * GIB },
          ramAvailable: { inBytes: (id === "spark" ? 15 : 80) * GIB },
        },
      ]),
    ),
    nodeBackends: {
      spark: ["MlxCuda"],
      a: ["MlxMetal"],
      b: ["MlxMetal"],
      studio: ["MlxMetal"],
    },
    nodeRdmaCtl: { a: { enabled: true }, b: { enabled: true } },
    topology: {
      nodes: ids,
      connections: {
        a: {
          spark: [{ sinkMultiaddr: { ip_address: "192.168.1.150" } }],
          b: [{ sourceRdmaIface: "rdma_en2", sinkRdmaIface: "rdma_en1" }],
        },
        b: {
          spark: [{ sinkMultiaddr: { ip_address: "192.168.1.150" } }],
          a: [{ sourceRdmaIface: "rdma_en1", sinkRdmaIface: "rdma_en2" }],
        },
        studio: { spark: [{ sinkMultiaddr: { ip_address: "192.168.1.150" } }] },
      },
    },
    instances: {
      prefill: {
        MlxRingInstance: {
          shardAssignments: {
            modelId: model.id,
            shards: [["spark", "rp", shard("PipelineShardMetadata", 0, 1)]],
          },
        },
      },
      decode: {
        MlxJacclInstance: {
          shardAssignments: {
            modelId: model.id,
            shards: [
              ["a", "ra", shard("TensorShardMetadata", 0, 2)],
              ["b", "rb", shard("TensorShardMetadata", 1, 2)],
            ],
          },
        },
      },
    },
    runners: {
      rp: { RunnerReady: { prefillServerPort: 1234 } },
      ra: { RunnerReady: {} },
      rb: { RunnerReady: {} },
    },
    instanceLinks: {
      link: { prefillInstances: ["prefill"], decodeInstances: ["decode"] },
    },
  };
}

test("the verified resident route stays ready even when free memory is low", () => {
  const result = assessSparkRoute(model, "twins", fleet(), true, now);
  assert.equal(result.status, "ready");
  assert.equal(result.verified, true);
});

test("Studio capacity does not inherit the Twins verification", () => {
  const result = assessSparkRoute(model, "studio", fleet(), true, now);
  assert.equal(result.verified, false);
  assert.equal(result.capacityCandidate, true);
  assert.equal(result.label, "Free memory needed");
});

test("Studio candidates can reuse the full checkpoint already resident on Spark", () => {
  const state = fleet();
  state.nodeMemory!.studio.ramAvailable!.inBytes = 125 * GIB;
  const result = assessSparkRoute(model, "studio", state, true, now);
  assert.equal(result.label, "Memory candidate");
  assert.equal(result.verified, false);
  assert.match(result.detail, /loaded copy can be reused/);
});

test("resident but unlinked copies are not charged for their weights twice", () => {
  const state = fleet();
  state.instanceLinks = {};
  state.nodeMemory!.a.ramAvailable!.inBytes = 20 * GIB;
  state.nodeMemory!.b.ramAvailable!.inBytes = 20 * GIB;
  const result = assessSparkRoute(model, "twins", state, true, now);
  assert.equal(result.label, "Memory candidate");
  assert.notEqual(result.status, "ready");
});

test("verification belongs to the exact quantization variant", () => {
  const result = assessSparkRoute(
    { ...model, id: "pipenetwork/Qwen3.8-Flash-Next-MLX-4bit" },
    "twins",
    fleet(),
    true,
    now,
  );
  assert.equal(result.verified, false);
  assert.notEqual(result.status, "ready");
});

test("a model that fits combined Mac memory can still be too large for Spark", () => {
  const result = assessSparkRoute(
    { ...model, id: "large", storage_size_megabytes: 160 * 1024 },
    "twins",
    fleet(),
    true,
    now,
  );
  assert.equal(result.label, "Too large for Spark");
  assert.equal(result.capacityCandidate, false);
});

test("Twins capacity checks each rank rather than pooling RAM", () => {
  const state = fleet();
  state.nodeMemory!.spark.ramTotal!.inBytes = 256 * GIB;
  state.nodeMemory!.a.ramTotal!.inBytes = 64 * GIB;
  assert.equal(
    assessSparkRoute(
      { ...model, id: "uneven", storage_size_megabytes: 120 * 1024 },
      "twins",
      state,
      true,
      now,
    ).label,
    "Too large for decoder",
  );
});

test("stale identities cannot keep an unavailable route green", () => {
  const result = assessSparkRoute(model, "twins", fleet(), true, now + 46000);
  assert.equal(result.status, "offline");
  assert.equal(result.verified, true);
  const state = fleet();
  delete state.lastSeen!.b;
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).status,
    "offline",
  );
});

test("loss of API connectivity overrides a cached ready state", () => {
  const state = fleet();
  state.reachable = false;
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "Cluster unavailable",
  );
});

test("Twins require bidirectional RDMA and enabled RDMA control", () => {
  const state = fleet();
  state.topology!.connections!.b.a = [];
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "RDMA link unavailable",
  );
  const disabled = fleet();
  disabled.nodeRdmaCtl!.a.enabled = false;
  assert.equal(
    assessSparkRoute(model, "twins", disabled, true, now).label,
    "RDMA link unavailable",
  );
});

test("every decode rank needs a path to Spark", () => {
  const state = fleet();
  state.topology!.connections!.b.spark = [];
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "No prefill network path",
  );
});

test("a mismatched model link is not a ready route", () => {
  const state = fleet();
  state.instances!.prefill.MlxRingInstance!.shardAssignments!.modelId =
    "another-model";
  assert.notEqual(
    assessSparkRoute(model, "twins", state, true, now).status,
    "ready",
  );
});

test("a live Studio route is not automatically historical verification", () => {
  const state = fleet();
  state.instances!.decode = {
    MlxRingInstance: {
      shardAssignments: {
        modelId: model.id,
        nodeToRunner: { studio: "rs" },
        runnerToShard: { rs: shard("PipelineShardMetadata", 0, 1) },
      },
    },
  };
  state.runners!.rs = { RunnerReady: {} };
  const result = assessSparkRoute(model, "studio", state, true, now);
  assert.equal(result.status, "ready");
  assert.equal(result.verified, false);
});

test("a ready route wins over another matching route still loading", () => {
  const state = fleet();
  state.runners!.ra = { RunnerLoading: {} };
  state.instances!.second = {
    MlxJacclInstance: {
      shardAssignments: {
        modelId: model.id,
        nodeToRunner: { a: "ra2", b: "rb2" },
        runnerToShard: {
          ra2: shard("TensorShardMetadata", 0, 2),
          rb2: shard("TensorShardMetadata", 1, 2),
        },
      },
    },
  };
  state.runners!.ra2 = { RunnerReady: {} };
  state.runners!.rb2 = { RunnerReady: {} };
  state.instanceLinks!.second = {
    prefillInstances: ["prefill"],
    decodeInstances: ["second"],
  };
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).status,
    "ready",
  );
});

test("a route with a loading rank or no prefill listener is preparing", () => {
  const state = fleet();
  state.runners!.ra = { RunnerLoading: {} };
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).status,
    "loading",
  );
  state.runners!.ra = { RunnerReady: {} };
  state.runners!.rp = { RunnerRunning: {} };
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).status,
    "loading",
  );
  state.prefillServerPorts = { rp: 1234 };
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).status,
    "ready",
  );
});

test("pipeline-over-JACCL is not a supported Twins prefill decoder", () => {
  const state = fleet();
  state.instances!.decode.MlxJacclInstance!.shardAssignments!.shards = [
    ["a", "ra", shard("PipelineShardMetadata", 0, 2, 0, 24)],
    ["b", "rb", shard("PipelineShardMetadata", 1, 2, 24, 48)],
  ];
  const result = assessSparkRoute(model, "twins", state, true, now);
  assert.equal(result.label, "Unsupported linked layout");
  assert.equal(result.status, "blocked");
});

test("missing shard metadata cannot establish readiness or resident credit", () => {
  const state = fleet();
  state.instances!.decode.MlxJacclInstance!.shardAssignments!.shards![0][2] =
    {};
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "Shard layout unavailable",
  );
  state.instanceLinks = {};
  state.nodeMemory!.a.ramAvailable!.inBytes = 20 * GIB;
  state.nodeMemory!.b.ramAvailable!.inBytes = 20 * GIB;
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "Free memory needed",
  );
});

test("partial single-rank prefill and Studio decoders are rejected", () => {
  const state = fleet();
  state.instances!.prefill.MlxRingInstance!.shardAssignments!.shards![0][2] =
    shard("PipelineShardMetadata", 0, 1, 0, 24);
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "Unsupported linked layout",
  );
  const studio = fleet();
  studio.instances!.decode = {
    MlxRingInstance: {
      shardAssignments: {
        modelId: model.id,
        shards: [["studio", "rs", shard("PipelineShardMetadata", 0, 1, 0, 24)]],
      },
    },
  };
  studio.runners!.rs = { RunnerReady: {} };
  assert.equal(
    assessSparkRoute(model, "studio", studio, true, now).label,
    "Unsupported linked layout",
  );
});

test("prefill and decode must cover the same complete layer count", () => {
  const state = fleet();
  state.instances!.prefill.MlxRingInstance!.shardAssignments!.shards![0][2] =
    shard("PipelineShardMetadata", 0, 1, 0, 64, 64);
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "Unsupported linked layout",
  );
});

test("known cache, task and backend limitations are explicit", () => {
  assert.equal(
    assessSparkRoute(
      { ...model, id: "mlx-community/DeepSeek-V4-Flash" },
      "studio",
      fleet(),
      true,
      now,
    ).label,
    "Unsupported cache",
  );
  assert.equal(
    assessSparkRoute(
      { ...model, tasks: ["ImageGeneration"] },
      "studio",
      fleet(),
      true,
      now,
    ).label,
    "Text models only",
  );
  const state = fleet();
  state.nodeBackends!.spark = ["Vllm"];
  assert.equal(
    assessSparkRoute(model, "twins", state, true, now).label,
    "Backend unavailable",
  );
});

test("tensor support is required only for Twins", () => {
  const noTensor = { ...model, id: "single-mac", supports_tensor: false };
  assert.equal(
    assessSparkRoute(noTensor, "twins", fleet(), true, now).label,
    "No tensor support listed",
  );
  assert.equal(
    assessSparkRoute(noTensor, "studio", fleet(), true, now).capacityCandidate,
    true,
  );
});

test("small candidates fit current free memory but remain unverified", () => {
  const result = assessSparkRoute(
    { ...model, id: "small", storage_size_megabytes: 1024 },
    "studio",
    fleet(),
    true,
    now,
  );
  assert.equal(result.label, "Memory candidate");
  assert.equal(result.verified, false);
  assert.equal(
    assessSparkRoute(model, "twins", fleet(), false, now).label,
    "Not enabled",
  );
});
