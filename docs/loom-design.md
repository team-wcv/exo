# Loom: A Framework for Confluent Inference

| | |
| --- | --- |
| **Status** | Draft (grounded in shipped exo code on `feature/gemma4-drafter-tuning` / PR #15) |
| **Author(s)** | JJ et al. |
| **Created** | 2026-05-07 |
| **Paradigm** | Confluent Inference |
| **Reference host** | [exo](https://github.com/exo-explore/exo) |
| **Reference PR** | [team-wcv/exo#15](https://github.com/team-wcv/exo/pull/15) |

---

## TL;DR

**Loom** is the productisable extraction of the speculative-decoding architecture
shipping in exo PR #15. It is the framework realisation of a paradigm we call
**Confluent Inference**: distributed AI computation in which speculative
branches are explored in parallel and collapsed through cross-branch
interference, replacing naive parallelism with amplitude-weighted consensus.

This document is grounded in **what we actually built and measured**. The
honest summary:

- **The architecture is the durable artifact.** The `Drafter` Protocol (the
  spinner contract), the `DrafterTransport` Protocol (the shuttle wire), the
  `PipelinedModelDrafter` cross-round speculation pattern, the asymmetric N+1
  placement model, and the round-robin admission scheduler are all shipping,
  type-clean, test-covered, and documented. They are the right seams.
- **Runtime profitability is hardware × topology × pattern conditional.** On
  single-device Apple Silicon 4-bit inference, every speculative pattern we
  measured is currently a **net loss** (-14% to -69% generation throughput).
  On distributed asymmetric placement where target-verify includes a network
  round-trip, the same code path becomes profitable as the network latency is
  fully overlapped by speculative drafter work.
- **Tree-attention wins are gated on upstream.** EAGLE / Medusa / Lookahead's
  published 3-6× wins all depend on `position_ids` being a per-position vector
  rather than a single `KVCache.offset` int. We've shipped scaffolding plus
  the offline EAGLE-3 head converter; the runtime is held until
  [`ml-explore/mlx-lm#846`](https://github.com/ml-explore/mlx-lm/issues/846) lands.
- **Concurrency is a separate concern from speculation.** The 5× slot-1 TTFT
  win we measured this week came from making `SequentialGenerator` round-robin
  rather than singular-slot — independent of every drafter strategy and not
  blocked on anything upstream.

Loom's value proposition is therefore not "make every model faster" — it is
*"a framework that makes Confluent Inference patterns measurable, swappable,
and distributable, with honest profile data per (pattern × topology ×
hardware) cell."*

---

## 1. Motivation

### 1.1 The bottleneck

Modern LLM inference is bottlenecked by **memory bandwidth**, not arithmetic
throughput. A single autoregressive forward pass loads tens to hundreds of
gigabytes of weights from HBM/unified memory into on-chip SRAM and produces a
single token. The matmul units sit at single-digit utilisation for the
duration of the load.

This means: a forward pass that produces 1 token costs almost the same as a
forward pass that verifies $K + 1$ tokens — *as long as the forward can
actually verify $K + 1$ positions in parallel*. The **break-even acceptance
fraction** for speculative decoding is therefore:

$$\alpha^* = \frac{K}{K + 1}$$

For $K = 4$, $\alpha^* = 80\%$. Most workloads do not clear this. This is the
quantitative version of "it's not free" that the literature usually elides.

### 1.2 The Apple Silicon ceiling

The CUDA literature's 3-6× speedups (EAGLE-3 paper, Medusa-2 paper, Lookahead
on H100) come from **tree verification**: dozens of candidate continuations
verified in a single batched forward, sharing prefix tokens via tree
attention, with each sibling getting a different RoPE position in the same
forward.

`mlx_lm` derives every position's RoPE id from `KVCache.offset`, which is a
single `int`. Two siblings at the same depth cannot get different RoPE
positions in the same forward. **Until upstream lands `position_ids`** (see
`ml-explore/mlx-lm#846`, `#250`), every speculative pattern on MLX collapses
to a *linear* verify, and the break-even math above kicks in.

A community MLX EAGLE-3 prototype confirmed this at **1.05×** on
LLaMA-3.1-8B-4bit on an M3 Ultra (mlx-lm discussion #890). Inside the noise
of our own measurements.

### 1.3 The naive parallelism trap

The natural response — "run N nodes in parallel" — does not address the
per-request bottleneck. Each node still runs its own bandwidth-bound forward
passes; throughput goes up, per-request latency does not.

The same trap applies inside one node: tensor parallelism shards the matmul
but does not reduce the number of sequential decoding steps.

### 1.4 The branch-and-collapse alternative

Speculative decoding (Leviathan et al. 2023; Chen et al. 2023) was the first
production-grade demonstration that you can break the serial barrier
*without loss*: cheaply draft multiple candidate tokens, then verify them all
in one forward pass through the heavy model. Accepted tokens advance the
sequence; rejected tokens are resampled. The output distribution is
mathematically identical to serial decoding.

EAGLE / Medusa / Lookahead generalised this with trained drafters and tree
attention. vLLM, SGLang, and TensorRT-LLM made it production-default.

**Our claim**: this same pattern — *explore many cheap branches, collapse
through cross-branch interference* — is not just a token-level decoding
trick. It is a general computational paradigm with several layers, each with
its own profitability profile and its own collapse mechanism. exo PR #15
implements the token layer. Loom is the framework that makes *every layer*
shareable, measurable, and swappable behind one interface.

| Layer | Branching unit | Collapse mechanism | Status in exo |
| --- | --- | --- | --- |
| Token | Speculative draft tokens | Greedy accept / rejection sampling | **Shipped** (PR #15) |
| Verify shape | Tree of candidates per round | Tree attention | Blocked (`mlx_lm#846`) |
| Concurrency | In-flight target requests | Round-robin admission | **Shipped** (PR #15, `456bbb32`) |
| Topology | Drafter rank vs target rank(s) | Asymmetric placement | **Shipped** (PR #15, Layer B) |
| Reasoning | Chain-of-thought paths | Verifier vote / logprob aggregation | Future Pattern |
| Routing | MoE expert candidates | Gating amplitudes | Future Pattern |
| State | Speculative cluster placements | Score-based commit | Future Pattern |
| Cache | Tree-attention KV branches | Promotion of accepted prefix | Future Pattern (gated on tree verify) |

---

## 2. The Confluent Inference Paradigm

### 2.1 Definition

> **Confluent Inference** is a model of distributed computation in which:
>
> 1. A current state is *fanned* into many candidate next-states (branches),
> 2. Branches evolve in parallel under a cheap process,
> 3. Branches are *bound* through cross-branch interference — verification,
>    voting, or scoring against a target distribution,
> 4. Bound branches collapse into a single committed extension of the state.

The principle that distinguishes Confluent Inference from naive parallelism
is **interference**: the parallel processes are not independent producers,
they are mutually constrained candidates whose disagreement carries
information.

### 2.2 Two contracts, separated honestly

A Confluent Inference operation commits to one of two correctness contracts:

- **Lossless**: the marginal distribution over outputs equals the
  distribution of serial execution under the target model. Token-level
  speculative decoding with greedy / rejection sampling is the canonical
  example. **What exo PR #15 implements.**
- **Bounded-divergence**: the marginal distribution differs from serial
  execution by a quantified amount (e.g., self-consistency over $N$
  samples, best-of-$N$ with a verifier). Useful for inference-time-scaling
  regimes where lossy aggregation is the goal. **Future Loom Patterns.**

Loom requires every collapse operator to declare which contract it
implements. Mixing them by accident is the dominant failure mode of
ad-hoc speculative systems; making the contract a typed property of every
Pattern prevents it.

### 2.3 Lossless ≠ profitable

A lossless Pattern is **structurally correct** but not necessarily
**runtime-profitable**. Profitability is a function of:

1. **The break-even acceptance fraction $\alpha^* = K / (K + 1)$** for the
   chosen $K$.
2. **The verify shape**: linear (current MLX) vs tree (CUDA, future MLX).
3. **The marginal cost of one extra verified position**: ~0 on bandwidth-bound
   forward passes with tree verify; ~$\frac{1}{K+1}$ of a forward on linear
   verify; *negative* (faster than baseline) on distributed pipeline-parallel
   targets where network latency is the dominant cost.

Loom's contract is to surface this honestly per (Pattern × topology ×
hardware) cell, not to claim universal speedups. See section 7.

---

## 3. Vocabulary → Code

Loom uses a consistent weaving vocabulary because the alternative is naming
the same concept five different ways across five different layers of the
implementation. Every term maps to a concrete artifact in the shipped code:

| Term | Meaning | exo module |
| --- | --- | --- |
| **Warp** | The committed sequence of state. Once bound, immutable. | Token sequence + KV cache offset; `KVCacheType` |
| **Weft** | A speculative branch woven across the warp. | Drafter proposals (`list[int]` of K candidate tokens) |
| **Thread** | A single candidate token / continuation. | Element of the proposal list |
| **Spinner** | The cheap branch generator. | `Drafter` Protocol (`drafter.py`) |
| **Shuttle** | The expensive branch verifier. | The target `Model` forward pass |
| **Shuttle wire** | The IPC primitive between spinner and shuttle. | `DrafterTransport` Protocol (`drafter_transport.py`) |
| **Bind** | The collapse step (greedy accept / rejection sampling). | The accept loop in `_ngram_speculative_step` / `_pipelined_speculative_step` |
| **Fabric** | The committed output. | `GenerationResponse` stream / `mlx_generate` output |
| **Spool** | A node in the cluster. | exo `Worker` / MLX rank |
| **Mill** | The full distributed cluster. | exo cluster / MLX `Group` |
| **Pattern** | A specific Confluent Inference policy. | `DraftMode` Literal (`drafter.py`) |
| **Loom** | The framework. | This document. |

The weaving language is load-bearing in the docs, comments, and APIs. It is
not decoration — it gives every concept *one* canonical name across every
layer.

---

## 4. Architecture

The architecture matches what shipped in PR #15. Below are the actual seams,
with real type signatures.

### 4.1 The Spinner Protocol (`Drafter`)

The cheap branch-generating tier is captured by a single Protocol that lives
at the **stream-factory level** — not at a finer-grained `propose / accept`
level. This was a deliberate decision: it lets the well-tested upstream
`mlx_lm.speculative_generate_step` keep owning the model-drafter path,
while in-house spinners (n-gram, EAGLE, lookahead) plug in by yielding
`GenerationResponse` the same way `stream_generate` does.

```python
DraftMode = Literal["model", "pipelined", "ngram", "eagle", "lookahead", "none"]

@runtime_checkable
class Drafter(Protocol):
    """Stream factory that runs one generation with a chosen drafting strategy."""

    @property
    def mode(self) -> DraftMode: ...

    def stream(
        self,
        *,
        model: Model,
        tokenizer: TokenizerWrapper,
        prompt: mx.array,
        context_tokens: Sequence[int],
        prompt_cache: KVCacheType,
        max_tokens: int,
        sampler: Callable[[mx.array], mx.array],
        logits_processors: Sequence[Callable[[mx.array, mx.array], mx.array]],
        prefill_step_size: int = 1,
    ) -> Generator[GenerationResponse, None, None]: ...
```

This is the actual shipping signature in
`src/exo/worker/engines/mlx/generator/drafter.py`. Six concrete spinners are
in tree today (two as scaffolding stubs); see section 5.

### 4.2 The Shuttle Wire Protocol (`DrafterTransport`)

For pipelined and remote speculation, the spinner is decoupled from its
*location* via a second Protocol that handles the IPC primitives. This is the
seam that lets the same `PipelinedModelDrafter` spec loop run locally
(in-process drafter on the same Metal device) or remotely (drafter on a
different MLX rank, communicating via `mx.distributed.send/recv` over RDMA
or TCP):

```python
DraftFuture = Future[list[int]]

@runtime_checkable
class DrafterTransport(Protocol):
    @property
    def num_draft_tokens(self) -> int:
        """K — the typical number of drafts per round."""
        ...

    def forward(self, inputs: list[int], num_forwards: int) -> DraftFuture:
        """Run num_forwards drafter forwards starting from inputs.

        Returns a Future so the caller can dispatch target verify in parallel
        — the source of cross-round speculation's win."""
        ...

    def trim_cache(self, n_positions: int) -> None: ...
    def reset_and_prefill(self, prompt_tokens: list[int]) -> None: ...
    def shutdown(self) -> None: ...
```

Two concrete transports ship: `InProcessTransport` (drafter colocated, same
Metal command queue) and `RemoteTransport` (drafter on a different rank,
wire protocol over `mx.distributed`).

The Future is `concurrent.futures.Future`, not `asyncio.Future` — the spec
loop is a synchronous generator and threading asyncio through it would have
been invasive. The remote transport's IPC thread sets the Future from
outside the calling thread, which `concurrent.futures.Future` natively
supports.

### 4.3 Cross-round speculation (`PipelinedModelDrafter`)

The cleanest demonstration of why the transport seam earns its complexity.

While the target rank verifies round $t$'s drafts, the drafter speculatively
starts round $t + 1$ by predicting the would-be bonus token and continuing
for $K$ more forwards. If the target's actual bonus matches the drafter's
predicted bonus, **round $t + 1$'s drafts are already in hand** by the time
round $t$'s verify finishes; if not, the speculative work is rolled back via
`trim_cache(K + 1)` and the standard non-speculative path runs.

The cache accounting is the only intricate bit (full notation lives in
`pipelined_drafter.py`'s module docstring), but the win is structural: the
overlap factor is bounded only by how parallel the two forwards can actually
run. On Apple Silicon's serialised Metal command queue, the in-process
overlap is ~0.1-0.3. On distributed asymmetric placement where target verify
includes a Thunderbolt-bridge RDMA round-trip, the overlap is ~1.0 and the
gain unlocks.

**Same code path. Different transport. Different topology. Different
profitability.** This is what Loom's discipline buys.

### 4.4 Round-robin admission (independent of speculation)

Speculative decoding is *one* knob; concurrent target requests is a *separate*
knob, and conflating them was the source of a 5× slot-1 TTFT regression.

The previous `SequentialGenerator._active: tuple | None` admitted exactly one
task at a time. With `EXO_NO_BATCH=1` (which spec decoding requires today),
that meant the second concurrent request waited for the first to fully
complete — extrapolated 5300 ms TTFT on a typical 512-token decode.

Round-robin admission (`456bbb32`) replaces the singular slot with
`OrderedDict[TaskId, ...]` capped by `max_concurrent_tasks` (default 8). Each
tick admits up to the cap from the queue, then advances every active task by
one `next(gen)`. Measured on `wc-smbp` with `gemma-4-26b-a4b-it-4bit`:

| Configuration | Pre-fix | Post-fix | Win |
| --- | --- | --- | --- |
| Slot 0 TTFT | ~1000 ms | 1029 ms | parity |
| Slot 1 TTFT | ~5300 ms (extrapolated) | 1015 ms | **5.2×** |
| Per-request gen tps | 120 t/s | 92.6 t/s | shared, by design |
| Aggregate gen tps | 120 t/s | 156.25 t/s | **1.3×** |

The asymmetric pipelined+remote path stays at `max_concurrent_tasks=1`
because `RemoteTransport`'s wire protocol is **session-aware but per-session
serial** — concurrent target requests would interleave `OP_PREFILL` /
`OP_FORWARD` frames on the same socket and corrupt the drafter rank's
per-session KV state. Lifting that cap requires extending the wire protocol's
command frame (already 9 `uint32` slots, with `session_id` populated; more
work needed for full multi-session interleaving). Tracked.

This is the kind of distinction Loom forces into the open: tree attention
needs `position_ids`; concurrent target requests do not. Conflating them
costs you 5× on slot-1 TTFT for a year.

### 4.5 Asymmetric N+1 placement (the distributed weave)

The hardest-won architectural piece: a topology where rank 0 is **drafter-only**
and ranks 1..N are pipeline-parallel target ranks.

Key design choices, all shipped in PR #15 Layer B:

- **No new transport stack.** `RemoteTransport` rides on the existing
  `mx.distributed.Group`, so the wire backend is automatically JACCL (RDMA
  over IB-verbs / Thunderbolt-bridge) for `MlxJacclInstance` or ring (TCP)
  for `MlxRingInstance`. The group is the transport.
- **Subgroup split.** The target's pipeline-parallel collectives operate on
  `group.split(...)` so they don't drag the drafter rank into every
  all-reduce. The drafter rank uses send/recv against the *parent* group for
  cross-subgroup point-to-point.
- **DrafterRunner state machine.** Mirrors the target runner state machine:
  `ConnectToGroup → LoadModel → StartWarmup → drafter_serve_loop`.
- **Wire protocol v2 (session-aware).** Command frame is fixed-shape
  `uint32[9]`: `[op, num_inputs, num_forwards, input_0, input_1,
  trim_amount, session_id, _, _]`. Drafter rank routes each op to its
  per-session KV cache via `session_id`. `OP_PREFILL` allocates the session;
  `OP_END_SESSION` frees it.
- **V1 boundary**: single target rank + single drafter rank ("twins"
  topology), RDMA or TCP. Multi-target asymmetric (N>1 target ranks + 1
  drafter) gated behind `NotImplementedError` until draft broadcast on the
  target subgroup ships.

This is the topology where Confluent Inference's value proposition is most
legible: the same `PipelinedModelDrafter` spec loop, the same
`DrafterTransport` interface, the same Pattern — but the speculative work
runs on a physically separate machine, fully overlapping the network round-
trip the target rank would otherwise wait on.

---

## 5. Patterns: the `DraftMode` Literal

Loom Patterns are not abstractions waiting to be filled in — they are the
concrete `DraftMode` Literal shipping in `drafter.py`. Each is a complete
spinner+collapse policy with measured profile data:

```python
DraftMode = Literal["model", "pipelined", "ngram", "eagle", "lookahead", "none"]
```

| Pattern | Spinner | Shuttle wire | Status | Profile (M5 Max, gemma-4-26b-a4b-it-4bit, 119 t/s baseline) |
| --- | --- | --- | --- | --- |
| `none` | None | n/a | Default for fast single-device | Baseline (119 t/s) |
| `ngram` | Suffix-match in running context | n/a (no second model) | Shipped | -14% to -23% (102 t/s @ K=2; degrades with K) |
| `model` | Smaller distilled drafter via `mlx_lm.stream_generate(draft_model=...)` | n/a (in-process) | Shipped | -25% to -45% (varies by workload) |
| `pipelined` (in-process) | `PipelinedModelDrafter` + `InProcessTransport` | Same Metal device | Shipped | -62% (Metal command queue serialises) |
| `pipelined` (remote, q4) | Same spinner + `RemoteTransport` over RDMA | `mx.distributed.send/recv` | Shipped | -26% to -54% (K=2 best; e2b @ q4 doubles throughput vs bf16) |
| `eagle` | EAGLE auxiliary head (depth-only or tree) | EAGLE/Medusa shareable tree verifier | **Scaffolding only**; converter ships | Blocked on `mlx_lm#846` (community prototype: 1.05×) |
| `lookahead` | Jacobi iteration on target's own forward | n/a (no second model) | **Scaffolding only** | Blocked on `mlx_lm#846` (same ceiling as `ngram`) |

### 5.1 What ships today vs what is gated

**Shipped, type-clean, test-covered, used in production paths:**
- `none`, `ngram`, `model`, `pipelined` (both in-process and remote)
- Asymmetric N+1 placement with RDMA + TCP
- `convert_eagle3_to_mlx.py` (offline tool) — downloads the RedHat EAGLE3
  head for `gemma-4-26B-A4B-it`, applies layer taps at `(2, 15, 27)`
  following EAGLE's `(2, N//2, N-3)` heuristic for Gemma-4-26b's 30 layers,
  writes MLX safetensors. Output is durable for whenever the runtime lands.

**Scaffolding only, runtime gated upstream:**
- `eagle`: `EagleDrafter` class is in tree with full integration seam, raises
  `NotImplementedError` on `stream`. Resumes when `mlx_lm` accepts
  `position_ids`.
- `lookahead`: `LookaheadDrafter` class same shape. Same gate.

The deliberate decision: ship the scaffolding and the offline converter
*now* so when the upstream blocker lifts, only the runtime fork has to land
— the integration seam, the factory dispatch, the model-card field shape,
and the head artifact are all already there.

### 5.2 Pattern resolution

`resolve_draft_mode` enforces precedence:

1. Per-request `TaskParams.draft_mode` — explicit override
2. Per-request `use_drafter is False` — opt-out shortcut
3. `EXO_DRAFT_MODE` env var
4. Implicit default: `"model"` if a drafter model loaded, else `"none"`

A `"model"` or `"pipelined"` mode without a loaded drafter degrades to
`"none"` with a warning, so misconfiguration fails *loudly* but does not
crash a serving node.

---

## 6. Topologies

Confluent Inference's profitability is dominated by the relationship between
spinner cost and shuttle cost — which is dominated by topology. Three
topologies matter today.

### 6.1 Single-device

One Metal device, one process, target + (optional) drafter both resident.

- **In-process drafter** (`InProcessTransport`): drafter and target compete
  for the same Metal command queue. MLX serialises GPU work per device, so
  the in-process pipelining overlap factor is 0.1-0.3.
- **Profitability**: every `DraftMode` except `none` is currently a net
  loss on this topology for fast small targets (4-bit Apple Silicon at >100
  t/s baseline). The break-even acceptance fraction $K/(K+1)$ is not met by
  the workloads measured.
- **Right default**: `none`.

### 6.2 Pipeline-parallel target

Target sharded across multiple ranks (either pipeline-parallel via
`MlxRingInstance` over TCP or `MlxJacclInstance` over RDMA). No dedicated
drafter rank; if drafting is enabled, the drafter is in-process on rank 0.

- **Profitability**: target verify includes cross-shard collective latency,
  which makes per-token cost much higher than single-device. The
  break-even acceptance fraction is easier to clear because each accepted
  draft saves a full cross-shard round-trip.
- **n-gram on TCP/Ring** (the only configuration we measured) showed
  roughly **1.95× scaling** on prompts where suffix-match drafting hits,
  for models that don't fit single-host (Qwen3.6-27B+).
- **Right default**: opt-in `ngram`; `none` when in doubt.

### 6.3 Asymmetric N+1

Rank 0 is drafter-only; ranks 1..N are pipeline-parallel target. The
spinner runs on physically separate hardware from the shuttle.

- **Profitability** is bounded by network latency overlap. On
  Thunderbolt-bridge RDMA between twin Macs, the speculative drafter
  forward fully overlaps the per-token network round-trip, and the
  pipelined-remote pattern is the **only configuration where the gain is
  structurally inevitable rather than empirically marginal**.
- **Measured today** (M5 Max twin pair, `gemma-4-26b-a4b-it-4bit`, q4
  drafter): K=2 reaches 74% of single-host baseline (vs 45% with bf16
  drafter). Still net-negative as standalone, but the limiting factor is
  the LAN/RDMA layer maturity, not the architecture. Twin bench is
  blocked at the LAN/RDMA layer, not the new code path.
- **The architectural payoff is durable** even if today's twin numbers
  are limited: the same code path runs over Thunderbolt-bridge RDMA, IB
  RDMA, and TCP, with no spec-loop change.

---

## 7. The Honest Hardware Matrix

Numbers measured this week. `wc-smbp` is M5 Max, single-device, 4-bit.

### 7.1 Single-device drafter sweep

| Drafter | K | Generation t/s | vs baseline |
| --- | --- | --- | --- |
| `none` (baseline) | — | **119.18** | — |
| `ngram` | 2 | 101.60 | -14% |
| `ngram` | 4 | 98.48 | -17% |
| `ngram` | 6 | 95.58 | -20% |
| `ngram` | 8 | 92.05 | -23% |
| `pipelined+remote` (q4 e2b, RDMA) | 2 | **87.91** | -26% |
| `pipelined+remote` (q4 e2b, RDMA) | 3 | 64.76 | -46% |
| `pipelined+remote` (q4 e2b, RDMA) | 5 | 54.88 | -54% |
| `pipelined+remote` (bf16 e2b, RDMA) | 5 | 37.26 | -69% |

### 7.2 What flipped, what didn't

- **Quantizing the asymmetric drafter** (bf16 e2b → q4 e2b): doubled K=5
  throughput (37 → 55 t/s); at K=2 reaches 74% of single-host baseline (vs
  45% before). Wire payload 3.5× lighter; drafter forward ~4× cheaper.
- **Concurrency (round-robin)**: 5.2× slot-1 TTFT win, 1.3× aggregate
  throughput. Independent of every drafter strategy.
- **Tree attention**: not measurable; gated upstream.

### 7.3 What this means for Loom

The honest read: **on this hardware class, for this model size, the
durable win shipped this week was concurrency, not speculation**. Speculation
remains the right architecture for distributed asymmetric topologies and for
the post-`position_ids` future, but it is not a free lunch on the most common
single-device exo deployment today.

Loom's contract is to surface this matrix per Pattern, per topology, per
hardware tier — not to obscure it behind a generic "speedup" claim.

---

## 8. Distribution Layer

### 8.1 Mapping to exo

Loom is designed as a layer over exo's existing infrastructure. The mapping
is direct because the exo work in PR #15 *is* Layer A and Layer B of Loom.

| Loom concept | exo module |
| --- | --- |
| Spinner Protocol | `src/exo/worker/engines/mlx/generator/drafter.py::Drafter` |
| `DraftMode` Literal | `src/exo/worker/engines/mlx/generator/drafter.py::DraftMode` |
| Shuttle wire Protocol | `src/exo/worker/engines/mlx/generator/drafter_transport.py::DrafterTransport` |
| In-process shuttle wire | `drafter_transport.py::InProcessTransport` |
| Remote shuttle wire | `src/exo/worker/engines/mlx/generator/remote_drafter.py::RemoteTransport` |
| Cross-round speculation pattern | `src/exo/worker/engines/mlx/generator/pipelined_drafter.py::PipelinedModelDrafter` |
| Round-robin admission | `src/exo/worker/runner/llm_inference/batch_generator.py::SequentialGenerator` |
| Drafter rank state machine | `src/exo/worker/runner/.../DrafterRunner` |
| Drafter placement | `instance.drafter_placement` field |
| Mill (cluster) | exo cluster |
| Spool (node) | exo `Worker` / MLX rank |
| Pub/sub topics | `src/exo/routing/topics.py` |
| Event-sourced state | `src/exo/shared/types/state.py` + `src/exo/shared/apply.py` |

### 8.2 Spool roles

A spool can hold one or more roles per Pattern:

- **Spinner-spool**: runs a `Drafter`. Cheap; can be co-located with the API
  node or live on a separate machine entirely (asymmetric N+1).
- **Shuttle-spool**: holds the target model. Can be one rank or sharded
  across N ranks via `MlxJacclInstance` / `MlxRingInstance`.
- **Loom-spool**: the master, by exo's bully-election protocol. Coordinates
  the Pattern, holds the canonical warp digest, broadcasts bind decisions
  via `GLOBAL_EVENTS`.

Single physical machine can play any combination. The default development
experience is all three on one process; the production frontier is the
asymmetric N+1 twin topology.

### 8.3 Failure semantics

Loom inherits exo's bully-election master replacement. The warp is
content-addressed via the running token sequence; a new master can resume
mid-flight by replaying the event log up to the last `BIND_DECISIONS`
broadcast. Speculative weft trees in flight at the moment of failure are
discarded — no correctness consequence, only a latency cost.

The remote shuttle wire's session model means a target-rank crash drops only
its own session's KV cache on the drafter rank; sibling sessions continue.
A drafter-rank crash drops every session and the target ranks degrade to
`none` with a warning (already wired in `resolve_draft_mode`).

---

## 9. API Surface

### 9.1 Building a drafter

The factory is the single entry point. It dispatches on `DraftMode` and
honours the asymmetric placement context if present:

```python
def make_drafter(
    *,
    mode: DraftMode,
    num_draft_tokens: int,
    draft_model: Model | None,
    draft_cache: KVCacheType | None,
    remote_parent_group: mx.distributed.Group | None = None,
    remote_drafter_rank: int | None = None,
    remote_target_rank: int | None = None,
    target_subgroup_size: int = 1,
    pipelined_transport: object | None = None,
) -> Drafter: ...
```

- For `mode == "pipelined"` and `remote_parent_group` is set: builds a
  `RemoteTransport` (or accepts a long-lived one passed in), wraps it in
  `PipelinedModelDrafter`.
- For `mode == "pipelined"` and no remote group: honours
  `EXO_DRAFTER_TRANSPORT`, defaulting to in-process.
- For `mode == "model"`: standard mlx_lm spec loop.
- For `mode == "ngram"`: in-house spec loop with adaptive K cap.
- For `mode in ("eagle", "lookahead")`: scaffolding stub that raises
  `NotImplementedError` on `stream` (see section 5.1).

### 9.2 K clamping for the wire-protocol budget

The asymmetric path allocates a long-lived `RemoteTransport` at builder time
with a fixed `num_draft_tokens` budget. A per-request override above the
budget is **silently clamped** rather than raising — a previous regression
(an aborted K=8 sweep) demonstrated that raising mid-flight kills the runner
subprocess and wedges the peer rank:

```python
def clamp_num_draft_tokens_to_transport(
    requested_num_draft_tokens: int,
    transport: DrafterTransport,
) -> tuple[int, bool]:
    """Clamp K against the transport's wire-protocol budget.
    Returns (clamped_K, was_clamped) so callers can emit a structured warning."""
```

This is the kind of "small but load-bearing" defensive behaviour that lives
in Loom because the alternative cost a real outage.

### 9.3 Streaming

Every Drafter produces a `Generator[GenerationResponse, None, None]` matching
`mlx_lm.stream_generate`. This means `mlx_generate`'s call site is uniform
across every Pattern; the only thing that changes is which `Drafter` was
constructed.

---

## 10. Comparison to Prior Work

| System | What it is | How Loom relates |
| --- | --- | --- |
| **vLLM, SGLang, TGI, TensorRT-LLM** | Production CUDA inference servers with built-in spec decoding | Different hardware tier (CUDA tree-attention available). Loom's vocabulary applies; concrete Patterns differ. |
| **EAGLE-1/2/3, Medusa, Lookahead** | Specific spec-decoding algorithms | Each becomes a Pattern in Loom's `DraftMode` registry. EAGLE-3 scaffolding + offline converter for Gemma-4 already shipped. |
| **mlx_lm `speculative_generate_step`** | The upstream MLX spec loop | Wrapped by `ModelDrafter`. Loom does not reinvent. |
| **vLLM's `--speculative-model='[ngram]'`, SGLang's n-gram drafter** | Production n-gram spec implementations | Equivalent to `NgramDrafter`. Loom adds adaptive K + match-strength biasing. |
| **mlx_lm community EAGLE-3 prototype (gist via discussion #890)** | Reference MLX EAGLE port (Llama-3.1 only, no Gemma-4 adapter, no tree verify, 1.05× ceiling) | Loom's `EagleDrafter` shell documents this as the integration starting point; offline converter ships, runtime forked when `position_ids` lands. |
| **Self-consistency (Wang et al.)** | Inference-time technique | Future Pattern (`confluent-reasoning`); not in scope for v0. |
| **Tree of Thoughts, Graph of Thoughts** | Reasoning-time search | Future Patterns with custom spinner + shuttle roles. |
| **Ray, Dask** | General distributed-computing frameworks | Different abstraction level. Loom is opinionated about *what* the parallelism is for. |
| **JAX `pmap` / `vmap`** | Functional batched parallelism | Loom borrows the same purity discipline at the cluster level. |

---

## 11. Roadmap

What's done is what shipped. What's planned is what's planned.

### Layer A — single-device foundation (✅ shipped, PR #15)

- `Drafter` Protocol + `DraftMode` Literal
- `NoSpecDrafter`, `ModelDrafter`, `NgramDrafter`
- `EXO_DRAFT_MODE` env + `TaskParams.draft_mode` per-request override
- `GenerationStats.draft_mode` telemetry
- 29 unit tests covering parse / resolve / propose / dispatch
- Bench harness (`bench/drafter_bench.py`) producing the matrix in section 7

### Layer B — distributed shuttle wire (✅ shipped, PR #15)

- `DrafterTransport` Protocol + `InProcessTransport`
- `PipelinedModelDrafter` with cross-round speculation
- `RemoteTransport` over `mx.distributed.send/recv` (JACCL/RDMA + ring/TCP)
- `DrafterRunner` state machine for the asymmetric drafter rank
- Wire protocol v2 (session-aware, fixed-shape `uint32[9]` command frame)
- `instance.drafter_placement` field + asymmetric N+1 placement
- Subgroup split so target's pipeline-parallel collectives don't drag the
  drafter rank in
- Twin-machine recipe documented in `remote_drafter.py`

### Layer B+ — concurrency (✅ shipped this week, `456bbb32`)

- Round-robin `SequentialGenerator` (`OrderedDict[TaskId, ...]` capped by
  `EXO_MAX_CONCURRENT_REQUESTS`, default 8)
- Asymmetric pipelined+remote stays at cap=1 (wire-protocol guard)
- Per-task error isolation
- 5.2× slot-1 TTFT win measured

### Layer C — EAGLE / Lookahead runtime (🚧 blocked upstream)

- `EagleDrafter` / `LookaheadDrafter` scaffolding ships in PR #15
- `scripts/convert_eagle3_to_mlx.py` ships, durable
- Recommended `ModelCard.eagle_head_repo` field + bootstrap-side download
- Runtime fork of `eagle_generate.py` from the gist linked in
  mlx-lm discussion #890 once `mlx_lm` accepts `position_ids` (`#846`)
- Tree verifier shareable between EAGLE and Medusa

### Layer D — multi-target asymmetric (planned)

- N>1 target ranks + 1 drafter requires draft broadcast on the target subgroup
- Currently gated behind `NotImplementedError` to keep telemetry honest
- Wire protocol extension TBD

### Layer E — concurrent asymmetric (planned)

- Wire protocol command frame already has `session_id` populated for V2
- Lifting `max_concurrent_tasks=1` on the asymmetric path requires full
  multi-session interleaving on the drafter rank (per-session executor or
  cooperative scheduler) — not just session tagging

### Layer F — beyond tokens (research)

- `confluent-reasoning` Pattern (cluster-level self-consistency)
- `placement-search` Pattern (speculative cluster scheduling using exo's
  pure `apply()`)
- Tensor-network compressed shuttles (long-term differentiator on
  heterogeneous hardware)

Each layer is independently shippable. Layer A and B together define the
"Loom v0.1" surface that could be extracted to a standalone framework.

---

## 12. Non-Goals

Loom is deliberately not trying to be:

- **A training framework.** Loom does not train models. It hosts trained
  spinners and shuttles.
- **A new model architecture.** Loom is architecture-agnostic. Any model
  exposing logits and KV-cache hooks can be a shuttle.
- **A general-purpose distributed-computing framework.** Loom's branching
  primitives are tuned for inference's specific shape. Don't use Loom to
  train a weather model.
- **Approximate by default.** Lossless is the default contract. Lossy
  Patterns must declare themselves and quantify their divergence.
- **Tied to exo.** exo is the reference host because its event-sourced
  pure-functional substrate is uniquely well-shaped for Confluent
  Inference. But Loom's APIs are designed to host on top of any inference
  server that exposes the required hooks.
- **A vendor of unsubstantiated speedup claims.** Every Pattern's
  profitability is measured per (topology × hardware) cell; "Loom makes X
  faster" is only ever a claim about a specific cell.

---

## 13. Open Questions

The ones that are actually open, today, with real stakes:

1. **Upstream `position_ids`.** When `mlx_lm` accepts per-position RoPE
   indices, every tree-attention Pattern (EAGLE, Medusa, Lookahead) becomes
   profitable on Apple Silicon. Track `ml-explore/mlx-lm#846` and `#250`.
   Loom's job until then: keep the integration seam warm, ship the offline
   artifacts, do not promise wins we cannot deliver.

2. **Multi-target asymmetric draft broadcast.** The V1 boundary
   (single target rank + single drafter) is real. The N+1 generalisation
   (N target ranks pipeline-parallel + 1 drafter rank) requires
   broadcasting drafts on the target subgroup so all ranks stay in
   lockstep through the verify forward. Open question: whether the
   broadcast belongs in the spec loop (Loom-level) or the runner
   (exo-level).

3. **Per-session interleaving on the asymmetric drafter rank.** Wire
   protocol v2 already tags sessions; the drafter rank's serve loop is
   still single-threaded. A per-session executor would lift
   `max_concurrent_tasks` from 1 on the asymmetric path. Open question:
   single thread-pool with `session_id` keying the cache lookup, or
   per-session worker thread? The session count is bounded by the target
   rank's `EXO_MAX_CONCURRENT_REQUESTS` (default 8), so neither approach
   has unbounded fan-out.

4. **Batched verify (independent of speculation).** One forward, B>1
   requests. This *does* benefit from `position_ids` and is a different
   problem from speculation. Worth cleanly separating in the doc and
   roadmap.

5. **Adaptive K.** The n-gram drafter's `adaptive_k` heuristic (cap
   proposal length to match length) is the cheapest form. A more
   principled version would tune K per-(topology × workload) using the
   live acceptance fraction. Open question: where does this live — in
   the Drafter? In the spec loop? In a new `KStrategy` Protocol?

6. **Cross-Pattern interference.** Can two Patterns running concurrently
   on the same warp share weft information? Rejected n-gram drafts might
   inform a future `confluent-reasoning` Pattern's branching decisions.
   Speculative; worth flagging.

7. **Quantum-inspired classical algorithms as Patterns.** Tensor-network
   compressed shuttles, amplitude-amplification-style search Patterns.
   Where is the line between "an interesting Pattern" and "a research
   paper looking for a framework"?

---

## Appendix A: Vocabulary → Code cheat sheet

| If you would normally say… | In Loom we say… | exo file |
| --- | --- | --- |
| The committed prompt + generated tokens | The **warp** | (KV cache + token sequence) |
| A speculative draft / proposal | A **weft** | `Drafter.stream` output / `propose` return |
| A single candidate token | A **thread** | element of `list[int]` |
| The drafter strategy | The **spinner** | `Drafter` Protocol |
| The verifier | The **shuttle** | target `Model` forward |
| The drafter IPC layer | The **shuttle wire** | `DrafterTransport` Protocol |
| Accepting a draft token | **Binding** a thread | accept loop |
| The full output | The **fabric** | `GenerationResponse` stream |
| A node in the cluster | A **spool** | `Worker` / MLX rank |
| The cluster | The **mill** | exo cluster / `mx.distributed.Group` |
| A spec-decoding algorithm | A **Pattern** | `DraftMode` |
| The collapse rule | The **interference operator** | greedy / rejection sampling |

---

## Appendix B: The asymmetric N+1 lifecycle (worked example)

The full lifecycle of one request on the V1 twin topology — `wc-smbp` as
target rank, `wc-smbpt` as drafter rank, RDMA over Thunderbolt-bridge.

1. **API request** arrives at `wc-smbp` (target, also playing API role).
2. **Pattern resolution**: `resolve_draft_mode` returns `"pipelined"` from
   `EXO_DRAFT_MODE=pipelined`.
3. **Session allocation**: `RemoteTransport.open_session()` returns a
   session-scoped `DrafterTransport` view with a fresh `session_id`.
4. **Prefill**: target announces `num_prompt_tokens` in an `OP_PREFILL`
   command frame, then sends the prompt token array. Drafter rank trims
   its KV cache to offset 0 *for that session*, runs prefill in 4096-token
   chunks, sends back an ack frame.
5. **Round 0 propose**: target sends `OP_FORWARD` with `[seed]`, requesting
   $K$ forwards. Drafter rank executes K drafter forwards on its session's
   draft cache, sends back a `(K+1,)` `uint32` drafts frame (last slot
   zero-padded since only K were requested).
6. **Cross-round speculation**: simultaneously, drafter rank receives a
   second `OP_FORWARD` with `[drafts[-1]]` requesting $K + 1$ forwards
   (the bonus-prediction + speculative round t+1). It dispatches in parallel
   with target's verify.
7. **Verify**: target rank runs its forward on `[y, *drafts]`, samples each
   position, walks the drafts and accepts on first mismatch (or accepts all
   K + bonus token).
8. **Bind**: target compares `bonus` against drafter's predicted bonus
   (first slot of the speculative output).
   - **Hit**: round t+1's drafts are already in hand; skip its propose call.
   - **Miss**: send `OP_TRIM_CACHE` with $K + 1$ to roll back the
     speculative work; round t+1 is a standard length-2-seed forward.
9. **Stream**: bound threads stream to the API as `GenerationResponse`
   chunks; `from_draft` flag marks which were drafter-accepted.
10. **End of request**: target sends `OP_END_SESSION` with the `session_id`.
    Drafter frees that session's KV cache. Drafter rank stays warm for
    other sessions / future requests.
11. **Shutdown**: at process teardown, target sends `OP_SHUTDOWN`. Drafter
    rank's serve loop drains and exits cleanly.

The full cycle, on twin M5 Max machines with q4 e2b drafter and bf16 26b
target, today: 87.91 t/s at K=2 (74% of single-host baseline). The limiting
factor is currently the LAN/RDMA layer maturity, not the spec loop.

---

*"Many threads, one fabric."*
