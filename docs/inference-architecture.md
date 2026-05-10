# exo Inference Architecture

| | |
| --- | --- |
| **Status** | Internal architecture reference (not a framework launch) |
| **Author(s)** | JJ et al. |
| **Created** | 2026-05-07 |
| **Last revised** | 2026-05-09 |
| **Working name for the design discipline** | **Branch-Bind Inference** |
| **Reference PR** | [team-wcv/exo#15](https://github.com/team-wcv/exo/pull/15), follow-ups on `feature/drafter-asymmetric-pipeline-v2` |

> This is the architecture reference for exo's distributed speculative-decoding
> subsystem. It is not a framework launch document. The work described is
> shipped, type-clean, test-covered, and currently used inside exo. Extraction
> as a standalone framework is **not** an active goal — see §11.
>
> The team also informally calls this subsystem **Loom**; the weaving
> vocabulary (warp, weft, shuttle, bind) appears in code comments and
> design discussions. Public APIs use the technical names.

---

## §0 Lead — Where The Architecture Wins Today

If you take one thing from this document, take this configuration:

> **n-gram drafting on TCP/Ring pipeline-parallel target, Qwen3.6-27B+ class
> model that does not fit single-host.** Measured ~1.95× scaling on prompts
> where suffix-match drafting hits.

This is the configuration where the architecture pays for itself **today**,
without waiting on any upstream change. Reasoning:

- The target's per-token cost is dominated by cross-shard collective latency
  (TCP between nodes, or Ring all-reduce inside one machine).
- Each accepted draft saves a full cross-shard round-trip, not just a verify
  forward — so the break-even acceptance fraction $\alpha^* = K/(K+1)$ is
  much easier to clear than on single-device.
- n-gram drafting costs zero MLX work, zero KV cache, zero warmup; if no
  match is found, throughput is identical to baseline. Worst-case downside is
  zero.
- The asymmetric N+1 architecture (drafter on its own RDMA-connected node)
  generalises the same idea further: the drafter forward fully overlaps the
  network round-trip the target rank would otherwise wait on.

Every other configuration in the matrix is currently a net loss on
single-device fast Apple Silicon. See §7 for the full numbers and §6 for the
topological reasoning.

---

## §TL;DR

What the work shipped over the past month established:

1. **A clean Drafter abstraction** (`Drafter` Protocol, `DraftMode` Literal)
   that lets `mlx_generate` dispatch on speculative strategy without
   branching at every site. Six modes registered (three concrete, two
   scaffolding stubs, one no-op).
2. **A transport-agnostic spec loop** (`PipelinedModelDrafter` over a
   `DrafterTransport` Protocol) that runs the same code path locally
   (in-process drafter, same Metal device) or remotely (drafter on a
   different MLX rank, communicating via `mx.distributed.send/recv` over
   RDMA or TCP).
3. **Asymmetric N+1 placement** (rank 0 = drafter-only, ranks 1..N =
   pipeline-parallel target) with a session-aware wire protocol, drafter
   runner state machine, and subgroup-split discipline so the target's
   pipeline-parallel collectives don't drag the drafter rank in.
4. **Round-robin admission** in `SequentialGenerator`, lifting the
   singular-slot ceiling that was costing slot-1 TTFT 5×.
5. **EAGLE / Lookahead scaffolding** + an offline EAGLE-3 head converter
   for Gemma-4. Runtime gated on upstream `mlx_lm#846` (per-position
   `position_ids`); seam ready for when it lands.
6. **Honest measurement discipline**: every Pattern's profile reported per
   (topology × hardware) cell, including the cells where it loses.

What this is *not*:
- A framework. Public extraction would require backend portability work
  that is not done.
- A net throughput win on single-device 4-bit Apple Silicon for fast small
  targets — it is currently a net loss in that cell.
- A speculative-decoding research contribution by itself. The cross-round
  pipelining pattern in `PipelinedModelDrafter` (§4.3, §15) is closer to a
  novel contribution and is worth a future write-up on its own.

---

## §1 Motivation

### 1.1 The bottleneck

Modern LLM inference is bottlenecked by **memory bandwidth**, not arithmetic
throughput. A single autoregressive forward pass loads tens to hundreds of
gigabytes of weights from HBM/unified memory into on-chip SRAM and produces
a single token. The matmul units sit at single-digit utilisation for the
duration of the load.

This means: a forward pass that produces 1 token costs almost the same as a
forward pass that verifies $K + 1$ tokens — *as long as the forward can
actually verify $K + 1$ positions in parallel*. The **break-even acceptance
fraction** for speculative decoding is therefore:

$$\alpha^* = \frac{K}{K + 1}$$

For $K = 4$, $\alpha^* = 80\%$. Most workloads do not clear this on
fast single-device decoding. This is the quantitative version of *"it's
not free"* that the literature usually elides.

### 1.2 The Apple Silicon ceiling

The CUDA literature's 3-6× speedups (EAGLE-3 paper, Medusa-2 paper,
Lookahead on H100) come from **tree verification**: dozens of candidate
continuations verified in a single batched forward, sharing prefix tokens
via tree attention, with each sibling getting a different RoPE position in
the same forward.

`mlx_lm` derives every position's RoPE id from `KVCache.offset`, which is
a single `int`. Two siblings at the same depth cannot get different RoPE
positions in the same forward. **Until upstream lands `position_ids`** (see
[`ml-explore/mlx-lm#846`](https://github.com/ml-explore/mlx-lm/issues/846),
[`#250`](https://github.com/ml-explore/mlx-lm/issues/250)), every speculative
pattern on MLX collapses to a *linear* verify, and the break-even math
above kicks in.

A community MLX EAGLE-3 prototype confirmed this at **1.05×** on
LLaMA-3.1-8B-4bit on an M3 Ultra (mlx-lm discussion #890) — inside the
noise of our own measurements.

### 1.3 The naive parallelism trap

The natural response — *"run N nodes in parallel"* — does not address the
per-request bottleneck. Each node still runs its own bandwidth-bound
forward passes; throughput goes up, per-request latency does not.

The same trap applies inside one node: tensor parallelism shards the
matmul but does not reduce the number of sequential decoding steps.

### 1.4 What this architecture covers

The shipped scope is three concrete layers:

| Layer | Branching unit | Collapse mechanism | Status |
| --- | --- | --- | --- |
| **Token** | Speculative draft tokens | Greedy accept / rejection sampling | Shipped (PR #15) |
| **Concurrency** | In-flight target requests | Round-robin admission | Shipped (PR #15, `456bbb32`) |
| **Topology** | Drafter rank vs target rank(s) | Asymmetric placement + remote shuttle wire | Shipped (PR #15 Layer B + follow-ups on `feature/drafter-asymmetric-pipeline-v2`) |

Other potential layers (verify-shape tree attention, reasoning-level
self-consistency, MoE routing, speculative cluster placement, KV-tree
promotion) are interesting research directions but are **not** part of the
shipped architecture. They live in §14 as future patterns, not in the
roadmap.

---

## §2 The Branch-Bind Inference Discipline

### 2.1 What it is

The team uses **Branch-Bind Inference** as a working name for the design
discipline this architecture implements. (Earlier drafts called it
*Confluent Inference*; renamed because *Confluent* has a $4B+ Kafka brand
that owns the SEO.)

> Branch-Bind Inference is a model of distributed computation in which:
>
> 1. A current state is *fanned* into many candidate next-states (branches),
> 2. Branches evolve in parallel under a cheap process,
> 3. Branches are *bound* through cross-branch verification, voting, or
>    scoring against a target distribution,
> 4. Bound branches collapse into a single committed extension of the state.

The principle that distinguishes it from naive parallelism is that the
parallel processes are **mutually constrained candidates whose disagreement
carries information**, not independent producers.

This is not a new conceptual primitive — speculative decoding (Leviathan
et al. 2023; Chen et al. 2023) is the canonical instance. The naming is
internal vocabulary that lets us discuss the discipline without retyping
*"speculative decoding plus its variants and the orthogonal concurrency
and topology layers"* every time.

### 2.2 Two contracts, separated honestly

A Branch-Bind operation commits to one of two correctness contracts:

- **Lossless**: the marginal distribution over outputs equals the
  distribution of serial execution under the target model. Token-level
  speculative decoding with greedy / rejection sampling is the canonical
  example. **What this architecture implements.**
- **Bounded-divergence**: the marginal distribution differs from serial
  execution by a quantified amount (e.g., self-consistency over $N$
  samples, best-of-$N$ with a verifier). Useful for inference-time-scaling
  regimes where lossy aggregation is the goal. **Not in the shipped scope.**

Mixing these by accident is the dominant failure mode of ad-hoc
speculative systems. Making the contract a typed property of every Pattern
prevents it.

### 2.3 Lossless ≠ profitable

A lossless Pattern is **structurally correct** but not necessarily
**runtime-profitable**. Profitability is a function of:

1. **The break-even acceptance fraction $\alpha^* = K / (K + 1)$** for the
   chosen $K$.
2. **The verify shape**: linear (current MLX) vs tree (CUDA, future MLX).
3. **The marginal cost of one extra verified position**: ~0 on
   bandwidth-bound forward passes with tree verify; ~$\frac{1}{K+1}$ of a
   forward on linear verify; *negative* (faster than baseline) on
   distributed pipeline-parallel targets where network latency is the
   dominant cost.

The discipline is to surface this honestly per (Pattern × topology ×
hardware) cell, not to claim universal speedups. See §7.

---

## §3 Internal Vocabulary

The team uses a consistent weaving vocabulary in code comments and design
discussions. It is **internal** — not load-bearing in public APIs, not a
brand. It appears here because reading the code is easier with the
mapping in front of you.

| Internal term | Technical concept | exo module |
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
| **Mill** | The full distributed cluster. | exo cluster / `mx.distributed.Group` |
| **Pattern** | A specific Branch-Bind policy. | `DraftMode` Literal (`drafter.py`) |
| **Loom** | The subsystem itself (informal codename). | This document. |

If you find the metaphor distracting, ignore it. The technical names are
sufficient. The metaphor is preserved because it is already in code
comments and removing it now would be churn for no benefit.

---

## §4 Architecture

The shipped seams, with real type signatures.

### 4.1 The Spinner Protocol (`Drafter`)

The cheap branch-generating tier is captured by a single Protocol that
lives at the **stream-factory level** — not at a finer-grained
`propose / accept` level. This was a deliberate decision: it lets the
well-tested upstream `mlx_lm.speculative_generate_step` keep owning the
model-drafter path, while in-house spinners (n-gram, EAGLE, lookahead)
plug in by yielding `GenerationResponse` the same way `stream_generate`
does.

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

Actual shipping signature in
`src/exo/worker/engines/mlx/generator/drafter.py`. Six concrete spinners
in tree (two as scaffolding stubs); see §5.

**Limitation**: this Protocol is intentionally MLX-specific. The
parameter list is dense with MLX types (`Model`, `mx.array`,
`KVCacheType`, `mlx_lm.GenerationResponse`). Porting to a non-MLX backend
would require redefining the Protocol. That is the right tradeoff for an
internal architecture; it is the wrong tradeoff for an extracted
framework. See §11 and §12.5.

### 4.2 The Shuttle Wire Protocol (`DrafterTransport`)

For pipelined and remote speculation, the spinner is decoupled from its
*location* via a second Protocol that handles the IPC primitives. This is
the seam that lets the same `PipelinedModelDrafter` spec loop run locally
(in-process drafter on the same Metal device) or remotely (drafter on a
different MLX rank, communicating via `mx.distributed.send/recv` over
RDMA or TCP).

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

Two concrete transports ship: `InProcessTransport` (drafter colocated,
same Metal command queue) and `RemoteTransport` (drafter on a different
rank, wire protocol over `mx.distributed`).

The Future is `concurrent.futures.Future`, not `asyncio.Future` — the
spec loop is a synchronous generator and threading asyncio through it
would have been invasive. The remote transport's IPC thread sets the
Future from outside the calling thread, which `concurrent.futures.Future`
natively supports.

This Protocol is the cleanest abstraction in the architecture. Its surface
is small (4 methods) and its types are mostly portable (`int`, `list[int]`,
`Future`). It is the most likely candidate for eventual extraction if a
framework story ever materialises.

### 4.3 Cross-round speculation (`PipelinedModelDrafter`) — *the conceptual contribution worth elevating*

This is the only piece of the architecture I would call genuinely novel
relative to the published spec-decoding literature. See §15 for the full
treatment; the short version:

While the target rank verifies round $t$'s drafts, the drafter
**speculatively starts round $t + 1$** by predicting the would-be bonus
token and continuing for $K$ more forwards. If the target's actual bonus
matches the drafter's predicted bonus, **round $t + 1$'s drafts are
already in hand** by the time round $t$'s verify finishes; if not, the
speculative work is rolled back via `trim_cache(K + 1)` and the standard
non-speculative path runs.

The cache accounting is the only intricate bit (full notation in
`pipelined_drafter.py`'s module docstring). The win is structural: the
overlap factor is bounded only by how parallel the two forwards can
actually run.

- On Apple Silicon's serialised Metal command queue, the in-process
  overlap is ~0.1-0.3.
- On distributed asymmetric placement where target verify includes a
  Thunderbolt-bridge RDMA round-trip, the overlap is ~1.0 and the gain
  unlocks.

**Same code path. Different transport. Different topology. Different
profitability.** This is what the discipline buys.

### 4.4 Round-robin admission (independent of speculation)

Speculative decoding is *one* knob; concurrent target requests is a
*separate* knob, and conflating them was the source of a 5× slot-1 TTFT
regression.

The previous `SequentialGenerator._active: tuple | None` admitted exactly
one task at a time. With `EXO_NO_BATCH=1` (which spec decoding requires
today), that meant the second concurrent request waited for the first to
fully complete — extrapolated 5300 ms TTFT on a typical 512-token decode.

Round-robin admission (`456bbb32`) replaces the singular slot with
`OrderedDict[TaskId, ...]` capped by `max_concurrent_tasks` (default 8).
Each tick admits up to the cap from the queue, then advances every active
task by one `next(gen)`. Measured on `wc-smbp` with
`gemma-4-26b-a4b-it-4bit`:

| Configuration | Pre-fix | Post-fix | Win |
| --- | --- | --- | --- |
| Slot 0 TTFT | ~1000 ms | 1029 ms | parity |
| Slot 1 TTFT | ~5300 ms (extrapolated) | 1015 ms | **5.2×** |
| Per-request gen tps | 120 t/s | 92.6 t/s | shared, by design |
| Aggregate gen tps | 120 t/s | 156.25 t/s | **1.3×** |

The asymmetric pipelined+remote path stays at `max_concurrent_tasks=1`
because `RemoteTransport`'s wire protocol is **session-aware but
per-session serial** — concurrent target requests would interleave
`OP_PREFILL` / `OP_FORWARD` frames on the same socket and corrupt the
drafter rank's per-session KV state. Lifting that cap requires extending
the wire protocol's command frame (already 9 `uint32` slots, with
`session_id` populated; more work needed for full multi-session
interleaving). Tracked.

The architectural lesson worth preserving: **tree attention needs
`position_ids`; concurrent target requests do not**. Conflating them
costs you 5× on slot-1 TTFT for a year.

### 4.5 Asymmetric N+1 placement (the distributed weave)

The hardest-won architectural piece: a topology where rank 0 is
**drafter-only** and ranks 1..N are pipeline-parallel target ranks.

Key design choices, all shipped in PR #15 Layer B:

- **No new transport stack.** `RemoteTransport` rides on the existing
  `mx.distributed.Group`, so the wire backend is automatically JACCL
  (RDMA over IB-verbs / Thunderbolt-bridge) for `MlxJacclInstance` or
  ring (TCP) for `MlxRingInstance`. The group is the transport.
  (Follow-up V3 transport on `feature/drafter-asymmetric-pipeline-v2`
  decouples from `mx.distributed` entirely via a socket transport for
  cases where the parent group is inappropriate.)
- **Subgroup split.** The target's pipeline-parallel collectives operate
  on `group.split(...)` so they don't drag the drafter rank into every
  all-reduce. The drafter rank uses send/recv against the *parent* group
  for cross-subgroup point-to-point.
- **DrafterRunner state machine.** Mirrors the target runner state
  machine: `ConnectToGroup → LoadModel → StartWarmup →
  drafter_serve_loop`.
- **Wire protocol v2 (session-aware).** Command frame is fixed-shape
  `uint32[9]`: `[op, num_inputs, num_forwards, input_0, input_1,
  trim_amount, session_id, _, _]`. Drafter rank routes each op to its
  per-session KV cache via `session_id`. `OP_PREFILL` allocates the
  session; `OP_END_SESSION` frees it.
- **V1 boundary**: single target rank + single drafter rank ("twins"
  topology), RDMA or TCP. Multi-target asymmetric (N>1 target ranks + 1
  drafter) shipped on the V2 branch with target-subgroup draft
  broadcast.

This is the topology where the discipline's value proposition is most
legible: the same `PipelinedModelDrafter` spec loop, the same
`DrafterTransport` interface, the same Pattern — but the speculative
work runs on a physically separate machine, fully overlapping the network
round-trip the target rank would otherwise wait on.

---

## §5 Patterns: the `DraftMode` Literal

Patterns are the concrete `DraftMode` Literal shipping in `drafter.py`.
Each is a complete spinner+collapse policy with measured profile data.

```python
DraftMode = Literal["model", "pipelined", "ngram", "eagle", "lookahead", "none"]
```

| Pattern | Spinner | Shuttle wire | Status | Profile (M5 Max, gemma-4-26b-a4b-it-4bit, 119 t/s baseline) |
| --- | --- | --- | --- | --- |
| `none` | None | n/a | Default for fast single-device | Baseline (119 t/s) |
| `ngram` | Suffix-match in running context | n/a (no second model) | Shipped | -14% to -23% (102 t/s @ K=2; degrades with K). **Note: net-positive on pipeline-parallel target topologies, see §0 and §6.2.** |
| `model` | Smaller distilled drafter via `mlx_lm.stream_generate(draft_model=...)` | n/a (in-process) | Shipped | -25% to -45% (varies by workload) |
| `pipelined` (in-process) | `PipelinedModelDrafter` + `InProcessTransport` | Same Metal device | Shipped | -62% (Metal command queue serialises) |
| `pipelined` (remote, q4) | Same spinner + `RemoteTransport` over RDMA | `mx.distributed.send/recv` (or socket transport on V3) | Shipped | -26% to -54% (K=2 best; e2b @ q4 doubles throughput vs bf16) |
| `eagle` | EAGLE auxiliary head (depth-only or tree) | EAGLE/Medusa shareable tree verifier | **Scaffolding only**; converter ships | Blocked on `mlx_lm#846` (community prototype: 1.05×) |
| `lookahead` | Jacobi iteration on target's own forward | n/a (no second model) | **Scaffolding only** | Blocked on `mlx_lm#846` (same ceiling as `ngram`) |

### 5.1 What ships today vs what is gated

**Shipped, type-clean, test-covered, used in production paths:**
- `none`, `ngram`, `model`, `pipelined` (both in-process and remote)
- Asymmetric N+1 placement with RDMA + TCP, V1 (twins) and V2
  (multi-target) boundaries
- V3 socket transport (decouples wire from `mx.distributed`)
- `convert_eagle3_to_mlx.py` (offline tool) — downloads the RedHat
  EAGLE3 head for `gemma-4-26B-A4B-it`, applies layer taps at
  `(2, 15, 27)` following EAGLE's `(2, N//2, N-3)` heuristic for
  Gemma-4-26b's 30 layers, writes MLX safetensors. Output is durable for
  whenever the runtime lands.

**Scaffolding only, runtime gated upstream:**
- `eagle`: `EagleDrafter` class is in tree with full integration seam,
  raises `NotImplementedError` on `stream`. Resumes when `mlx_lm`
  accepts `position_ids`.
- `lookahead`: `LookaheadDrafter` class same shape. Same gate.

The deliberate decision: ship the scaffolding and the offline converter
**now** so when the upstream blocker lifts, only the runtime fork has to
land — the integration seam, the factory dispatch, the model-card field
shape, and the head artifact are all already there.

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

## §6 Topologies

Profitability is dominated by the relationship between spinner cost and
shuttle cost — which is dominated by topology. Three topologies matter
today.

### 6.1 Single-device

One Metal device, one process, target + (optional) drafter both
resident.

- **In-process drafter** (`InProcessTransport`): drafter and target
  compete for the same Metal command queue. MLX serialises GPU work per
  device, so the in-process pipelining overlap factor is 0.1-0.3.
- **Profitability**: every `DraftMode` except `none` is currently a net
  loss on this topology for fast small targets (4-bit Apple Silicon at
  >100 t/s baseline). The break-even acceptance fraction $K/(K+1)$ is
  not met by the workloads measured.
- **Right default**: `none`.

### 6.2 Pipeline-parallel target — *the lead-win cell*

Target sharded across multiple ranks (either pipeline-parallel via
`MlxRingInstance` over TCP or `MlxJacclInstance` over RDMA). No
dedicated drafter rank; if drafting is enabled, the drafter is
in-process on rank 0.

- **Profitability**: target verify includes cross-shard collective
  latency, which makes per-token cost much higher than single-device.
  The break-even acceptance fraction is much easier to clear because
  each accepted draft saves a full cross-shard round-trip.
- **Measured**: n-gram on TCP/Ring shows **~1.95× scaling** on prompts
  where suffix-match drafting hits, for models that don't fit
  single-host (Qwen3.6-27B+).
- **Right default**: opt-in `ngram` for Qwen3.6-27B+ class models;
  `none` when the target fits single-host.

This is the configuration that makes the whole architecture pay for
itself today. Promote it in deployment guides.

### 6.3 Asymmetric N+1

Rank 0 is drafter-only; ranks 1..N are pipeline-parallel target. The
spinner runs on physically separate hardware from the shuttle.

- **Profitability** is bounded by network latency overlap. On
  Thunderbolt-bridge RDMA between twin Macs, the speculative drafter
  forward fully overlaps the per-token network round-trip, and the
  pipelined-remote pattern is the **only configuration where the gain
  is structurally inevitable rather than empirically marginal**.
- **Measured today** (M5 Max twin pair, `gemma-4-26b-a4b-it-4bit`, q4
  drafter): K=2 reaches 74% of single-host baseline (vs 45% with bf16
  drafter). Still net-negative as standalone, but the limiting factor
  is the LAN/RDMA layer maturity, not the architecture.
- **The architectural payoff is durable** even if today's twin numbers
  are limited: the same code path runs over Thunderbolt-bridge RDMA, IB
  RDMA, and TCP, with no spec-loop change.

---

## §7 The Honest Hardware Matrix

Numbers measured on `wc-smbp` (M5 Max, single-device, 4-bit) unless
otherwise noted.

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

### 7.2 Concurrency sweep (independent of speculation)

Round-robin `SequentialGenerator` measured on `wc-smbp`, same target,
512-token decode:

| Metric | Pre-fix | Post-fix | Win |
| --- | --- | --- | --- |
| Slot 0 TTFT | ~1000 ms | 1029 ms | parity |
| Slot 1 TTFT | ~5300 ms (extrapolated) | 1015 ms | **5.2×** |
| Aggregate gen tps | 120 t/s | 156.25 t/s | **1.3×** |

### 7.3 What flipped, what didn't

- **Quantizing the asymmetric drafter** (bf16 e2b → q4 e2b): doubled
  K=5 throughput (37 → 55 t/s); at K=2 reaches 74% of single-host
  baseline (vs 45% before). Wire payload 3.5× lighter; drafter forward
  ~4× cheaper.
- **Concurrency (round-robin)**: 5.2× slot-1 TTFT win, 1.3× aggregate
  throughput. Independent of every drafter strategy.
- **n-gram on TCP/Ring** for Qwen3.6-27B+ class models that don't fit
  single-host: ~1.95× scaling. The lead win.
- **Tree attention**: not measurable; gated upstream.

### 7.4 What this means

The honest read: **on this hardware class, for this model size, the
durable win shipped this month was concurrency on single-device and
n-gram on pipeline-parallel topology**. Single-device speculative
patterns remain a net loss until tree attention lands; the asymmetric
N+1 architecture is a durable artifact whose runtime profitability is
gated on LAN/RDMA layer maturity for distributed deployments.

The discipline's contract is to surface this matrix per Pattern, per
topology, per hardware tier — not to obscure it behind a generic
"speedup" claim.

### 7.5 Coverage limitations of this matrix

This matrix is **one model on one hardware tier**. See §12.5.

---

## §8 Mapping to exo

This architecture is implemented inside exo and depends on exo's
infrastructure. The shipped exo modules are:

| Concept | exo module |
| --- | --- |
| Spinner Protocol | `src/exo/worker/engines/mlx/generator/drafter.py::Drafter` |
| `DraftMode` Literal | `src/exo/worker/engines/mlx/generator/drafter.py::DraftMode` |
| Shuttle wire Protocol | `src/exo/worker/engines/mlx/generator/drafter_transport.py::DrafterTransport` |
| In-process shuttle wire | `drafter_transport.py::InProcessTransport` |
| Remote shuttle wire (mx.distributed) | `src/exo/worker/engines/mlx/generator/remote_drafter.py::RemoteTransport` |
| Remote shuttle wire (socket, V3) | follow-up commit on `feature/drafter-asymmetric-pipeline-v2` |
| Cross-round speculation pattern | `src/exo/worker/engines/mlx/generator/pipelined_drafter.py::PipelinedModelDrafter` |
| Round-robin admission | `src/exo/worker/runner/llm_inference/batch_generator.py::SequentialGenerator` |
| Drafter rank state machine | `src/exo/worker/runner/drafter_runner.py::DrafterRunner` |
| Drafter placement | `instance.drafter_placement` field |
| Pub/sub topics | `src/exo/routing/topics.py` |
| Event-sourced state | `src/exo/shared/types/state.py` + `src/exo/shared/apply.py` |

### 8.1 Spool roles

A spool can hold one or more roles per Pattern:

- **Spinner-spool**: runs a `Drafter`. Cheap; can be co-located with
  the API node or live on a separate machine entirely (asymmetric N+1).
- **Shuttle-spool**: holds the target model. Can be one rank or sharded
  across N ranks via `MlxJacclInstance` / `MlxRingInstance`.
- **Master-spool**: by exo's bully-election protocol. Coordinates the
  Pattern, holds the canonical state, broadcasts events via
  `GLOBAL_EVENTS`.

Single physical machine can play any combination. The default
development experience is all three on one process; the production
frontier is the asymmetric N+1 twin topology.

### 8.2 Failure semantics

Inherits exo's bully-election master replacement. The committed token
sequence is content-addressed; a new master can resume mid-flight by
replaying the event log. Speculative weft trees in flight at the moment
of failure are discarded — no correctness consequence, only a latency
cost.

The remote shuttle wire's session model means a target-rank crash drops
only its own session's KV cache on the drafter rank; sibling sessions
continue. A drafter-rank crash drops every session and the target ranks
degrade to `none` with a warning (already wired in `resolve_draft_mode`).

---

## §9 API Surface

### 9.1 Building a drafter

Single entry-point factory:

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
  `RemoteTransport` (or accepts a long-lived one passed in), wraps it
  in `PipelinedModelDrafter`.
- For `mode == "pipelined"` and no remote group: honours
  `EXO_DRAFTER_TRANSPORT`, defaulting to in-process.
- For `mode == "model"`: standard mlx_lm spec loop.
- For `mode == "ngram"`: in-house spec loop with adaptive K cap.
- For `mode in ("eagle", "lookahead")`: scaffolding stub that raises
  `NotImplementedError` on `stream` (see §5.1).

### 9.2 K clamping for the wire-protocol budget

The asymmetric path allocates a long-lived `RemoteTransport` at builder
time with a fixed `num_draft_tokens` budget. A per-request override
above the budget is **silently clamped** rather than raising — a
previous regression (an aborted K=8 sweep) demonstrated that raising
mid-flight kills the runner subprocess and wedges the peer rank:

```python
def clamp_num_draft_tokens_to_transport(
    requested_num_draft_tokens: int,
    transport: DrafterTransport,
) -> tuple[int, bool]:
    """Clamp K against the transport's wire-protocol budget.
    Returns (clamped_K, was_clamped) so callers can emit a structured warning."""
```

Small but load-bearing defensive behaviour; the alternative cost a real
outage.

### 9.3 Streaming

Every Drafter produces a `Generator[GenerationResponse, None, None]`
matching `mlx_lm.stream_generate`. This means `mlx_generate`'s call site
is uniform across every Pattern; the only thing that changes is which
`Drafter` was constructed.

---

## §10 Comparison to Prior Work

| System | What it is | How this architecture relates |
| --- | --- | --- |
| **vLLM, SGLang, TGI, TensorRT-LLM** | Production CUDA inference servers with built-in spec decoding | Different hardware tier (CUDA tree-attention available). Vocabulary applies; concrete Patterns differ. We deliberately do not compete on CUDA. |
| **EAGLE-1/2/3, Medusa, Lookahead** | Specific spec-decoding algorithms | Each becomes a Pattern in the `DraftMode` registry. EAGLE-3 scaffolding + offline converter for Gemma-4 already shipped. |
| **mlx_lm `speculative_generate_step`** | The upstream MLX spec loop | Wrapped by `ModelDrafter`. Not reinvented. |
| **vLLM's `--speculative-model='[ngram]'`, SGLang's n-gram drafter** | Production n-gram spec implementations | Equivalent to `NgramDrafter`. Adaptive K + match-strength biasing are local additions. |
| **mlx_lm community EAGLE-3 prototype (gist via discussion #890)** | Reference MLX EAGLE port (Llama-3.1 only, no Gemma-4 adapter, no tree verify, 1.05× ceiling) | `EagleDrafter` shell documents this as the integration starting point; offline converter ships, runtime forked when `position_ids` lands. |
| **DeepSeek MTP (Multi-Token Prediction)** | Architecture-baked multi-token prediction | Different shape — bakes drafting into the model rather than splitting drafter/target. Could in principle become a `DraftMode = "mtp"` if a Gemma-4-class MTP model lands. Not a near-term plan. |
| **Sequoia / SpecInfer (tree-based)** | Optimal-tree-shape speculative decoding | Same upstream gate (tree attention requires `position_ids` on MLX). |
| **Self-consistency (Wang et al.)** | Inference-time technique | Not implemented. Listed in §14 as a future pattern, not roadmap. |
| **Tree of Thoughts, Graph of Thoughts** | Reasoning-time search | Not implemented. §14. |
| **Ray, Dask** | General distributed-computing frameworks | Different abstraction level. We are opinionated about *what* the parallelism is for. |
| **JAX `pmap` / `vmap`** | Functional batched parallelism | Borrows the same purity discipline at the cluster level. |

---

## §11 Roadmap

These are **internal milestones, not framework releases**. Extraction as
a standalone framework is gated on Layer C (post-`position_ids` runtime)
*and* a non-MLX backend adapter that does not exist today. There is no
v1.0 framework launch on the calendar.

### Layer A — single-device foundation (✅ shipped, PR #15)

- `Drafter` Protocol + `DraftMode` Literal
- `NoSpecDrafter`, `ModelDrafter`, `NgramDrafter`
- `EXO_DRAFT_MODE` env + `TaskParams.draft_mode` per-request override
- `GenerationStats.draft_mode` telemetry
- 29 unit tests covering parse / resolve / propose / dispatch
- Bench harness (`bench/drafter_bench.py`) producing the matrix in §7

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

### Layer B+ — concurrency (✅ shipped, `456bbb32`)

- Round-robin `SequentialGenerator` (`OrderedDict[TaskId, ...]` capped by
  `EXO_MAX_CONCURRENT_REQUESTS`, default 8)
- Asymmetric pipelined+remote stays at cap=1 (wire-protocol guard)
- Per-task error isolation
- 5.2× slot-1 TTFT win measured

### Layer B++ — V2/V3 transport hardening (✅ shipped, `feature/drafter-asymmetric-pipeline-v2`)

- Multi-target asymmetric (N>1 target ranks + 1 drafter) with target-
  subgroup draft broadcast
- V3 socket transport (decouples wire from `mx.distributed` for cases
  where the parent group is inappropriate)
- Reverse-K-drift fix
- `find_ip_prioritised` arg-order fix in drafter placement

### Layer C — EAGLE / Lookahead runtime (🚧 blocked upstream)

- `EagleDrafter` / `LookaheadDrafter` scaffolding ships in PR #15
- `scripts/convert_eagle3_to_mlx.py` ships, durable
- Recommended `ModelCard.eagle_head_repo` field + bootstrap-side
  download
- Runtime fork of `eagle_generate.py` from the gist linked in mlx-lm
  discussion #890 once `mlx_lm` accepts `position_ids` (`#846`)
- Tree verifier shareable between EAGLE and Medusa

### Layer D — concurrent asymmetric (planned, scoped)

- Wire protocol command frame already has `session_id` populated
- Lifting `max_concurrent_tasks=1` on the asymmetric path requires full
  multi-session interleaving on the drafter rank (per-session executor
  or cooperative scheduler) — not just session tagging

### Layer E — bench-bar expansion (planned, near-term)

- Multi-model coverage (≥3 sizes, ≥2 quantizations)
- Multi-hardware coverage (Studio M3 Ultra in addition to M5 Max)
- Reproduction recipes published alongside results
- Required to credibly support the "honest matrix per cell" discipline
  beyond the current single-cell coverage

### Future patterns

Reasoning, routing, state, cache layers are deliberately **not** here —
they live in §14 as research directions, not roadmap.

---

## §12 Non-Goals

This architecture is deliberately not trying to be:

- **A training framework.** Does not train models. Hosts trained
  spinners and shuttles.
- **A new model architecture.** Architecture-agnostic *within MLX*. Any
  MLX model exposing logits and KV-cache hooks can be a shuttle.
- **A general-purpose distributed-computing framework.** Branching
  primitives are tuned for inference's specific shape. Don't use this
  to train a weather model.
- **Approximate by default.** Lossless is the default contract. Lossy
  Patterns must declare themselves and quantify their divergence.
- **A vendor of unsubstantiated speedup claims.** Every Pattern's
  profitability is measured per (topology × hardware) cell;
  *"speculation makes X faster"* is only ever a claim about a specific
  cell.
- **An extracted standalone framework.** See §11.

---

## §12.5 Limitations (read this before promoting to anyone else)

These are the things you should mention out loud whenever you describe
this work to someone outside the team:

1. **Single hardware tier benchmarked.** Every number in §7 is from
   `wc-smbp` (M5 Max). Studio M3 Ultra, M2 Ultra, M1 Max, MacBook Pro M
   variants are unmeasured. Layer E in §11 plans to fix this; it is
   not fixed yet.
2. **Single model benchmarked.** Every number in §7 is from
   `gemma-4-26b-a4b-it-4bit`. Qwen, Llama, smaller / larger Gemma, and
   non-4-bit quantizations are unmeasured. The cited 1.95× pipeline-
   parallel scaling on Qwen3.6-27B+ is from anecdotal earlier
   measurement and needs to be re-run with the current code path before
   it is quoted publicly.
3. **MLX-only abstractions.** The `Drafter` Protocol's parameter list
   leaks MLX types (`Model`, `mx.array`, `KVCacheType`,
   `mlx_lm.GenerationResponse`). Porting to `transformers`, `vLLM`,
   `SGLang`, or `llama.cpp` would require redefining the Protocol. The
   `DrafterTransport` Protocol is more portable in shape.
4. **Asymmetric topology hardware requirements are real.**
   Thunderbolt-bridge RDMA between Macs needs cabling and OS-level
   configuration that most users do not have. TCP/Ring works
   everywhere but pays a higher per-token network cost. The asymmetric
   N+1 cell is *structurally* the most exciting one but is the least
   reachable for casual deployments.
5. **Tree attention is gated upstream and we do not control the
   timeline.** EAGLE / Medusa / Lookahead's published 3-6× wins are
   unreachable on MLX until `mlx_lm#846` lands. We have shipped the
   scaffolding and the offline converter; the runtime fork waits.
6. **Single-device speculative patterns are currently net-negative on
   fast small Apple Silicon targets.** This is the most-deployed cell
   and it currently loses 14-69% of throughput. The `none` default
   prevents this from hurting users in production, but it is the
   reason we should not pitch this work as *"speculative decoding in
   exo"* without immediate qualification.
7. **The "1.95× on TCP/Ring" lead-win number** (§0, §6.2, §7.3) is
   our best deployment story but its bench provenance predates the
   current code path. Re-run on the current spec loop with the current
   admission scheduler before quoting it externally.
8. **Cross-round speculation's overlap factor** is hardware-dependent
   and not measured per-platform. The 0.1-0.3 in-process number and
   the ~1.0 RDMA number are estimates / implications from §7's data,
   not directly measured overlap factors.

If a recommendation in this doc would be embarrassing to defend in
public, this section is where the embarrassment lives. Read it before
showing this doc to anyone outside the team.

---

## §13 Open Questions

The ones that are actually open today, with real stakes:

1. **Upstream `position_ids`.** When `mlx_lm` accepts per-position
   RoPE indices, every tree-attention Pattern (EAGLE, Medusa,
   Lookahead) becomes profitable on Apple Silicon. Track
   `ml-explore/mlx-lm#846` and `#250`.

2. **Per-session interleaving on the asymmetric drafter rank.** Wire
   protocol v2 already tags sessions; the drafter rank's serve loop is
   still single-threaded. A per-session executor would lift
   `max_concurrent_tasks` from 1 on the asymmetric path. Open
   question: single thread-pool with `session_id` keying the cache
   lookup, or per-session worker thread? Session count is bounded by
   the target rank's `EXO_MAX_CONCURRENT_REQUESTS` (default 8), so
   neither approach has unbounded fan-out.

3. **Batched verify (independent of speculation).** One forward, B>1
   requests. This *does* benefit from `position_ids` and is a
   different problem from speculation. Worth cleanly separating in
   the doc and roadmap.

4. **Adaptive K.** The n-gram drafter's `adaptive_k` heuristic (cap
   proposal length to match length) is the cheapest form. A more
   principled version would tune K per-(topology × workload) using
   the live acceptance fraction. Where does it live — in the Drafter,
   in the spec loop, or in a new `KStrategy` Protocol?

5. **Multi-model bench coverage.** Layer E in §11. Without this the
   honest-matrix discipline is aspirational.

6. **Whether to publish a standalone write-up of the cross-round
   pipelining pattern (§15).** It is the closest to a novel
   contribution we have. The cost is some hours of prose; the upside
   is establishing prior art.

---

## §14 Future Patterns (research, not roadmap)

These are interesting research directions, not committed work.
Implementing any of them would be its own project.

| Pattern | Branching unit | Collapse mechanism | Why it would be interesting |
| --- | --- | --- | --- |
| **Verify-shape tree attention** | Tree of candidates per round | Tree attention | Unblocks EAGLE / Medusa / Lookahead's published wins on MLX. Gated on `mlx_lm#846`. |
| **Reasoning self-consistency** | Chain-of-thought paths | Verifier vote / logprob aggregation | Cluster-level Branch-Bind: same paradigm, different scale. Needs a verifier model and an aggregator. |
| **MoE routing as Branch-Bind** | Expert candidates | Gating amplitudes | Reframes existing MoE as the same pattern; not obviously useful as code, possibly useful as conceptual unification. |
| **Speculative cluster placement** | Candidate placement decisions | Score-based commit | Uses exo's pure `apply()` to explore placement branches before committing. Architecturally clean; profitability uncertain. |
| **Tree-attention KV branches** | KV-cache subtrees per branch | Promotion of accepted prefix | Caching layer for tree-verify Patterns. Gated on the same upstream blocker. |

These are listed because the conceptual unification is interesting and
the team has discussed each. They are **not** committed work and the
matrix discipline does not apply to them yet.

---

## §15 Cross-Round Speculation: The Conceptual Contribution

The closest thing this work has to a novel contribution worth a future
write-up. Promoted to its own section because it is the part of the
architecture that is doing real work the literature does not already
cover end-to-end.

### 15.1 The standard speculative-decoding round

Standard speculative decoding is round-by-round:

1. Draft $K$ candidate tokens.
2. Verify them in one target forward.
3. Accept the longest matching prefix; emit + bonus token.
4. Trim drafter cache to match acceptance.
5. Begin next round.

Steps 1 and 2 are sequential. The drafter is idle while the target
verifies; the target is idle while the drafter drafts.

### 15.2 Cross-round speculation

`PipelinedModelDrafter` issues, **simultaneously with the target's
verify forward**, a second drafter forward of $K + 1$ positions:

- Position 0: drafter's prediction of the *would-be bonus token* for
  this round.
- Positions 1..K: drafter's continuation as if that bonus prediction is
  correct — i.e., round $t + 1$'s drafts.

When the target's verify completes:

- **Hit case** (target's actual bonus matches drafter's predicted
  bonus): round $t + 1$'s drafts are already in hand. Skip the propose
  call for round $t + 1$ entirely. The drafter cache is already at the
  correct offset.
- **Miss case** (bonus mismatch): roll back the speculative work via
  `trim_cache(K + 1)`. Round $t + 1$ proceeds standardly with a length-
  2 seed forward.

The accounting (offsets, partial vs full accept × spec hit vs miss × K
truncation interactions) is non-trivial but tractable. It is fully
spelled out in `pipelined_drafter.py`'s module docstring.

### 15.3 Why this is interesting

The win is **structural**: the speculative drafter forward fully
overlaps the target verify forward, bounded only by how parallel the
two forwards can actually run on the chosen transport.

| Topology | Overlap factor (estimated) | Implication |
| --- | --- | --- |
| Single-device, in-process (Apple Silicon) | 0.1-0.3 | Limited by Metal command queue serialisation |
| Single-device, in-process (CUDA) | 0.5-0.8 | Concurrent kernel launches available |
| Distributed asymmetric, RDMA | ~1.0 | Drafter forward fully overlaps network round-trip |
| Distributed asymmetric, TCP | ~1.0 | Same; even higher absolute win because TCP latency is larger |

The conceptual point: **the same code path, expressed against the same
`DrafterTransport` interface, has wildly different profitability
profiles depending on the transport's natural concurrency**. This is
what motivates the transport seam to exist at all.

### 15.4 Relation to prior work

The literature on speculative decoding focuses on:
- Drafter quality (EAGLE, Medusa heads, MTP)
- Verify shape (tree vs linear; Sequoia, SpecInfer)
- Acceptance rules (lossless rejection sampling, lossy thresholds)

Cross-round speculation is **orthogonal to all three**. It is a
scheduling pattern: how to keep the drafter busy during target verify.
The closest published work is the *async* / *pipelined* speculative
decoding sketches in vLLM and SGLang internal discussion, but a clean
formulation against a transport-abstracted interface, with the
hit/miss/cache-accounting fully worked out, does not appear in any
paper or framework documentation we have found.

This is the part worth a standalone write-up — short blog post first,
possibly a short paper later, after Layer E expands the bench coverage
enough to substantiate the claim.

### 15.5 Status

- Implemented: `PipelinedModelDrafter` in
  `src/exo/worker/engines/mlx/generator/pipelined_drafter.py`
- Tested: cache-accounting tests in
  `src/exo/worker/tests/unittests/test_mlx/test_pipelined_drafter.py`
- Documented: full docstring in the module
- Measured: in-context within §7 (the cell where it currently loses on
  single-device and gates on transport latency on distributed)

---

## §16 Internal Adoption Notes

How to use this architecture inside exo today:

### 16.1 Default configuration

Do nothing. `none` is the default; speculative decoding is opt-in. Users
get standard `mlx_lm.stream_generate` performance.

### 16.2 When to opt into `ngram`

Set `EXO_DRAFT_MODE=ngram` at process start, or pass `draft_mode="ngram"`
on individual requests, when:

- Running a model that does not fit single-host (Qwen3.6-27B+,
  Llama-3.3-70B class) on a pipeline-parallel topology, **or**
- Workload is dominated by RAG / structured output / code completion
  that the model echoes back from prompt content.

Do **not** opt into `ngram` for fast single-device inference on
creative or novel-content workloads — it loses 14-23% throughput.

### 16.3 When to set up asymmetric N+1

The twin-Mac topology (one drafter, one target) is currently a research
artifact, not a production recommendation. Set it up if:

- You have two RDMA-capable machines connected via Thunderbolt-bridge,
  IB, or low-latency LAN, **and**
- You want to validate the architecture end-to-end, **and**
- You are willing to debug the LAN/RDMA layer when the twin bench
  doesn't reach predicted numbers.

Documented twin-machine recipe in `remote_drafter.py`'s module
docstring.

### 16.4 When to wait

- For tree-attention wins (EAGLE / Medusa runtime): wait for
  `mlx_lm#846`.
- For single-device speculation that actually wins: wait for
  `mlx_lm#846` (tree attention is the path) or for a 30B+ model that
  doesn't fit single-host (pipeline-parallel + n-gram is the path
  today).

### 16.5 Telemetry

`GenerationStats.draft_mode` records the resolved mode per request. Use
it when comparing benchmarks; the mode actually run is not always the
mode requested (factory may downgrade silently if a drafter model isn't
loaded).

---

## Appendix A: Internal Vocabulary Cheat Sheet

| If you would normally say… | Internal term | exo file |
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
| The subsystem itself | **Loom** (informal) | this document |

---

## Appendix B: The asymmetric N+1 lifecycle (worked example)

Full lifecycle of one request on the V1 twin topology — `wc-smbp` as
target rank, `wc-smbpt` as drafter rank, RDMA over Thunderbolt-bridge.

1. **API request** arrives at `wc-smbp` (target, also playing API role).
2. **Pattern resolution**: `resolve_draft_mode` returns `"pipelined"`
   from `EXO_DRAFT_MODE=pipelined`.
3. **Session allocation**: `RemoteTransport.open_session()` returns a
   session-scoped `DrafterTransport` view with a fresh `session_id`.
4. **Prefill**: target announces `num_prompt_tokens` in an `OP_PREFILL`
   command frame, then sends the prompt token array. Drafter rank
   trims its KV cache to offset 0 *for that session*, runs prefill in
   4096-token chunks, sends back an ack frame.
5. **Round 0 propose**: target sends `OP_FORWARD` with `[seed]`,
   requesting $K$ forwards. Drafter rank executes K drafter forwards
   on its session's draft cache, sends back a `(K+1,)` `uint32` drafts
   frame (last slot zero-padded since only K were requested).
6. **Cross-round speculation**: simultaneously, drafter rank receives
   a second `OP_FORWARD` with `[drafts[-1]]` requesting $K + 1$
   forwards (the bonus-prediction + speculative round t+1). It
   dispatches in parallel with target's verify.
7. **Verify**: target rank runs its forward on `[y, *drafts]`, samples
   each position, walks the drafts and accepts on first mismatch (or
   accepts all K + bonus token).
8. **Bind**: target compares `bonus` against drafter's predicted bonus
   (first slot of the speculative output).
   - **Hit**: round t+1's drafts are already in hand; skip its
     propose call.
   - **Miss**: send `OP_TRIM_CACHE` with $K + 1$ to roll back the
     speculative work; round t+1 is a standard length-2-seed forward.
9. **Stream**: bound threads stream to the API as `GenerationResponse`
   chunks; `from_draft` flag marks which were drafter-accepted.
10. **End of request**: target sends `OP_END_SESSION` with the
    `session_id`. Drafter frees that session's KV cache. Drafter rank
    stays warm for other sessions / future requests.
11. **Shutdown**: at process teardown, target sends `OP_SHUTDOWN`.
    Drafter rank's serve loop drains and exits cleanly.

The full cycle, on twin M5 Max machines with q4 e2b drafter and bf16
26b target, today: 87.91 t/s at K=2 (74% of single-host baseline). The
limiting factor is currently the LAN/RDMA layer maturity, not the spec
loop.

---

## Appendix C: Document History

- 2026-05-07: initial draft as `docs/loom-design.md` framed as a
  framework launch (*"Loom: A Framework for Confluent Inference"*).
- 2026-05-09: critical review; renamed to
  `docs/inference-architecture.md`; framework framing dropped;
  paradigm renamed *Confluent Inference → Branch-Bind Inference*; lead-
  win cell promoted to §0; layer table cut to shipped (Token /
  Concurrency / Topology); future patterns moved to §14; Limitations
  §12.5 added; §15 cross-round speculation deep dive added.

---

*Internal architecture reference. Read the limitations section before
quoting any number externally.*
