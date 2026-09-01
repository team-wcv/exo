---
branch: fix/6a95c4b0-resilient-downloads-vq
created: 2026-08-31
owner: codex-agent
status: active
scope: "Stack peer-safe stale-download recovery and VQ codebook replication on the hardware-validated exo upstream merge"
orchestraitor:
  ticket: 6a95c4b0e6b6efd51fc8569d
  task_url: orchestraitor://task/6a95c4b0e6b6efd51fc8569d
pr:
  url: https://github.com/team-wcv/exo/pull/43
  state: open
cleanup:
  merged_into: team-wcv/chore/6a959fcc-sync-exo-upstream
  archived_after: null
  successor_branch: null
source_branches:
  - chore/6a959fcc-sync-exo-upstream (stack base at 7763d51e, PR #42)
  - exo-explore/exo PR #2278 (adapted HTTP 416 recovery)
  - exo-explore/exo PR #2268 (codebook replication guard only)
---

- Why this branch exists: add two bounded upstream improvements without changing the exact PR #42 build currently running on smbp/smbpt.
- Changed paths: peer-safe partial-download reset and bounded HTTP 416 recovery; vector-quantized codebook replication policy; focused recovery and sharding-policy tests.
- Validation run: all 153 download/policy tests pass; the sandbox-safe suite reports 594 passed, 5 skipped, and 2 Metal-only cases deselected, plus 8 stacked-base tests pass; BasedPyright reports 0 errors; whole-repo Ruff check and format check pass; Cargo workspace tests and fmt check pass. Nix is unavailable on this host.
- Known follow-ups: Deliberaitor plan creation is unavailable because its workspace-local route rejects this bastion-scoped Codex session; the PR will use the required outage note. VQ model cards from upstream #2268 are intentionally out of scope.
- Why this branch exists: preserve the known-good two-node exo deployment, then bring the fork current with official upstream without losing team-wcv custom behavior.
- Changed paths: upstream Zenoh/exo_rs transition; Python package and dashboard locks; persistent scoped identity; event routing and peer-aware election recovery; restart-safe custom model cards; model-card/backend compatibility; peer-download, custom drafting, placement, runner supervision, API serialization, dashboard shard-assignment compatibility, benchmark packaging, tests, and formatting.
- Validation run: matching smbp/smbpt snapshots archived with Git bundle at f033e597; official and fork tips are ancestors of the merge head; uv lock/sync succeeded; Ruff and Ruff format pass; BasedPyright reports 0 errors; Cargo check/test/fmt pass; dashboard npm ci/build, Svelte check, and Prettier pass. The P1 review round passes 48 focused tests (1 skipped), 601 sandbox-safe tests (5 skipped, 2 Metal-only deselected), and 9 stacked-base tests; an isolated exo-bench environment imports exo-tools successfully. The latest P1 follow-up also passes the Cargo workspace tests and 566 non-Metal Python tests (5 skipped, 2 Metal-only deselected), including regressions for explicit IPv4 bootstrap without IPv6 discovery, frozen-app distribution metadata, and repository-wide Nix source-path integrity. Earlier live twin testing found and fixed two merge regressions: the legacy `--bootstrap-peers` multiaddr is translated to an explicit Zenoh TCP endpoint, and MLX cache typing no longer imports a protocol absent from the pinned runtime. The validated build formed a shared two-node cluster, loaded all 48 layers of the cached 192 GB Qwen model as a Tensor/JACCL instance, and completed the same 66-token inference probe used against the stable baseline. Dashboard validation also covers the current serialized `shards` tuples so ready runners no longer appear as `Unknown`/`PREPARING`; legacy shard maps remain supported. The live twins run detached PR-head validation worktrees, with 7763d51e retained as the rollback lane. Nix is unavailable on this host.
- Known follow-ups: Deliberaitor plan creation is unavailable because its workspace-local route rejects this bastion-scoped Codex session; the PR will use the required outage note. The legacy `--libp2p-port` flag remains a Zenoh-port alias, and supported `/ip4` or `/ip6` TCP bootstrap multiaddrs are translated to Zenoh endpoints for existing launch scripts.
