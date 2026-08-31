---
branch: chore/6a959fcc-sync-exo-upstream
created: 2026-08-31
owner: codex-agent
status: active
scope: "Merge current exo-explore/exo main into the team-wcv fork while preserving the fork's custom inference, fleet, and twin-recovery work"
orchestraitor:
  ticket: 6a959fcce6b6efd51fc85507
  task_url: orchestraitor://task/6a959fcce6b6efd51fc85507
pr:
  url: https://github.com/team-wcv/exo/pull/42
  state: open
cleanup:
  merged_into: team-wcv/main
  archived_after: null
  successor_branch: null
source_branches:
  - team-wcv/main (fork integration baseline at 0909a1ed)
  - origin/main (official upstream at 21a54c5e)
---

- Why this branch exists: preserve the known-good two-node exo deployment, then bring the fork current with official upstream without losing team-wcv custom behavior.
- Changed paths: upstream Zenoh/exo_rs transition; Python package and dashboard locks; persistent scoped identity; event routing and peer-aware election recovery; restart-safe custom model cards; model-card/backend compatibility; peer-download, custom drafting, placement, runner supervision, API serialization, benchmark packaging, tests, and formatting.
- Validation run: matching smbp/smbpt snapshots archived with Git bundle at f033e597; official and fork tips are ancestors of the merge head; uv lock/sync succeeded; Ruff and Ruff format pass; BasedPyright reports 0 errors; Cargo check/test/fmt pass; dashboard npm ci/build, Svelte check, and Prettier pass. The P1 review round passes 47 focused tests (1 skipped), 599 sandbox-safe tests (5 skipped, 2 Metal-only deselected), and 9 stacked-base tests; an isolated exo-bench environment imports exo-tools successfully. Earlier live twin testing found and fixed two merge regressions: the legacy `--bootstrap-peers` multiaddr is translated to an explicit Zenoh TCP endpoint, and MLX cache typing no longer imports a protocol absent from the pinned runtime. The validated build formed a shared two-node cluster, loaded all 48 layers of the cached 192 GB Qwen model as a Tensor/JACCL instance, and completed the same 66-token inference probe used against the stable baseline. The live twins remain pinned to validated head 7763d51e; the P1 follow-up head is not deployed. Nix is unavailable on this host.
- Known follow-ups: Deliberaitor plan creation is unavailable because its workspace-local route rejects this bastion-scoped Codex session; the PR will use the required outage note. The legacy `--libp2p-port` flag remains a Zenoh-port alias, and supported `/ip4` or `/ip6` TCP bootstrap multiaddrs are translated to Zenoh endpoints for existing launch scripts.
