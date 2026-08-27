# Exo Cluster Operations Reference

Detailed topology, troubleshooting, and model constraint reference for JJ's
default Mac exo cluster.

Current runtime ownership lives in this repo under `ops/team-wcv`. Codex-facing
workflow notes live in `team-wcv/cdx/skills/exo-fleet`. Older references to
`bb rdma` are historical Big Brain helper commands and should not be treated as
current EXO launch instructions unless that helper has been intentionally
restored.

## Cluster Topology

| Node        | Chip     | RAM    | Wired LAN IP (NIC) | Wi-Fi IP (fallback) | Thunderbolt | RDMA Interfaces                                    | Notes              |
| ----------- | -------- | ------ | ------------------ | ------------------- | ----------- | -------------------------------------------------- | ------------------ |
| `wc-smbp`   | M5 Max   | 128 GB | `192.168.1.63` | `192.168.0.2` (TB control) | Yes         | direct TB5 edge to `wc-smbpt` | 14-inch MBP; current launchd master/rank-zero preference. |
| `wc-smbpt`  | M5 Max   | 128 GB | `192.168.1.120` | `192.168.0.1` (TB control) | Yes         | direct TB5 edge to `wc-smbp` | 14-inch MBP; current launchd worker. |
| `wc-bmbp`   | M5 Max   | 48 GB  | see live discovery | see live discovery | Yes         | optional / not in default twin launchd | Historical asymmetric-drafter candidate. Do not include in the default twin fleet unless deliberately testing that topology. |
| `wc-studio` | M1 Ultra | 128 GB | `192.168.1.29` / `100.73.100.108` | n/a                 | No          | None — TCP/LAN only                                | Mac Studio |
| `wc-spark`  | NVIDIA DGX Spark | 128 GB | `192.168.1.150` | n/a              | No          | None — TCP/LAN only                                | Opt-in only; excluded by default |

> **NIC vs Wi-Fi:** All three M5 Max nodes have a real wired ethernet adapter
> in addition to Wi-Fi. After the 2026-05-11 classifier fix in
> `exo/utils/info_gatherer/system_info.py` (PR team-wcv/exo#28), the placement
> engine correctly classifies the USB-LAN dongles on smbp / smbpt and the TB
> dock NIC on bmbp as `ethernet` rather than `maybe_ethernet`, so
> `find_ip_prioritised(ring=False)` (the JACCL coordinator wire) now picks the
> symmetric wired path (~0.5–0.85 ms RTT bidirectional) instead of the
> asymmetric-routed Wi-Fi return path (43 ms RTT). For human-driven curl /
> bootstrap-peer use, prefer the wired LAN IPs in the first column.

The **current default Thunderbolt topology is a 2-edge mesh**:

| Edge | Cable | Negotiated speed | Min RTT | Avg RTT |
| ---- | ----- | ---------------: | ------: | ------: |
| `smbp ↔ smbpt` direct | TB5 | 80 Gb/s | 0.317 ms | 0.389 ms |
| `smbp ↔ bmbp` direct  | TB4 | 40 Gb/s | 0.400 ms | 0.448 ms |
| `smbpt ↔ bmbp`        | — (no RDMA edge; control traffic falls back to LAN TCP) | — | — | — |

The `smbpt ↔ bmbp` leg is intentionally omitted from the RDMA mesh. The
asymmetric drafter dispatch (drafter on bmbp, target on smbp+smbpt TP2)
only requires bmbp ↔ smbp connectivity — drafts ship to target rank 0
(smbp) which then broadcasts internally to smbpt over the TP2 group.
Control-plane heartbeats between smbpt and bmbp ride LAN TCP and tolerate
the higher latency.

**TB4 (40 Gb/s) vs TB5 (80 Gb/s) on the bmbp ↔ smbp leg is functionally
equivalent** for the drafter workload. Drafter dispatch packets are KB-
scale and latency-bound, not bandwidth-bound (peak sustained usage is
~500 KB/s ≈ 4 Mbit/s, ~0.01% of TB4 capacity). Swap to a TB5-rated cable
on this leg only if you ever pivot to a TP3 workload that includes bmbp.

### Optional hubs and the iVANKY Fusiondock Ultra

An iVANKY Fusiondock Ultra can optionally be inserted on the `smbp ↔
bmbp` leg to validate the nested-hub discovery code (`_ConnectivityItem`
tree walk in `src/exo/shared/types/thunderbolt.py`). Measured hub
overhead is in the noise floor:

| Path | Min RTT | Avg RTT | iperf3 TCP (4 streams, 10 s) |
| ----- | ------: | ------: | ---------------------------: |
| `smbp ↔ smbpt` direct TB5 (pre-fanout-test bench) | 0.341 ms | 0.575 ms | 52.6 – 56.1 Gbit/s |
| `smbp ↔ bmbp` via iVANKY (single hub) | 0.354 ms | 0.482 ms | 63.6 Gbit/s |
| `bmbp ↔ smbp` reverse via iVANKY (single hub) | 0.345 ms | 0.616 ms | 60.9 Gbit/s |

The fork's `main` already carries the equivalent nested-hub parser fix;
the same fix is in upstream review as [exo-explore/exo#2063](https://github.com/exo-explore/exo/pull/2063).
Without the fix, `bmbp`'s side of any hub-attached connection silently
drops because the iVANKY entry has no `domain_uuid_key` of its own and
the old parser only looked at top-level `_items`.

### Known-bad TB topologies

Two patterns reproducibly break TB-Bridge networking and / or trigger
macOS kernel panics in `AppleThunderboltIPConnection`. Avoid both:

1. **CalDigit TS5 multi-host fanout** (one TS5 hub brokering two Mac
   peers on a single upstream Mac port). TS5 enumerates both peers in
   `system_profiler`, but only one TB-IP link establishes at a time, and
   any disturbance — heavy concurrent iperf3 load, a TB port cable
   swap, or a `networksetup -setnetworkserviceenabled "Thunderbolt
   Bridge" off`/`on` cycle — can panic the upstream host. Two
   reproducible kernel panics observed during PR #2063 hub validation
   (panic-full-2026-05-11-12*.panic and panic-full-2026-05-11-13*.panic
   on wc-bmbp).
2. **2-hub chain** (e.g. `bmbp ↔ iVANKY ↔ TS5 ↔ smbp`). Thunderbolt
   enumeration walks the chain fine (system_profiler shows the full
   tree), but TB-Bridge IPv4 does not pass through two active hubs on
   macOS 26.5. Stick to at most one hub in any TB-Bridge path.

Failure mode is the same in both cases: macOS's
`AppleThunderboltIPConnection` driver appears to have a use-after-free
or similar fault on multi-host TB hub configurations. The driver bug is
not exo's responsibility to work around — exo's nested-hub *discovery*
remains correct (PR #2063), but operators should not run the data plane
through these topologies.

### Interconnect Summary

- **TCP mesh**: Built by `check_reachable()` in `net_profile.py`. It HTTP-pings
  every peer's advertised interfaces at `GET /node_id` on `api_port`. Each node
  **must** have its API server running (`--api-port 52415`) for this to work.
- **RDMA mesh**: Detected by the Thunderbolt scanner via
  `system_profiler SPThunderboltDataType`. Only the 3 TB-connected nodes
  participate.

Healthy targets:

- Default 4-node cluster (`wc-smbp`, `wc-smbpt`, `wc-bmbp`, `wc-studio`):
  12/12 TCP, **4/12 RDMA** (2-edge RDMA mesh: smbp↔smbpt + smbp↔bmbp = 4
  directional pairs; studio has no TB so contributes zero RDMA edges).
- 3-node Thunderbolt-only cluster (no studio/Spark): 6/6 TCP, **4/6 RDMA**.
- 5-node Spark opt-in cluster: 20/20 TCP, 4/20 RDMA. Use only when
  `wc-spark` is intentionally included for Spark-specific testing.

RDMA pair count was 6/6 prior to 2026-05-11 (3-edge full mesh including
smbpt↔bmbp). The smbpt↔bmbp leg was dropped after the macOS
AppleThunderboltIPConnection panics on TS5 multi-host fanout testing,
because the drafter workload only requires bmbp↔smbp (drafter dispatches
to target rank 0 = smbp, which broadcasts to smbpt internally over TP2).

## Startup

The current twin fleet is managed by the launchd wrapper in
`ops/team-wcv/launchd/teamwcv-exo-bigbrain`.

For an isolated branch deployment, set `EXO_BIGBRAIN_SOURCE_ROOT` in the
launchd manager environment to a prepared worktree containing
`dashboard/build/index.html`, then restart the service through the normal
consensus flow. When the variable is unset, the wrapper continues to prefer
`~/.orchestraitor/worktrees/exo/cluster-main` and falls back to the primary
checkout.

- **Master**: `wc-smbp`, `--api-port 52415`, `--libp2p-port 52418`, `-m`.
- **Worker**: `wc-smbpt`, `--api-port 52415`,
  `--bootstrap-peers /ip4/192.168.1.63/tcp/52418`.
- **Client/API alias**: `http://bigbrain.localhost:18888/v1`, via Caddy on
  `wc-studio`, proxying to `http://192.168.1.63:52415`.
- **Never** use Tailscale or `.local` hostnames for the EXO workload/control
  path. The stable operator path is numeric LAN for API/control and
  Thunderbolt/RDMA for tensor traffic.

## Topology Sanity Check

```bash
curl -sS --max-time 6 http://192.168.1.63:52415/state -o /tmp/state.json && python3 - <<'PY'
import json
d = json.load(open("/tmp/state.json"))
ids = d.get("nodeIdentities") or {}
top = d.get("topology") or {}
conns = top.get("connections") or {}
hostmap = {k: ((d.get("nodeSystem") or {}).get(k) or {}).get("hostname") or k[:10] for k in ids}
print(f"{len(ids)} nodes")
for src in sorted(ids, key=lambda k: hostmap[k]):
    for dst in sorted(ids, key=lambda k: hostmap[k]):
        if src == dst: continue
        links = (conns.get(src) or {}).get(dst) or []
        r = sum(1 for l in links if "sourceRdmaIface" in l)
        t = sum(1 for l in links if "sinkMultiaddr" in l)
        print(f"  {hostmap[src]:>14} -> {hostmap[dst]:<14}  RDMA={r}  TCP={t}")
PY
```

Diagnostic thresholds:

- Below the expected TCP count for the intended node set → a worker is running
  with `--no-api` or a requested node has not joined the cluster.
- Below expected RDMA count → a Thunderbolt cable is missing, or the TB scanner
  hasn't converged yet (wait longer), or the `bridge0` issue on bmbp is blocking
  RDMA GID population.

## `wc-bmbp` Thunderbolt Bridge (RDMA Only)

`wc-bmbp` has a `bridge0` that captures TB interfaces (`en6`, `en2`) as
members. **This is only an issue when RDMA/JACCL sharding is needed** — the
cluster starts and runs fine over TCP/Pipeline without touching bridge0.

When bridge0 is active and RDMA is needed, it causes:

- Missing RDMA GIDs (only 1 per device instead of 3).
- JACCL QP RTR transitions fail with errno 96 (EPROTOTYPE).
- Unplugging/replugging TB cables can corrupt kernel RDMA state (reboot needed).

### Fix Procedure (only when RDMA/JACCL is needed)

Historical Big Brain helper commands existed for repairing this state:

```bash
bb rdma repair all
bb rdma jaccl-status all
bb rdma verify
```

Those helpers are now legacy unless reintroduced under `ops/team-wcv`. The
important invariant remains: RDMA device presence is not enough. JACCL must be
able to move queue pairs to RTR, and stale 169.254 routes through LAN are enough
to break that even while `ibv_devinfo` reports active RDMA ports.

Manual commands remain useful when repairing one machine from a console session:

```bash
sudo ifconfig bridge0 destroy
sudo ifconfig en6 up
sudo ifconfig en2 up
sudo ifconfig en6 169.254.100.6 netmask 255.255.0.0
sudo ifconfig en2 169.254.100.2 netmask 255.255.0.0
sudo ifconfig en6 inet6 fe80::3479:2cff:fe66:984 prefixlen 64
sudo ifconfig en2 inet6 fe80::3479:2cff:fe66:988 prefixlen 64
sudo ifconfig en6 inet6 fe80::1:2:3:4 prefixlen 64
sudo ifconfig en2 inet6 fe80::5:6:7:8 prefixlen 64
sudo arp -d 169.254.222.72 2>/dev/null || true
sudo arp -d 169.254.136.105 2>/dev/null || true
```

Verify:

```bash
ibv_devinfo -d rdma_en6 -v | grep GID   # expect 3 GIDs
ibv_devinfo -d rdma_en2 -v | grep GID   # expect 3 GIDs
ping -c 1 169.254.222.72                 # smbp via en6
ping -c 1 169.254.136.105               # smbpt via en2
```

The other two Macs (smbp, smbpt) have persistent "RDMA en\*" NetworkServices
and don't need this.

### Making It Permanent

The proper fix is creating persistent NetworkServices, but this is blocked by
SIP on bmbp:

```bash
sudo networksetup -createnetworkservice "RDMA en6" en6   # fails under SIP
```

Alternatives: boot Recovery OS, or create a LaunchDaemon that runs the manual
procedure at boot.

## Clearing Stale State

If the master gets stuck replaying a massive event log (you'll see
`RequestEventLog since_idx=700000+` scrolling endlessly), or phantom nodes
appear with `ramTotal=0`:

```bash
# On all nodes (master and workers, including opt-in wc-spark)
rm -rf ~/.exo/event_log ~/.exo/exo_log
```

Always wipe `wc-spark` too even when it isn't part of the active topology: a
prior opt-in session can leave stale membership/event-log state that re-enters
the cluster the next time Spark is added.

Then restart the cluster (master first, then workers). Models are preserved in
`~/.exo/models`.

## Model Constraints

| Model                                                | Size   | Tensor 3-way                | Pipeline 3-way                 | Notes                                                                                                  |
| ---------------------------------------------------- | ------ | --------------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------ |
| GLM-4.6-4bit                                         | 185 GB | No (`supportsTensor=False`) | Yes (needs ~2x working memory) | Freshly-rebooted cluster recommended                                                                   |
| gpt-oss-120b-MXFP4-Q8                                | 71 GB  | No (8 KV heads % 3 != 0)    | Yes                            | 2-way Tensor OK                                                                                        |
| Step-3.5-Flash-8Bit                                  | varies | Depends on KV heads         | Yes                            | Supports thinking/reasoning                                                                            |
| Qwen3.5-122B-A10B-mlx-8bit (MoE)                     | ~122 GB | n/a (use 2-way Tensor)     | Yes                            | **Headline DFlash configuration**: 2-way Tensor + `MlxJaccl` on smbp+smbpt yields 159 t/s (3.02x speedup) via per-rank coupled DFlash drafter. See `bench/results/dflash/REPORT.md` in the exo repo. |
| Qwen3.5-397B-A17B-4bit (MoE)                         | ~224 GB | n/a (use 2-way Tensor)     | No (per-rank shard too large for bmbp) | TP2 on smbp+smbpt with asymmetric `Qwen3.5-2B-MLX-4bit` drafter on bmbp via TB-5 wire. Post-classifier-fix bench (PR team-wcv/exo#28): target-only 50.0 t/s; K=3 drafter +38% (code), +60% (list), +22% (explain); K=5 hits 104 t/s on list (83% accept, x2.10) but goes net-negative on lower-accept-rate workloads. Custom card override at `~/.exo/custom_model_cards/mlx-community--Qwen3.5-397B-A17B-4bit.toml` (declares `drafter_model_ids` + `drafter_eligible_nodes`). |
| Qwen3.5-35B-A3B-4bit (MoE)                           | ~20 GB | Single-device fits         | Single-device fits             | Single-node Pipeline + `MlxRing` on smbp with in-process `z-lab/Qwen3.5-35B-A3B-DFlash` coupled drafter delivers 221–278 t/s inference rate across workloads (+105% / +143% on code / explain). Throughput champion among models tested on the 3-node cluster. Custom card declares `coupled_drafter = "z-lab/Qwen3.5-35B-A3B-DFlash"`. |
| Qwen3.6-35B-A3B-8bit (MoE)                           | ~36 GB | Single-device fits         | Single-device fits             | DFlash coupled drafter delivers 4.30x (377 t/s) on a single M5 Max via in-process drafter.             |
| gemma-4-31b-it-bf16                                  | ~62 GB | Yes (2-way + asym drafter)  | Yes                            | Verified TP2 config: smbp+smbpt + asymmetric standard drafter on bmbp, +13–24% speedup.                |

## Speculative-Decoding Drafters

The `SKILL.md` section "Drafter strategies" covers placement
semantics in detail. Key operational points:

- **Coupled drafters** (MTP for Gemma 4, DFlash for Qwen 3.5/3.6)
  are declared on the model card's `coupled_drafter` field. They
  run in-process with every target rank that holds the LM head and
  replicate per-rank on TP placements (no asymmetric attachment).
- **Standard drafters** are declared on the model card's
  `drafter_model_ids` list and can be asymmetric-attached on a
  separate runner for `Tensor` / `AsymmetricTensor` placements.
- When a card declares both, the loader tries coupled first and
  falls back to standard at runtime (e.g. if `mlx-vlm` is missing
  or the coupled weights aren't on disk).
- The wired-memory limit is bumped by the eligible drafter size
  before the target loads. On TP placements the bump tracks the
  *coupled* drafter only (standard drafter is gated off on TP).

Full A/B benchmark methodology and per-scenario numbers live in
the exo repo:

- `bench/results/mtp/REPORT.md`    — Gemma 4 MTP coupled drafter
- `bench/results/dflash/REPORT.md` — Qwen 3.5 / 3.6 DFlash coupled drafter (incl. 122B TP2)
- `bench/results/drafter/local/`   — raw asymmetric standard-drafter runs (e.g. gemma-4-31b-it-bf16 + e2b on bmbp)

## Capacity Limits

`wc-bmbp` has 48 GB total RAM. Three-way placements where each shard exceeds
~40 GB will pass the placement preview but OOM at runtime when bmbp tries to
load its slice. Use 2-way for large models, or 3-way only when bmbp's slice
fits comfortably.

`wc-studio` has 128 GB but is TCP-only — Tensor sharding over TCP is slower
than RDMA. Best used for Pipeline placements or as a standalone node for large
models that fit in its memory.

## Troubleshooting: Target throughput regressed after adding a third node

**Symptom:** TP2 target-only throughput on smbp+smbpt was healthy
(e.g. 50 t/s on Qwen3.5-397B-A17B-4bit) before bmbp joined the cluster as a
drafter-eligible node, and dropped sharply (e.g. to 17 t/s) afterwards.

**Likely cause:** The macOS interface classifier in
`exo/utils/info_gatherer/system_info.py` is misclassifying one of the wired
NICs. `find_ip_prioritised(ring=False)` (used by `get_mlx_jaccl_coordinators`)
ranks `ethernet > maybe_ethernet > wifi > unknown > thunderbolt`; if a USB-LAN
dongle or TB-dock NIC is classified as `maybe_ethernet` instead of `ethernet`,
it ties with `wifi` and the rank-0 coordinator dial can leak onto the Wi-Fi
return path (asymmetric routing, 43+ ms RTT).

**Quick check:**

```bash
curl -sS http://192.168.1.63:52415/state | python3 -c "
import json, sys
d = json.load(sys.stdin)
for nid, info in d.get('nodeNetwork', {}).items():
    print(f'--- {nid[-12:]} ---')
    for iface in info.get('interfaces', []):
        addr = iface.get('ipAddress', '')
        if ':' in addr or addr.startswith(('169.254.', '127.')): continue
        print(f\"  {iface['name']:8s} type={iface['interfaceType']:15s} addr={addr}\")
"
```

Each node's USB-LAN dongle (`en16` on smbp, `en9` on smbpt) and bmbp's TB-dock
NIC (`en7`) should report `type=ethernet`. If they report `maybe_ethernet`,
the cluster is running a stale build — pull the team-wcv/exo PR #28 fix and
**restart every exo process, including the master on `wc-smbp`**, not just the
worker processes. `get_mlx_jaccl_coordinators` runs on the master and consumes
the master node's own interface typing when it picks the rank-0 dial target,
so a stale master keeps publishing `maybe_ethernet` for its own dongle and
the coordinator wire can still leak onto the Wi-Fi return path even after
every worker is restarted. After a full restart, the JACCL coordinator wire
lands on the symmetric LAN dongle/dock-NIC path (~0.85 ms RTT both ways) and
target-only throughput recovers.

## Troubleshooting: 397B placement fails with "No tensor sharding found"

**Symptom:** `POST /place_instance` for `mlx-community/Qwen3.5-397B-A17B-4bit`
with `sharding=Tensor` returns
`No tensor sharding found for model with hidden_size=4096, num_key_value_heads=2`,
despite the 2-cycle `(smbp, smbpt)` passing the divisibility check
trivially (`4096 % 2 == 0`, `2 % 2 == 0`).

**Likely cause:** `filter_cycles_by_memory` rejected the 2-cycle because the
sum of `ramAvailable` across smbp+smbpt is less than the 223.86 GB model
storage size. Each node should have ~115 GB free; if smbp shows ~95 GB free
or less, an orphaned Python process from a previous run is holding RAM that
the OS hasn't reclaimed yet.

**Recovery:**

```bash
ssh wc-smbp 'ps aux | awk "{ if (\$6 > 1000000) print \$0 }"'
# Look for stale `multiprocessing.spawn` / `mlx_lm` / `Qwen3.5` workers
# left behind by a previous spec-decode run that didn't clean up. Kill
# with `kill -9 <pid>`; the node memory will free within ~5 s. Re-place.
```

## Key Exo Source Locations

| Component             | Path                                                |
| --------------------- | --------------------------------------------------- |
| Node entry point      | `src/exo/main.py`                                   |
| Placement engine      | `src/exo/master/placement.py`, `placement_utils.py` |
| TCP mesh builder      | `src/exo/utils/info_gatherer/net_profile.py`        |
| TB scanner            | `src/exo/utils/info_gatherer/thunderbolt.py`        |
| JACCL init            | `src/exo/worker/engines/mlx/utils_mlx.py`           |
| Peer discovery (Rust) | `rust/networking/src/discovery.rs`                  |
| Topics/messaging      | `src/exo/routing/topics.py`                         |
| State model           | `src/exo/shared/types/state.py`                     |
| Event apply           | `src/exo/shared/apply.py`                           |
