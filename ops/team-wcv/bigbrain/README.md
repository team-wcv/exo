# Bigbrain EXO fleet

## Current service allocation (2026-10-24)

The Twins serve Qwen with local prefill and RDMA tensor decode. Bigbrain’s root
and `/twins/` now choose a healthy API on smbp or smbpt; `/spark/` serves native
Laguna and `/studio/` serves Studio. The `/laguna/` path remains compatible. Spark and Studio
now provide independent, persistent 118B Laguna workers with dedicated browser
and API paths. Spark remains visible in EXO as a reserved observer with no
inference backend. See the [always-on service runbook](../always-on/README.md)
for the current allocation, startup pins, client profiles, and acceptance.

The original disaggregated configuration below records the October 16–18
validation and its recovery procedure. It is not the current default placement.

## Original shared deployment (2026-10-16)

The operator dashboard at `http://bigbrain.localhost:18888/` serves one shared
`wc-bigbrain` cluster. Studio is the preferred master; smbp, smbpt, and Spark
join the same cluster. All four APIs use port `52415`, and peer-file servers use
`24216`. Using different API ports breaks the deployed reachability probes even
when discovery succeeds.

| Function | Host / transport |
|---|---|
| Dashboard/API ingress | Local Caddy, Studio preferred, health-checked MacBook fallbacks |
| Control plane | Studio; inference/control topology uses LAN addresses |
| Large-model prefill | Spark, MLX CUDA, cache transfer over LAN |
| Tensor decode | smbp + smbpt, JACCL/RDMA over Thunderbolt |
| JACCL coordinator | Reachable Thunderbolt IP, `10.254.42.1` or `10.254.42.2` |

The current serving model is
`pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit`, pinned at
`b2c422f3c643e36f04227a64d61796b44a4b1029`. Its fleet-specific custom card names
Spark in `prefill_eligible_nodes`; both automatic placement and explicit dashboard
placement create or reuse a compatible prefill instance and link it to decode.
The remote path applies to more than 1,000 uncached prompt tokens.

Persisted launchd/systemd settings use `ENABLE_DISAGGREGATION=true`,
`EXO_HOME=.exo-bigbrain-cluster`, and allowed inference CIDRs
`192.168.1.0/24,192.168.0.0/24,10.254.42.0/30`. Service labels are
`ai.team-wcv.exo-studio-bigbrain` on Studio, `com.teamwcv.exo.bigbrain` on both
MacBooks, and `exo-spark-bigbrain.service` on Spark. MacBook jobs hold a
`caffeinate` assertion. During a bootout/bootstrap maintenance cycle, hold a
separate temporary assertion until the replacement job is running.

The checked-in [shared startup snapshots](shared/) reproduce the verified
startup configuration and fleet-specific model card. Compare them with live
files and preserve local overrides; back up before replacing a startup file.
They refer to the existing validated dependency environments and deployment
source/native directories, so confirm those paths before installation. The
model card belongs in `~/.exo-bigbrain-cluster/custom_model_cards/`; the worker
publishes persisted cards to the cluster on startup.

The validated deployment is under
`~/.orchestraitor/deployments/exo-qwen38-prefill-20261016/`: `current/` contains
source based on `2c707493` plus the recorded overlay, and `native-current/`
contains the matching platform build. `deployment-manifest.json` records the
source overlay hashes. Startup files select both with `PYTHONPATH`; preserve
that pairing during upgrades. Prior startup configurations and source revisions
are backed up alongside the deployment.

The standalone Spark Laguna unit was stopped for this original EXO validation.
It is now enabled and running with failure recovery, and Spark is reserved from
EXO inference. The October 24 runbook above supersedes the former stopped state.

Validation must use the operator URL: inspect `/state`, `/v1/feature-flags`, and
`/v1/instance-links`, then send a long uncached prompt and verify successful remote
cache import on **both** decode ranks. Repeat after a master restart and worker
restarts. A model card or topology edge alone is not inference proof. See
[cache transport and restart details](../docs/exo-cluster-ops.md#qwen38-spark-prefill-cache-compatibility).

The named Tailnet alias `https://bigbrain.taile43e67.ts.net` is verified on Studio's
existing tagged `wc-studio-gw` gateway. Its private service forwards to Studio's
API without changing the ownership/SSH identity of the operator's Macs. The
host advertisement is approved, TLS is provisioned, and a real Qwen3.8 request
completed through HTTPS with Spark cache ingestion on both Mac ranks. See the
[gateway configuration and acceptance procedure](shared/tailscale.md).

## Network recovery and prefill role safety (2026-10-18)

The direct MacBook Thunderbolt subnet is now `10.254.42.0/30`: smbpt uses
`10.254.42.1` on `RDMA Thunderbolt 1` (`en1`), and smbp uses `10.254.42.2` on
`RDMA en2` (`en2`). Both have mask `255.255.255.252` and no router. The previous
`192.168.0.1` Thunderbolt address conflicted with smbpt's Wi-Fi gateway after it
joined `192.168.0.0/24`. Keep the direct link disjoint from every LAN subnet.
Backups and rollback values are in each host's
`~/.orchestraitor/deployments/exo-network-recovery-20261018/`.

Both LAN subnets are allowed for API reachability. Both MacBooks bootstrap through
Studio's wired address `192.168.1.21:52618` and the direct Thunderbolt link. The
previous Wi-Fi address `192.168.0.33` was no longer reachable after Studio's
October 22 update; avoid using a transient Wi-Fi address as the only master peer.
A healthy Tailscale connection does not prove the inference LAN path.

If a decode instance disappears, its surviving prefill instances retain their
role in replayable state. New requests return a terminal error until a decoder
is restored, rather than selecting the prefill server for generation. Explicit
role reassignment through a new link remains supported.

Spark sets `EXO_PREFILL_CHUNK_SIZE=256` to bound temporary activation memory for
the large model. The remote-prefill default remains 4096 elsewhere. Large chunks
on a nearly full Spark can trigger swapping and leave the UI at zero progress
while the first chunk evaluates. Validate a long uncached prompt as well as short
chat, checking memory and cache ingestion on both decode ranks.

Recovery acceptance through the operator URL used the original 5,815-token
conversation: cold first text took 90.8 seconds and completion took 96.2 seconds.
Spark computed 5,813 prefix tokens in 84.4 seconds; both Mac ranks imported them
and decoded the response over JACCL. Spark retained about 14 GiB available with
no new swap-out in sampled intervals. This is a functional recovery, not a claim
that cold prefill on this large model is fast. A fresh 29-token story request
returned first text in 1.7 seconds; a 6,039-token follow-up reused 5,813 cached
tokens and returned first text in 1.9 seconds. These latency checks used thinking
disabled. The old demo conversation includes thousands of diagnostic tokens;
start a fresh conversation for ordinary short chat.

The API registers its response channel before dispatching a generation command,
so an immediate missing-decoder error is buffered even before SSE consumption
starts. Tests cover that ordering and cleanup when command dispatch fails.

## Peer-download listener recovery (2026-10-18)

The old peer-download port `52416` collided with macOS `rapportd` on smbpt,
causing EXO to exit and launchd to retry every ten seconds. All four nodes now
use `24216`, below the observed macOS dynamic range `49152–65535` and Linux
range `32768–60999`. Keep this port common across the fleet: peer discovery
currently assumes the local configured port is also the peer's port.

Changing this port also changes EXO's identity-file scope. Copy the previous
`node_zid.zenoh-<port>.api-52415.peer-52416` into the corresponding
`peer-24216` path before starting the new configuration, verifying its contents
match; never overwrite an existing different identity. Preserve the original
file and the backed-up startup configuration for rollback. Verify `/node_id`,
the peer server's `/health`, stable launchd PIDs, and four-node topology after
restart. Do not stop `rapportd` to make EXO's old port available.

## Dashboard asset continuity

An existing Chrome tab can load cached bootstrap code that references a previous
hashed JavaScript module. On 2026-10-18, requests for
`/_app/immutable/nodes/2.-Ujo2dWt.js` returned 404 and caused an import/reload loop
with a blank page, even while fresh browser tests and the API succeeded.
The original module and its matching immutable dependencies were recovered from
the prior local build and added to each deployed dashboard without replacing the
current HTML or any existing asset. Per-host `dashboard-compat-assets.json` in
the recovery deployment directory records the added file hashes.

Retain the previous `_app/immutable` assets when replacing a dashboard build.
Merge missing hashed files into the new build; if an existing path has different
bytes, stop and investigate rather than overwriting it. Preserve the complete
dependency set, not just the first missing chunk. Check both a fresh bootstrap
and a previous cached bootstrap against the production URL before declaring a
dashboard deployment healthy. Existing conversations remain in browser storage.

The checked-in Caddy configuration sends `Cache-Control: no-store` for `/`,
`/index.html`, and `/_app/version.json`, so future bootstrap documents do not
remain cached across deployments. Hashed immutable assets remain cacheable.
Inspect actual failing asset requests in the access log: successful `/state`
responses alone do not establish that the operator's existing tab can render.

## Reconnection and replay recovery (2026-10-22)

After Studio rebooted, both Mac services were running but isolated. Updating
smbpt's stale bootstrap and reconnecting restored discovery, but worker state
was still hours behind: a master with roughly 240,000 events was being replayed
in windows of 64. An initial router can now request a targeted snapshot once the
master has at least 64 events. The snapshot includes current state, its exact
global index, and producer acknowledgement watermarks; buffered newer events
continue in order. Existing consumers keep bounded replay, so active streams
never skip history. Increasing replay windows alone caused delivery-watchdog
restarts under load and was rolled back. Verify `lastEventAppliedIdx` and fresh `lastSeen` timestamps as well
as the master's topology. Being visible at the master is not proof that a worker
has caught up enough to accept new placements. Snapshot bootstrap introduces a
new control event: update all Python runtimes together before enabling it.
The native networking binary does not change. Snapshots are accepted only by
their target router, in the current master session, before any event is applied.
Any master session that has received image inputs retains ordinary replay,
including after the original task is deleted, because APIs can reuse image
hashes whose transient input chunks are not part of the state snapshot.
Producer watermarks clear only local
deliveries already accepted by the master, leaving newer deliveries retryable.

The [reconnection guard](reconnect_guard.py) is installed on the MacBooks by the
`com.teamwcv.exo.reconnect` LaunchAgent, using the corresponding
[smbp](shared/smbp-reconnect.plist) and [smbpt](shared/smbpt-reconnect.plist)
snapshots. It runs every 30 seconds and checks the local node identity, master
freshness, master API identity, and the master's Zenoh listener. These background
client probes use Studio's existing Tailscale address because smbpt's background
Python cannot reach the LAN address even while its EXO runner can. Bootstrap,
inference, and RDMA transports retain their LAN/Thunderbolt addresses. Master
health over Tailscale establishes service availability, not LAN reachability;
the cooldown limits unsuccessful recovery attempts during a LAN outage.
Only a persistently isolated idle worker is restarted after 120 seconds, with a
10-minute cooldown. A changed process ID gets a new startup grace period.
Advancing replay from the expected master resets the timer, even when replayed
timestamps are old; local singleton-master events do not count as catch-up.
Local model assignments, pending/ongoing downloads, uncertain probes, and an unavailable master prevent
automatic restart. The guard does not recreate model placements.

The deployed script is in `~/.orchestraitor/deployments/exo-reconnect-20261022/`.
Decision records and logs are under `~/.cache/exo/reconnect-guard*`. Running the
script without `--apply` only reports the recovery decision. To disable automatic
reconnection, boot out `gui/$(id -u)/com.teamwcv.exo.reconnect` and remove its
LaunchAgent from `~/Library/LaunchAgents` after saving a backup. Leave the main
`com.teamwcv.exo.bigbrain` job in place.

## Legacy isolated endpoint profiles

The checked-in profiles below predate the shared four-node deployment. They are
reference material for intentionally isolated endpoints, including BMBP; do not
copy them over the active shared-cluster startup files. Their `52615` ports and
standalone namespaces are not the current Bigbrain topology.

### Install legacy host services

Studio:

```bash
mkdir -p ~/.cache/exo ~/Library/LaunchAgents
cp ops/team-wcv/bigbrain/studio/ai.team-wcv.exo-studio-bigbrain.plist \
  ~/Library/LaunchAgents/
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/ai.team-wcv.exo-studio-bigbrain.plist
```

Spark:

```bash
mkdir -p ~/.config/systemd/user
cp ops/team-wcv/bigbrain/spark/exo-spark-bigbrain.service \
  ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now exo-spark-bigbrain.service
```

BMBP:

```bash
mkdir -p ~/.cache/exo ~/.local/share/bigbrain-endpoints/bmbp ~/Library/LaunchAgents
ln -sfn <validated-exo-worktree> ~/.local/share/bigbrain-endpoints/bmbp/exo-current
cp ops/team-wcv/bigbrain/bmbp/ai.team-wcv.exo-bmbp-bigbrain.plist \
  ~/Library/LaunchAgents/
launchctl bootstrap "gui/$(id -u)" \
  ~/Library/LaunchAgents/ai.team-wcv.exo-bmbp-bigbrain.plist
```

Studio and Spark use the hardware-tested PR44 runtime at commit
`0e9c429f2afee6f2412c1bd07787119968ceba85`. Update the absolute runtime path in the
service file only after validating a replacement build on the target host.

BMBP resolves its runtime through
`~/.local/share/bigbrain-endpoints/bmbp/exo-current`, allowing the validated checkout
to be advanced without changing the LaunchAgent definition.

Use `http://127.0.0.1:52615/v1` for runtimes running on BMBP itself and
`https://wc-bmbp.taile43e67.ts.net:8443/v1` for runtimes on other Tailnet devices.

### Publish legacy device endpoints

```bash
# Studio
tailscale serve --https=8443 --bg --yes http://127.0.0.1:52615

# Spark
tailscale serve --https=8443 --bg --yes http://127.0.0.1:52615

# BMBP
tailscale serve --https=8443 --bg --yes http://127.0.0.1:52615
```

Studio, Spark, and BMBP are intentionally operator-owned, untagged Tailscale devices. Named
Tailscale Services require tagged service hosts and would change device ownership/SSH
policy, so their dedicated endpoints use HTTPS 8443 on the existing device DNS names.
Capture the current configuration before changing it with
`tailscale serve status --json`.

### Validate isolated profiles

For each endpoint, verify `/node_id`, `/agents`, `/state`, `/v1/models`, and a short chat
completion after loading a model. The standalone endpoints must each report exactly one
topology node. The Twin endpoint must report exactly the two Twin nodes and keep both
existing Qwen runners `RunnerReady`.

Do not add bootstrap peers to the standalone profiles unless cross-host placement is an
intentional, reviewed change.
