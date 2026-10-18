# Bigbrain EXO fleet

## Current shared deployment (2026-10-16)

The operator dashboard at `http://bigbrain.localhost:18888/` serves one shared
`wc-bigbrain` cluster. Studio is the preferred master; smbp, smbpt, and Spark
join the same cluster. All four APIs use port `52415`, and peer-file servers use
`52416`. Using different API ports breaks the deployed reachability probes even
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

The standalone Spark Laguna unit `vllm-laguna-current-256k.service` is stopped and
disabled with operator approval to make memory available to EXO. Its restore
configuration is recorded in the earlier `exo-qwen27b-20261016` deployment backup.

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

Both LAN subnets are allowed for API reachability. Studio offers bootstrap
addresses `192.168.1.21:52618` and `192.168.0.33:52618`; MacBooks also bootstrap
over the direct Thunderbolt link. Verify the configured addresses after network
changes. A healthy Tailscale connection does not prove the inference LAN path.

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
