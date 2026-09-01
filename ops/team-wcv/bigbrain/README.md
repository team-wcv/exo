# Isolated Bigbrain endpoints

This profile keeps three Exo control planes active at the same time:

| Tailnet service | Owner | Local API | Namespace | Placement boundary |
|---|---|---:|---|---|
| `svc:bigbrain` | `wc-smbp` | `52415` | Twin production cluster | `wc-smbp` + `wc-smbpt` only |
| `svc:studio-bigbrain` | `wc-studio` | `52615` | `bigbrain-studio` | Studio only |
| `svc:spark-bigbrain` | `wc-spark` | `52615` | `bigbrain-spark` | Spark only |

Spark's Laguna vLLM endpoint on port 8888 is independent of Exo and remains active. A
Spark-local Exo endpoint can coexist with Laguna as a control plane, but the host does not
have enough free unified memory to load another heavyweight model while Laguna is loaded.

## Install the host services

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

Both profiles use the hardware-tested PR44 runtime at commit
`0e9c429f2afee6f2412c1bd07787119968ceba85`. Update the absolute runtime path in the
service file only after validating a replacement build on the target host.

## Publish Tailnet services

```bash
# Studio
tailscale serve --service=svc:studio-bigbrain --bg --yes http://127.0.0.1:52615

# Spark
tailscale serve --service=svc:spark-bigbrain --bg --yes http://127.0.0.1:52615
```

The tailnet policy must grant the host permission to advertise the service. Capture the
current configuration before changing it with `tailscale serve status --json` or
`tailscale serve get-config <path> --all`.

## Validation

For each endpoint, verify `/node_id`, `/agents`, `/state`, `/v1/models`, and a short chat
completion after loading a model. The standalone endpoints must each report exactly one
topology node. The Twin endpoint must report exactly the two Twin nodes and keep both
existing Qwen runners `RunnerReady`.

Do not add bootstrap peers to the standalone profiles unless cross-host placement is an
intentional, reviewed change.
