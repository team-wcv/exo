# Big Brain localhost proxy

`bigbrain.localhost:18888` is a wc-studio convenience alias for the EXO
dashboard and OpenAI-compatible API. It is a local Caddy reverse proxy; it is
not part of the EXO tensor/RDMA path.

## Current target

The proxy must point directly at the EXO master LAN/control address:

```text
http://bigbrain.localhost:18888 -> http://192.168.1.63:52415
```

Do not proxy to `wc-smbp:52415`, `wc-smbp.local:52415`, or a Tailscale name.
On wc-studio, `wc-smbp` can resolve through Tailscale MagicDNS, for example:

```text
wc-smbp.taile43e67.ts.net -> 100.82.178.117
```

That path is not the Big Brain data/control path and can produce dashboard
`502` errors, slow model startup, or misleading operator symptoms while EXO is
healthy on the LAN address.

## Install or repair

Use the tracked Caddyfile as the source of truth:

```bash
mkdir -p ~/.config/caddy
cp ~/Development/Tooling/exo/ops/team-wcv/caddy/bigbrain.localhost.Caddyfile \
  ~/.config/caddy/Caddyfile.bigbrain
```

Point the user LaunchAgent at that file:

```text
~/Library/LaunchAgents/homebrew.mxcl.caddy.plist
ProgramArguments:
  /opt/homebrew/opt/caddy/bin/caddy
  run
  --config
  /Users/JJ/.config/caddy/Caddyfile.bigbrain
```

Then reload or restart Caddy from a normal user shell:

```bash
/opt/homebrew/opt/caddy/bin/caddy validate \
  --config ~/.config/caddy/Caddyfile.bigbrain \
  --adapter caddyfile

launchctl kickstart -k "gui/$(id -u)/homebrew.mxcl.caddy"
```

If `launchctl` is unavailable from an agent sandbox, the Caddy admin API can
repair the live process without touching EXO:

```bash
/opt/homebrew/opt/caddy/bin/caddy adapt \
  --config ~/.config/caddy/Caddyfile.bigbrain \
  --adapter caddyfile \
  --pretty > /tmp/caddy-bigbrain-runtime.json

curl -sS -X POST \
  -H 'Content-Type: application/json' \
  --data-binary @/tmp/caddy-bigbrain-runtime.json \
  http://localhost:2019/load
```

## Verify

Check the backend first:

```bash
curl -sS -o /dev/null -w 'direct=%{http_code}\n' \
  http://192.168.1.63:52415/
```

Then check the alias:

```bash
curl -sS -o /dev/null -w 'root=%{http_code}\n' \
  http://bigbrain.localhost:18888/
curl -sS -o /dev/null -w 'state=%{http_code}\n' \
  http://bigbrain.localhost:18888/state
curl -sS -o /dev/null -w 'models=%{http_code}\n' \
  http://bigbrain.localhost:18888/v1/models
```

Expected result:

```text
direct=200
root=200
state=200
models=200
```

The EXO state should still show the twin topology using LAN control and TB/RDMA
links:

```text
LAN control:
  wc-smbp  192.168.1.63
  wc-smbpt 192.168.1.120

RDMA:
  wc-smbp  rdma_en2 <-> wc-smbpt rdma_en1
```

## Guardrail

Tailscale remains useful for other operator workflows, but do not let the
Big Brain dashboard proxy choose a Tailscale upstream. The EXO cluster should
use LAN for coordination/control and Thunderbolt/RDMA for tensor traffic.
