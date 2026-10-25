# Bigbrain Tailscale ingress

## Host and current acceptance state

Studio's existing Docker container `watch-gateway-edge` owns the tagged
`wc-studio-gw` Tailscale identity (`tag:watch-gateway`). It can host the private
`svc:bigbrain` service while its separate device hostname retains the existing
watch gateway Funnel mapping. Do not add Bigbrain to `AllowFunnel`.

On 2026-10-16 the operator approved this service host. Its `service-host`
capability now includes `svc:bigbrain`, and Tailscale provisioned its TLS
certificate. The HTTPS `/node_id` matches Studio, and `/state`, `/v1/models`,
`/v1/feature-flags`, and `/v1/instance-links` all succeeded through the named URL.
A real Qwen3.8 mixed 4/8-bit code-review request used 1,691 uncached input tokens
and returned 242 output tokens at 40.4 generated tokens/sec. Both Mac ranks logged
successful remote prefill for 1,689 tokens before decoding the final boundary
tokens. Initial prefill and transfer took about 52 seconds; generation throughput
does not include that latency.

Evidence is retained on the operator machine under
`~/.cache/exo/spark-prefill-validation/tailscale-approved-inference-20261016.json`
and `tailscale-approved-rank-proof-20261016.json`. Startup configuration, mounted
configuration, and live service configuration were compared; a shared gateway
restart was not performed during this verification.

The existing Mac identities remain user-owned. The failed service configuration
on untagged smbp is not a working ingress path. Neither this gateway nor Tailscale
changes EXO's LAN prefill or Thunderbolt/JACCL tensor transport.

## Persistence and recovery

Since October 24 the private service targets Studio's Caddy dispatcher on 18889.
Its root and `/twins/` select a healthy MacBook EXO API on 52415; `/spark/`
(and legacy `/laguna/`) and `/studio/` serve the independent
native inference workers. See the [current service runbook](../../always-on/README.md).
The mounted file was edited in place to preserve its bind-mounted inode, and
the live service was reapplied without restarting the shared gateway.

The container has `restart: unless-stopped`, persistent Tailscale state at
`~/.orchestraitor/watch-gateway/state`, and a read-only bind mount of
`~/.orchestraitor/watch-gateway/serve.json` at `/config/serve.json`.
`TS_SERVE_CONFIG` loads that file during container startup. The existing watch
configuration was backed up as `serve.before-bigbrain-20261016.json` before adding
only this member to the top-level `Services` object:

```json
"svc:bigbrain": {
  "TCP": {"443": {"HTTPS": true}},
  "Web": {
    "bigbrain.taile43e67.ts.net:443": {
      "Handlers": {"/": {"Proxy": "http://host.docker.internal:18889"}}
    }
  }
}
```

Preserve all other members, especially the watch gateway's `TCP`, `Web`, and
`AllowFunnel`. Never replace the entire file with the service-only snapshot.
`orc gateway` owns the base configuration: after regenerating it, merge this
member back before a container recreation. Back up the current file first and
check the diff. A full gateway reset or deleted state requires renewed acceptance.

The [service-only snapshot](bigbrain-service.json) is the output of
`tailscale serve get-config --all`, retained for comparison and recovery.
To reapply the live configuration from Studio, use the verified CLI command:

```bash
docker exec watch-gateway-edge tailscale serve --service=svc:bigbrain --https=443 --bg --yes http://host.docker.internal:18889
```

CLI configuration also needs the `Services` entry in the mounted file so startup
configuration does not overwrite it. The existing container need not restart to
apply the live service. Do not restart the shared watch gateway without arranging
its service maintenance.

## Control-plane approval and verification

If a replacement host needs approval, open **Services → bigbrain** in the Tailscale
admin console and approve it. The current `wc-studio-gw` host is already approved.
This is separate from tagging or authorizing the device.
The currently installed Tailscailor MCP exposes no service-host approval action.

Verify the host acceptance and then the actual endpoint:

```bash
docker exec watch-gateway-edge tailscale status --json
docker exec watch-gateway-edge tailscale serve status --json
curl --fail --max-time 15 https://bigbrain.taile43e67.ts.net/node_id
curl --fail --max-time 15 https://bigbrain.taile43e67.ts.net/v1/instance-links
curl --fail --max-time 15 https://bigbrain.taile43e67.ts.net/v1/models
```

The status must include accepted `service-host` capability, and `/node_id` must
match the selected healthy Twin's live API (smbp or smbpt). Studio still owns
the EXO master and gateway. Check `/state` and `/v1/feature-flags`, then send a model
request. For remote-prefill proof use more than 1,000 uncached prompt tokens and
verify successful ingestion on both Mac decode ranks. Repeat endpoint acceptance
after the next approved gateway restart. Do not equate stored configuration or
successful advertisement with host approval or working HTTPS.

## Rollback

Remove only `Services["svc:bigbrain"]` from the current mounted configuration,
then run `docker exec watch-gateway-edge tailscale serve clear svc:bigbrain`.
Verify the service is absent and the original watch mapping remains. If no other
changes occurred, the backup can restore the original configuration after first
reviewing its diff against the current file. Keep the local Caddy endpoint
`http://bigbrain.localhost:18888/` available throughout.
