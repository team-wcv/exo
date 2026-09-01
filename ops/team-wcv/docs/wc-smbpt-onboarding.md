# wc-smbpt onboarding playbook

Step-by-step procedure to bring up the second MacBook (`wc-smbpt`) so it
slots cleanly into:

- the `dotfiles` operator fleet (joins `wc-smbp`, `wc-bmbp`, `wc-studio`,
  `wc-mba` as a peer)
- the Syncthing `~/.agents` + `~/.codex` real-time sync mesh
- the `Big Brain` 2-node JACCL cluster as **rank1**

This guide does NOT redefine the canonical workstation procedure — it
**sequences** the existing artifacts in the only order that avoids
known traps. Read each linked doc when you reach the step that points
to it.

## TL;DR

Do **not** restore `wc-smbpt` from a Time Machine backup of `wc-smbp`,
and do **not** use Migration Assistant against another tailnet-joined
Mac. Bring it up clean. Then run the dotfiles installer, join Tailscale
with its own hostname, then bootstrap the Big Brain role.

```
A. Out-of-box (clean) -> B. Prereqs -> C. Dotfiles -> D. Tailscale -> E. Sync mesh -> F. SSH wiring -> G. Big Brain cluster role -> H. Recovery RDMA -> I. Verify
```

## A. CRITICAL — what NOT to do

The single most expensive mistake on this onboarding is creating a
duplicate Tailscale node identity. It happens whenever `wc-smbpt`
inherits another Mac's `/Library/Tailscale` daemon state — Time Machine
restores, Migration Assistant transfers, and broad
`sync-workstation.sh --switch-ready` runs all carry that risk.

**Do not:**

- Restore `wc-smbpt` from `wc-smbp`'s Time Machine backup.
- Run Migration Assistant from `wc-smbp` -> `wc-smbpt` over Wi-Fi/cable.
- Copy `/Library/Tailscale` between Macs by any mechanism.
- Run `sync-workstation.sh --switch-ready` from `wc-smbp` -> `wc-smbpt`
  before Tailscale on `wc-smbpt` has its own distinct node identity.

**Do:**

- Boot `wc-smbpt` clean from the macOS installer (or Setup Assistant
  with a fresh user, no Migration Assistant).
- Sign in to your Apple ID, but skip "transfer information from another
  Mac".
- Bring up Tailscale on `wc-smbpt` with `--hostname=wc-smbpt` BEFORE any
  bulk sync from another Mac.

If you ever see two Macs share a Tailscale `ID` or `PublicKey`, the
issue is daemon-level and the recovery is in
[`~/Development/Tooling/dotfiles/docs/TAILSCALE-DEVICE-CONFLICT-RECOVERY.md`](../../dotfiles/docs/TAILSCALE-DEVICE-CONFLICT-RECOVERY.md).
That doc is the canonical fix; this guide only points at it.

## B. Prereqs (manual; can't be scripted)

Done physically on `wc-smbpt`:

1. Power on, run Setup Assistant.
2. Pick **"Don't transfer any information now"** when asked.
3. Sign in with your Apple ID.
4. Set the names so they match the canonical fleet identity:
   ```bash
   sudo scutil --set ComputerName  wc-smbpt
   sudo scutil --set LocalHostName wc-smbpt
   sudo scutil --set HostName      wc-smbpt
   ```
   (`WORKSTATION-SYNC.md` requires the macOS host name and the
   Tailscale device name to match, fleet-wide.)
5. Install Xcode Command Line Tools:
   ```bash
   xcode-select --install
   ```
6. Install Homebrew:
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
7. Generate (do NOT copy) a fresh ed25519 SSH key:
   ```bash
   ssh-keygen -t ed25519 -C "JJ@wc-smbpt" -f ~/.ssh/id_ed25519
   ```
   Add the public key to GitHub (Settings -> SSH keys) so the dotfiles
   clone in step C works.

## C. Dotfiles installer (this brings most of the workstation up)

This is the canonical fleet bring-up that already exists. Follow
[`docs/WORKSTATION-SYNC.md` -> "New Machine Bring-Up"](../../dotfiles/docs/WORKSTATION-SYNC.md#new-machine-bring-up)
verbatim. The short version, on `wc-smbpt`:

```bash
mkdir -p ~/Development/Tooling
cd ~/Development/Tooling
git clone git@github.com:team-wcv/dotfiles.git
cd dotfiles
./scripts/install.sh
```

The installer will:

- symlink `~/.agents`, `~/.cursor`, `~/.codex`, shell config, git
  config, `~/.ssh/config`, tmux, Hammerspoon, Terminal profiles
- render MCP templates
- install `~/.githooks`

After install:

```bash
orc config reconcile --force
```

Then apply the curated software inventory:

```bash
brew bundle --file=~/Development/Tooling/dotfiles/brew/Brewfile.essential
# Optional, full studio set:
brew bundle --file=~/Development/Tooling/dotfiles/Brewfile
```

`Brewfile.essential` includes Tailscale, Codex, Cursor, Docker, Ollama,
node, python, jq, etc. — everything `wc-smbpt` needs to participate.

## D. Tailscale — clean join with its own identity

Order matters. Tailscale must come up on `wc-smbpt` BEFORE you run any
broad sync from another Mac.

1. Tailscale was installed by the Brewfile. Start the daemon:
   ```bash
   sudo brew services start tailscale
   ```
2. Mint a one-shot auth key on the operator Mac via Tailscailor (no
   browser flow needed). The `tailscailor` MCP / skill at
   [`~/.agents/skills/tailscailor/SKILL.md`](~/.agents/skills/tailscailor/SKILL.md)
   mints reusable, scoped keys — see that skill for the current verb
   set. Otherwise:
   - https://login.tailscale.com/admin/settings/keys -> "Generate auth
     key" (one-time, ephemeral OFF, tagged `tag:operator-mac`)
3. Join with the explicit hostname so MagicDNS lands at `wc-smbpt`
   without a `-1` suffix:
   ```bash
   tailscale up \
     --auth-key=tskey-auth-... \
     --hostname=wc-smbpt \
     --ssh
   ```
4. Verify the new node is distinct from every other Mac in the fleet:
   ```bash
   python3 - <<'PY'
   import json, subprocess
   self = json.loads(subprocess.check_output(["tailscale","status","--json"]))["Self"]
   print("ID         :", self["ID"])
   print("PublicKey  :", self["PublicKey"])
   print("HostName   :", self["HostName"])
   print("DNSName    :", self["DNSName"])
   for ip in self["TailscaleIPs"]: print("IP         :", ip)
   PY
   ```
   Cross-check against `wc-smbp`'s `Self.ID` / `Self.PublicKey` (they
   MUST differ). If they match, stop and run the recovery in
   [`TAILSCALE-DEVICE-CONFLICT-RECOVERY.md`](../../dotfiles/docs/TAILSCALE-DEVICE-CONFLICT-RECOVERY.md).
5. From the operator Mac, scan the tailnet and remove any stale
   `wc-smbpt` placeholders or duplicates that may have been auto-created:
   ```bash
   ~/Development/Tooling/Homelab/skills/homelab/scripts/remove_tailnet_device.sh wc-smbpt-1
   # Or interactively via the Tailscale admin console.
   ```

## E. Real-time sync mesh (Syncthing for ~/.agents and ~/.codex)

Once Tailscale on `wc-smbpt` has its own identity, opt into the
multi-Mac live sync (per
[`WORKSTATION-SYNC.md` -> "Real-Time Sync (Syncthing)"](../../dotfiles/docs/WORKSTATION-SYNC.md#real-time-sync-syncthing)):

```bash
orc config sync-setup
```

This syncs:

- `~/.agents/skills`, `~/.agents/rules`, `~/.agents/agents`, `AGENTS.md`,
  `agentkit.json`
- `~/.codex/config.toml`, `~/.codex/rules/`

It deliberately does NOT sync auth tokens, sqlite caches, bastion
state, or Cursor reconciler outputs — same allowlist as the rest of the
fleet.

After `wc-smbpt` joins the Syncthing mesh, run:

```bash
orc config reconcile-watch
```

so any incoming changes from other Macs reconcile into Cursor/Codex on
`wc-smbpt` automatically.

## F. SSH wiring on the operator Mac

On the operator Mac (this MacBook), add `wc-smbpt` to `~/.ssh/config`
mirroring the `wc-smbp` pattern. The dotfiles `home/.ssh/config`
already follows this convention; add this block to it (or to your local
`~/.ssh/config` if not yet committed):

```sshconfig
Host wc-smbpt smbpt smbpt-lan
HostName WC-Smbpt.local
User JJ
IdentityFile ~/.ssh/id_ed25519
IdentitiesOnly yes
StrictHostKeyChecking accept-new
ServerAliveInterval 30
ServerAliveCountMax 3

Host wc-smbpt-ts smbpt-ts
HostName wc-smbpt.taile43e67.ts.net
User JJ
IdentityFile ~/.ssh/id_ed25519
IdentitiesOnly yes
StrictHostKeyChecking accept-new
ServerAliveInterval 30
ServerAliveCountMax 3
```

Confirm both lanes work from the operator Mac:

```bash
ssh -o BatchMode=yes wc-smbpt 'hostname && uname -srm'
ssh -o BatchMode=yes wc-smbpt-ts 'hostname && uname -srm'
```

Then add the operator Mac's pubkey to `wc-smbpt` so passwordless ssh
works. Easiest path:

```bash
ssh-copy-id wc-smbpt
```

And — crucially for the cluster — make sure `wc-smbp` (rank0) can
passwordless-ssh to `wc-smbpt` (rank1). `mlx.launch` and
`bb model sync` both depend on it:

```bash
ssh wc-smbp 'ssh-copy-id wc-smbpt'
ssh wc-smbp 'ssh -o BatchMode=yes wc-smbpt true && echo OK'
```

## G. Historical Big Brain role install

This section is retained for historical onboarding context. The active EXO
runtime now lives in `~/Development/Tooling/exo`, and the supported twin
supervisor is `ops/team-wcv/launchd/teamwcv-exo-bigbrain`.

The old Big Brain helper flow was `./bin/bb bootstrap wc-smbpt` from the
retired Big Brain repo.

This installs (idempotently): `coreutils rsync jq git wget curl hf`,
the `miniforge` cask, the `mlxjccl` conda env (Python 3.12), and pip
installs `mlx>=0.30.4 mlx-lm==0.30.5 fastapi uvicorn
transformers==5.0.0rc3 tokenizers mistral_common`. It also creates
`~/.lmstudio/models`, the default model root. Same shape that ran on
`wc-smbp`.

Confirm:

```bash
./bin/bb status
# expect: wc-smbpt ssh=ok rdma=disabled env=present models-dir=present rank1
```

## H. Recovery-mode RDMA enable (one-time, BOTH Macs)

This is the only part that can't be done over SSH. JACCL needs RDMA
over Thunderbolt enabled at the OS level, and that switch lives in
macOS Recovery.

The old helper rendered recovery instructions with `./bin/bb rdma instructions`.
If that CLI is not present, use Apple's Recovery terminal directly and run
`rdma_ctl enable`.

Then on each Mac (`wc-smbp` and `wc-smbpt`), physically:

1. Apple menu -> Restart, hold power until "Loading startup options".
2. Options -> Continue -> Utilities -> Terminal.
3. `rdma_ctl enable`
4. Reboot normally.
5. Verify in macOS: `rdma_ctl status` -> `enabled`, `ibv_devices` ->
   `rdma_en*` device(s).

After both Macs are back up:

Historical post-boot checks were `./bin/bb rdma verify` and
`./bin/bb topology check`. In the current EXO fleet, validate with `rdma_ctl
status`, `ibv_devices`, and the EXO dashboard/topology state.

The topology check should show the direct TB5 cable as `Status: Device
connected`, `Speed: 80 Gb/s` (TB4) or `Speed: 120 Gb/s` (TB5),
`Device Name: MacBook Pro` on the peer side.

## I. Update cluster.json + verify cluster

Legacy Big Brain installs edited [`config/cluster.json`](../config/cluster.json):

- `wc-smbpt`'s `ips[0]` = its LAN IP from `tailscale status` or
  `ifconfig en11 inet | awk '/inet / {print $2}'`
- `rdma` matrix entries = the actual `rdma_enN` device names from
  `ibv_devices` on each Mac

Then:

```bash
./bin/bb cluster verify
./bin/bb model sync qwen3-4b-instruct-4bit
./bin/bb bench --smoke
./bin/bb model sync qwen2.5-coder-32b-4bit
./bin/bb use qwen2.5-coder-32b-4bit
```

Pre-pulled on `wc-smbp` already, so the syncs are LAN-local and fast.

`bb use` returns the OpenAI base URL — wire Cursor and Codex CLI to
it (see [`big-brain.md`](big-brain.md)).

## J. Final readiness audit

Run the canonical doctor against the new Mac (from the operator Mac):

```bash
cd ~/Development/Tooling/dotfiles
./scripts/doctor-workstation.sh --host wc-smbpt.local --user JJ --expect-host wc-smbpt
```

It checks: host naming + Tailscale identity, core CLI availability,
Docker daemon, GitHub CLI auth, homelab SSH reachability, key launch
agents, expected NAS mounts, canonical `~/Development` repo roots.

And run the cluster doctor:

```bash
cd ~/Development/Tooling/exo
git status --short --branch
```

## What's intentionally NOT here

- macOS configuration steps that already live in
  [`WORKSTATION-SYNC.md`](../../dotfiles/docs/WORKSTATION-SYNC.md) — go
  read that doc, this guide just sequences it.
- Bulk `sync-workstation.sh --switch-ready` instructions — `wc-smbpt`
  is NEW, not a switchable peer of `wc-smbp` yet. Do that later, after
  Tailscale and Syncthing are clean, if you ever want fast cutover
  between the two.
- Time Machine restore steps — explicitly out of scope; see section A.
- Anything specific to Salesforce, signing, provisioning — restore
  those local-only items per `WORKSTATION-SYNC.md` -> "Local" section.

## Follow-up TODOs (operator-side, not blocking)

- Add `wc-smbpt` to `WORKSTATION-SYNC.md` -> "Current Mac Fleet" list.
- Add `wc-smbpt` to `TAILSCALE-DEVICE-CONFLICT-RECOVERY.md` ->
  "Typical target names in this setup".
- If the dotfiles repo's `home/.ssh/config` is the canonical source,
  commit the `wc-smbpt` / `wc-smbpt-ts` blocks there instead of editing
  `~/.ssh/config` by hand.

## See also

- [`AGENTS.md`](../AGENTS.md) — repo-level agent rules
- [`docs/from-scratch-2node.md`](from-scratch-2node.md) — 2-node JACCL walkthrough
- [`docs/agent-runbook.md`](agent-runbook.md) — failure modes
- [`docs/topology.md`](topology.md) — Thunderbolt cabling
- `~/Development/Tooling/dotfiles/docs/WORKSTATION-SYNC.md`
- `~/Development/Tooling/dotfiles/docs/TAILSCALE-DEVICE-CONFLICT-RECOVERY.md`
- `~/.agents/skills/tailscailor/SKILL.md`
- `~/Development/Tooling/Homelab/skills/homelab/scripts/remove_tailnet_device.sh`
