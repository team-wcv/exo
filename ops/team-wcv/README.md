# team-wcv EXO Fleet Ops

This directory contains JJ/team-wcv operator assets for running the local EXO
fleet. These files moved here from the retired `team-wcv/big-brain` repo so the
runtime and its operational glue live together.

## Contents

- `launchd/`: per-user macOS launchd supervisor for the `wc-smbp` / `wc-smbpt`
  EXO twin cluster.
- `caddy/`: `bigbrain.localhost:18888` Caddy alias for the active EXO master.
- `docs/`: local fleet topology, proxy, and onboarding notes.

## Boundaries

EXO owns model serving, placement, tensor/RDMA behavior, dashboard state, and
fleet lifecycle.

`team-wcv/cdx` owns Codex-facing UX: `cdx exo`, the EXO TUI, and the Codex/EXO
shim.

The old Big Brain model server, router, and web UI are retired and should not be
revived as part of this directory.

## Install The Twin LaunchAgent

Run this on each supported twin:

```bash
cd ~/Development/Tooling/exo
ops/team-wcv/launchd/install_exo_launchd.sh
```

The service label is intentionally unchanged:

```text
com.teamwcv.exo.bigbrain
```

That preserves the existing installed launchd identity while moving the tracked
source into EXO.
