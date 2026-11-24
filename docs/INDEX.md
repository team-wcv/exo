# EXO documentation index

<a id="branch-state"></a>
## Branch State

Team-wcv operational repair branch: `fix/6b053e26-control-delivery`, based on
`main` commit `43bfab7`. Refreshed November 24, 2026. Production acceptance is
recorded in the linked operational task, separately from source validation.

<a id="quick-entry-points"></a>
## Quick Entry Points

- [API reference](api.md)
- [Fleet operations](../ops/team-wcv/README.md)
- [Bigbrain deployment](../ops/team-wcv/bigbrain/README.md)
- [Control-delivery repair](../ops/team-wcv/bigbrain/control-delivery-20261124.md)

<a id="documentation-surface-map"></a>
## Documentation Surface Map

Upstream technical documents live here. Team-wcv startup, deployment, model
placement, acceptance and rollback records live under `ops/team-wcv/`.

<a id="code-surface-map"></a>
## Code Surface Map

- `rust/networking/src/swarm.rs`: network operations and inbound event delivery.
- `src/exo/routing/event_router.py`: event ordering, retries and acknowledgements.
- `src/exo/api/adapters/responses.py`: Responses streaming and collected output.

<a id="agent-navigation"></a>
## Agent Navigation

Read the fleet runbooks before changing active startup files or native modules.
Match Python and native artifacts; verify actual inference after deployment.

<a id="todo-audit"></a>
## TODO Audit

This repair introduces no TODO markers. Existing independent TODOs are retained.

<a id="open-risks"></a>
## Open Risks

The blocked-publish regression establishes a receive-starvation defect; live
Metal failures, scheduling load and fleet stability need separate observations.
Retain the configured context windows and tensor/RDMA transport during testing.
