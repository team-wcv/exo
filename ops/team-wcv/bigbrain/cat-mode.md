# Cat mode

The dashboard's **Cat mode** button runs short inference requests against the
selected loaded text model (or the best running model if none is selected).
It uses that instance's existing placement, including the Twins' tensor/RDMA
configuration. It does not load models or change thermal controls.

Each run lasts at most 10 minutes, sends one request at a time with a 512-token
limit and thinking disabled, and pauses three seconds between requests. It
waits when the instance reports busy. A user request can still arrive between
the readiness check and submission; this is not a priority scheduler.

The API owns the timer, so closing the browser does not end the run. **Stop cat
mode** cancels the active generation. Expiry also cancels any remaining request.
The run ends on generation failure, missing/stale workers, API reset, or API
shutdown. A controller restart leaves Cat mode off; runs are not auto-resumed.
The countdown, request count, selected model, and connection errors are visible
in the dashboard. A lost connection never reports a confirmed stop.

## Single controller

Enable `EXO_CAT_MODE_ENABLED=1` only on Studio's existing EXO API service
(`ai.team-wcv.exo-studio-bigbrain`, port 52415). Keep it unset on worker APIs.
This setting is in `shared/studio.plist`.

The endpoints are `GET`, `POST`, and `DELETE /v1/cat-mode`. POST accepts only
`{"model_id":"<loaded-model-id>"}` and returns 409 for a duplicate job, missing
model, or ambiguous eligible instances. All timestamps are Unix seconds.
The disabled endpoint returns `enabled:false` on GET and 404 on POST.

Both root and `/twins/` dashboard control requests must route to Studio, even
though chat and dashboard assets can fail over between Twins APIs:

- `always-on/Caddyfile.studio.fragment` routes both control paths to Studio.
- `always-on/Caddyfile.bigbrain-alias.fragment` sends those paths through the
  private Bigbrain gateway. Keep these fragments together when deploying.
- Validate/reload Studio Caddy using its actual startup configuration
  `~/.config/caddy/Caddyfile.bigbrain`, which imports `/opt/homebrew/etc/Caddyfile`.

Deploy the API module and dashboard build, set the Studio flag, and reload the
Studio EXO service under the normal restart protocol. No native artifact or
Twins runtime change is needed. Preserve old hashed dashboard assets for tabs
that were open before deployment. Back up affected files before replacing them.

Verify the exact operator URL and the private `/twins/` URL: start in one tab,
observe the same job in another, close the first tab, observe a completed request,
stop from the second tab, and confirm the workers return ready. Check disabled,
busy, failure, expiry, duplicate-start, and cancellation behavior in tests.

Rollback: stop the job, remove the Studio flag, restore the backed-up API and
dashboard files, and reload the scoped API service. The controller routes can
remain in place with the feature disabled, or restore both Caddy fragments and
reload Caddy. Do not alter the Twins' stable MLX environment or RDMA settings.
