# Qwen Twins: opt-in YaRN context extension

Task: `6b03e17e3e896bda5f6343df`.
Target: 524,288 total tokens on the two tensor/JACCL Twins.
The built-in model card remains 262,144 because unmodified checkpoints still
use default rotary embeddings. Never advertise a larger window on an unscaled
runtime or infer support just from a client context setting.

## Configuration and implementation

For the deployed `pipenetwork/Qwen3.8-Flash-Next-MLX-mixed-4_8bit` checkpoint,
preserve the existing `text_config.rope_parameters` fields and set:

```json
{
  "max_position_embeddings": 524288,
  "rope_parameters": {
    "rope_type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 262144,
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10]
  }
}
```

This fragment belongs inside `text_config`, not at the model config root.
[Qwen's guidance](https://huggingface.co/Qwen/Qwen3.8-Flash-Next#best-practices)
describes the 2x configuration. Static scaling can affect shorter-context
quality, so keep the unscaled configuration for rollback.

The bundled remote-code model ignores `rope_type`. EXO's `apply_qwen_yarn`
replaces its shared rotary object immediately after loading and before tensor
or pipeline sharding. Both full attention and the sparse-attention indexer use
that object. Default rotary checkpoints are untouched. Unsupported scaling
modes and malformed parameters fail rather than silently running unscaled.
No model weights, quantization settings or native network binaries change.

The frequency blend and amplitude match Transformers' public YaRN registry.
The tiny float32 frequency table is initialized on the CPU because Metal fast
power rounding can produce phase error at long absolute positions. Rotary
evaluation and inference remain on the model's MLX device.

Extended Qwen prefill uses 512-token chunks instead of 4,096, bounding the
dense masked-attention working set at long contexts. Recurrent checkpoint
history is thinned above 64 records while retaining the first and latest two
exact states. Prefix-cache reuse may replay a longer suffix; rollback never
uses interpolated or fabricated states. Other models retain existing settings.

## Fleet deployment record

On each Twin, deployment records live at:

```text
~/.orchestraitor/deployments/qwen-yarn-512k-20261122/
```

`before/` holds the original model config and loader. `applied.json` records
before/after hashes. Loader changes are applied narrowly to the live overlay;
do not replace the complete runtime with a different checkout and lose unrelated
local fixes. The serving venv and native MLX/exo_rs artifacts remain unchanged.

Stop admission to the model before changing its live configuration. Check the
runner IDs belonging to the current instance; historical Shutdown tasks may
remain in the event state. Temporarily pause the smbpt fallback watcher only
during instance replacement, then restore its prior state. Recreate the model
with smbpt rank 0 and the existing Thunderbolt/JACCL device matrix. Verify both
runner logs show YaRN enabled and that a short real request succeeds.

Only after a real extended-context request succeeds should the fleet-specific
model card advertise 524,288. Keep its previous card for rollback; do not raise
the repository's default card for installations without this config. The
standalone Laguna fallback has its own context capacity and does not acquire
the Twins' extension automatically.

## Validation

The rotary suite checks Transformers parity at factors 1/2/4, optional scaling
parameters, positions through 524,287, cached/batched positions, unchanged
default behavior, malformed configs and checkpoint-thinning invariants.

`check_long_context.py` submits a prepared retrieval prompt through the real
Twins route and records progress, memory samples, usage and the exact response.
It stops if either Twin has less than 16 GiB available or adds over 2 GiB swap.
Inputs and receipts belong in a private deployment/evidence directory, not Git.
The prepared test has approximately 500K tokens and three exact synthetic codes
near the beginning, middle and end. A failed or interrupted test is not proof
of a usable 512K window.

## Rollback

Wait for the test/user request to finish or explicitly cancel it. Remove the
current model instance through the normal API. Compare current files with the
deployment manifest before restoring anything; stop if another change has
superseded these files. Restore the saved config, loader and prefill implementation,
remove the newly introduced helper only when its hash matches this deployment,
and restore any fleet-specific model-card override to 262,144. Recreate the
two-rank instance, restore the watcher state and verify a real request. Do not
restore old weights, erase event history or restart unrelated services.
