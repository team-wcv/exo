import json
import os
import re
import sys
import tempfile
import time
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast, final

if TYPE_CHECKING:
    from exo.worker.engines.mlx.vision import VisionProcessor

# Monkey-patch for transformers 5.x compatibility
# Kimi's tokenization_kimi.py imports bytes_to_unicode from the old location
# which was moved in transformers 5.0.0rc2
try:
    import transformers.models.gpt2.tokenization_gpt2 as gpt2_tokenization
    from transformers.convert_slow_tokenizer import bytes_to_unicode

    if not hasattr(gpt2_tokenization, "bytes_to_unicode"):
        gpt2_tokenization.bytes_to_unicode = bytes_to_unicode  # type: ignore[attr-defined]
except ImportError:
    pass  # transformers < 5.0 or bytes_to_unicode not available

from mlx_lm.models.cache import KVCache
from mlx_lm.models.deepseek_v3 import DeepseekV3Model
from mlx_lm.tokenizer_utils import TokenizerWrapper

from exo.shared.models.model_cards import ModelCard, ModelId
from exo.worker.engines.mlx.constants import TRUST_REMOTE_CODE

try:
    from mlx_lm.tokenizer_utils import load_tokenizer
except ImportError:
    from mlx_lm.tokenizer_utils import load as load_tokenizer
import contextlib

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_model
from pydantic import RootModel

from exo.download.download_utils import build_model_path, resolve_existing_model
from exo.shared.types.common import Host
from exo.shared.types.memory import Memory
from exo.shared.types.tasks import TaskId, TextGeneration
from exo.shared.types.text_generation import ChatTemplateValue, TextGenerationTaskParams
from exo.shared.types.worker.instances import (
    BoundInstance,
    MlxJacclInstance,
    MlxRingInstance,
)
from exo.shared.types.worker.runner_response import ModelLoadingResponse
from exo.shared.types.worker.shards import (
    AsymmetricTensorShardMetadata,
    CfgShardMetadata,
    PipelineShardMetadata,
    ShardMetadata,
    TensorShardMetadata,
)
from exo.worker.engines.mlx.asymmetric_parallel import (
    asymmetric_tensor_auto_parallel,
)
from exo.worker.engines.mlx.auto_parallel import (
    get_inner_model,
    get_layers,
    pipeline_auto_parallel,
    tensor_auto_parallel,
)
from exo.worker.engines.mlx.types import Model
from exo.worker.runner.bootstrap import logger


def get_weights_size(model_shard_meta: ShardMetadata) -> Memory:
    if isinstance(model_shard_meta, AsymmetricTensorShardMetadata):
        rank_weight_fraction = (
            model_shard_meta.ratio
            if model_shard_meta.device_rank == 0
            else 1.0 - model_shard_meta.ratio
        )
        return Memory.from_float_kb(
            (model_shard_meta.end_layer - model_shard_meta.start_layer)
            / model_shard_meta.n_layers
            * model_shard_meta.model_card.storage_size.in_kb
            * rank_weight_fraction
        )

    return Memory.from_float_kb(
        (model_shard_meta.end_layer - model_shard_meta.start_layer)
        / model_shard_meta.n_layers
        * model_shard_meta.model_card.storage_size.in_kb
        / (
            1
            if isinstance(model_shard_meta, PipelineShardMetadata)
            else model_shard_meta.world_size
        )
    )


class HostList(RootModel[list[str]]):
    @classmethod
    def from_hosts(cls, hosts: list[Host]) -> "HostList":
        return cls(root=[str(host) for host in hosts])


def _bound_rank(bound_instance: BoundInstance) -> int:
    """Rank of this runner inside the parent ``mx.distributed`` group.

    Target ranks read this from their bound shard metadata; the drafter
    rank reads it from :class:`DrafterPlacement` since the drafter has
    no target shard.
    """
    if bound_instance.is_drafter_rank:
        placement = bound_instance.instance.drafter_placement
        assert placement is not None  # type narrowed by is_drafter_rank
        return placement.drafter_rank
    return bound_instance.bound_shard.device_rank


def mlx_distributed_init(
    bound_instance: BoundInstance,
) -> mx.distributed.Group:
    """Initialize MLX distributed for this rank's parent group.

    The parent group spans every rank declared by the instance: target
    ranks plus, for asymmetric placement, the trailing drafter rank.
    Target ranks split off into a subgroup at runtime via
    :func:`initialize_mlx`; this helper just brings up the parent.
    """
    rank = _bound_rank(bound_instance)
    logger.info(f"Starting initialization for rank {rank}")

    with tempfile.TemporaryDirectory() as tmpdir:
        coordination_file = str(
            Path(tmpdir) / f"hosts_{bound_instance.instance.instance_id}_{rank}.json"
        )
        group: mx.distributed.Group | None = None
        # TODO: singleton instances
        match bound_instance.instance:
            case MlxRingInstance(hosts_by_node=hosts_by_node, ephemeral_port=_):
                hosts_for_node = hosts_by_node[bound_instance.bound_node_id]
                hosts_json = HostList.from_hosts(hosts_for_node).model_dump_json()

                with open(coordination_file, "w") as f:
                    _ = f.write(hosts_json)

                logger.info(
                    f"rank {rank} hostfile: {coordination_file} hosts: {hosts_json}"
                )

                os.environ["MLX_HOSTFILE"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_RING_VERBOSE"] = "1"
                group = mx.distributed.init(backend="ring", strict=True)

            case MlxJacclInstance(
                jaccl_devices=jaccl_devices, jaccl_coordinators=jaccl_coordinators
            ):
                assert all(
                    jaccl_devices[i][i] is None for i in range(len(jaccl_devices))
                )
                jaccl_devices_json = json.dumps(jaccl_devices)

                with open(coordination_file, "w") as f:
                    _ = f.write(jaccl_devices_json)

                jaccl_coordinator = jaccl_coordinators[bound_instance.bound_node_id]

                logger.info(
                    f"rank {rank} MLX_IBV_DEVICES: {coordination_file} with devices: {jaccl_devices_json}"
                )
                logger.info(f"rank {rank} MLX_JACCL_COORDINATOR: {jaccl_coordinator}")
                os.environ["MLX_IBV_DEVICES"] = coordination_file
                os.environ["MLX_RANK"] = str(rank)
                os.environ["MLX_JACCL_COORDINATOR"] = jaccl_coordinator

                max_jaccl_attempts = 8
                for attempt in range(1, max_jaccl_attempts + 1):
                    try:
                        group = mx.distributed.init(backend="jaccl", strict=True)
                        break
                    except (RuntimeError, ValueError) as exc:
                        if attempt == max_jaccl_attempts:
                            raise
                        backoff = min(2.0 * attempt, 10.0)
                        logger.warning(
                            f"rank {rank} JACCL init attempt {attempt}/{max_jaccl_attempts} "
                            f"failed ({exc}), retrying in {backoff:.0f}s"
                        )
                        time.sleep(backoff)

        logger.info(f"Rank {rank} mlx distributed initialization complete")
        if group is None:
            raise RuntimeError("MLX distributed initialization did not return a group")

        return group


@final
@dataclass(frozen=True)
class MlxGroupSplit:
    """Parent ``mx.distributed`` group split into target subgroup + (optional) drafter rank.

    Symmetric placement: ``parent is target_subgroup`` (all ranks are
    target ranks; ``drafter_rank_in_parent`` is ``None``).

    Asymmetric placement: ``parent`` spans N+1 ranks (N target + 1
    drafter at rank ``parent.size() - 1``). Target ranks see a
    ``target_subgroup`` of size N for tensor / pipeline collectives;
    the drafter rank sees ``target_subgroup is None`` because the
    drafter never runs target-side collectives. ``parent`` is reserved
    for ``RemoteTransport.send/recv`` between target rank 0 and the
    drafter rank.

    For V1 N=1 (single target rank, parent_size == 2) the target rank
    also sets ``target_subgroup = None``. ``None`` is the well-known
    "single rank, no collectives needed" signal that
    :func:`load_mlx_items`, :func:`mx_barrier`, :func:`mx_any`, etc.
    already understand and short-circuit on. We don't synthesize a
    size-1 ``mx.distributed.Group`` because MLX's ring and jaccl
    backends do not implement ``Group.split`` on Apple Silicon, and a
    pure-Python stub fails the C++ ``mx.distributed.all_sum`` nanobind
    type check.

    The drafter rank is always last in the parent group, by placement
    convention. This avoids needing to broadcast the drafter rank index
    out of band -- callers can read it from the instance metadata
    (``DrafterPlacement.drafter_rank``) and trust that it equals
    ``parent.size() - 1``.
    """

    parent: mx.distributed.Group
    target_subgroup: mx.distributed.Group | None
    drafter_rank_in_parent: int | None

    @property
    def is_asymmetric(self) -> bool:
        return self.drafter_rank_in_parent is not None


def initialize_mlx(bound_instance: BoundInstance) -> MlxGroupSplit:
    """Bring up the parent group and split off a target subgroup.

    For symmetric placement (no drafter) returns a split where parent
    and target subgroup are the same object. For asymmetric placement
    returns three things in one struct: the parent (used for drafter
    IPC), the target subgroup (used for tensor / pipeline collectives,
    or ``None`` for V1 N=1), and the drafter rank index in the parent
    group.
    """
    # should we unseed it?
    # TODO: pass in seed from params
    mx.random.seed(42)

    parent_size = bound_instance.instance.parent_group_size
    assert parent_size > 1, (
        f"Tried to initialize mlx for a single-rank instance "
        f"(parent_group_size={parent_size}); the single-device runner skips "
        "mx.distributed entirely."
    )
    parent = mlx_distributed_init(bound_instance)
    placement = bound_instance.instance.drafter_placement
    if placement is None:
        return MlxGroupSplit(
            parent=parent,
            target_subgroup=parent,
            drafter_rank_in_parent=None,
        )

    drafter_rank = placement.drafter_rank
    parent_size = parent.size()

    # V1 boundary: single target rank + drafter rank (parent_size == 2).
    # MLX's ring and jaccl backends do not implement ``Group.split`` on
    # Apple Silicon (only the MPI backend does), so we cannot construct
    # a real target-only subgroup. Fortunately V1 only supports a
    # single target rank, where TP=1 / PP=1 means the target never
    # invokes target-subgroup collectives at all -- the existing
    # ``group is None`` short-circuits in ``load_mlx_items``,
    # ``mx_barrier``, and ``mx_any`` cover us.
    #
    # When V2 lands (multi-target asymmetric, parent_size > 2), the
    # target subgroup MUST exclude the drafter rank because TP/PP
    # collectives would otherwise hang waiting on the drafter. At that
    # point either MPI is required or we add a backend-level split
    # implementation; the assertion below pins the V1 contract.
    if parent_size > 2:
        raise NotImplementedError(
            f"Asymmetric placement with multi-target subgroup "
            f"(parent_size={parent_size}) requires a backend that "
            "implements ``Group.split``. MLX's ring and jaccl backends "
            "do not support split on Apple Silicon today; V1 supports a "
            "single target rank (parent_size == 2)."
        )
    return MlxGroupSplit(
        parent=parent,
        target_subgroup=None,
        drafter_rank_in_parent=drafter_rank,
    )


EXO_DISABLE_DRAFTER_ENV = "EXO_DISABLE_DRAFTER"
EXO_DRAFTER_PREFERENCE_ENV = "EXO_DRAFTER_PREFERENCE"

# Allowed values for ``EXO_DRAFTER_PREFERENCE``. ``fastest`` picks the first
# drafter declared on the card (smallest by convention); ``highest_acceptance``
# picks the last (largest by convention); ``auto`` defaults to ``fastest`` but
# may be tuned by future heuristics (e.g. observed acceptance rate).
_DRAFTER_PREFERENCE_VALUES: frozenset[str] = frozenset(
    {"fastest", "highest_acceptance", "auto"}
)


def _drafter_disabled_by_env() -> bool:
    return os.environ.get(EXO_DISABLE_DRAFTER_ENV, "").lower() in {"1", "true", "yes"}


def _drafter_preference() -> str:
    raw = os.environ.get(EXO_DRAFTER_PREFERENCE_ENV, "auto").lower()
    if raw not in _DRAFTER_PREFERENCE_VALUES:
        logger.warning(
            f"Unknown {EXO_DRAFTER_PREFERENCE_ENV}={raw!r}, falling back to 'auto'"
        )
        return "auto"
    return raw


def _select_drafter_id(candidates: list[ModelId], preference: str) -> ModelId | None:
    """Pick a drafter id from a card's preference-ordered list.

    The card lists drafters in `[fastest, ..., highest_acceptance]` order. We
    prefer drafters that are already on disk (so the chooser doesn't force a
    surprise download); within the on-disk subset we honor the user's
    preference. If nothing is on disk we fall back to the head of the list,
    leaving the loader to log a "weights missing" warning.
    """
    if not candidates:
        return None

    on_disk = [cid for cid in candidates if resolve_existing_model(cid) is not None]
    pool = on_disk if on_disk else candidates

    if preference == "highest_acceptance":
        return pool[-1]
    return pool[0]


def _maybe_load_drafter(model_card: ModelCard) -> tuple[ModelId, Model] | None:
    """Load a drafter model declared on ``model_card``, if any.

    Returns the chosen ``(drafter_id, drafter_model)`` pair on success, or
    ``None`` when the card declares no drafter, the chosen drafter's weights
    are not on disk, ``EXO_DISABLE_DRAFTER`` is set, or the load itself
    fails. Drafter loading failures are logged and swallowed: the target
    model continues to load and inference falls back to standard
    (non-speculative) decoding.

    This helper is intentionally single-device only. Multi-device distributed
    inference does not pass ``draft_model`` through to ``stream_generate``
    today (see ``mlx_generate``), so loading a drafter on those ranks would
    just waste memory.
    """
    candidates = list(model_card.drafter_model_ids)
    if not candidates:
        return None
    if _drafter_disabled_by_env():
        logger.info(
            f"Drafter declared by {model_card.model_id} but "
            f"{EXO_DISABLE_DRAFTER_ENV} is set; skipping drafter load."
        )
        return None

    preference = _drafter_preference()
    drafter_id = _select_drafter_id(candidates, preference)
    if drafter_id is None:
        return None

    drafter_path = resolve_existing_model(drafter_id)
    if drafter_path is None:
        logger.warning(
            f"Drafter {drafter_id} (preferred '{preference}') declared by "
            f"{model_card.model_id} is not downloaded; falling back to "
            "standard decoding. Pre-download the drafter to enable "
            "speculative decoding."
        )
        return None

    drafter_start = time.perf_counter()
    try:
        drafter_model, _ = load_model(drafter_path, lazy=True, strict=False)
        mx.eval(drafter_model)
    except Exception as exc:
        logger.opt(exception=exc).warning(
            f"Failed to load drafter {drafter_id}; continuing without "
            "speculative decoding."
        )
        return None
    logger.info(
        f"Loaded drafter {drafter_id} (preferred '{preference}') for "
        f"{model_card.model_id} in {(time.perf_counter() - drafter_start):.2f}s"
    )
    return drafter_id, cast(Model, drafter_model)


def _drafter_weight_size_bytes(drafter_id: ModelId) -> int:
    """Best-effort drafter-on-disk size for the wired-memory bump.

    Walks the drafter directory and sums file sizes. Returns 0 on any error
    (the drafter weights aren't critical-path so we'd rather under-wire than
    crash).
    """
    drafter_path = resolve_existing_model(drafter_id)
    if drafter_path is None:
        return 0
    try:
        return sum(p.stat().st_size for p in drafter_path.rglob("*") if p.is_file())
    except OSError:
        return 0


def load_mlx_items(
    bound_instance: BoundInstance,
    group: mx.distributed.Group | None,
) -> Generator[
    ModelLoadingResponse,
    None,
    tuple[
        Model,
        TokenizerWrapper,
        "VisionProcessor | None",
        Model | None,
        ModelId | None,
    ],
]:
    target_card = bound_instance.bound_shard.model_card
    target_size = get_weights_size(bound_instance.bound_shard)

    # Pre-include drafter size in the wired-memory limit so the OS doesn't
    # page out drafter weights between requests. We have to make this decision
    # *before* loading the target because `set_wired_limit_for_model` configures
    # the limit once. Skip the bump for asymmetric placements: the drafter
    # weights live on a different node so they don't draw from this rank's
    # wired pool.
    combined_size = target_size
    if (
        group is None
        and bound_instance.instance.drafter_placement is None
        and not _drafter_disabled_by_env()
        and target_card.drafter_model_ids
    ):
        chosen = _select_drafter_id(
            list(target_card.drafter_model_ids), _drafter_preference()
        )
        if chosen is not None:
            drafter_bytes = _drafter_weight_size_bytes(chosen)
            if drafter_bytes > 0:
                combined_size = target_size + Memory.from_bytes(drafter_bytes)

    set_wired_limit_for_model(combined_size)

    drafter_model: Model | None = None
    drafter_id: ModelId | None = None

    if group is None:
        logger.info(f"Single device used for {bound_instance.instance}")
        model_path = build_model_path(target_card.model_id)
        start_time = time.perf_counter()
        model, _ = load_model(model_path, lazy=True, strict=False)
        # Eval layers one by one for progress reporting
        try:
            inner = get_inner_model(model)
            layers = get_layers(inner)
            total = len(layers)
            for i, layer in enumerate(layers):
                mx.eval(layer)  # type: ignore
                yield ModelLoadingResponse(layers_loaded=i, total=total)
        except ValueError as e:
            logger.opt(exception=e).debug(
                "Model architecture doesn't support layer-by-layer progress tracking",
            )
        mx.eval(model)
        end_time = time.perf_counter()
        logger.info(f"Time taken to load model: {(end_time - start_time):.2f}s")
        tokenizer = get_tokenizer(model_path, bound_instance.bound_shard)

        # Skip the local in-process drafter when an asymmetric drafter
        # rank exists for this instance: ``DrafterPlacement`` means the
        # drafter is a separate ``DrafterRunner`` reachable via
        # ``RemoteTransport`` over the parent group, and loading a
        # second copy locally would just duplicate the weights and
        # confuse the spec-decode loop.
        if bound_instance.instance.drafter_placement is None:
            drafter_pair = _maybe_load_drafter(target_card)
            if drafter_pair is not None:
                drafter_id, drafter_model = drafter_pair

    else:
        logger.info("Starting distributed init")
        start_time = time.perf_counter()
        model, tokenizer = yield from shard_and_load(
            bound_instance.bound_shard,
            group=group,
        )
        end_time = time.perf_counter()
        logger.info(
            f"Time taken to shard and load model: {(end_time - start_time):.2f}s"
        )

    mx.clear_cache()

    vision_config = bound_instance.bound_shard.model_card.vision

    if vision_config is not None:
        from exo.worker.engines.mlx.vision import VisionProcessor

        vision_start_time = time.perf_counter()
        try:
            vision_processor: VisionProcessor | None = VisionProcessor(
                vision_config, bound_instance.bound_shard.model_card.model_id
            )
            vision_processor.load()
            logger.info(
                f"Time taken to load vision weights: {(time.perf_counter() - vision_start_time):.2f}s"
            )
        except Exception as e:
            logger.opt(exception=e).error(
                "Failed to load vision weights — disabling vision for this runner"
            )
            vision_processor = None
    else:
        vision_processor = None

    return cast(Model, model), tokenizer, vision_processor, drafter_model, drafter_id


def shard_and_load(
    shard_metadata: ShardMetadata,
    group: mx.distributed.Group,
) -> Generator[ModelLoadingResponse, None, tuple[nn.Module, TokenizerWrapper]]:
    model_path = build_model_path(shard_metadata.model_card.model_id)

    model, _ = load_model(model_path, lazy=True, strict=False)
    logger.debug(model)
    if hasattr(model, "model") and isinstance(model.model, DeepseekV3Model):  # type: ignore
        pass
        # TODO: See if we should quantize the model.
        # def is_attention_layer(path: str) -> bool:
        #     path = path.lower()

        #     return "self_attn" in path and "layernorm" not in path

        # def quant_predicate(path: str, module: nn.Module):
        #     if not isinstance(module, nn.Linear):
        #         return False

        #     return is_attention_layer(path)
        # model, config = quantize_model(
        #        model, config, group_size=KV_GROUP_SIZE, bits=ATTENTION_KV_BITS, quant_predicate=quant_predicate, mode=QUANTIZE_MODEL_MODE
        #    )

    assert isinstance(model, nn.Module)

    tokenizer = get_tokenizer(model_path, shard_metadata)

    logger.info(f"Group size: {group.size()}, group rank: {group.rank()}")

    match shard_metadata:
        case TensorShardMetadata():
            logger.info(f"loading model from {model_path} with tensor parallelism")
            model = yield from tensor_auto_parallel(model, group)
        case AsymmetricTensorShardMetadata():
            rank_zero_ratio = shard_metadata.ratio
            ratios_list = [rank_zero_ratio, 1.0 - rank_zero_ratio]
            logger.info(
                f"loading model from {model_path} with asymmetric tensor parallelism "
                f"(ratios={[f'{r:.0%}' for r in ratios_list]})"
            )
            model = yield from asymmetric_tensor_auto_parallel(
                model, group, ratios_list
            )
        case PipelineShardMetadata():
            logger.info(f"loading model from {model_path} with pipeline parallelism")
            model = yield from pipeline_auto_parallel(model, group, shard_metadata)
            mx.eval(model.parameters())
        case CfgShardMetadata():
            raise ValueError(
                "CfgShardMetadata is not supported for text model loading - "
                "this metadata type is only for image generation models"
            )

    # TODO: Do we need this?
    mx.eval(model)

    logger.debug("SHARDED")
    logger.debug(model)

    # Synchronize processes before generation to avoid timeout
    mx_barrier(group)

    return model, tokenizer


def get_tokenizer(model_path: Path, shard_metadata: ShardMetadata) -> TokenizerWrapper:
    """Load tokenizer for a model shard. Delegates to load_tokenizer_for_model_id."""
    return load_tokenizer_for_model_id(
        shard_metadata.model_card.model_id,
        model_path,
        trust_remote_code=shard_metadata.model_card.trust_remote_code,
    )


def get_eos_token_ids_for_model(model_id: ModelId) -> list[int] | None:
    """
    Get the EOS token IDs for a model based on its ID.

    Some models require explicit EOS token configuration that isn't in their
    tokenizer config. This function returns the known EOS token IDs for such models.

    Args:
        model_id: The HuggingFace model ID

    Returns:
        List of EOS token IDs, or None if the model uses standard tokenizer config
    """
    model_id_lower = model_id.lower()
    if "kimi-k2" in model_id_lower:
        return [163586]
    elif "glm-5" in model_id_lower:
        # For GLM-5
        # 154820: <|endoftext|>, 154827: <|user|>, 154829: <|observation|>
        return [154820, 154827, 154829]
    elif "glm-4.7" in model_id_lower:
        # For GLM-4.7
        # 151336: <|user|>, 151329: <|endoftext|>, 151338: <|observation|>
        return [151336, 151329, 151338]
    elif "glm" in model_id_lower:
        # For GLM-4.5 and older
        return [151336, 151329, 151338]
    elif "gpt-oss" in model_id_lower:
        return [200002, 200012]
    elif (
        "qwen3.5" in model_id_lower
        or "qwen-3.5" in model_id_lower
        or "qwen3.6" in model_id_lower
        or "qwen-3.6" in model_id_lower
    ):
        # For Qwen3.5 / Qwen3.6: 248046 (<|im_end|>), 248044 (<|endoftext|>)
        return [248046, 248044]
    elif "gemma-4" in model_id_lower or "gemma-3" in model_id_lower:
        return [1, 106, 50]
    return None


def load_tokenizer_for_model_id(
    model_id: ModelId, model_path: Path, *, trust_remote_code: bool = TRUST_REMOTE_CODE
) -> TokenizerWrapper:
    """
    Load tokenizer for a model given its ID and local path.

    This is the core tokenizer loading logic, handling special cases for different
    model families (Kimi, GLM, etc.) and transformers 5.x compatibility.

    Args:
        model_id: The HuggingFace model ID (e.g., "moonshotai/Kimi-K2-Instruct")
        model_path: Local path where the model/tokenizer files are stored

    Returns:
        TokenizerWrapper instance configured for the model
    """
    model_id_lower = model_id.lower()
    eos_token_ids = get_eos_token_ids_for_model(model_id)

    # Kimi uses a custom TikTokenTokenizer that transformers 5.x can't load via AutoTokenizer
    if "kimi-k2" in model_id_lower:
        import importlib.util
        import types

        sys.path.insert(0, str(model_path))

        # Load tool_declaration_ts first (tokenization_kimi imports it with relative import)
        tool_decl_path = model_path / "tool_declaration_ts.py"
        if tool_decl_path.exists():
            spec = importlib.util.spec_from_file_location(
                "tool_declaration_ts", tool_decl_path
            )
            if spec and spec.loader:
                tool_decl_module = importlib.util.module_from_spec(spec)
                sys.modules["tool_declaration_ts"] = tool_decl_module
                spec.loader.exec_module(tool_decl_module)

        # Load tokenization_kimi with patched source (convert relative to absolute import)
        tok_path = model_path / "tokenization_kimi.py"
        source = tok_path.read_text()
        source = source.replace("from .tool_declaration_ts", "from tool_declaration_ts")
        spec = importlib.util.spec_from_file_location("tokenization_kimi", tok_path)
        if spec:
            tok_module = types.ModuleType("tokenization_kimi")
            tok_module.__file__ = str(tok_path)
            sys.modules["tokenization_kimi"] = tok_module
            exec(compile(source, tok_path, "exec"), tok_module.__dict__)  # noqa: S102
            TikTokenTokenizer = tok_module.TikTokenTokenizer  # type: ignore[attr-defined]  # noqa: N806
        else:
            from tokenization_kimi import TikTokenTokenizer  # type: ignore[import-not-found]  # noqa: I001

        hf_tokenizer: Any = TikTokenTokenizer.from_pretrained(model_path)  # pyright: ignore[reportUnknownVariableType,reportUnknownMemberType]

        # Patch encode to use internal tiktoken model directly
        # transformers 5.x has a bug in the encode->pad path for slow tokenizers
        def _patched_encode(text: str, **_kwargs: object) -> list[int]:
            # Pass allowed_special="all" to handle special tokens like <|im_user|>
            return list(hf_tokenizer.model.encode(text, allowed_special="all"))  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]

        hf_tokenizer.encode = _patched_encode
        return TokenizerWrapper(
            hf_tokenizer,
            eos_token_ids=eos_token_ids,
            tool_call_start="<|tool_calls_section_begin|>",
            tool_call_end="<|tool_calls_section_end|>",
            tool_parser=_parse_kimi_tool_calls,
        )

    # We should really consider going back to mlx lm load to get tokenizer
    tokenizer = load_tokenizer(
        model_path,
        tokenizer_config_extra={"trust_remote_code": trust_remote_code},
        eos_token_ids=eos_token_ids,
    )

    return tokenizer


def _normalize_tool_calls(msg_dict: dict[str, Any]) -> None:
    """Normalize tool_calls in a message dict.

    OpenAI format has tool_calls[].function.arguments as a JSON string,
    but some chat templates (e.g., GLM) expect it as a dict.
    """
    tool_calls = msg_dict.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        return

    for tc in tool_calls:  # pyright: ignore[reportUnknownVariableType]
        if not isinstance(tc, dict):
            continue
        func = tc.get("function")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if not isinstance(func, dict):
            continue
        args = func.get("arguments")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        if isinstance(args, str):
            with contextlib.suppress(json.JSONDecodeError):
                func["arguments"] = json.loads(args)


def _collect_nested_property_names(schema: dict[str, Any]) -> set[str]:
    names: set[str] = set()
    properties: dict[str, Any] = schema.get("properties", {})  # type: ignore[reportAny]
    for prop_spec in properties.values():  # pyright: ignore[reportAny]
        if not isinstance(prop_spec, dict):
            continue
        if prop_spec.get("type") == "array":  # type: ignore[reportAny]
            items: dict[str, Any] | None = prop_spec.get("items")  # type: ignore[reportAny]
            if isinstance(items, dict) and items.get("type") == "object":  # type: ignore[reportAny]
                inner_props: dict[str, Any] = items.get("properties", {})  # type: ignore[reportAny]
                for k in inner_props:  # pyright: ignore[reportUnknownVariableType]
                    names.add(str(k))  # pyright: ignore[reportUnknownArgumentType]
                names.update(_collect_nested_property_names(items))  # pyright: ignore[reportUnknownArgumentType]
    return names


def _schemas_lost_in_prompt(prompt: str, tools: list[dict[str, Any]]) -> bool:
    """Return True if nested property names from any tool schema are absent."""
    for tool in tools:
        fn: dict[str, Any] = tool.get("function", {})  # type: ignore
        params: dict[str, Any] = fn.get("parameters", {})  # type: ignore
        nested = _collect_nested_property_names(params)
        if nested and not all(name in prompt for name in nested):
            return True
    return False


_LOSSY_TEMPLATE_PATTERN = re.compile(
    r"""inner_type\s*==\s*["']object \| object["']\s*or\s*inner_type\|length\s*>\s*\d+""",
)


def _patch_lossy_chat_template(template: str) -> str | None:
    """Patch chat templates that collapse nested object schemas to ``any[]``.

    Some templates (e.g., GPT-OSS) have a guard like::

        inner_type == "object | object" or inner_type|length > 50

    The length check silently drops complex array-of-object schemas.
    We remove the length guard, keeping only the object-union check.
    Returns the patched template, or *None* if no patch was needed.
    """
    patched, n = _LOSSY_TEMPLATE_PATTERN.subn(
        lambda m: m.group(0).split(" or ")[0],  # keep only the object-union check
        template,
    )
    return patched if n > 0 else None


def _needs_dsml_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v3.2" in task_params.model.lower()


def _needs_v4_encoding(task_params: TextGenerationTaskParams) -> bool:
    return "deepseek-v4" in task_params.model.lower()


def _v4_reasoning_effort(task_params: TextGenerationTaskParams) -> str | None:
    effort = task_params.reasoning_effort
    if effort == "xhigh":
        return "max"
    if effort == "high":
        return "high"
    return None


def _strip_v4_thinking_markers(content: str) -> str:
    """Remove `<think>…</think>` blocks and any stray `<think>`/`</think>` tags
    from prior-turn assistant content.

    The V4 encoder drops `reasoning_content` for older turns when
    `drop_thinking=True`"""
    block = re.compile(r"<think>.*?</think>", re.DOTALL)
    if not content:
        return content
    cleaned = block.sub("", content)
    return cleaned.replace("<think>", "").replace("</think>", "")


def consolidate_system_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    System messages almost exclusively must go at the start of a message
    and there must only be a single one.

    Also, Codex sends "developer" messages which are just system prompts.
    """
    system_parts: list[str] = []
    non_system: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") in ("system", "developer"):
            content = cast(str, msg.get("content", ""))
            if content:
                system_parts.append(content)
        else:
            non_system.append(msg)
    formatted_messages = non_system
    if system_parts:
        formatted_messages.insert(
            0, {"role": "system", "content": "\n".join(system_parts)}
        )
    return formatted_messages


def render_chat_template(
    tokenizer: TokenizerWrapper,
    messages: list[dict[str, Any]],
    task_params: TextGenerationTaskParams,
) -> str:
    """
    Convert TextGenerationTaskParams to a chat template prompt.

    Converts the internal format (input + instructions) to a messages list
    that can be processed by the tokenizer's chat template.

    When chat_template_messages is available (from Chat Completions API),
    uses those directly to preserve tool_calls, thinking, and other fields.
    """
    formatted_messages = consolidate_system_messages(messages)

    # For assistant prefilling, append content after templating to avoid a closing turn token.
    partial_assistant_content: str | None = None
    if formatted_messages and formatted_messages[-1].get("role") == "assistant":
        partial_assistant_content = cast(str, formatted_messages[-1].get("content", ""))
        formatted_messages = formatted_messages[:-1]

    if _needs_dsml_encoding(task_params):
        from exo.worker.engines.mlx.vendor.dsml_encoding import encode_messages

        prompt = encode_messages(
            messages=formatted_messages,
            # Only use chat mode if enable thinking is explicitly Fakse.
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            tools=task_params.tools,
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    if _needs_v4_encoding(task_params):
        from exo.worker.engines.mlx.vendor.deepseek_v4_encoding import (
            encode_messages as encode_messages_v4,
        )

        v4_messages = [dict(m) for m in formatted_messages]
        for msg in v4_messages:
            if msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    msg["content"] = _strip_v4_thinking_markers(content)
        if task_params.tools:
            for msg in v4_messages:
                if msg.get("role") in ("system", "developer"):
                    msg["tools"] = task_params.tools
                    break
            else:
                v4_messages.insert(
                    0, {"role": "system", "content": "", "tools": task_params.tools}
                )

        prompt = encode_messages_v4(
            messages=v4_messages,
            thinking_mode="chat"
            if task_params.enable_thinking is False
            else "thinking",
            reasoning_effort=_v4_reasoning_effort(task_params),
        )
        if partial_assistant_content:
            prompt += partial_assistant_content
        return prompt

    for msg in formatted_messages:
        _normalize_tool_calls(msg)

    # Put reasoning content in thinking block for GPT OSS
    if "gpt-oss" in task_params.model.lower():
        for msg in formatted_messages:
            if msg.get("role") == "assistant" and "thinking" not in msg:
                rc = msg.get("reasoning_content")
                if isinstance(rc, str) and rc:
                    msg["thinking"] = rc

    extra_kwargs: dict[str, Any] = {}
    if task_params.enable_thinking is not None:
        # Qwen3 and GLM use "enable_thinking"; DeepSeek uses "thinking".
        # Jinja ignores unknown variables, so passing both is safe.
        extra_kwargs["enable_thinking"] = task_params.enable_thinking
        extra_kwargs["thinking"] = task_params.enable_thinking
    if task_params.reasoning_effort is not None:
        extra_kwargs["reasoning_effort"] = task_params.reasoning_effort

    patched_template: str | None = None
    if task_params.tools:
        original_template: str | None = getattr(tokenizer, "chat_template", None)
        if isinstance(original_template, str):
            patched_template = _patch_lossy_chat_template(original_template)
            if patched_template is not None:
                logger.info(
                    "Patched lossy chat template (removed inner_type length guard)"
                )

    prompt: str = tokenizer.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=task_params.tools,
        **({"chat_template": patched_template} if patched_template is not None else {}),
        **extra_kwargs,
    )

    if task_params.tools and _schemas_lost_in_prompt(prompt, task_params.tools):
        logger.warning("Chat template lost nested tool schemas even after patching")

    if partial_assistant_content:
        prompt += partial_assistant_content

    return prompt


def apply_chat_template(
    tokenizer: TokenizerWrapper,
    task_params: TextGenerationTaskParams,
) -> str:
    messages: list[dict[str, ChatTemplateValue]] = []
    if task_params.chat_template_messages is not None:
        # Use pre-formatted messages that preserve tool_calls, thinking, etc.
        messages = task_params.chat_template_messages
    else:
        # Add system message (instructions) if present
        if task_params.instructions:
            messages.append({"role": "system", "content": task_params.instructions})

        # Convert input to messages
        for msg in task_params.input:
            if not msg.content:
                logger.warning("Received message with empty content, skipping")
                continue
            messages.append({"role": msg.role, "content": msg.content})

    prompt = render_chat_template(tokenizer, messages, task_params)
    logger.debug(prompt)

    return prompt


def system_prompt_token_count(
    task_params: TextGenerationTaskParams,
    tokenizer: TokenizerWrapper,
) -> int:
    """Approximate token count of the system prompt portion of the input."""
    parts: list[str] = []
    if task_params.chat_template_messages is not None:
        for msg in task_params.chat_template_messages:
            if msg.get("role") in ("system", "developer"):
                content = msg.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
    else:
        if task_params.instructions:
            parts.append(task_params.instructions)
        for msg in task_params.input:
            if msg.role in ("system", "developer"):
                parts.append(msg.content)
    if len(parts) == 0:
        return 0
    return len(tokenizer.encode(" ".join(parts), add_special_tokens=False))


def detect_thinking_prompt_suffix(prompt: str, tokenizer: TokenizerWrapper) -> bool:
    """
    Detect if prompt ends with a thinking opening tag that should be
    prepended to the output stream.
    """
    think_token = tokenizer.think_start

    return think_token is not None and prompt.rstrip().endswith(think_token)


def fix_unmatched_think_end_tokens(
    tokens: mx.array, tokenizer: TokenizerWrapper
) -> mx.array:
    if not tokenizer.has_thinking:
        return tokens
    assert tokenizer.think_start_tokens
    assert tokenizer.think_end_tokens
    think_start_tokens: list[int] = tokenizer.think_start_tokens
    think_end_tokens: list[int] = tokenizer.think_end_tokens
    token_list: list[int] = cast(list[int], tokens.tolist())
    result: list[int] = []

    depth = 0
    accumulated_think_start_length = 0
    accumulated_think_end_length = 0

    for token in token_list:
        if token == think_start_tokens[accumulated_think_start_length]:
            accumulated_think_start_length += 1
            if accumulated_think_start_length == len(think_start_tokens):
                depth += 1
                accumulated_think_start_length = 0

        elif token == think_end_tokens[accumulated_think_end_length]:
            accumulated_think_end_length += 1
            if accumulated_think_end_length == len(think_end_tokens):
                if depth == 0:
                    result.extend(think_start_tokens)
                else:
                    depth -= 1
                accumulated_think_end_length = 0

        else:
            accumulated_think_start_length = 0
            accumulated_think_end_length = 0

        result.append(token)
    return mx.array(result)


class NullKVCache(KVCache):
    """
    A KVCache that pretends to exist but holds zero tokens.
    It satisfies .state/.meta_state and never allocates real keys/values.
    """

    def __init__(self, dtype: mx.Dtype = mx.float16):
        super().__init__()
        # zero-length K/V so shapes/dtypes are defined but empty
        self.keys = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.values = mx.zeros((1, 1, 0, 1), dtype=dtype)
        self.offset = 0

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        # matches what mx.save_safetensors / mx.eval expect
        assert self.keys is not None and self.values is not None
        return self.keys, self.values

    @state.setter
    def state(self, v: tuple[mx.array, mx.array]) -> None:
        raise NotImplementedError("We should not be setting a NullKVCache.")


def mlx_force_oom(size: int = 200000) -> None:
    """
    Force an Out-Of-Memory (OOM) error in MLX by performing large tensor operations.
    """
    mx.set_default_device(mx.gpu)
    a = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    b = mx.random.uniform(shape=(size, size), dtype=mx.float32)
    mx.eval(a, b)
    c = mx.matmul(a, b)
    d = mx.matmul(a, c)
    e = mx.matmul(b, c)
    f = mx.sigmoid(d + e)
    mx.eval(f)


def set_wired_limit_for_model(model_size: Memory):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        return

    max_rec_size = Memory.from_bytes(
        int(mx.device_info()["max_recommended_working_set_size"])
    )
    if model_size > 0.9 * max_rec_size:
        logger.warning(
            f"Generating with a model that requires {model_size.in_float_mb:.1f} MB "
            f"which is close to the maximum recommended size of {max_rec_size.in_float_mb:.1f} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    mx.set_wired_limit(max_rec_size.in_bytes)
    logger.info(f"Wired limit set to {max_rec_size}.")


def mlx_cleanup(
    model: Model | None,
    tokenizer: TokenizerWrapper | None,
    group: mx.distributed.Group | None,
) -> None:
    del model, tokenizer, group
    mx.clear_cache()
    import gc

    gc.collect()


def mx_any(bool_: bool, group: mx.distributed.Group | None) -> bool:
    if group is None:
        return bool_
    num_true = mx.distributed.all_sum(
        mx.array(bool_), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
    )
    mx.eval(num_true)
    return num_true.item() > 0


def mx_barrier(group: mx.distributed.Group | None):
    if group is None:
        return
    mx.eval(
        mx.distributed.all_sum(
            mx.array(1.0), group=group, stream=mx.default_stream(mx.Device(mx.cpu))
        )
    )


def _parse_kimi_tool_calls(text: str):
    import regex as re

    # kimi has a fixed function naming scheme, with a json formatted arg
    #   functions.multiply:0<|tool_call_argument_begin|>{"a": 2, "b": 3}
    _func_name_regex = re.compile(
        r"^\s*((?:functions\.)?(.+?):\d+)\s*<\|tool_call_argument_begin\|>", re.DOTALL
    )
    _func_arg_regex = re.compile(r"<\|tool_call_argument_begin\|>\s*(.*)\s*", re.DOTALL)
    _tool_call_split_regex = re.compile(
        r"<\|tool_call_begin\|>(.*?)<\|tool_call_end\|>", re.DOTALL
    )

    def _parse_single_tool(text: str) -> dict[str, Any]:
        func_name_match = _func_name_regex.search(text)
        if func_name_match is None:
            raise ValueError("No tool call found.")
        tool_call_id = func_name_match.group(1)  # e.g. "functions.get_weather:0"
        func_name = func_name_match.group(2)  # e.g. "get_weather"

        func_args_match = _func_arg_regex.search(text)
        if func_args_match is None:
            raise ValueError("No tool call arguments found.")
        func_args = func_args_match.group(1)
        arg_dct = json.loads(func_args)  # pyright: ignore[reportAny]

        return dict(id=tool_call_id, name=func_name, arguments=arg_dct)  # pyright: ignore[reportAny]

    tool_matches = _tool_call_split_regex.findall(text)
    if tool_matches:
        return [_parse_single_tool(match) for match in tool_matches]  # pyright: ignore[reportAny]
    else:
        return [_parse_single_tool(text)]


def mx_all_gather_tasks(
    tasks: list[TextGeneration],
    group: mx.distributed.Group | None,
) -> tuple[list[TextGeneration], list[TextGeneration]]:
    # Single-rank short-circuit. ``mx.distributed.all_gather(group=None)``
    # delegates to the MLX *default* group, which on an asymmetric runner
    # is the parent (target+drafter) group. The drafter rank is busy in
    # ``drafter_serve_loop`` doing its own ``recv`` on that same default
    # group, so an unguarded all-gather here cross-talks with the
    # drafter's wire protocol and corrupts the next command frame the
    # drafter decodes (manifesting as ``num_forwards must be >= 1, got
    # 0``). When ``group is None`` we are by construction the only
    # participating rank, so every task is trivially "agreed".
    if group is None:
        return list(tasks), []

    def encode_task_id(task_id: TaskId) -> list[int]:
        utf8_task_id = task_id.encode()
        return [
            int.from_bytes(utf8_task_id[i : i + 1]) for i in range(len(utf8_task_id))
        ]

    def decode_task_id(encoded_task_id: list[int]) -> TaskId:
        return TaskId(
            bytes.decode(b"".join((x).to_bytes(length=1) for x in encoded_task_id))
        )

    uuid_byte_length = 36

    n_tasks = len(tasks)
    all_counts = cast(
        list[int],
        mx.distributed.all_gather(mx.array([n_tasks]), group=group).tolist(),
    )
    max_tasks = max(all_counts)
    world_size: int = group.size()

    if max_tasks == 0:
        return [], []

    padded = [encode_task_id(task.task_id) for task in tasks] + [
        [0] * uuid_byte_length
    ] * (max_tasks - n_tasks)

    assert all(len(encoded_task_id) == uuid_byte_length for encoded_task_id in padded)

    gathered = cast(
        list[list[list[int]]],
        mx.distributed.all_gather(mx.array(padded), group=group)
        .reshape(world_size, max_tasks, -1)
        .tolist(),
    )
    all_task_ids: list[list[TaskId]] = [
        [decode_task_id(encoded_task_id) for encoded_task_id in rank_tasks[:count]]
        for rank_tasks, count in zip(gathered, all_counts, strict=True)
    ]

    agreed_ids = set[TaskId].intersection(*(set(tids) for tids in all_task_ids))

    local_tasks = {task.task_id: task for task in tasks}
    agreed = [local_tasks[tid] for tid in sorted(agreed_ids)]
    different = [task for task in tasks if task.task_id not in agreed_ids]
    return agreed, different
