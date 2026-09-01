# Chat inference engine wrapping exllamav3's Generator/Job API.
#
# Adapted from exllamav3 examples/chat.py
# Original author: turboderp (https://github.com/turboderp-org/exllamav3)

from __future__ import annotations

import asyncio
import functools
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import torch

from .templates import prompt_formats, infer_mode


# ---------------------------------------------------------------------------
# Defensive exllamav3 symbol resolution
# ---------------------------------------------------------------------------
# Some exllamav3 installs (e.g. partial / editable builds inside containers)
# resolve the package as a PEP-420 namespace package, so the top-level
# re-exports from exllamav3/__init__.py aren't available. These helpers try
# the top-level first, then fall back to explicit submodule paths.

def _import_model_init():
    try:
        from exllamav3 import model_init  # type: ignore
        return model_init
    except ImportError:
        import exllamav3.model_init as model_init  # type: ignore
        return model_init


def _import_generator():
    try:
        from exllamav3 import Generator  # type: ignore
        return Generator
    except ImportError:
        from exllamav3.generator import Generator  # type: ignore
        return Generator


def _import_job():
    try:
        from exllamav3 import Job  # type: ignore
        return Job
    except ImportError:
        from exllamav3.generator import Job  # type: ignore
        return Job


def _import_config():
    try:
        from exllamav3 import Config  # type: ignore
        return Config
    except ImportError:
        from exllamav3.model import Config  # type: ignore
        return Config


def _import_model():
    try:
        from exllamav3 import Model  # type: ignore
        return Model
    except ImportError:
        from exllamav3.model import Model  # type: ignore
        return Model


def _import_cache():
    try:
        from exllamav3 import Cache  # type: ignore
        return Cache
    except ImportError:
        from exllamav3.cache import Cache  # type: ignore
        return Cache


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

@dataclass
class ChatSettings:
    system_prompt: str = ""
    max_response_tokens: int = 1000
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    think: bool = False
    no_think: bool = False
    think_budget: Optional[int] = None
    amnesia: bool = False
    banned_strings: List[str] = field(default_factory=list)
    mode: str = "chatml"
    user_name: str = "User"
    bot_name: str = "Assistant"

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChatSettings":
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Chat engine
# ---------------------------------------------------------------------------

class ChatEngine:
    """Wraps exllamav3 Generator/Job for streaming chat inference."""

    # Sensible defaults so we don't blow VRAM on a 128k model card context
    DEFAULT_CACHE_SIZE = 32768      # must be multiple of 256
    DEFAULT_CACHE_QUANT = "6,6"     # Q6 for both K and V

    # Recurrent (hybrid linear-attn / SWA-state) models clamp the
    # generator's batch size to the cache's recurrent state slots, which
    # default to 1 — serializing DPO duel candidates. Two slots cover the
    # n=2 duel; each extra slot costs real VRAM (~1.3 GB on a 31B Gemma4).
    BATCH_SLOTS = 2

    def __init__(
        self,
        model_dir: str | None = None,
        devices: list[int] | None = None,
        device_ratios: str | None = None,
        cache_size: int | None = None,
        cache_quant: str | None = None,
        draft_model_dir: str | None = None,
        use_mtp: bool = False,
        ngram_min: int = 0,
        batch_slots: int | None = None,
        cpu_offload: dict | None = None,
        ngram_ram: bool = False,
    ):
        if sum([bool(draft_model_dir), bool(use_mtp), bool(ngram_min)]) > 1:
            raise ValueError(
                "Specify only one of: draft model directory, MTP drafting, "
                "or n-gram drafting"
            )
        self.model_dir = os.path.abspath(model_dir) if model_dir else None
        self._devices = devices or []
        self._device_ratios = device_ratios
        self._cache_size = cache_size or self.DEFAULT_CACHE_SIZE
        self._cache_quant = cache_quant or self.DEFAULT_CACHE_QUANT
        self._cpu_offload = dict(cpu_offload or {})
        # Load the hashed n-gram embedding table of PLE models (e.g.
        # Qwen3.8-Flash-Next) fully into system RAM instead of streaming
        # rows from disk per forward (exllamav3 -ngr, needs >= 1.4.5).
        self.ngram_ram = bool(ngram_ram)

        # Populated by load()
        self.model = None
        self.config = None
        self.cache = None
        self.tokenizer = None
        self.generator = None
        self.loras: list = []
        self.lora_dirs: list[str] = []
        self.lora_weights: list[float] = []
        self.draft_model = None
        self.draft_cache = None
        self.draft_config = None
        self.draft_model_dir: str | None = None
        self.draft_model_name: str = ""
        self.use_mtp: bool = False
        self.ngram_min: int = 0
        self.context_length: int = 0
        self.model_name: str = os.path.basename(self.model_dir) if self.model_dir else ""

        # Draft source requested at construction (loaded together with the
        # model — required for recurrent models, whose caches must be sized
        # with draft headroom at creation).
        if draft_model_dir:
            self.draft_model_dir = os.path.abspath(draft_model_dir)
            self.draft_model_name = os.path.basename(self.draft_model_dir)
        self.use_mtp = bool(use_mtp)
        self.ngram_min = int(ngram_min or 0)

        # Concurrent DPO-duel candidates. Sets the recurrent cache's state
        # slots (-ambs) at load, capping the batch size for hybrid/recurrent
        # models; non-recurrent models batch freely and ignore it. See
        # BATCH_SLOTS. Changing it needs a reload to re-allocate the cache.
        self.batch_slots = int(batch_slots) if batch_slots else self.BATCH_SLOTS

        # Chat state
        self.settings = ChatSettings()
        self.context: list[tuple[str, Optional[str]]] = []
        self._current_job = None
        self._is_generating = False

    @property
    def is_loaded(self) -> bool:
        return self.generator is not None

    @property
    def is_generating(self) -> bool:
        return self._is_generating

    def load(self):
        """Load model synchronously (called at startup)."""
        model_init = _import_model_init()

        # Set visible devices before model_init touches CUDA
        devices = self._devices or list(range(torch.cuda.device_count()))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in devices)

        # Build a minimal args namespace that model_init.init() expects
        args = _build_model_args(
            self.model_dir,
            devices,
            self._device_ratios,
            self._cache_size,
            self._cache_quant,
            self.batch_slots,
            self._cpu_offload,
            self.ngram_ram,
        )

        # Recurrent (hybrid linear-attn) models like Qwen3.5 size their
        # recurrent-state history at cache creation (Cache max_history);
        # speculative decoding needs history >= draft length. DFlash drafts
        # a whole block per round (block_size - 1 tokens), other draft
        # sources use the generator's default window of 4. Older exllamav3
        # builds lack the parameter, so only pass it when supported.
        init_kwargs = {}
        if self.draft_model_dir or self.use_mtp or self.ngram_min:
            import inspect
            if "min_draft_len" in inspect.signature(model_init.init).parameters:
                init_kwargs["min_draft_len"] = self._draft_len_hint()

        torch.set_grad_enabled(False)
        self.model, self.config, self.cache, self.tokenizer = model_init.init(args, **init_kwargs)
        self.context_length = self.cache.max_num_tokens

        if self.draft_model_dir or self.use_mtp:
            self._load_draft_model()

        self._create_generator()
        self._load_loras()

        # Set default mode/system prompt based on model
        self._auto_detect_mode()
        print(f"  Model loaded: {self.model_name}")
        print(f"  Context length: {self.context_length:,} tokens")
        if self.draft_model:
            print(f"  Draft model: {self.draft_model_name}")
        print(f"  Prompt mode: {self.settings.mode}")

    def _create_generator(self):
        Generator = _import_generator()
        kwargs = dict(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
            max_chunk_size=4096,
        )
        if self.draft_model is not None:
            kwargs["draft_model"] = self.draft_model
            kwargs["draft_cache"] = self.draft_cache
        elif self.ngram_min:
            # Draft-model-free speculative decoding: match the last N+
            # generated tokens against prior context (exllamav3 SAM ngram).
            # Mutually exclusive with a draft model (Generator asserts).
            kwargs["ngram_match_min"] = self.ngram_min
        self.generator = Generator(**kwargs)

    def unload(self):
        """Unload the current model, freeing GPU memory."""
        if self._is_generating:
            self.cancel()
        self.generator = None
        self.loras = []
        self.lora_dirs = []
        self.lora_weights = []
        self.draft_model = None
        self.draft_cache = None
        self.draft_config = None
        self.draft_model_dir = None
        self.draft_model_name = ""
        self.use_mtp = False
        self.ngram_min = 0
        self.model = None
        self.cache = None
        self.tokenizer = None
        self.config = None
        self.context_length = 0
        self.context = []
        self.model_name = ""
        self.model_dir = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def load_model(
        self,
        model_dir: str,
        lora_dirs: list[str] | None = None,
        lora_weights: list[float] | None = None,
        draft_model_dir: str | None = None,
        use_mtp: bool = False,
        ngram_min: int = 0,
        devices: list[int] | None = None,
        device_ratios: str | None = None,
        cache_size: int | None = None,
        cache_quant: str | None = None,
        batch_slots: int | None = None,
        cpu_offload: dict | None = None,
        ngram_ram: bool = False,
    ):
        """Load a model (callable from the UI after startup)."""
        if use_mtp and draft_model_dir:
            raise ValueError("Cannot specify both a draft model directory and MTP drafting")
        if ngram_min and (use_mtp or draft_model_dir):
            raise ValueError("Cannot combine n-gram drafting with a draft model or MTP drafting")
        if self.is_loaded:
            self.unload()
        self.model_dir = os.path.abspath(model_dir)
        self.model_name = os.path.basename(self.model_dir)
        self.lora_dirs = [os.path.abspath(p) for p in (lora_dirs or [])]
        self.lora_weights = list(lora_weights or [1.0] * len(self.lora_dirs))
        self.use_mtp = use_mtp
        self.ngram_min = int(ngram_min or 0)
        if draft_model_dir:
            self.draft_model_dir = os.path.abspath(draft_model_dir)
            self.draft_model_name = os.path.basename(self.draft_model_dir)
        self._devices = devices or []
        self._device_ratios = device_ratios
        self._cache_size = cache_size or self.DEFAULT_CACHE_SIZE
        self._cache_quant = cache_quant or self.DEFAULT_CACHE_QUANT
        self.batch_slots = int(batch_slots) if batch_slots else self.BATCH_SLOTS
        self._cpu_offload = dict(cpu_offload or {})
        self.ngram_ram = bool(ngram_ram)
        self.settings = ChatSettings()
        self.load()

    def _load_loras(self):
        if not self.lora_dirs:
            return
        try:
            from exllamav3.model.lora import LoRA  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "LoRA support is unavailable in this exllamav3 install"
            ) from e
        self.loras = []
        for i, lora_dir in enumerate(self.lora_dirs):
            weight = self.lora_weights[i] if i < len(self.lora_weights) else 1.0
            if weight == 0:
                self.loras.append(None)
                continue
            self.loras.append(
                LoRA.from_directory(self.model, lora_dir, lora_scaling=weight)
            )

    def update_loras(
        self,
        lora_configs: list[dict],
    ):
        """
        Dynamically update LoRAs on a loaded model without reloading.

        Each entry: {"dir": "/path/to/lora", "weight": 0.5}
        Weight of 0 means inactive. Requires model to be loaded.
        """
        if not self.is_loaded:
            raise RuntimeError("No model loaded")
        if self._is_generating:
            raise RuntimeError("Cannot update LoRAs while generating")

        from exllamav3.model.lora import LoRA  # type: ignore

        for lora in self.loras:
            if lora is not None:
                lora.unload()
        self.loras = []
        self.lora_dirs = []
        self.lora_weights = []

        for cfg in lora_configs:
            d = os.path.abspath(cfg["dir"])
            w = float(cfg.get("weight", 1.0))
            self.lora_dirs.append(d)
            self.lora_weights.append(w)
            if w == 0:
                self.loras.append(None)
                continue
            self.loras.append(
                LoRA.from_directory(self.model, d, lora_scaling=w)
            )

        active = sum(1 for l in self.loras if l is not None)
        print(f"  LoRAs updated: {active} active / {len(self.loras)} total")

    def _read_json_config(self, model_dir: str) -> dict:
        try:
            with open(os.path.join(model_dir, "config.json")) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _dflash_cfg(cfg: dict) -> dict | None:
        """The DFlash settings of a draft config, flattened, or None when the
        config isn't a DFlash draft at all.

        The original DFlash release nests these keys under ``dflash_config``;
        later drafters built on the same architecture (Meta's
        MuseGlimmerAssistantModel for Muse Glimmer) put them at the top level,
        so match on the keys DFlashConfig actually requires rather than on
        the wrapper or the architecture name.
        """
        sub = cfg.get("dflash_config")
        merged = {**cfg, **sub} if isinstance(sub, dict) else cfg
        if all(k in merged for k in
               ("mask_token_id", "target_layer_ids", "block_size")):
            return merged
        return None

    def _draft_len_hint(self, draft_model_dir: str | None = None) -> int:
        """Draft length the recurrent-state history must cover.

        DFlash drafts block_size - 1 tokens per round; every other draft
        source uses the generator's default window of 4.
        """
        d = draft_model_dir or self.draft_model_dir
        if d:
            cfg = self._dflash_cfg(self._read_json_config(d))
            if cfg and cfg.get("block_size"):
                return max(4, int(cfg["block_size"]) - 1)
        return 4

    def _check_draft_compat(self):
        """DFlash drafts run on the target's embedding and lm_head, so
        their hidden size must match the target's. A mismatched pair loads
        cleanly and then dies on the first draft forward with a cryptic
        shape error, so catch it here with a readable message."""
        dcfg = self._dflash_cfg(self._read_json_config(self.draft_model_dir))
        if dcfg is None:
            return  # regular draft models legitimately differ in size
        tcfg = self._read_json_config(self.model_dir)
        d_hidden = dcfg.get("hidden_size")
        t_hidden = (tcfg.get("text_config") or {}).get("hidden_size") \
            or tcfg.get("hidden_size")
        if d_hidden and t_hidden and d_hidden != t_hidden:
            raise RuntimeError(
                f"DFlash draft {self.draft_model_name} was trained for a "
                f"target with hidden size {d_hidden}, but "
                f"{self.model_name} has hidden size {t_hidden}. "
                f"Wrong base model for this draft?"
            )

    def _apply_draft_cpu_offload(self, config, prefix: str):
        """Set the draft model's MoE CPU-offload budget on its config.

        exllamav3 mirrors this in model_init when --draft_moe_cpu_layers is
        used, but we load draft models ourselves, so we write infer_params
        directly. Silently a no-op on builds predating the fields.
        """
        layers = int(self._cpu_offload.get("draft_moe_layers") or 0)
        if layers <= 0:
            return
        ip = getattr(config, "infer_params", None)
        if ip is None or not hasattr(ip, f"{prefix}_offload"):
            return
        setattr(ip, f"{prefix}_offload", layers)
        threads = int(self._cpu_offload.get("draft_moe_threads") or 0)
        if threads > 0 and hasattr(ip, f"{prefix}_threads"):
            setattr(ip, f"{prefix}_threads", threads)

    def _load_draft_model(self):
        Config = _import_config()
        Model = _import_model()
        Cache = _import_cache()

        if self.use_mtp:
            # Standard exllamav3 --mtp behavior: the MTP head lives inside
            # the main model's checkpoint, loaded as its "mtp" component
            # with the main config (exllamav3 dev, Qwen3.5/3.6).
            if "mtp" not in getattr(self.config, "model_classes", {}):
                self.use_mtp = False
                raise RuntimeError(
                    f"{self.model_name} does not expose an MTP draft component. "
                    "MTP drafting needs a model with MTP weights (e.g. Qwen3.5) "
                    "and an exllamav3 build with MTP support."
                )
            self.draft_model_dir = self.model_dir
            self.draft_config = self.config
            # The MTP head shares the target's config, so its expert budget
            # rides on the draft_* fields (component != "text").
            self._apply_draft_cpu_offload(self.draft_config, "draft_moe_cpu")
            self.draft_model = Model.from_config(self.draft_config, component="mtp")
            self.draft_model_name = f"{self.model_name} (MTP)"
        else:
            self._check_draft_compat()
            self.draft_config = Config.from_directory(self.draft_model_dir)
            # A standalone draft model gets its own config, and its own text
            # component takes the budget from the plain moe_cpu_* fields.
            self._apply_draft_cpu_offload(self.draft_config, "moe_cpu")
            self.draft_model = Model.from_config(self.draft_config)
        self.draft_cache = Cache(
            self.draft_model,
            max_num_tokens=self.cache.max_num_tokens,
        )
        self.draft_model.load()

    def _unload_draft_model(self):
        self.draft_model = None
        self.draft_cache = None
        self.draft_config = None
        self.draft_model_dir = None
        self.draft_model_name = ""
        self.use_mtp = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def needs_load_time_draft(self, draft_model_dir: str | None = None) -> bool:
        """True if enabling a draft source requires a full model reload.

        Recurrent (hybrid linear-attn) models size their state history at
        cache creation; enabling drafting post-load on a cache without
        draft headroom corrupts recurrent state allocation. The required
        headroom depends on the draft source (DFlash needs block_size - 1).
        """
        caps = getattr(self.model, "caps", None)
        if isinstance(caps, dict) and caps.get("recurrent_states"):
            need = self._draft_len_hint(
                os.path.abspath(draft_model_dir) if draft_model_dir else None
            )
            return getattr(self.cache, "max_history", 0) < need
        return False

    def load_draft(
        self,
        draft_model_dir: str | None = None,
        mtp: bool = False,
        ngram_min: int = 0,
    ):
        """Enable a draft source (model dir, MTP head, or n-gram) on an already-loaded model."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")
        if self._is_generating:
            raise RuntimeError("Cannot load draft model while generating")
        selected = sum([bool(draft_model_dir), bool(mtp), bool(ngram_min)])
        if selected > 1:
            raise RuntimeError(
                "Specify only one of: draft model directory, MTP drafting, or n-gram drafting"
            )
        if selected == 0:
            raise RuntimeError(
                "draft_model_dir is required unless MTP or n-gram drafting is enabled"
            )

        if self.needs_load_time_draft(draft_model_dir):
            raise RuntimeError(
                "This model uses recurrent states, so speculative decoding "
                "must be enabled at load time (the cache needs draft "
                "headroom). Reload the model with the draft option selected."
            )

        if self.draft_model is not None:
            self.unload_draft()
        self.ngram_min = 0

        if ngram_min:
            # No extra weights to load — just recreate the generator with
            # ngram drafting enabled.
            self.ngram_min = int(ngram_min)
            self._create_generator()
            print(f"  N-gram drafting enabled (min match {self.ngram_min})")
            return

        self.use_mtp = mtp
        if draft_model_dir:
            self.draft_model_dir = os.path.abspath(draft_model_dir)
            self.draft_model_name = os.path.basename(self.draft_model_dir)
        self._load_draft_model()
        self._create_generator()
        print(f"  Draft model loaded: {self.draft_model_name}")

    def unload_draft(self):
        """Disable the current draft source without touching the main model."""
        if not self.is_loaded:
            raise RuntimeError("No model loaded")
        if self._is_generating:
            raise RuntimeError("Cannot unload draft model while generating")
        if self.draft_model is None and not self.ngram_min:
            return

        had_ngram = self.ngram_min > 0
        self._unload_draft_model()
        self.ngram_min = 0
        self._create_generator()
        print("  N-gram drafting disabled" if had_ngram else "  Draft model unloaded")

    @staticmethod
    def detect_gpus() -> list[dict]:
        """Return list of available GPUs with name and VRAM."""
        gpus = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpus.append({
                    "index": i,
                    "name": props.name,
                    "vram_gb": round(props.total_memory / (1024**3), 1),
                })
        return gpus

    def _auto_detect_mode(self):
        """Try to pick a sensible default prompt format from model config."""
        self.settings.mode = infer_mode(self.model_name)

        # Set default system prompt from the format
        pf_cls = prompt_formats.get(self.settings.mode, prompt_formats["chatml"])
        pf = pf_cls(self.settings.user_name, self.settings.bot_name)
        self.settings.system_prompt = pf.default_system_prompt(self.settings.think)

    def _get_prompt_format(self):
        pf_cls = prompt_formats.get(self.settings.mode, prompt_formats["chatml"])
        pf = pf_cls(self.settings.user_name, self.settings.bot_name)
        spc = {}
        if self.settings.think_budget is not None:
            spc["thinking_budget"] = self.settings.think_budget
        # The 'jinja' format reads the model's own chat template off disk,
        # and needs the tokenizer to spot a BOS the template writes that
        # encode() would then add a second time.
        spc["model_dir"] = self.model_dir
        spc["tokenizer"] = self.tokenizer
        pf.set_special(spc)
        return pf

    def _get_stop_conditions(self, prompt_format):
        stop_conditions = [
            sc for sc in prompt_format.stop_conditions(self.tokenizer)
            if sc is not None
        ]
        if self.config.eos_token_id_list and all(
            x is not None for x in self.config.eos_token_id_list
        ):
            stop_conditions += self.config.eos_token_id_list
        return stop_conditions

    def _get_sampler(self):
        model_init = _import_model_init()
        import argparse

        s = self.settings
        ns = argparse.Namespace(
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            min_p=s.min_p,
            repetition_penalty=s.repetition_penalty,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            penalty_range=1024,
            temperature_first=False,
            adaptive_target=1.0,
            adaptive_decay=0.9,
        )
        return model_init.get_arg_sampler(ns)

    def _build_input_ids(self, prompt_format, prefix: str = "",
                         system_prompt: str | None = None,
                         context: list | None = None):
        """Tokenize full context, trimming from head if too long.

        *system_prompt* overrides the settings' system prompt for this
        build only (used for per-candidate DPO generation prompts);
        None means use the trained prompt from settings.

        *context* overrides the persistent chat context for this build
        (bulk generation builds fresh single-turn contexts per prompt);
        when None the persistent self.context is used AND head-trimmed
        in place if too long, matching interactive-chat behavior.
        """
        think = self.settings.think
        sys_prompt = (self.settings.system_prompt if system_prompt is None
                      else system_prompt)
        persistent = context is None
        ctx = self.context if persistent else context
        frm_context = prompt_format.format(sys_prompt, ctx, think)
        if prefix:
            frm_context += prefix
        elif think and prompt_format.thinktag()[0] is not None:
            frm_context += prompt_format.thinktag()[0]

        add_bos = prompt_format.add_bos()
        ids = self.tokenizer.encode(
            frm_context, add_bos=add_bos, encode_special_tokens=True
        )
        exp_len = ids.shape[-1] + self.settings.max_response_tokens + 1

        # Trim from head if context too long
        if exp_len > self.context_length:
            while exp_len > self.context_length - 2 * self.settings.max_response_tokens:
                if len(ctx) <= 1:
                    break
                ctx = ctx[1:]
                if persistent:
                    self.context = ctx
                frm_context = prompt_format.format(sys_prompt, ctx, think)
                if prefix:
                    frm_context += prefix
                elif think and prompt_format.thinktag()[0] is not None:
                    frm_context += prompt_format.thinktag()[0]
                ids = self.tokenizer.encode(
                    frm_context, add_bos=add_bos, encode_special_tokens=True
                )
                exp_len = ids.shape[-1] + self.settings.max_response_tokens + 1

        return ids

    @staticmethod
    def _tps_event(r: dict, elapsed: float, cand: int) -> dict:
        new_tokens = r.get("new_tokens", 0)
        prompt_tokens = r.get("prompt_tokens", 0)
        tps = new_tokens / elapsed if elapsed > 0 else 0
        prefill_tps = (
            prompt_tokens / r["time_prefill"]
            if r.get("time_prefill", 0) > 0
            else 0
        )
        ev = {
            "type": "tps",
            "cand": cand,
            "new_tokens": new_tokens,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": r.get("cached_tokens", 0),
            "tps": round(tps, 2),
            "prefill_tps": round(prefill_tps, 2),
            "elapsed": round(elapsed, 2),
        }
        if r.get("accepted_draft_tokens", 0) > 0:
            accepted = r["accepted_draft_tokens"]
            rejected = r.get("rejected_draft_tokens", 0)
            total = accepted + rejected
            ev["draft_accepted"] = accepted
            ev["draft_rejected"] = rejected
            ev["draft_acceptance_rate"] = round(accepted / total, 3) if total > 0 else 0
        return ev

    async def generate(self, user_message: str, prefix: str = "",
                       n: int = 1,
                       system_prompts: list | None = None,
                       ) -> AsyncGenerator[dict, None]:
        """
        Stream response(s) for *user_message*. With n > 1 all candidates
        generate CONCURRENTLY as batched jobs in one generator pass —
        the chat UI's DPO duel mode uses n=2.

        *system_prompts* optionally overrides the system prompt per
        candidate (list of str-or-None aligned with n; None or "" falls
        back to the settings' trained prompt). Used by DPO duels to bias
        each candidate's generation while the recorded dataset keeps the
        trained prompt.

        Yields dicts; "cand" indexes the candidate (always 0 when n=1):
            {"type": "token", "cand": 0, "text": "..."}
            {"type": "tps", "cand": 0, ...}
            {"type": "done", "cand": 0, "eos_reason": "..."}
            {"type": "error", "message": "..."}
        """
        Job = _import_job()

        if not self.is_loaded:
            yield {"type": "error", "message": "Model not loaded"}
            return

        if self._is_generating:
            yield {"type": "error", "message": "Already generating"}
            return

        self._is_generating = True
        try:
            # Amnesia mode
            if self.settings.amnesia:
                self.context = []

            # Add user message
            self.context.append((user_message, None))

            prompt_format = self._get_prompt_format()
            stop_conditions = self._get_stop_conditions(prompt_format)

            # Per-candidate input ids: candidates share everything except
            # an optional per-candidate system-prompt override. Identical
            # prompts still share cache pages via the generator's dedup.
            n_jobs = max(1, int(n))
            ids_per_cand = []
            for i in range(n_jobs):
                override = None
                if system_prompts and i < len(system_prompts):
                    override = system_prompts[i] or None
                ids_per_cand.append(self._build_input_ids(
                    prompt_format, prefix=prefix, system_prompt=override))

            # Banned strings
            banned = list(self.settings.banned_strings)
            if self.settings.no_think:
                tt = prompt_format.thinktag()
                if tt[0]:
                    banned.append(tt[0])
                if tt[1]:
                    banned.append(tt[1])

            jobs = [
                Job(
                    input_ids=ids,
                    max_new_tokens=self.settings.max_response_tokens,
                    stop_conditions=stop_conditions,
                    sampler=self._get_sampler(),
                    banned_strings=list(banned) if banned else None,
                )
                for ids in ids_per_cand
            ]
            cand_of = {id(job): i for i, job in enumerate(jobs)}
            self._current_job = list(jobs)
            for job in jobs:
                self.generator.enqueue(job)

            texts = [""] * len(jobs)
            finals: list = [None] * len(jobs)
            pending = len(jobs)
            t_start = time.time()

            while pending and self.generator.num_remaining_jobs():
                for r in self.generator.iterate():
                    cand = cand_of.get(id(r.get("job")), 0)
                    chunk = r.get("text", "")
                    if chunk:
                        texts[cand] += chunk
                        yield {"type": "token", "cand": cand, "text": chunk}

                    if r.get("eos") and finals[cand] is None:
                        finals[cand] = r
                        pending -= 1
                        yield self._tps_event(r, time.time() - t_start, cand)
                        yield {"type": "done", "cand": cand,
                               "eos_reason": r.get("eos_reason", "unknown")}

                # Let the event loop breathe
                await asyncio.sleep(0)

            # Close out candidates that never reported eos (cancelled)
            for cand, r in enumerate(finals):
                if r is None:
                    yield self._tps_event({}, time.time() - t_start, cand)
                    yield {"type": "done", "cand": cand, "eos_reason": "unknown"}

            # Save to context. With n > 1 candidate 0 is a placeholder —
            # the client re-syncs the real context (the duel winner's
            # branch) on its next request.
            full_response = (prefix + texts[0]).strip()
            self.context[-1] = (user_message, full_response)

        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            self._is_generating = False
            self._current_job = None

    async def generate_bulk(self, prompts: list, n: int = 1,
                            system_prompt: str | None = None,
                            ) -> AsyncGenerator[dict, None]:
        """
        Generate *n* completions for EACH prompt in *prompts*, all as one
        job pool — the generator batches across prompts up to its ceiling
        (recurrent models: the -ambs cache slots allocated at load), so
        throughput is bounded by the model, not by per-prompt turnaround.

        Every prompt is a fresh single-turn conversation; the persistent
        chat context is neither read nor written. *system_prompt*
        overrides the settings' system prompt for every job (None/"" =
        the trained prompt).

        Yields dicts (no per-token streaming — completions arrive whole):
            {"type": "row_done", "item": i, "cand": j, "text": "...",
             "eos_reason": "..."}
            {"type": "progress", "done": k, "total": m,
             "new_tokens": t, "elapsed": s, "tps": r}
            {"type": "error", "message": "..."}
        """
        Job = _import_job()

        if not self.is_loaded:
            yield {"type": "error", "message": "Model not loaded"}
            return
        if self._is_generating:
            yield {"type": "error", "message": "Already generating"}
            return
        if not prompts:
            return

        self._is_generating = True
        try:
            n = max(1, int(n))
            prompt_format = self._get_prompt_format()
            stop_conditions = self._get_stop_conditions(prompt_format)
            override = system_prompt or None

            banned = list(self.settings.banned_strings)
            if self.settings.no_think:
                tt = prompt_format.thinktag()
                if tt[0]:
                    banned.append(tt[0])
                if tt[1]:
                    banned.append(tt[1])

            jobs = []
            slot_of = {}      # id(job) -> (item, cand)
            for i, message in enumerate(prompts):
                ids = self._build_input_ids(
                    prompt_format, system_prompt=override,
                    context=[(message, None)])
                for j in range(n):
                    job = Job(
                        input_ids=ids,
                        max_new_tokens=self.settings.max_response_tokens,
                        stop_conditions=stop_conditions,
                        sampler=self._get_sampler(),
                        banned_strings=list(banned) if banned else None,
                    )
                    slot_of[id(job)] = (i, j)
                    jobs.append(job)

            self._current_job = list(jobs)
            for job in jobs:
                self.generator.enqueue(job)

            texts = {id(job): "" for job in jobs}
            done_of = {id(job): False for job in jobs}
            pending = len(jobs)
            total = len(jobs)
            new_tokens = 0
            t_start = time.time()
            t_beat = t_start

            while pending and self.generator.num_remaining_jobs():
                for r in self.generator.iterate():
                    job = r.get("job")
                    if id(job) not in slot_of:
                        continue
                    chunk = r.get("text", "")
                    if chunk:
                        texts[id(job)] += chunk
                    if r.get("eos") and not done_of[id(job)]:
                        done_of[id(job)] = True
                        pending -= 1
                        new_tokens += r.get("new_tokens", 0)
                        item, cand = slot_of[id(job)]
                        yield {"type": "row_done", "item": item, "cand": cand,
                               "text": texts[id(job)],
                               "eos_reason": r.get("eos_reason", "unknown")}

                # Heartbeat: throughput + counts every ~2 s so the UI can
                # show live progress without per-token traffic.
                now = time.time()
                if now - t_beat >= 2.0:
                    t_beat = now
                    elapsed = now - t_start
                    yield {"type": "progress",
                           "done": total - pending, "total": total,
                           "new_tokens": new_tokens,
                           "elapsed": round(elapsed, 2),
                           "tps": round(new_tokens / elapsed, 2)
                                  if elapsed > 0 else 0}
                await asyncio.sleep(0)

            # Cancelled jobs never report eos — surface them as empty rows
            # so the caller's accounting closes out.
            for job in jobs:
                if not done_of[id(job)]:
                    item, cand = slot_of[id(job)]
                    yield {"type": "row_done", "item": item, "cand": cand,
                           "text": texts[id(job)], "eos_reason": "cancelled"}
            elapsed = time.time() - t_start
            yield {"type": "progress",
                   "done": total - pending, "total": total,
                   "new_tokens": new_tokens, "elapsed": round(elapsed, 2),
                   "tps": round(new_tokens / elapsed, 2) if elapsed > 0 else 0}

        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            self._is_generating = False
            self._current_job = None

    def cancel(self):
        """Cancel the current generation (all candidates, if a duel)."""
        jobs = self._current_job
        if jobs is not None and self.generator is not None:
            for job in (jobs if isinstance(jobs, list) else [jobs]):
                try:
                    self.generator.cancel(job)
                except Exception:
                    pass
            self._is_generating = False
            self._current_job = None

    def clear_context(self):
        self.context = []

    def save_session(self) -> dict:
        return {
            "system_prompt": self.settings.system_prompt,
            "banned_strings": self.settings.banned_strings,
            "context": self.context,
            "settings": self.settings.to_dict(),
        }

    def load_session(self, data: dict):
        if "settings" in data:
            self.settings = ChatSettings.from_dict(data["settings"])
        if "system_prompt" in data:
            self.settings.system_prompt = data["system_prompt"]
        if "banned_strings" in data:
            self.settings.banned_strings = data["banned_strings"]
        if "context" in data:
            self.context = [tuple(pair) for pair in data["context"]]

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "generating": self.is_generating,
            "model_name": self.model_name,
            "model_dir": self.model_dir or "",
            "context_length": self.context_length,
            "context_turns": len(self.context),
            "lora_dirs": getattr(self, "lora_dirs", []),
            "lora_weights": getattr(self, "lora_weights", []),
            "lora_count": sum(
                1 for l in getattr(self, "loras", []) if l is not None
            ),
            "draft_model_dir": self.draft_model_dir or "",
            "draft_model_name": self.draft_model_name,
            "draft_model_loaded": self.draft_model is not None,
            "draft_mtp": getattr(self, "use_mtp", False),
            "ngram_min": getattr(self, "ngram_min", 0),
            "available_modes": {
                k: v.description for k, v in prompt_formats.items()
            },
            "gpus": self.detect_gpus(),
            "cpu_cores": os.cpu_count() or 0,
            "cpu_offload": cpu_offload_support(),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def cpu_offload_support() -> dict:
    """Which CPU-offload knobs the installed exllamav3 exposes.

    MoE expert offload (-mcl/-mclt) and the second-tier CPU KV cache (-ccs)
    landed in exllamav3 1.3.0. Older builds don't have them, so the chat UI
    disables the controls rather than silently dropping the setting. The
    draft-side knobs live on config.infer_params rather than argparse (we
    load draft models ourselves), but they shipped in the same release, so
    -mcl is a sound proxy for both.
    """
    try:
        import argparse
        model_init = _import_model_init()
        parser = argparse.ArgumentParser()
        model_init.add_args(parser, cache=True)
        opts = {o for a in parser._actions for o in a.option_strings}
    except Exception:
        return {"moe": False, "moe_threads": False, "cache": False}
    return {
        "moe": "-mcl" in opts,
        "moe_threads": "-mclt" in opts,
        "cache": "-ccs" in opts,
    }


def ngram_ram_support() -> bool:
    """Whether the installed exllamav3 exposes -ngr (n-gram table in RAM,
    PLE models; landed in 1.4.5). The chat UI disables the toggle on older
    builds rather than silently dropping the setting."""
    try:
        import argparse
        model_init = _import_model_init()
        parser = argparse.ArgumentParser()
        model_init.add_args(parser, cache=True)
        opts = {o for a in parser._actions for o in a.option_strings}
    except Exception:
        return False
    return "-ngr" in opts


def _build_model_args(
    model_dir: str,
    devices: list[int],
    device_ratios: str | None,
    cache_size: int | None,
    cache_quant: str | None,
    batch_slots: int | None = None,
    cpu_offload: dict | None = None,
    ngram_ram: bool = False,
) -> object:
    """
    Build a namespace object mimicking the argparse args that
    exllamav3.model_init.init() expects.
    """
    import argparse
    model_init = _import_model_init()

    # Build the parser to discover defaults, then feed it the required args
    parser = argparse.ArgumentParser()
    model_init.add_args(parser, cache=True)

    argv = ["-m", model_dir]
    if device_ratios:
        # User-provided GB ratios per device, e.g. "20,20"
        argv += ["-gs", device_ratios]
    elif len(devices) > 1:
        # Multi-GPU: let exllamav3 auto-split across visible devices
        argv += ["-gs", "auto"]
    if cache_size:
        argv += ["-cs", str(cache_size)]
    if cache_quant:
        argv += ["-cq", cache_quant]
    # Recurrent models get one state slot per concurrent job; the default
    # of 1 serializes DPO duel candidates (Generator clamps its batch size
    # to the cache's slot count). Non-recurrent models ignore this. Older
    # exllamav3 builds don't expose the argument.
    slots = int(batch_slots) if batch_slots else ChatEngine.BATCH_SLOTS
    if any("-ambs" in a.option_strings for a in parser._actions):
        argv += ["-ambs", str(max(1, slots))]

    # CPU offload (exllamav3 >= 1.3.0). Each knob is gated on the running
    # build actually exposing it, so a stale exllamav3 loads as before
    # instead of dying in parse_args. The draft-side knobs aren't part of
    # add_args(cache=True) — those are applied in _load_draft_model().
    off = cpu_offload or {}
    opts = {o for a in parser._actions for o in a.option_strings}
    moe_layers = int(off.get("moe_layers") or 0)
    if moe_layers > 0 and "-mcl" in opts:
        argv += ["-mcl", str(moe_layers)]
        moe_threads = int(off.get("moe_threads") or 0)
        if moe_threads > 0 and "-mclt" in opts:
            argv += ["-mclt", str(moe_threads)]
    cache_gb = float(off.get("cache_gb") or 0)
    if cache_gb > 0 and "-ccs" in opts:
        argv += ["-ccs", str(cache_gb)]

    # N-gram table in RAM (exllamav3 >= 1.4.5). Gated the same way; only
    # matters for PLE models, which older builds can't load anyway.
    if ngram_ram and "-ngr" in opts:
        argv += ["-ngr"]

    args = parser.parse_args(argv)
    # exllamav3 dev reads args.mtp in init() even when draft model args
    # aren't requested; older builds ignore the extra attribute.
    if not hasattr(args, "mtp"):
        args.mtp = False
    return args
