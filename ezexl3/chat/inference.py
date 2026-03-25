# Chat inference engine wrapping exllamav3's Generator/Job API.
#
# Adapted from exllamav3 examples/chat.py
# Original author: turboderp (https://github.com/turboderp-org/exllamav3)

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import AsyncGenerator, List, Optional

import torch

from .templates import prompt_formats


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

    def __init__(
        self,
        model_dir: str,
        devices: list[int] | None = None,
        device_ratios: str | None = None,
        cache_size: int | None = None,
        cache_quant: str | None = None,
    ):
        self.model_dir = os.path.abspath(model_dir)
        self._devices = devices or [0]
        self._device_ratios = device_ratios
        self._cache_size = cache_size or self.DEFAULT_CACHE_SIZE
        self._cache_quant = cache_quant or self.DEFAULT_CACHE_QUANT

        # Populated by load()
        self.model = None
        self.config = None
        self.cache = None
        self.tokenizer = None
        self.generator = None
        self.context_length: int = 0
        self.model_name: str = os.path.basename(self.model_dir)

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
        from exllamav3 import Generator, model_init

        # Set visible devices before model_init touches CUDA
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in self._devices)

        # Build a minimal args namespace that model_init.init() expects
        args = _build_model_args(
            self.model_dir,
            self._devices,
            self._device_ratios,
            self._cache_size,
            self._cache_quant,
        )

        torch.set_grad_enabled(False)
        self.model, self.config, self.cache, self.tokenizer = model_init.init(args)
        self.context_length = self.cache.max_num_tokens
        self.generator = Generator(
            model=self.model,
            cache=self.cache,
            tokenizer=self.tokenizer,
        )

        # Set default mode/system prompt based on model
        self._auto_detect_mode()
        print(f"  Model loaded: {self.model_name}")
        print(f"  Context length: {self.context_length:,} tokens")
        print(f"  Prompt mode: {self.settings.mode}")

    def _auto_detect_mode(self):
        """Try to pick a sensible default prompt format from model config."""
        name_lower = self.model_name.lower()
        # Simple heuristic matching
        mode_hints = {
            "llama": "llama3",
            "qwen": "chatml",
            "phi": "phi",
            "mistral": "mistral3",
            "gemma": "gemma",
            "glm": "glm",
            "cohere": "cohere",
            "command": "commanda",
            "exaone": "exaone",
            "reka": "reka",
            "dots": "dots",
            "ernie": "ernie",
            "smollm": "smollm3",
            "seed": "seed",
            "apertus": "apertus",
            "minimax": "minimax",
        }
        for hint, mode in mode_hints.items():
            if hint in name_lower:
                self.settings.mode = mode
                break
        else:
            self.settings.mode = "chatml"  # safe default

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
        pf.set_special(spc)
        return pf

    def _get_stop_conditions(self, prompt_format):
        stop_conditions = [
            sc for sc in prompt_format.stop_conditions(self.tokenizer) if sc
        ]
        if self.config.eos_token_id_list and all(self.config.eos_token_id_list):
            stop_conditions += self.config.eos_token_id_list
        return stop_conditions

    def _get_sampler(self):
        from exllamav3 import Sampler

        sampler = Sampler()
        s = self.settings
        sampler.temperature = s.temperature
        sampler.top_k = s.top_k
        sampler.top_p = s.top_p
        sampler.min_p = s.min_p
        if s.repetition_penalty != 1.0:
            sampler.repetition_penalty = s.repetition_penalty

        # Ensure Sampler.forward accepts logit_mask (needed by banned_strings).
        # Older exllamav3 builds lack this parameter; patch it in so the
        # generator can pass logit_mask without crashing.
        import inspect
        sig = inspect.signature(sampler.forward)
        if "logit_mask" not in sig.parameters:
            _orig_forward = sampler.forward
            def _patched_forward(*args, logit_mask=None, **kwargs):
                # Apply mask to logits BEFORE sampling
                if logit_mask is not None and len(args) > 0:
                    args[0][logit_mask] = float("-inf")
                return _orig_forward(*args, **kwargs)
            sampler.forward = _patched_forward

        return sampler

    def _build_input_ids(self, prompt_format, prefix: str = ""):
        """Tokenize full context, trimming from head if too long."""
        think = self.settings.think
        frm_context = prompt_format.format(
            self.settings.system_prompt, self.context, think
        )
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
                if len(self.context) <= 1:
                    break
                self.context = self.context[1:]
                frm_context = prompt_format.format(
                    self.settings.system_prompt, self.context, think
                )
                if prefix:
                    frm_context += prefix
                elif think and prompt_format.thinktag()[0] is not None:
                    frm_context += prompt_format.thinktag()[0]
                ids = self.tokenizer.encode(
                    frm_context, add_bos=add_bos, encode_special_tokens=True
                )
                exp_len = ids.shape[-1] + self.settings.max_response_tokens + 1

        return ids

    async def generate(self, user_message: str) -> AsyncGenerator[dict, None]:
        """
        Stream a response for *user_message*.

        Yields dicts:
            {"type": "token", "text": "..."}
            {"type": "tps", ...}
            {"type": "done", "eos_reason": "..."}
            {"type": "error", "message": "..."}
        """
        from exllamav3 import Job

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
            sampler = self._get_sampler()
            ids = self._build_input_ids(prompt_format)

            # Banned strings
            banned = list(self.settings.banned_strings)
            if self.settings.no_think:
                tt = prompt_format.thinktag()
                if tt[0]:
                    banned.append(tt[0])
                if tt[1]:
                    banned.append(tt[1])

            job = Job(
                input_ids=ids,
                max_new_tokens=self.settings.max_response_tokens,
                stop_conditions=stop_conditions,
                sampler=sampler,
                banned_strings=banned if banned else None,
            )
            self._current_job = job
            self.generator.enqueue(job)

            response_text = ""
            t_start = time.time()
            r = None

            while self.generator.num_remaining_jobs():
                for r in self.generator.iterate():
                    chunk = r.get("text", "")
                    if chunk:
                        response_text += chunk
                        yield {"type": "token", "text": chunk}

                    if r.get("eos"):
                        break

                # Let the event loop breathe
                await asyncio.sleep(0)

                if r and r.get("eos"):
                    break

            # Stats
            elapsed = time.time() - t_start
            eos_reason = r.get("eos_reason", "unknown") if r else "unknown"
            new_tokens = r.get("new_tokens", 0) if r else 0
            prompt_tokens = r.get("prompt_tokens", 0) if r else 0
            cached_tokens = r.get("cached_tokens", 0) if r else 0
            tps = new_tokens / elapsed if elapsed > 0 else 0
            prefill_tokens = prompt_tokens - cached_tokens
            prefill_tps = (
                prefill_tokens / r["time_prefill"]
                if r and r.get("time_prefill", 0) > 0
                else 0
            )

            yield {
                "type": "tps",
                "new_tokens": new_tokens,
                "prompt_tokens": prompt_tokens,
                "cached_tokens": cached_tokens,
                "tps": round(tps, 2),
                "prefill_tps": round(prefill_tps, 2),
                "elapsed": round(elapsed, 2),
            }

            yield {"type": "done", "eos_reason": eos_reason}

            # Save to context
            self.context[-1] = (user_message, response_text.strip())

        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            self._is_generating = False
            self._current_job = None

    def cancel(self):
        """Cancel the current generation."""
        if self._current_job is not None and self.generator is not None:
            try:
                self.generator.cancel(self._current_job)
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
            "model_dir": self.model_dir,
            "context_length": self.context_length,
            "context_turns": len(self.context),
            "available_modes": {
                k: v.description for k, v in prompt_formats.items()
            },
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_model_args(
    model_dir: str,
    devices: list[int],
    device_ratios: str | None,
    cache_size: int | None,
    cache_quant: str | None,
) -> object:
    """
    Build a namespace object mimicking the argparse args that
    exllamav3.model_init.init() expects.
    """
    import argparse
    from exllamav3 import model_init

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

    return parser.parse_args(argv)
