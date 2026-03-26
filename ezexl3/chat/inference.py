# Chat inference engine wrapping exllamav3's Generator/Job API.
#
# Adapted from exllamav3 examples/chat.py
# Original author: turboderp (https://github.com/turboderp-org/exllamav3)

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
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
# Conversation tree
# ---------------------------------------------------------------------------

@dataclass
class MessageNode:
    """A single message in the conversation tree."""
    id: str
    role: str               # "user" or "assistant"
    content: Optional[str]  # None while assistant is streaming
    children: List[str] = field(default_factory=list)   # child node IDs
    active_child: int = -1  # index into children, -1 = no children
    parent_id: Optional[str] = None


class ConversationTree:
    """Tree-structured conversation supporting branching via regen/edit."""

    def __init__(self):
        self.nodes: dict[str, MessageNode] = {}
        self.root_children: list[str] = []  # first-level (user) node IDs
        self.active_root: int = -1

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:12]

    def add_node(
        self, role: str, content: Optional[str], parent_id: Optional[str] = None
    ) -> str:
        """Create a new message node and attach it to the tree. Returns node ID."""
        nid = self._new_id()
        node = MessageNode(
            id=nid, role=role, content=content, parent_id=parent_id
        )
        self.nodes[nid] = node

        if parent_id is None:
            self.root_children.append(nid)
            self.active_root = len(self.root_children) - 1
        else:
            parent = self.nodes[parent_id]
            parent.children.append(nid)
            parent.active_child = len(parent.children) - 1

        return nid

    def get_active_path(self) -> list[tuple[str, Optional[str]]]:
        """Walk the active branch and return flat (user_msg, asst_msg) tuples."""
        if not self.root_children or self.active_root < 0:
            return []

        path: list[tuple[str, Optional[str]]] = []
        current_id: Optional[str] = self.root_children[self.active_root]

        while current_id is not None:
            node = self.nodes[current_id]
            if node.role != "user":
                break

            user_content = node.content
            # Follow to active assistant child
            if node.children and node.active_child >= 0:
                asst_id = node.children[node.active_child]
                asst_node = self.nodes[asst_id]
                path.append((user_content, asst_node.content))
                # Follow to next user message
                if asst_node.children and asst_node.active_child >= 0:
                    current_id = asst_node.children[asst_node.active_child]
                else:
                    current_id = None
            else:
                path.append((user_content, None))
                current_id = None

        return path

    def get_path_to_node(self, node_id: str) -> list[tuple[str, Optional[str]]]:
        """Get conversation path from root to the given node (inclusive)."""
        # Build ancestry chain
        chain: list[str] = []
        nid: Optional[str] = node_id
        while nid is not None:
            chain.append(nid)
            nid = self.nodes[nid].parent_id
        chain.reverse()

        path: list[tuple[str, Optional[str]]] = []
        i = 0
        while i < len(chain):
            node = self.nodes[chain[i]]
            if node.role == "user":
                user_content = node.content
                if i + 1 < len(chain):
                    asst_node = self.nodes[chain[i + 1]]
                    path.append((user_content, asst_node.content))
                    i += 2
                else:
                    path.append((user_content, None))
                    i += 1
            else:
                # Lone assistant at start (shouldn't happen normally)
                i += 1

        return path

    def delete_node(self, node_id: str):
        """Delete a node and all its descendants."""
        if node_id not in self.nodes:
            return

        # Collect all descendant IDs
        to_delete: set[str] = set()
        queue = [node_id]
        while queue:
            nid = queue.pop()
            to_delete.add(nid)
            n = self.nodes.get(nid)
            if n:
                queue.extend(n.children)

        # Detach from parent
        node = self.nodes[node_id]
        if node.parent_id is None:
            if node_id in self.root_children:
                self.root_children.remove(node_id)
                if self.root_children:
                    self.active_root = min(
                        self.active_root, len(self.root_children) - 1
                    )
                else:
                    self.active_root = -1
        else:
            parent = self.nodes[node.parent_id]
            if node_id in parent.children:
                idx = parent.children.index(node_id)
                parent.children.remove(node_id)
                if parent.children:
                    parent.active_child = min(
                        parent.active_child, len(parent.children) - 1
                    )
                else:
                    parent.active_child = -1

        # Purge
        for nid in to_delete:
            self.nodes.pop(nid, None)

    def navigate(self, node_id: str, sibling_index: int):
        """Switch which sibling is active at the given node's parent level."""
        node = self.nodes[node_id]
        if node.parent_id is None:
            if 0 <= sibling_index < len(self.root_children):
                self.active_root = sibling_index
        else:
            parent = self.nodes[node.parent_id]
            if 0 <= sibling_index < len(parent.children):
                parent.active_child = sibling_index

    def get_sibling_info(self, node_id: str) -> tuple[list[str], int]:
        """Return (sibling_ids, index_of_this_node)."""
        node = self.nodes[node_id]
        if node.parent_id is None:
            siblings = self.root_children
        else:
            siblings = self.nodes[node.parent_id].children
        return siblings, siblings.index(node_id)

    def get_last_active_leaf(self) -> Optional[str]:
        """Return the ID of the last node on the active path."""
        if not self.root_children or self.active_root < 0:
            return None
        nid = self.root_children[self.active_root]
        while True:
            node = self.nodes[nid]
            if node.children and node.active_child >= 0:
                nid = node.children[node.active_child]
            else:
                return nid

    def clear(self):
        self.nodes.clear()
        self.root_children.clear()
        self.active_root = -1

    # -- Serialisation -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "nodes": {
                nid: {
                    "id": n.id,
                    "role": n.role,
                    "content": n.content,
                    "children": n.children,
                    "active_child": n.active_child,
                    "parent_id": n.parent_id,
                }
                for nid, n in self.nodes.items()
            },
            "root_children": self.root_children,
            "active_root": self.active_root,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationTree":
        tree = cls()
        for nid, nd in data.get("nodes", {}).items():
            tree.nodes[nid] = MessageNode(
                id=nd["id"],
                role=nd["role"],
                content=nd.get("content"),
                children=nd.get("children", []),
                active_child=nd.get("active_child", -1),
                parent_id=nd.get("parent_id"),
            )
        tree.root_children = data.get("root_children", [])
        tree.active_root = data.get("active_root", -1)
        return tree

    @classmethod
    def from_flat_context(
        cls, context: list[tuple[str, Optional[str]]]
    ) -> "ConversationTree":
        """Upgrade a legacy flat context list to a tree."""
        tree = cls()
        last_asst_id: Optional[str] = None
        for user_msg, asst_msg in context:
            user_id = tree.add_node("user", user_msg, parent_id=last_asst_id)
            if asst_msg is not None:
                last_asst_id = tree.add_node(
                    "assistant", asst_msg, parent_id=user_id
                )
            else:
                last_asst_id = None
        return tree


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
        self.tree = ConversationTree()
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
            max_chunk_size=4096,
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
        from exllamav3 import model_init
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

    def _build_input_ids(
        self, prompt_format, prefix: str = "", context: list | None = None
    ):
        """Tokenize full context, trimming from head if too long."""
        think = self.settings.think
        ctx = context if context is not None else self.tree.get_active_path()

        frm_context = prompt_format.format(
            self.settings.system_prompt, ctx, think
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

        # Trim from head if context too long (trim the local copy, not the tree)
        while exp_len > self.context_length and len(ctx) > 1:
            ctx = ctx[1:]
            frm_context = prompt_format.format(
                self.settings.system_prompt, ctx, think
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

    # -- Generation helpers --------------------------------------------------

    async def _stream_response(
        self, context: list[tuple[str, Optional[str]]], asst_node_id: str
    ) -> AsyncGenerator[dict, None]:
        """Core streaming loop. Populates the assistant node with generated text."""
        from exllamav3 import Job

        prompt_format = self._get_prompt_format()
        stop_conditions = self._get_stop_conditions(prompt_format)
        sampler = self._get_sampler()
        ids = self._build_input_ids(prompt_format, context=context)

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

        loop = asyncio.get_running_loop()
        while self.generator.num_remaining_jobs():
            results = await loop.run_in_executor(
                None, self.generator.iterate
            )
            for r in results:
                chunk = r.get("text", "")
                if chunk:
                    response_text += chunk
                    yield {"type": "token", "text": chunk}

                if r.get("eos"):
                    break

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
        prefill_tps = (
            prompt_tokens / r["time_prefill"]
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

        # Save response to tree node
        self.tree.nodes[asst_node_id].content = response_text.strip()

    # -- Public generation methods -------------------------------------------

    async def generate(
        self, user_message: str, parent_id: Optional[str] = None
    ) -> AsyncGenerator[dict, None]:
        """
        Stream a response for *user_message*.

        parent_id: the assistant node after which to append this user turn.
                   None means append at root level (or after the current
                   last active assistant node).

        Yields dicts:
            {"type": "token", "text": "..."}
            {"type": "tps", ...}
            {"type": "done", "eos_reason": "...", "user_node_id": ..., "asst_node_id": ...}
            {"type": "error", "message": "..."}
        """
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
                self.tree.clear()

            # If no parent_id given, auto-detect: attach after last active leaf
            if parent_id is None:
                leaf = self.tree.get_last_active_leaf()
                if leaf is not None and self.tree.nodes[leaf].role == "assistant":
                    parent_id = leaf

            # Add user message and empty assistant node
            user_id = self.tree.add_node("user", user_message, parent_id=parent_id)
            asst_id = self.tree.add_node("assistant", None, parent_id=user_id)

            # Build context path up to the new user message
            context = self.tree.get_path_to_node(asst_id)

            # Emit start event so frontend knows the node IDs before tokens arrive
            yield {
                "type": "start",
                "user_node_id": user_id,
                "asst_node_id": asst_id,
            }

            async for event in self._stream_response(context, asst_id):
                if event["type"] == "done":
                    event["user_node_id"] = user_id
                    event["asst_node_id"] = asst_id
                yield event

        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            self._is_generating = False
            self._current_job = None

    async def regenerate(self, asst_node_id: str) -> AsyncGenerator[dict, None]:
        """
        Create a new sibling for the given assistant node and generate a
        fresh response.  The prefix (all turns before the branch point)
        stays identical, so the KV cache gives full prefix hits.
        """
        if not self.is_loaded:
            yield {"type": "error", "message": "Model not loaded"}
            return
        if self._is_generating:
            yield {"type": "error", "message": "Already generating"}
            return

        self._is_generating = True
        try:
            old_node = self.tree.nodes.get(asst_node_id)
            if old_node is None or old_node.role != "assistant":
                yield {"type": "error", "message": "Invalid assistant node"}
                return

            user_parent_id = old_node.parent_id
            if user_parent_id is None:
                yield {"type": "error", "message": "Orphan assistant node"}
                return

            # Create new assistant sibling under the same user parent
            new_asst_id = self.tree.add_node(
                "assistant", None, parent_id=user_parent_id
            )

            yield {"type": "start", "asst_node_id": new_asst_id}

            # Build context path
            context = self.tree.get_path_to_node(new_asst_id)

            async for event in self._stream_response(context, new_asst_id):
                if event["type"] == "done":
                    event["asst_node_id"] = new_asst_id
                yield event

        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            self._is_generating = False
            self._current_job = None

    async def edit_and_generate(
        self, user_node_id: str, new_content: str
    ) -> AsyncGenerator[dict, None]:
        """
        Edit a user message: create a sibling user node with *new_content*,
        then generate an assistant response for it.
        """
        if not self.is_loaded:
            yield {"type": "error", "message": "Model not loaded"}
            return
        if self._is_generating:
            yield {"type": "error", "message": "Already generating"}
            return

        self._is_generating = True
        try:
            old_node = self.tree.nodes.get(user_node_id)
            if old_node is None or old_node.role != "user":
                yield {"type": "error", "message": "Invalid user node"}
                return

            # Create sibling user node under the same parent
            parent_id = old_node.parent_id  # may be None for root-level
            new_user_id = self.tree.add_node(
                "user", new_content, parent_id=parent_id
            )
            new_asst_id = self.tree.add_node(
                "assistant", None, parent_id=new_user_id
            )

            yield {
                "type": "start",
                "user_node_id": new_user_id,
                "asst_node_id": new_asst_id,
            }

            context = self.tree.get_path_to_node(new_asst_id)

            async for event in self._stream_response(context, new_asst_id):
                if event["type"] == "done":
                    event["user_node_id"] = new_user_id
                    event["asst_node_id"] = new_asst_id
                yield event

        except Exception as e:
            yield {"type": "error", "message": str(e)}
        finally:
            self._is_generating = False
            self._current_job = None

    def delete_message(self, node_id: str):
        """Delete a message node and all its descendants."""
        self.tree.delete_node(node_id)

    def navigate_branch(self, node_id: str, sibling_index: int):
        """Switch which sibling is active at the given node's level."""
        self.tree.navigate(node_id, sibling_index)

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
        self.tree.clear()

    def save_session(self) -> dict:
        return {
            "settings": self.settings.to_dict(),
            "tree": self.tree.to_dict(),
        }

    def load_session(self, data: dict):
        if "settings" in data:
            self.settings = ChatSettings.from_dict(data["settings"])
        if "system_prompt" in data:
            self.settings.system_prompt = data["system_prompt"]
        if "banned_strings" in data:
            self.settings.banned_strings = data["banned_strings"]

        # Support both new tree format and legacy flat context
        if "tree" in data:
            self.tree = ConversationTree.from_dict(data["tree"])
        elif "context" in data:
            flat = [tuple(pair) for pair in data["context"]]
            self.tree = ConversationTree.from_flat_context(flat)

    def get_status(self) -> dict:
        return {
            "loaded": self.is_loaded,
            "generating": self.is_generating,
            "model_name": self.model_name,
            "model_dir": self.model_dir,
            "context_length": self.context_length,
            "context_turns": len(self.tree.get_active_path()),
            "available_modes": {
                k: v.description for k, v in prompt_formats.items()
            },
        }

    def get_tree(self) -> dict:
        """Return the full conversation tree for the frontend."""
        return self.tree.to_dict()


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
