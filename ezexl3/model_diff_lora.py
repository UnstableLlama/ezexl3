# model_diff_lora.py - Layerwise KL-div of a base model vs the same model
# with a PEFT LoRA adapter applied, derived from Turboderp/exllamav3
# Source: https://github.com/turboderp-org/exllamav3/blob/master/eval/model_diff.py
# Rebuilt for ezexl3: loads ONE copy of the base model and runs each module
# twice per batch (adapter detached, then attached), so VRAM use matches
# ppl_layer_v2.py rather than the two-model model_diff.py. The adapter is
# parsed from PEFT format up front (mirroring exllamav3.model.lora.LoRA's
# transforms) because Linear.unload() clears attached LoRA tensors, which
# makes the stock LoRA class incompatible with a load/unload-per-module loop.

import sys, os
# Must be set before importing torch — otherwise the CUDA allocator is
# already initialized and the option is silently ignored.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import json
import math
import torch


def get_test_tokens(tokenizer, rows, eval_len=2048, eval_stride=2048):
    from datasets import load_dataset
    from exllamav3.util.progress import ProgressBar

    print(" -- Tokenizing dataset...")
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )

    eval_tokens = tokenizer.encode(dataset_text)
    if not torch.is_tensor(eval_tokens):
        eval_tokens = torch.tensor(eval_tokens)
    if len(eval_tokens.shape) == 1:
        eval_tokens = eval_tokens.unsqueeze(0)

    num_tokens = eval_tokens.shape[-1]
    seqs = []

    with ProgressBar("Tokenizing", rows) as pb:
        for a in range(0, num_tokens - eval_len, eval_stride):
            b = a + eval_len
            seqs.append(eval_tokens[:, a:b])
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break

    if not seqs:
        raise ValueError(f"Dataset too short for eval_len={eval_len}. Only {num_tokens} tokens found.")

    return torch.cat(seqs, dim=0)


def ppl(input_ids_, logits_, vocab_size_):
    from exllamav3.util.measures import compute_target_log_probs
    logprob_sum_ = 0.0
    logprob_count_ = 0
    chunksize = 10240
    b_ = 0
    while b_ < logits_.shape[0]:
        a_ = b_
        b_ = min(b_ + chunksize, logits_.shape[0])
        logits_f = logits_[a_:b_, :]
        target_ids = input_ids_[a_ + 1:b_ + 1].to(logits_.device)
        token_log_probs = compute_target_log_probs(logits_f, target_ids, vocab_size_)
        logprob_sum_ += token_log_probs.sum().item()
        logprob_count_ += target_ids.numel()
    return logprob_sum_, logprob_count_


def load_lora_map(model, lora_dir, lora_weight):
    """
    Parse a PEFT adapter into {linear_module_key: (A, B)} with fp16 CPU
    tensors, pre-transposed for x @ A @ B and with alpha/r scaling baked
    into B, mirroring exllamav3.model.lora.LoRA. Kept on CPU so each pair
    can be moved to the right device after its module loads.
    """
    from safetensors.torch import load_file as safe_load_file
    try:
        from exllamav3.modules import Linear
    except ImportError:
        from exllamav3.modules.linear import Linear

    config_path = os.path.join(lora_dir, "adapter_config.json")
    with open(config_path, encoding="utf8") as f:
        config = json.load(f)

    lora_r = config["r"]
    lora_alpha = float(config["lora_alpha"])
    if config.get("use_rslora", False):
        lora_alpha *= math.sqrt(lora_r)
    lora_scaling = lora_weight * lora_alpha / lora_r

    if config.get("fan_in_fan_out", False):
        raise ValueError("fan_in_fan_out mode is not supported")

    weights_st = os.path.join(lora_dir, "adapter_model.safetensors")
    weights_bin = os.path.join(lora_dir, "adapter_model.bin")
    if os.path.exists(weights_st):
        raw_tensors = safe_load_file(weights_st, device="cpu")
    elif os.path.exists(weights_bin):
        raw_tensors = torch.load(weights_bin, map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(f"No LoRA adapter found in {lora_dir}")

    modules_dict = {m.key: m for m in model}

    lora_map = {}
    skipped_keys = []

    for key, tensor in raw_tensors.items():
        if ".lora_A." not in key and ".lora_B." not in key:
            continue

        # Extract dotted path and lora half from the PEFT key, e.g.
        # "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
        parts = key.split(".")
        full_path = lora_half = None
        for j, p in enumerate(parts):
            if p in ("lora_A", "lora_B"):
                full_path, lora_half = ".".join(parts[:j]), p
                break
        if full_path is None:
            skipped_keys.append(key)
            continue

        # Match against model modules by suffix to handle any PEFT prefix
        target = None
        module_key = None
        path_parts = full_path.split(".")
        for start in range(len(path_parts)):
            candidate = ".".join(path_parts[start:])
            t = modules_dict.get(candidate)
            if t is not None and isinstance(t, Linear):
                target = t
                module_key = candidate
                break

        if target is None or target.is_sliced:
            skipped_keys.append(key)
            continue

        if tensor.dtype in (torch.bfloat16, torch.float32):
            tensor = tensor.to(torch.float16)

        # PEFT stores lora_A as [rank, in] and lora_B as [out, rank];
        # transpose for x @ A @ B
        tensor = tensor.T.contiguous()

        if lora_half == "lora_B" and lora_scaling != 1.0:
            tensor.mul_(lora_scaling)

        # Pad to match target dims (quantized layers may pad features)
        if lora_half == "lora_A" and tensor.shape[0] < target.in_features:
            padded = torch.zeros(target.in_features, tensor.shape[1], dtype=tensor.dtype)
            padded[:tensor.shape[0]] = tensor
            tensor = padded
        elif lora_half == "lora_B" and tensor.shape[1] < target.out_features:
            padded = torch.zeros(tensor.shape[0], target.out_features, dtype=tensor.dtype)
            padded[:, :tensor.shape[1]] = tensor
            tensor = padded

        a, b = lora_map.get(module_key, (None, None))
        if lora_half == "lora_A":
            a = tensor
        else:
            b = tensor
        lora_map[module_key] = (a, b)

    incomplete = [k for k, (a, b) in lora_map.items() if a is None or b is None]
    for k in incomplete:
        del lora_map[k]

    print(
        f" -- LoRA: {len(lora_map)} target modules "
        f"(r={lora_r}, alpha={lora_alpha:.0f}, scaling={lora_scaling:.4f})"
    )
    if skipped_keys:
        print(f" -- LoRA: skipped {len(skipped_keys)} unmatched keys")
    if incomplete:
        print(f" -- LoRA: dropped {len(incomplete)} modules missing an A or B half")
    if not lora_map:
        raise ValueError("No LoRA tensors matched the model; wrong adapter for this model?")

    return lora_map


@torch.inference_mode()
def main(args):
    # Defensive imports: some exllamav3 installs (e.g. partial/editable builds
    # inside containers) resolve the package as a PEP-420 namespace package, so
    # the top-level re-exports from __init__.py are not available. Fall back to
    # the explicit submodule paths in that case.
    try:
        from exllamav3 import Config, Model, Tokenizer
    except ImportError:
        from exllamav3.model.config import Config
        from exllamav3.model.model import Model
        from exllamav3.tokenizer import Tokenizer
    from exllamav3.util.memory import free_mem
    from exllamav3.util.measures import compute_kl_div, cosine_error, sqnr
    import time

    print(f" -- Loading model from: {args.model}")
    device = torch.device(f"cuda:{args.device}")
    print(f" -- Using device: {device}")

    # Inter-layer states for all rows (x2 variants) are the main VRAM cost at
    # large model sizes; optionally park them on another GPU or in system RAM
    # and move each batch to the compute device on demand.
    storage_device = None
    if args.storage_device is not None:
        sd = args.storage_device.strip().lower()
        storage_device = torch.device("cpu") if sd == "cpu" else torch.device(f"cuda:{int(sd)}")
        print(f" -- Storing inter-layer states on: {storage_device}")

    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(2048)
    tokenizer = Tokenizer.from_config(config)
    model = Model.from_config(config)
    vocab_size = tokenizer.actual_vocab_size
    print(f" -- Model created")

    print(f" -- Loading LoRA from: {args.lora}")
    lora_map = load_lora_map(model, args.lora, args.lora_weight)

    # Dataset
    all_eval_ids = get_test_tokens(tokenizer, args.rows)
    if args.gen_prompt:
        from exllamav3.util.misc import prepend_hf_chat_context
        all_eval_ids = prepend_hf_chat_context(tokenizer, all_eval_ids)
    print(f" -- Processing {len(model.modules)} layers...")

    # Inputs: A = base model, B = base model + LoRA
    states_a = list(all_eval_ids.split(args.batch_size))
    states_b = list(all_eval_ids.split(args.batch_size))
    all_eval_ids = list(all_eval_ids.split(args.batch_size))

    # Inference (layerwise: load -> forward all batches twice -> unload)
    for idx, module in enumerate(model.modules):

        logits_layer = module == model.modules[-1]
        layer_start = time.time()

        # Load module
        config.stc.begin_deferred_load()
        module.load(device if not module.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()

        # Collect this module's LoRA tensors, moved to wherever each target
        # Linear actually loaded. Attached/detached around the B forward pass
        # so the same loaded weights serve both variants.
        lora_pairs = []
        for m in module:
            ab = lora_map.get(m.key)
            if ab is not None:
                lora_pairs.append((m, ab[0].to(m.device), ab[1].to(m.device)))

        # Error measures
        max_diff = 0
        rfn_error_sum = 0
        cos_error_sum = 0
        sqnr_sum = 0

        # Similarity measures
        topk_max = args.topk_max
        logprob_sum = [0, 0]
        logprob_count = [0, 0]
        kl_div_sum_ab = 0
        kl_div_sum_ba = 0
        topk_hits_sum = [[0] * topk_max, [0] * topk_max]
        topk_hits_count = [[0] * topk_max, [0] * topk_max]
        topk_agreement_sum = [0] * topk_max
        topk_agreement_count = [0] * topk_max

        for b in range(len(states_a)):

            eval_ids = all_eval_ids[b]

            # Base forward (adapter detached)
            params_a = {}
            state_a = module.prepare_for_device(states_a[b], params_a)
            state_a = module.forward(state_a, params_a)

            # LoRA forward (adapter attached)
            for m, la, lb in lora_pairs:
                m.lora_a_tensors["lora"] = la
                m.lora_b_tensors["lora"] = lb
            params_b = {}
            state_b = module.prepare_for_device(states_b[b], params_b)
            state_b = module.forward(state_b, params_b)
            for m, la, lb in lora_pairs:
                m.lora_a_tensors.pop("lora", None)
                m.lora_b_tensors.pop("lora", None)

            # Measure error (on the compute device, before states move out)
            if not logits_layer:
                rows = state_a.shape[0]
                for j in range(rows):
                    sa = state_a[j].to(float)
                    sb = state_b[j].to(float)
                    cos_error_sum += cosine_error(sa, sb)
                    sqnr_sum += sqnr(sa, sb)
                    sa -= sb
                    rfn_error_sum += (torch.linalg.norm(sa, 'fro') / torch.linalg.norm(sb, 'fro').mean()).item()
                    sa.abs_()
                    md = ((sa.max().item()) / torch.linalg.norm(sb, 'fro').mean()).item()
                    max_diff = max(max_diff, md)
                    del sa, sb

            # Drop logits on last iteration
            if not logits_layer:
                if storage_device is not None:
                    state_a = state_a.to(storage_device)
                    state_b = state_b.to(storage_device)
                states_a[b] = state_a
                states_b[b] = state_b

            # Perplexity, KL-div
            if logits_layer:
                rows = state_a.shape[0]
                for j in range(rows):
                    x = (state_a[j], state_b[j])
                    input_ids = eval_ids[j]
                    top_indices = []

                    for i in [0, 1]:
                        logits = x[i][:-1, :]
                        logprob_sum__, logprob_count__ = ppl(input_ids, logits, vocab_size)
                        logprob_sum[i] += logprob_sum__
                        logprob_count[i] += logprob_count__

                        _, top_index = torch.topk(logits, topk_max, dim=-1)
                        top_index = top_index.cpu().view(-1, topk_max)
                        top_indices.append(top_index)
                        targets = input_ids[1:].view(-1, 1)

                        for t in range(topk_max):
                            top_slice = top_index[:, :t + 1]
                            hits = torch.eq(targets, top_slice)
                            row_hits = hits.any(dim=1)
                            topk_hits_sum[i][t] += row_hits.sum().item()
                            topk_hits_count[i][t] += top_slice.shape[0]

                    for t in range(topk_max):
                        top_slice_a = top_indices[0][:, :t + 1]
                        top_slice_b = top_indices[1][:, :t + 1]
                        hits = torch.eq(top_slice_a, top_slice_b)
                        row_hits = hits.all(dim=1)
                        topk_agreement_sum[t] += row_hits.sum().item()
                        topk_agreement_count[t] += top_slice_a.shape[0]

                    kl_vocab_size = min(vocab_size, x[0].shape[-1], x[1].shape[-1])
                    kl_div_sum_ab += compute_kl_div(x[0], x[1], kl_vocab_size).mean().item()
                    kl_div_sum_ba += compute_kl_div(x[1], x[0], kl_vocab_size).mean().item()

        # Final ppl, kld
        if logits_layer:
            perplexity = [math.exp(-logprob_sum[i] / logprob_count[i]) for i in (0, 1)]
            kl_div_ab = kl_div_sum_ab / args.rows
            kl_div_ba = kl_div_sum_ba / args.rows

        # Unload module (mirror upstream model_diff: close + free per module)
        module.unload()
        config.stc.close()
        free_mem()

        # Print error
        layer_time = time.time() - layer_start
        if not logits_layer:
            rfn_error = rfn_error_sum / args.rows
            cos_error = cos_error_sum / args.rows
            sqnr_ = sqnr_sum / args.rows
            print(
                f" -- {module.key:40}"
                f"   lora_mods: {len(lora_pairs):3}"
                f"   rfn_err: {rfn_error:.6f}"
                f"   max_diff/norm: {max_diff:.6f}"
                f"   sqnr: {sqnr_:9.6f}"
                f"   cos_err: {cos_error:.6f}"
                f"   time: {layer_time:6.2f}s"
            )

    # Perplexity for each variant
    print(f" -- Base perplexity: {perplexity[0]:11.8f}")
    print(f" -- LoRA perplexity: {perplexity[1]:11.8f}")

    # Probability of the test label being in the top K tokens, for each variant
    print(f" -- Base label in top-K:")
    for t in range(topk_max):
        a_acc_ = topk_hits_sum[0][t] / topk_hits_count[0][t]
        print(f"      K = {t+1}: {a_acc_:6.4f}")
    print(f" -- LoRA label in top-K:")
    for t in range(topk_max):
        a_acc_ = topk_hits_sum[1][t] / topk_hits_count[1][t]
        print(f"      K = {t+1}: {a_acc_:6.4f}")

    # Probability of exact top-K token match between variants
    print(f" -- Top-K agreement, base vs LoRA:")
    for t in range(topk_max):
        topk_agree_ = topk_agreement_sum[t] / topk_agreement_count[t]
        print(f"      K = {t+1}: {topk_agree_:6.4f}")

    # KLD, either way around
    print(f" -- KL divergence (base, LoRA): {kl_div_ab:11.8f}")
    print(f" -- KL divergence (LoRA, base): {kl_div_ba:11.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, help="Base model directory", required=True)
    parser.add_argument("-l", "--lora", type=str, help="PEFT LoRA adapter directory", required=True)
    parser.add_argument("-lw", "--lora_weight", type=float, help="Extra LoRA scaling factor", default=1.0)
    parser.add_argument("-r", "--rows", type=int, help="Number of rows", default=100)
    parser.add_argument("-tkm", "--topk_max", type=int, default=5, help="Max top-K interval to test")
    parser.add_argument("-d", "--device", type=int, help="CUDA device index", default=0)
    parser.add_argument("-sd", "--storage_device", type=str, default=None,
                        help="Where to keep inter-layer states between modules: a CUDA index (e.g. 1) or 'cpu'. "
                             "Default: keep them on the compute device")
    parser.add_argument("-bsz", "--batch_size", type=int, help="Batch size", default=1)
    parser.add_argument("-gp", "--gen_prompt", action="store_true", help="Prepend chat template generation prompt to every row")
    _args = parser.parse_args()
    main(_args)
