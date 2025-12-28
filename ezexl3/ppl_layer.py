# This is a stripped down version https://github.com/turboderp-org/exllamav3/blob/dev/eval/model_diff.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from exllamav3.util.file import disk_lru_cache
from exllamav3.util.progress import ProgressBar
from exllamav3.util.memory import free_mem
from exllamav3 import Config, Model, Tokenizer
from exllamav3.loader import SafetensorsCollection, VariantSafetensorsCollection
from datasets import load_dataset
import torch
import torch.nn.functional as F
import math
import yaml
from safetensors.torch import save_file

def save_tensor(tensor, path: str, tensor_name: str = None):
    if isinstance(tensor, dict):
        save_file({
            k: v for k, v in tensor.items()
        }, path)
    elif isinstance(tensor, list):
        save_file({
            f"tensor.{i}": t for i, t in enumerate(tensor)
        }, path)
    else:
        save_file({
            tensor_name or f"tensor": tensor
        }, path)


@disk_lru_cache("get_dataset_text")
def get_dataset_text(spec: dict):
    assert spec["dataset"] == "wiki2", "Only wiki2 implemented atm"
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split = "test")
        ["text"]
    )
    return dataset_text


def get_test_tokens(tokenizer, rows, eval_len = 2048, eval_stride = 512):
    with ProgressBar("Tokenizing", rows) as pb:
        dataset_spec = { "dataset": "wiki2" }
        eval_tokens = tokenizer.encode(get_dataset_text(dataset_spec))
        num_tokens = eval_tokens.shape[-1]
        seqs = []
        for a in range(0, num_tokens - eval_len, eval_stride):
            b = a + eval_len
            seqs.append(eval_tokens[:, a:b])
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break
    return torch.cat(seqs, dim = 0)[:, :]


def ppl(input_ids_, logits_):
    logprob_sum_ = 0.0
    logprob_count_ = 0
    chunksize = logits_.shape[1] * 10240 // logits_.shape[1]
    b_ = 0
    while b_ < logits_.shape[1]:
        a_ = b_
        b_ = min(b_ + chunksize, logits_.shape[1])
        logits_f = logits_[a_:b_, :].float() + 1e-10
        target_ids = input_ids_[a_ + 1:b_ + 1].to(logits_.device)
        log_probs = F.log_softmax(logits_f, dim = -1)
        token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        logprob_sum_ += token_log_probs.sum().item()
        logprob_count_ += target_ids.numel()
    return logprob_sum_, logprob_count_


@torch.inference_mode()
def main(args):

    print(f" -- Loading model from: {args.model}")
    device = torch.device(f"cuda:{args.device}")
    print(f" -- Using device: {device}")

    config = Config.from_directory(args.model)
    print(f" -- Config loaded")
    config.override_dynamic_seq_len(2048)
    print(f" -- Creating tokenizer...")
    tokenizer = Tokenizer.from_config(config)
    print(f" -- Creating model...")
    model = Model.from_config(config)
    print(f" -- Model created")

    # Override tensors
    if args.override:
        with open(args.override, "r") as f:
            comp = yaml.safe_load(f)
        sources = {s["id"]: s["model_dir"] for s in comp["sources"]}
        overrides = {o["key"]: sources[o["source"]] for o in comp["overrides"]}
        collections = {}
        for o_key, o_dir in overrides.items():
            if o_dir not in collections:
                collections[o_dir] = []
            collections[o_dir].append(o_key)
        if len(collections):
            vstc = VariantSafetensorsCollection(config.stc)
            for o_dir, o_keys in collections.items():
                print(f" -- Overriding from: {o_dir}:")
                for o_key in o_keys:
                    print(f"      {o_key}")
                vstc.add_stc(o_keys, SafetensorsCollection(o_dir))
            config.stc = vstc

    # Dataset
    print(f" -- Loading dataset...")
    all_eval_ids = get_test_tokens(tokenizer, args.rows)
    print(f" -- Tokenization complete, processing {len(model.modules)} layers...")

    # Inputs
    states = list(all_eval_ids.split(args.batch_size))
    all_eval_ids = list(all_eval_ids.split(args.batch_size))

    # Save input IDs
    if args.save_input_ids:
        print(f" -- Saving input IDs to: {args.save_input_ids}")
        save_tensor(all_eval_ids, args.save_input_ids, "input_ids")

    # Output logits
    save_logits = []

    # Perplexity accumulation
    logprob_sum = 0
    logprob_count = 0

    # Inference
    import time
    for idx, module in enumerate(model.modules):

        logits_layer = module == model.modules[-1]
        layer_start = time.time()

        # Load module
        config.stc.begin_deferred_load()
        module.load(device if not module.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()

        for b in range(len(states)):

            # Advance state
            state = states[b]
            eval_ids = all_eval_ids[b]

            params = {}
            state = module.prepare_for_device(state, params)
            state = module.forward(state, params)

            # Drop logits on last iteration
            if not logits_layer:
                states[b] = state

            # Copy logits to CPU if saving
            else:
                if save_logits:
                    save_logits.append(state.cpu().split(1))

            # Calculate perplexity
            if logits_layer:
                rows = state.shape[0]
                for j in range(rows):
                    logits = state[j]
                    input_ids = eval_ids[j]
                    
                    logits_for_ppl = logits[:-1, :]
                    logprob_sum_, logprob_count_ = ppl(input_ids, logits_for_ppl)
                    logprob_sum += logprob_sum_
                    logprob_count += logprob_count_

        # Save logits
        if logits_layer:
            if args.save_logits:
                print(f" -- Saving logits to: {args.save_logits}")
                save_tensor(state, args.save_logits, "logits")

        # Unload module
        module.unload()
        config.stc.close()
        free_mem()
        
        # Print layer stats
        layer_time = time.time() - layer_start
        print(f" -- {module.key:40}   time: {layer_time:6.2f}s")

    # Final perplexity
    perplexity = math.exp(-logprob_sum / logprob_count)
    print(f" -- Perplexity: {perplexity:11.8f}")


if __name__ == "__main__":
    print("Script starting...")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type = str, help = "Model directory", required = True)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    parser.add_argument("-d", "--device", type = int, help = "CUDA device index", default = 0)
    parser.add_argument("-or", "--override", type = str, help = "Model tensor override spec (YAML)", default = None)
    parser.add_argument("-si", "--save_input_ids", type = str, help = "Save input IDs (filename)", default = None)
    parser.add_argument("-sl", "--save_logits", type = str, help = "Save logits (filename)", default = None)
    parser.add_argument("-bsz", "--batch_size", type = int, help = "Batch size", default = 1)
    _args = parser.parse_args()
    main(_args)