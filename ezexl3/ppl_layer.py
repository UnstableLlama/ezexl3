import sys, os
import argparse
import math

def get_test_tokens(tokenizer, rows, eval_len=2048, eval_stride=512):
    import torch
    from datasets import load_dataset
    from exllamav3.util.progress import ProgressBar
    
    print(" -- Tokenizing dataset...")
    dataset_text = "\n\n".join(
        load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
    )
    
    eval_tokens = tokenizer.encode(dataset_text)
    num_tokens = eval_tokens.shape[-1]
    seqs = []
    
    with ProgressBar("Tokenizing", rows) as pb:
        for a in range(0, num_tokens - eval_len, eval_stride):
            b = a + eval_len
            seqs.append(eval_tokens[:, a:b])
            pb.update(len(seqs))
            if len(seqs) >= rows:
                break
    
    return torch.cat(seqs, dim=0)

def ppl(input_ids_, logits_):
    import torch.nn.functional as F
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


def main(args):
    import torch
    from exllamav3 import Config, Model, Tokenizer
    from exllamav3.loader import SafetensorsCollection, VariantSafetensorsCollection
    from exllamav3.util.memory import free_mem
    import time
    import yaml

    print(f" -- Loading model from: {args.model}")
    device = torch.device(f"cuda:{args.device}")
    print(f" -- Using device: {device}")

    config = Config.from_directory(args.model)
    config.override_dynamic_seq_len(2048)
    tokenizer = Tokenizer.from_config(config)
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
                vstc.add_stc(o_keys, SafetensorsCollection(o_dir))
            config.stc = vstc

    # Dataset
    all_eval_ids = get_test_tokens(tokenizer, args.rows)
    print(f" -- Processing {len(model.modules)} layers...")

    # Inputs
    states = list(all_eval_ids.split(args.batch_size))
    all_eval_ids = list(all_eval_ids.split(args.batch_size))

    # Perplexity accumulation
    logprob_sum = 0
    logprob_count = 0

    # Inference
    for idx, module in enumerate(model.modules):
        logits_layer = module == model.modules[-1]
        layer_start = time.time()

        # Load module
        config.stc.begin_deferred_load()
        module.load(device if not module.caps.get("prefer_cpu") else "cpu")
        config.stc.end_deferred_load()

        for b in range(len(states)):
            state = states[b]
            eval_ids = all_eval_ids[b]

            params = {}
            state = module.prepare_for_device(state, params)
            state = module.forward(state, params)

            if not logits_layer:
                states[b] = state
            else:
                rows = state.shape[0]
                for j in range(rows):
                    logits = state[j]
                    input_ids = eval_ids[j]
                    logits_for_ppl = logits[:-1, :]
                    logprob_sum_, logprob_count_ = ppl(input_ids, logits_for_ppl)
                    logprob_sum += logprob_sum_
                    logprob_count += logprob_count_

        # Unload module
        module.unload()
        config.stc.close()
        free_mem()
        
        layer_time = time.time() - layer_start
        print(f" -- {module.key:40}   time: {layer_time:6.2f}s")

    # Final perplexity
    perplexity = math.exp(-logprob_sum / logprob_count)
    print(f" -- Perplexity: {perplexity:11.8f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type = str, help = "Model directory", required = True)
    parser.add_argument("-r", "--rows", type = int, help = "Number of rows", default = 100)
    parser.add_argument("-d", "--device", type = int, help = "CUDA device index", default = 0)
    parser.add_argument("-or", "--override", type = str, help = "Model tensor override spec (YAML)", default = None)
    parser.add_argument("-bsz", "--batch_size", type = int, help = "Batch size", default = 1)
    # Removing unused args to keep it clean
    _args = parser.parse_args()
    main(_args)