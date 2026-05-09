import torch
import time
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import numpy as np
from attention_patch import enable_streaming_llm, enable_cross_layer_kv

def calculate_metrics(model, tokenizer, dataset_name, num_samples, max_seq_length, device, method):
    model.eval()
    if dataset_name == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        # Assuming pg-19 or similar text dataset
        dataset = load_dataset(dataset_name, split="test")

    encodings = tokenizer("\n\n".join(dataset["text"][:num_samples * 10]), return_tensors="pt")

    nlls = []
    ttfts = []
    tpots = []
    throughputs = []

    seq_len = max_seq_length
    for i in range(min(num_samples, encodings.input_ids.size(1) // seq_len)):
        begin_loc = i * seq_len
        end_loc = begin_loc + seq_len
        trg_len = seq_len
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-1] = -100

        with torch.no_grad():
            if method == "baseline":
                outputs = model(input_ids, labels=target_ids)
                # target_ids has -100 at the last position, but model.loss computes average over valid tokens
                neg_log_likelihood = outputs.loss * (trg_len - 1)
                nlls.append(neg_log_likelihood)
            else:
                # Step-by-step autoregressive PPL to avoid information leakage from KV cache truncation
                past_key_values = None
                nll_sum = 0.0
                curr_input = input_ids[:, 0:1]
                
                # GPT-NeoX requires passing explicit position_ids to avoid RoPE misalignment 
                # when the KV Cache length is forcibly truncated.
                for step in range(trg_len - 1):
                    position_ids = torch.tensor([[step]], device=device)
                    outputs = model(
                        curr_input, 
                        past_key_values=past_key_values, 
                        use_cache=True,
                        position_ids=position_ids
                    )
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    target = input_ids[:, step+1]
                    loss = torch.nn.functional.cross_entropy(logits, target, reduction="none")
                    nll_sum += loss.item()
                    curr_input = input_ids[:, step+1:step+2]
                
                nlls.append(torch.tensor(nll_sum))

        # TTFT and TPOT measurement
        prompt_ids = encodings.input_ids[:, begin_loc:begin_loc+seq_len//2].to(device)
        attention_mask = torch.ones_like(prompt_ids)
        model.generate(prompt_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1)
        torch.cuda.synchronize()

        start_time = time.time()
        _ = model.generate(prompt_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1)
        torch.cuda.synchronize()
        ttft = time.time() - start_time
        ttfts.append(ttft)

        start_time = time.time()
        # Ensure we generate exactly 10 tokens for TPOT calculation
        out_ids = model.generate(prompt_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=11, min_new_tokens=11)
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # Total time -= TTFT, divide by tokens generated (10)
        tpot = (total_time - ttft) / 10.0
        tpots.append(tpot)

        throughput = 10.0 / total_time
        throughputs.append(throughput)
        print(f"Sample {i+1}: TTFT: {ttft*1000:.2f}ms, TPOT: {tpot*1000:.2f}ms, Throughput: {throughput:.2f} tokens/s")

    ppl = torch.exp(torch.stack(nlls).sum() / (len(nlls) * (seq_len - 1)))
    
    print("\n--- Final Results ---")
    print(f"PPL: {ppl.item():.4f}")
    print(f"Average TTFT: {np.mean(ttfts)*1000:.2f} ms")
    print(f"Average TPOT: {np.mean(tpots)*1000:.2f} ms")
    print(f"Average Throughput: {np.mean(throughputs):.2f} tokens/s")
    # Note: FLOPs estimation typically requires deepspeed or similar tools, excluded for simplicity in this baseline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "streamingllm", "cross_layer_kv"])
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    
    if args.method == "streamingllm":
        print("Applying StreamingLLM patch... (Window=128, Sink=4)")
        enable_streaming_llm(model, window_size=128, sink_size=4)
    elif args.method == "cross_layer_kv":
        print("Applying Cross-Layer KV patch... (group_size=2)")
        enable_cross_layer_kv(model, group_size=2)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    calculate_metrics(model, tokenizer, args.dataset, args.num_samples, args.max_seq_length, device, args.method)
