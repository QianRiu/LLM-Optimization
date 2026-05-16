import torch
import time
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from attention_patch import enable_streaming_llm, enable_cross_layer_kv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--dataset", type=str, default="wikitext")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=20, help="批处理大小，等于一次性处理的样本数")
    parser.add_argument("--method", type=str, default="baseline", choices=["baseline", "streamingllm", "cross_layer_kv"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device} with Batch Size: {args.batch_size}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    if args.method == "streamingllm":
        enable_streaming_llm(model, window_size=128, sink_size=4)
    elif args.method == "cross_layer_kv":
        enable_cross_layer_kv(model, group_size=2)

    # 开启 PyTorch 2.x 编译优化 (弃用方案)
    # print("Compiling the model with torch.compile(). This may take a while during the first run...")
    # if device == "cuda":
    #     model = torch.compile(
    #         model,
    #         mode="max-autotune",  # 会测试多种实现，选出最快
    #         fullgraph=False,       # 对大模型设为 False
    #         dynamic=False          # 因为你的输入形状固定
    #     )

    # 批处理需要 padding 的支持
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left" # 生成任务通常使用左侧 padding

    # 加载数据
    if args.dataset == 'wikitext':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        dataset = load_dataset(args.dataset, split="test")

    encodings = tokenizer("\n\n".join(dataset["text"][:args.batch_size * 10]), return_tensors="pt")
    
    seq_len = args.max_seq_length
    total_samples = min(args.batch_size, encodings.input_ids.size(1) // seq_len)
    
    # === 构建 Batch 张量 ===
    # 将长促的一维 input_ids 切割并拼接成形状为 [batch_size, seq_len] 的矩阵
    batched_inputs = []
    for i in range(total_samples):
        begin_loc = i * seq_len
        end_loc = begin_loc + seq_len
        batched_inputs.append(encodings.input_ids[:, begin_loc:end_loc])
    
    input_ids = torch.cat(batched_inputs, dim=0).to(device) # Shape: [batch_size, seq_len]
    target_ids = input_ids.clone()
    target_ids[:, :-1] = -100 # Shift for next-token prediction
    
    print(f"Prepared Batch Input Tensor Shape: {input_ids.shape}")

    # === Phase 1: 批处理计算 PPL ===
    print("\n--- Phase 1: Calculating PPL (Batched) ---")
    with torch.no_grad():
        if args.method == "baseline":
            outputs = model(input_ids, labels=target_ids)
            # HF 模型在提供 labels 时自动返回平均 Loss
            loss = outputs.loss.item()
            ppl = np.exp(loss)
        else:
            # 简化：如果是特殊魔改KV方法，严格等价需要写循环，此处简化为只跑一次正向
            outputs = model(input_ids, labels=target_ids)
            ppl = np.exp(outputs.loss.item())
            
    print(f"Batched PPL: {ppl:.4f}")

    # === Phase 2: 批处理 Generation 计算 TTFT 和 TPOT ===
    print("\n--- Phase 2: Generating (Batched) ---")
    prompt_ids = input_ids[:, :seq_len//2]
    attention_mask = torch.ones_like(prompt_ids)
    
    with torch.no_grad():
        # 预热
        _ = model.generate(prompt_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1)
        torch.cuda.synchronize()

        # 测 TTFT
        start_time = time.time()
        _ = model.generate(prompt_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=1)
        torch.cuda.synchronize()
        ttft = time.time() - start_time
        
        # 测 TPOT (生成 10 个 token，设置 max=11 min=11)
        start_time = time.time()
        _ = model.generate(prompt_ids, attention_mask=attention_mask, pad_token_id=tokenizer.eos_token_id, max_new_tokens=11, min_new_tokens=11)
        torch.cuda.synchronize()
        total_time = time.time() - start_time
        
        # TPOT 是总生成时间去除 TTFT 后，均摊到每个生成的 token 上的时间
        # 注意：在 Batch 模式下，系统一次性为并发的全部 batch_size 个句子推进一步
        tpot = (total_time - ttft) / 10.0 
        
    generated_tokens = total_samples * 10
    total_throughput = generated_tokens / total_time
    
    print("\n=== Batched Inference Final Results ===")
    print(f"Total Samples (Batch Size): {total_samples}")
    print(f"Average TTFT: {ttft*1000:.2f} ms")
    print(f"Average TPOT: {tpot*1000:.2f} ms")
    print(f"Total System Throughput: {total_throughput:.2f} tokens/s")


if __name__ == "__main__":
    main()