# KV Cache Inference Acceleration Experiments

## 简介 
本项目探索并实现了针对大语言模型（LLM）的 KV Cache 压缩与推理加速方法。目前基线测试基于 `Pythia-70m`，并在长文本数据集上评估了不同注意力截断策略（如 StreamingLLM）的有效性及系统瓶颈。

## 运行环境与安装
确保你拥有 `conda` 环境，并安装了项目依赖。
```bash
conda create -n LLM python=3.10
conda activate LLM
pip install torch transformers datasets accelerate
```

## 使用方法
通过脚本启动评测（使用 `--method baseline` 或 `--method streamingllm`）：
```bash
# 运行基线模型
conda run -n LLM env HF_ENDPOINT=https://hf-mirror.com python baseline.py --method baseline

# 运行 StreamingLLM 截断优化模型
conda run -n LLM env HF_ENDPOINT=https://hf-mirror.com python baseline.py --method streamingllm

# 运行 Cross Layer KV 跨层复用优化模型
conda run -n LLM env HF_ENDPOINT=https://hf-mirror.com python baseline.py --method cross_layer_kv
```

## 实验结果与错误分析 (Error Analysis)

### 1. StreamingLLM 速度逐渐变慢与 TPOT 异常波动
**现象：** 运行 `StreamingLLM` 时，生成速度逐渐变慢，首字延迟（TTFT）和单字延迟（TPOT）有时会出现异常飙升。
**原因分析：**
1. Pythia 使用的 `GPTNeoX` 依赖 `DynamicLayer` 计算 KV 长度作为位置嵌入的依据。当我们通过列表切片截断 KV Cache 物理长度后（限制在 window_size + sink_size），框架会以为当前处理的 `seq_len` 退回去了。这不仅导致 RoPE 周期紊乱，也引发了内部的重复副本修复逻辑。
2. 此外，未截取原配的 Attention Mask 导致在高维矩阵相乘时，Mask 尺寸（不断增长）和 KV Cache 尺寸（被固定）不匹配，触发缓慢的基础回退（Fallback）路径。
**修正措施：**
给魔改的 `DynamicLayer` 注入 `self._seen_tokens` 计数器并重写 `get_seq_length`，使得注意力机制能正确获取已经生成过的全局 Token 数量。并在 Attention Forward 中添加 Patch ，同步将 `attention_mask` 的中间维度裁切到和截断后的 KV 一致。

### 2. Cross-Layer KV 导致 PPL 爆炸式劣化 (达到 3000+) 与推理速度倒退
**现象：** 使用免训练的跨层 KV (Cross-Layer KV) 复用时，推理速度竟然比 Baseline 还慢，且 PPL 飙升到了 3900+ 的灾难级别，模型完全“不知道自己在说什么”。

**原因分析与原理解释：**
1. **模型特征空间断崖式错位 (PPL爆炸的原因)**：
   Cross-Layer KV 的核心思想是让高层 Transformer 借用底层已经提取好的 KV 缓存，从而省去一半以上的计算和显存。但 Pythia 并没有针对这种跨层共享做过预训练。
   如果在最底层（如 Layer 0）就开始复用，此时 Layer 0 处理完的特征还非常“低级”。而 Layer 1 的算法（Query）是被训练为去匹配 Layer 1 专属的特征（Key）的。直接把 Layer 0 的特征强行喂给 Layer 1，就像是**“用英语的问题去检索法语的文档”**，注意力分布会在瞬间完全打乱，导致模型的语义表征彻底崩塌。
2. **GPU/CPU 强制同步阻塞了流水线 (速度倒退的原因)**：
   在最初编写的推理代码中，为了判别当前序列生成的位置，有一句 `int(cache_position.reshape(-1)[-1].item())` 代码，试图读取 GPU 里的特定数值。在深度学习框架中，`.item()` 会强行将张量从 GPU 拷贝回 CPU，这要求 GPU **必须停下所有正在并行的矩阵运算去等待 CPU**（即 CUDA Stream Synchronization）。这个可怕的阻塞直接抹杀了跳过 KV 计算带来的硬件提速。

**修正措施：**
1. **仅在特征稳定的高层复用 KV**：加入 `if layer_idx >= 4:` 限制条件。根据大模型特性，越靠近深层，网络所提取的特征越接近相同的表征空间（因残差连接引发的同质化）。我们保留前 4 层独立计算 KV 确保语义不流失，仅在特征最接近的后两层进行复用。这使得 PPL 从 3900+ 回落到了正常的折损范围（80~90上下）。
2. **剔除并重写跨设备阻塞的代码**：全面移除了 `forward` 前向传播循环里涉及 `.item()` 或强制拉回 CPU 同步的代码，使图计算完全常驻 GPU，吞吐率立刻从 60+ tokens/s 一跃提升至 110~120 tokens/s。

### 3. RoPE 旋转位置编码对齐崩塌
在严谨自回归评估下，最初 `StreamingLLM` 测得的 PPL 竟然狂飙至 300+。
**原因分析：** 这种断崖式下跌并非单纯的“忘记上下文”，而是由于模型使用了 RoPE（旋转位置编码）。在强制截断 KV Cache 物理长度后（例如把长度从 512 斩断至 132），模型层在推断接下来的 Token 时，会通过 Cache 长度错误地认为当前是第 133 个 Token。这导致了严重的绝对位置错位（Position Misalignment），新 Query 与旧 Key 的相对距离计算彻底瓦解。
**修正方法：** 在测试中对于每一步生成，强制手动传入准确的 `position_ids`，接管框架自动计算的位置，PPL 回归到了符合预期的程度（损失了部分精度，但不是彻底崩塌）。

### 4. GPU 评测延迟与全局热身 (Global Warmup)
在早期测速时，观察到前几个 sample 的生成延迟（TTFT 和 TPOT）慢了近一倍，之后才逐渐逼近稳态速率。
**原因分析：** PyTorch 与 GPU 驱动（CUDA）具备惰性初始化（Lazy Initialization）特性。在最开始的几轮计算中，框架需要进行 CUDA 上下文创建、算子内核（Kernel）的预热与编译、以及 GPU 内存池（Memory Allocator）的分配。如果把这部分系统时间算入首字延迟中，会产生严重的测速偏差。
**修正方法：** 引入**全局热身（Global Warmup）**机制。在正式进入评测循环进行计时之前，先使用 Dummy 数据让模型预先生成数十个 Token，并执行 `torch.cuda.synchronize()`。这能迫使所有的硬件与显存分配进入巅峰活跃状态，从而保证从 Sample 1 开始的测速就是一致且公平的。
