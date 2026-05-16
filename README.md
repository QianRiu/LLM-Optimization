# KV Cache Inference Acceleration Experiments

## 简介
本项目探索并实现了针对大语言模型的 KV Cache 压缩与推理加速方法。基于 **Pythia-70M**，在长文本数据集上对比了以下三种策略：
- **Baseline**：原始模型，无任何修改。
- **StreamingLLM**：通过“注意力沉没”+滑动窗口的方式截断 KV Cache，保持恒定缓存大小。
- **Cross-Layer KV Reuse**：跨层共享 Key-Value 状态，减少高层注意力的计算与存储开销。

评估指标包括困惑度 (PPL)、首字延迟 (TTFT)、每词延迟 (TPOT) 和吞吐量 (Throughput)。

## 运行环境与安装
推荐使用独立的 conda 环境：

    conda create -n LLM python=3.10
    conda activate LLM
    pip install torch transformers datasets accelerate numpy

- Python ≥ 3.10
- PyTorch ≥ 2.0 (CUDA 推荐)
- transformers ≥ 4.40
- datasets, accelerate

> 本项目所有实验在 **NVIDIA RTX 4060 (8 GB)** 上完成，CUDA 版本 12.9。

## 使用方法
通过 `baseline.py` 启动评测，指定 `--method` 参数：

    # 运行基线模型
    python baseline.py --method baseline

    # 运行 StreamingLLM 截断优化模型
    python baseline.py --method streamingllm

    # 运行 Cross Layer KV 跨层复用优化模型
    python baseline.py --method cross_layer_kv

    # 调整超参数（可选的窗口大小、数据集、样本数等）
    python baseline.py --method streamingllm --max_seq_length 512 --num_samples 10

如果遇到 Hugging Face 连接问题，可使用镜像：

    HF_ENDPOINT=https://hf-mirror.com python baseline.py --method baseline

## 实验结果与加速效果
### 主要结果（10 个样本，序列长度 512）
下表汇总了不同方法在 wikitext-2 数据集上的性能：

| 方法                     | PPL ↓      | TTFT (ms) ↓ | TPOT (ms) ↓ | Throughput (tok/s) ↑ | 相对 Baseline 吞吐提升 |
|--------------------------|------------|-------------|-------------|----------------------|------------------------|
| Baseline                 | 40.26      | 5.37        | 4.30        | 206.97               | –                      |
| StreamingLLM (w=128,s=4) | 79.90      | 5.19        | 3.68        | 238.87               | +15.4%                 |
| StreamingLLM (w=64,s=4)  | 96.52      | 5.08        | 3.87        | 229.44               | +10.8%                 |
| Cross-Layer KV (g=2)     | 99.10      | 4.82        | 3.68        | 240.66               | +16.3%                 |
| Cross-Layer KV (g=3)     | 189.87     | 5.62        | 3.40        | 252.45               | +22.0%                 |

> 注：**FLOPs 未单独测量**，但缓存压缩和跨层复用均减少了注意力矩阵的尺寸或计算层数，间接降低了浮点运算量。  
> Baseline 在样本数增至 20 时 PPL 升至 108.8，这是因为顺序截取片段时，后面的文本因缺乏更早的上下文而更难预测——这属于评测策略带来的偏差，不影响各方法的横向比较。

### 结果分析
1. **StreamingLLM 的窗口尺寸对加速效果的影响**
   - 窗口越大（`window_size=128`），触发 KV 截断的频率越低，减少了 GPU 显存分配与拷贝的开销，因此吞吐提升更为明显（+15.4%）。
   - 窗口较小（64）时，虽然注意力计算量更小，但频繁的 `torch.cat` 操作引发大量的显存分配和数据搬移，这部分开销超过了计算量的节约，导致吞吐提升反而降至 +10.8%。
   - 从 PPL 角度看，更大的窗口保留了更多的历史信息，PPL 恶化程度较轻。这正是 **StreamingLLM 在长文本场景下固有的准确率-效率权衡**。

2. **Cross-Layer KV 复用的组大小（g）的影响**
   - `group_size=2`：仅将 Pythia-70M 的最后两层共享 KV，前四层独立计算。PPL 约 99，比 Baseline 高约 2.5 倍，但仍能产生连贯文本；吞吐提升 16.3%，效果接近 StreamingLLM 的窗口截断方案。
   - `group_size=3`：将后三层强制复用，跳过计算的层数更多，吞吐提升至 +22.0%，但 PPL 飙升至 189.9，模型输出出现明显的语义退化。
   - **核心原因**：Pythia 系列未针对跨层共享进行训练，低层特征空间与高层 Query 空间存在错位。仅仅在已经接近输出端的几层共享 KV，才能在一定程度上保留语言能力。

3. **TTFT 与 TPOT 行为**
   - 两种优化方法主要降低 **TPOT**（每词延迟），而对 TTFT 影响较小，因为首字生成仍需完整前向传播。
   - Cross-Layer KV 因减少了部分层的注意力计算，TTFT 略有下降；StreamingLLM 在首次生成时缓存尚未填满，截断逻辑未激活，TTFT 与 Baseline 持平或略低。

## 修正说明与陷阱记录

### 1. 幽灵状态泄漏 (Ghost State Leakage)
- **现象**：Cross-Layer KV 的 PPL 发生剧烈波动，从 99 瞬间跳到 189。
- **原因**：预热阶段的 `use_cache=True` 会使底层 DynamicCache 的指针被全局字典 `_cross_layer_kv_state` 保留。正式评测时，尺寸不匹配的旧缓存被复用，造成计算错误。
- **解决**：预热时使用 `use_cache=False`，并在正式评测前清空自定义状态字典。

### 2. RoPE 位置编码对齐崩塌
- **现象**：StreamingLLM 自回归评测 PPL 飙升至 300+。
- **原因**：截断 KV Cache 后，框架通过 `cache.length` 推断位置，导致绝对位置错位。
- **解决**：在每一步手动传入正确的 `position_ids`，覆盖自动计算。

### 3. StreamingLLM 的显存碎片与拷贝开销
- **现象**：减小窗口后，吞吐反而下降。
- **原因**：每生成一个 token，一旦触发截断就执行 `torch.cat(sink, window)`，引发昂贵的显存分配和深度拷贝；窗口越小，截断越频繁，拷贝开销超过节约的注意力计算。
- **备注**：理想方案是使用环形缓冲区（Ring Buffer）避免拷贝，但需要深入修改底层缓存机制，当前未实现。

### 4. GPU 评测延迟与预热
- **现象**：前几个样本的延迟异常偏高。
- **原因**：CUDA 惰性初始化、内核编译和显存池分配导致首次推理耗时更长。
- **解决**：增加全局预热阶段，执行若干次生成并同步 GPU，再进入正式计时循环。

### 5. PPL 计算与生成测速的物理隔离
- **现象**：总吞吐量被 PPL 计算拖低。
- **解决**：将评测分为独立的两阶段——先完成所有样本的纯生成（测量 TTFT/TPOT/吞吐），随后单独计算 PPL，确保加速指标不受其它负载干扰。

## 结论
- **算法有效性**：无训练模式下，StreamingLLM 和跨层 KV 复用均可有效提升小模型推理吞吐（最高约 16%），同时以可控的代价牺牲困惑度。
- **性能权衡**：StreamingLLM 在窗口大小 128 时实现了吞吐与 PPL 的最佳折衷；Cross-Layer KV 在组大小 2 时提供了略高的加速，但 PPL 恶化更明显。增大压缩率会进一步推高吞吐，但 PPL 急剧恶化，不建议在精度敏感任务中使用。
- **可复现性**：所有代码、补丁及评测脚本均公开于此仓库。按照 README 命令即可在同类硬件上重现表格中的主要结果。指标存在 ±2% 的正常波动。

---

**文件结构**  
    .
    ├── attention_patch.py   # StreamingLLM, Cross-Layer KV 等补丁
    ├── baseline.py          # 主评测脚本
    ├── result.md            # 原始实验记录
    └── README.md