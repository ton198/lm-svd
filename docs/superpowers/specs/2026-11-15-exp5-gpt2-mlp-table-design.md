# exp5：GPT-2 hidden state + MLP，直接输出 L_M 表的 top-50 列

日期：2026-11-15　状态：待批准

## 为什么做

目前两个 NeuralSVD 实验都没有把右奇异函数学好：

- **exp4**（`exp4_table_neural_svd`）的左右网络都是从零训练的 2 层 transformer，右网络的输入是 `f + z`（future 后面拼一个 header token）。
- **exp6**（已从仓库删除，代码在 `1842ca6^`）用冻结 GPT-2 + MLP，右网络输出 `W(f)`（rank × 768）再乘冻结的中心化 unembedding，损失覆盖全词表。结果是训练和验证一致、整体欠拟合。exp15 的诊断显示误差在右函数：保留学到的左特征、逐 future 重拟合右侧，probe KL 从 0.97 降到 0.07。

exp5 检验一个更直接的结构：右网络只看 `f`，一次前向直接输出这个 future 在论文 L_M 表里的全部 50 列，损失也只算这 50 列。它和 exp6 的差别有四点：输出从 rank × 768 降到 rank × 50；只拟合 top-50 header token，不拟合全词表；不经过 unembedding；第一模态固定，不进损失。

## 模型

```
左: h (64 tokens) ──冻结 GPT-2──> 最后一个 token 的 hidden state (768)
                  ──2 层 MLP──> (rank,)
右: f (32 tokens) ──冻结 GPT-2──> 最后一个 token 的 hidden state (768)
                  ──2 层 MLP──> (top_k, rank)   第 j 行 = 列 (f, j) 的 singular vector

重构: L_M[h, (f, j)] ≈ g(f, j) + u(h) · V(f)[j]
       g(f, j) = f 单独输入时第 j 个 header token 的 centered logit（固定的第一模态）
```

- **hidden state**：最后一层、过了 `ln_f` 的输出，即 `CausalLMInference.final_hidden_states(ids)[:, -1]`。左右两边各自单独过 GPT-2，网络从不读 `h + f`。
- **槽位 j 的含义**：`f` 单独输入时 logit 第 j 高的 token，和 exp1/exp4 的列顺序一致（future 为主序，future 内按 future-only logit 降序，并列时 token id 小的在前，同 `select_header_tokens`）。右网络不接收 header token 的 id。
- **top_k** 不写死为 50，从 exp4 inference 的 `val/tokens.npz` 读取（`header_token_ids.shape[1]`）。
- **GPT-2 冻结**，只训练两个 MLP。

## 固定第一模态

未处理的表里第一模态占 99% 以上的能量，它近似是"对所有 history 相同的常数 × 每列的列均值"。exp5 把它固定下来，不让网络学，也不进损失：

- 左边固定为 1，右边固定为 `g(f, j)`。
- NestedLoRA 的训练目标是残差 `K'(h, (f, j)) = K(h, (f, j)) − g(f, j)`，其中 `K` 是 centered logit（减去全词表均值）。

`g` 不需要学习：右网络本来就要把 `f` 单独过一次 GPT-2，同一次前向给出 `g` 和 header token。它对新 future 同样可用。

在本地 exp1 表（wikitext，GPT-2 Small，512 × 25,600）上实测：

| 从表里减掉 | 剩余能量 | 剩余部分里最大模态的占比 | σ₁/σ₂ 约为 |
| --- | ---: | ---: | ---: |
| 不减 | 100% | 99.6% | 46 |
| `g`（本设计） | 1.22% | 67% | 4 |
| 真正的列均值 | 0.42% | 11% | 1.2 |

`g` 和真正的列均值相关系数是 0.93，两者的差仍是一个对所有 history 相同的模态，会作为残差的第一模态留在损失里。用户已决定接受这一点，不加额外的可学习修正头。

## 目标与非目标

**目标**

1. 用 sample_store 的成对样本训练，每个样本都是新的 (h, f)，future 的种类不受样本文件限制。
2. 在 exp4 已有的训练表、验证表上评估：对两张表做同样的"减 `g`"处理，重算精确 SVD，用 exp4 的指标比较。
3. 回答两个问题：(a) 右函数能否拟合好；(b) 在没见过的 history 和 future 上是否保持（两张评估表的样本都不来自 sample_store 的同一次抽样，见"数据"）。

**非目标**

- 不输出全词表，不使用 unembedding，不做生成（lingen）。
- 不微调 GPT-2。
- 不加学习列均值的修正头，不做 column-mean 中心化（成对样本里每个 future 只配一个 history，算不了列均值）。
- 不新建表格式，不重跑 exp4 inference，不改 sample_store 的格式和 producer。
- 不改变 exp4 的行为和已有结果。

**反目标**（做成这样就算失败）

- 右网络以任何方式看到 header token 的 id 或 `h`。
- 评估表的处理和训练目标不一致（例如训练减了 `g`，评估却对照未处理表的 SVD）。

## 考虑过的方案

**右网络的输出**

| 方案 | 右网络输出 | 结论 |
| --- | --- | --- |
| **A（选定）** | MLP 直接输出 (top_k, rank) | 用户选定。输出维度小，只拟合论文关心的列，与 exp6 的差别最大 |
| B | W(f)（rank × 768）乘冻结的中心化 unembedding | 就是 exp6，已经跑过并欠拟合，不重复 |
| C | 把 50 个 header token 的 embedding 也输入右 MLP | 暂缓。A 的槽位 j 对应哪个 token 需要 MLP 自己从 hidden state 推断；如果 A 的结果显示这是瓶颈，再做 C |

**训练数据**

| 方案 | 结论 |
| --- | --- |
| **sample_store 成对样本（选定）** | 用户选定。期望上与网格相同（h、f 独立均匀抽取，损失的第一项只需要单个位置的表值，第二项只用特征），每个样本都是新的 future |
| exp4 的 fresh table（512 × 256 网格，每若干步换一张） | 可以缓存 hidden state、可以做列均值中心化，但样本池只有 9,488 个 future，一张表 13 万次前向只对应 256 个 future |

**第一模态**

| 方案 | 结论 |
| --- | --- |
| **固定为 1 × `g`，不进损失（选定）** | 用户选定。不需要学习，σ₁/σ₂ 从约 46 降到约 4 |
| 固定左边为 1，右边 `g` + 可学习修正头（回归到列均值） | 能把 σ₁/σ₂ 降到约 1.2，用户决定不做 |
| 不处理，让网络学第一模态 | 第一模态主导梯度，其余模态学得慢 |

## 已定的做法

### 数据

- **训练**：`data_pipline.sample_store.SampleSampler`，默认 store 是 `data/sample_stores/wikitext-103-train__gpt2-small__h64-f32`（`--store-dir`）。每个样本是 `(h, f, 全词表 raw logits)`，logits 是 GPT-2 读完 `h + f` 后最后一个位置的输出。
- **每个样本的训练目标**（top_k 个数）：
  1. `f` 单独过 GPT-2，得到最后一个 token 的 hidden state；由它算出 future-only logits，减全词表均值，取最高的 top_k 个 token 及其值 `g`。
  2. 把 store 里的 logits 减去全词表均值，取这 top_k 个 token 的值，得到 `K`。
  3. `K' = K − g`。
- **启动时检查**：store 的 `history_tokens`、`future_tokens`、`vocab_size` 必须与 exp4 评估表和 `--model-path` 的模型一致；store 记录的模型文件名和文件大小（`store.json` 的 `model_files`，不比较修改时间）必须与 `--model-path` 下的文件一致。否则报错。store 里还没有发布的 block 时等待，不报错。
- **与评估表的隔离**：store 从整个 wikitext-103 训练 stream 均匀抽窗口，样本里不记录起始位置，无法精确排除评估表的窗口。一个样本与验证表某个窗口部分重叠的概率约 0.1%。这一点写进 `metadata.json`，不做处理。存储的训练表同理，所以 exp5 里 `train` 和 `val` 两张评估表对模型来说都是"几乎没见过"的，区别只在大小。
- **评估表**：读 `data/experiment_results/exp4_table_neural_svd/<dataset>__<model>/inference/` 下的 `train/`、`val/`。目录里没有 `metadata.json` 时报错，提示先跑 `exp4 inference`。

### 损失

- `utils.nestedlora.NestedLoRA`，joint nesting，与 exp4 相同。
- 一个 batch 是 B 个成对样本：左特征 `(B, rank)`，右特征 `(B, top_k, rank)`，目标 `K'` 为 `(B, top_k)`。
- 第一项（算子项）只用配对的位置：`−2 · mean_i mean_j [ K'[i, j] · Σ_l mask_l · u_i[l] · v_ij[l] ]`。
- 第二项（度量项）用整个 batch：左边 B 个特征的二阶矩 × 右边全部 `B × top_k` 个特征的二阶矩。`h` 和 `f` 独立，所以这是无偏的，且比 `svd_inference/train.py` 里逐对计算的写法噪声小。
- 一种等价的实现：构造 `(B, B × top_k)` 的核，配对位置填 `B · K'[i, j]`，其余填 0，然后调用现有的 `compute_loss_kernel`。实现者可以选别的写法，但要有测试证明与上面的公式一致。

### 网络（以下是我的决定，用户未指定）

- 每个 MLP：`LayerNorm(768, 无仿射参数) → Linear(768, width) → GELU → Linear(width, out)`。左 `out = rank`，右 `out = top_k × rank`，reshape 成 `(top_k, rank)`。
- 输入加 LayerNorm 是沿用 exp6 的做法，GPT-2 hidden state 的尺度不适合直接进 MLP。
- `--mlp-width` 默认 4096，指中间层的大小。理由：右网络的输出（rank=128 时 6,400 个数）都是中间层的线性组合；exp6 的中间层是 1024、输出 196,608，我怀疑这是它欠拟合的原因之一。

### 训练

- `--batch-size` 默认 256 对（每步 12,800 个表值）。这是我的决定。
- `--neigs` 128，Adam，`--lr` 3e-4 常数，`--seed` 42，`--steps` 100000，`--eval-every` 1000，与 exp4 的默认值相同。
- 每一步都要把这批 `h` 和 `f` 各过一次 GPT-2（不缓存，样本不重复使用）。
- `SampleSampler` 每次在内存里放最多 100 个 block（6,400 个样本，约 1.3 GB），抽满 6,400 次后换一批。producer 跟不上时样本会重复，这是 store 本身的行为，exp5 不改。日志在每次评估时打印这一段的平均每步耗时。

### 评估

训练开始时对两张评估表各做一次：

1. 对表里每个 future 把 `f` 单独过 GPT-2，在 `tokens.npz` 里存的 header token 上取 centered logit 作为 `g`。不重新排序 header token：exp4 造表时是逐条前向，这里是成批前向，两者有 1e-5 量级的浮点差，相邻名次的 token 可能互换。校验改为带容差：每个 future 存的 header token 里最小的 `g`，不得比重算出的第 top_k 大的值低超过 1e-3，否则报错（说明模型或 top_k 对不上）。
2. `matrix' = matrix − g`（对每一行减同一个 `g`），重算精确 SVD（float64，保留前 256 个奇异向量），做法同 exp4 `load_table` 的 column-mean 分支。
3. 缓存表里所有 history、future 的 hidden state。
4. 在 `metadata.json` 里记录每张表减 `g` 后的剩余能量占比，以及残差第一模态的能量占比。

每 `--eval-every` 步，用 exp4 的 `compare_with_exact_svd` 把学到的特征与 `matrix'` 的精确 SVD 比较（rank 1, 2, 4, …, 128），打印用 `format_comparison`。指标：`sv_rel_err`、`left_overlap`、`right_overlap`、`energy`、`exact_energy`。

和 exp4 的可比性：exp5 的评估对象是"减 `g` 的表"，exp4 现有结果是"未处理的表"和"减列均值的表"，三者是不同的矩阵，`energy` 等数值不能直接横向比较，只能各自对照自己的 `exact_energy`。这一点写进 README。

### 输出

`data/experiment_results/exp5_gpt2_mlp_table/<dataset>__<model>/train/` 下：

- `eval_history.jsonl`：每次评估一行，字段 `step`、`train_loss`、`kernel_energy`（`K'` 的均方）、`train`、`val`。
- `checkpoint.pt`：MLP 权重、优化器、步数、历史、配置。配置不一致时拒绝续跑，`--overwrite` 重新开始。只保存 MLP，不保存 GPT-2。
- `metadata.json`：全部设置，另记 `mlp_width`、可训练参数量、hidden state 的定义、固定第一模态的定义、store 的配置、exp4 inference 目录、上面的隔离说明和剩余能量。

续跑：store 的内容会变，采样无法逐位复现。续跑时从 checkpoint 恢复权重、优化器和步数，采样器用 `(seed, step)` 重新播种。

### 入口与配套

- 模块 `src/experiments/exp5_gpt2_mlp_table/`，`pyproject.toml` 里加脚本 `exp5`，只有 `train` 一个子命令（`uv run exp5 train`）。
- `--dataset`（默认 `wikitext-large`，用来定位 exp4 的评估表）、`--model-path`（默认 `models/gpt2-small`）、`--results-root`、`--device`、`--overwrite` 与 exp4 一致。
- `bluehive_run/exp5-train.sbatch`，照 `exp4-train.sbatch` 写；注明需要 sample_store 已有数据（producer 可以同时在跑）。
- README 加一节 exp5，并更新开头的实验表和导航。

### 测试

照 `tests/test_exp4_table_neural_svd.py` 和 `tests/test_sample_store.py` 的方式，用随机初始化的小 GPT-2 和临时的小 store：

- 左输出形状 `(B, rank)`；右输出 `(B, top_k, rank)`，展平后第 `f × top_k + j` 行对应表的第 `f × top_k + j` 列。
- 训练时算出的 header token 和 `g` 与 `select_header_tokens` 对同一批 future 的结果一致。
- 从 store 样本得到的 `K` 等于直接用 GPT-2 算 `h + f` 的 centered logit 再取 header token；`K' = K − g`。
- 成对损失与上面的公式一致；当 batch 恰好构成完整网格的对角时，算子项与手算结果一致。
- 评估表减 `g` 后的 SVD 是 `matrix − g` 的 SVD；存的 header token 只是名次互换时照常加载，换成非 top_k 的 token 时报错。
- 评估时的 hidden state 缓存等于直接调用 `final_hidden_states(ids)[:, -1]`。
- 小规模训练若干步后，评估表的 `energy` 上升。
- 续跑能从 checkpoint 的步数继续；配置不同时报错；store 的窗口长度或词表大小不匹配时报错。
- GPT-2 的参数没有梯度，checkpoint 里没有 GPT-2 权重。
- exp4 和 sample_store 的现有测试全部通过。

## 留给实现者决定

- 模块内的文件划分。
- 训练循环自己写，复用 exp4 的 `compare_with_exact_svd`、`format_comparison`、`Table`、`table_columns`，以及 `utils.linalg.matrix_svd`。不改 exp4 的输出和 checkpoint 兼容性。
- 从 store 取 batch 的方式（主线程循环或 DataLoader worker）、评估时的分块大小、缓存放在哪个设备。
- 评估表减 `g` 后的 SVD 是否缓存到磁盘。

## 怎么读结果

| 现象 | 含义 | 下一步 |
| --- | --- | --- |
| 两张评估表的 `right_overlap`、`energy` 都接近精确值 | 右函数学好了，并且对没见过的 future 成立 | 加大 rank，接回生成 |
| rank 1 很好、更高的 rank 差 | 残差的第一模态（`g` 与列均值的差）仍在主导 | 加可学习的列均值修正头 |
| 所有 rank 都差，损失还在下降 | 没训练够，或 store 的新样本供应不足 | 延长训练，检查 producer 的速度 |
| 所有 rank 都差，损失已经停住 | 瓶颈在输入（f 单独过 GPT-2 的最后一个 token 信息不够），或槽位 j 对应哪个 token 难以推断 | 方案 C；改用多个位置的 hidden state；加宽 MLP |
