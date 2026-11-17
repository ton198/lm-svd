# exp7：每个奇异函数一个独立子网络，逐个训练、逐个冻结

日期：2026-11-16　状态：已批准（2026-11-16）；同日修订：子网络改为 BERT 形式（见“修订”一节）

## 为什么做

exp5 和 exp6 在没见过的 history 和 future 上都欠拟合。在 exp4 的两张评估表上（都减了 `g`），不管取多少阶，捕获的 energy 都停在 0.47–0.51，而精确 SVD 是 0.65（1 阶）到 0.95（128 阶）。

这两个实验的所有模式都共用一个网络（每侧一个 GPT-2 副本，128 个模式是它的 128 个输出），用 joint nesting 同时训练。exp6 的记录里有模式互相干扰的迹象：exp4 的 `val` 表上第 1 阶的 energy 在 5 万步时是 0.470，到 10 万步掉到 0.393，而 128 阶的总 energy 一直在 0.47–0.50。

exp7 检验用户提出的做法：

1. 每个奇异函数（每个 `u_ℓ`、每个 `v_ℓ`）有自己的子网络，子网络之间不共享任何可训练参数。
2. 按顺序训练：先训练第 1 个模式，训练好后冻结；再训练第 2 个，依此类推。
3. 全部训练完后做一次对齐，把学到的函数重新组合成正交、按大小排序的一组。

要回答的第一个问题是：**独立的子网络是否比共享网络学得更好**。这由第 1 阶段单独回答（见"怎么读结果"）。

### 和论文的关系

NeuralSVD 论文（arXiv:2402.03655）的 sequential nesting 就是这个思路的"同时训练"版本：第 ℓ 个模式的梯度是

```
∂L/∂f_ℓ = 2{ −T g_ℓ + Σ_{i≤ℓ} f_i ⟨g_i, g_ℓ⟩ }
        = −2 (T − Σ_{i<ℓ} f_i g_iᵀ) g_ℓ + 2 f_ℓ ‖g_ℓ‖²
```

也就是对"减掉前面模式之后的残差"做 1 阶拟合，并且前面的模式不接收这一项的梯度。论文说它适合参数互不共享的网络。论文和官方代码都不冻结，所有模式同时更新，靠梯度掩码实现（我们的 `utils.nestedlora.NestedLoRA(sequential=True)` 就是那份代码）。

exp7 改成分阶段冻结，原因有两个：所有子网络同时带梯度放不进一张显卡；在成对样本上，先在同一个样本上减掉已冻结模式的输出再训练，前面模式带来的噪声会被抵消，而掩码写法做不到这一点。

## 修订（2026-11-16）：子网络改为 BERT 形式

初版实现并合并后，用户决定更换子网络的结构：

1. **`scratch` 改成 BERT 的形式**：双向注意力的 transformer，对所有位置的 hidden state 做平均池化，从头训练。原来是因果注意力、取最后一个 token。
2. **新增预训练的双向模型 `roberta`**：RoBERTa-base 加平均池化。用户说的是“预训练的 BERT”；选 RoBERTa 是我的决定，原因是它和 GPT-2 用同一套 BPE（已核实：GPT-2 的 50257 个 token 在 RoBERTa 的词表里全部存在，只是编号不同），做一个编号映射就能直接输入，不用把文本解码后重新分词。原版 BERT 用 WordPiece，需要重新分词，而 history 的结尾经常切在词中间。
3. **`gpt2`（GPT-2 副本）暂时不用**：不再是默认项，也不在运行计划里。代码保留（我的决定，不花成本，以后可以直接对比；用户要求时再删除）。

理由（用户的判断）：每个奇异函数做的事是“读完一段文本，输出一个固定的向量”，这是双向编码器加池化的标准用途。我的保留意见：拟合的目标是 GPT-2 自己的内部量，预训练的双向模型不带这部分知识，可能不如 GPT-2 的副本；第 1 阶段的结果会回答这一点。

下面各节已按修订后的做法改写。训练目标、分阶段、冻结、对齐、评估、续跑都不变。

## 目标与非目标

**目标**

1. 实现分阶段训练：第 ℓ 阶段只训练第 ℓ 个模式的一对子网络，目标是减掉前 ℓ−1 个已冻结模式之后的残差。
2. 子网络类型可切换：从头训练的双向 transformer（BERT 结构），或预训练的 RoBERTa；GPT-2 副本保留为不使用的选项。
3. 训练目标、成对样本、评估表、评估指标都沿用 exp5/exp6，使数字可以直接和 exp6 比较。
4. 每个阶段结束时做一次对齐，同时报告对齐前和对齐后的结果。
5. 可以先只跑 1 个模式，之后在同一个输出目录里接着加模式，不重训已冻结的模式。

**非目标**

- 不改训练目标：仍是 `logit − 全词表均值 − g`。不做行中心化，不改 top-k。
- 不做"锚点 history"方案，不做可变权重的 joint nesting。
- 不做生成（lingen）。
- 不改 exp4、exp5、exp6 的行为、输出和 checkpoint 兼容性，只导入它们的函数。
- 不改 sample_store。

**反目标**（做成这样就算失败）

- 任何梯度流进已冻结的模式，或已冻结模式的参数在冻结后发生变化。
- 两个模式之间共享可训练参数。（冻结的教师模型的 unembedding 是共用的常数，不算。）
- 用报告表决定什么时候冻结或选哪个检查点。
- 子网络读到 `h + f`，或右子网络读到 `h`、左子网络读到 `f`。

## 考虑过的方案

**怎么让前面的模式不受后面的影响**

| 方案 | 结论 |
| --- | --- |
| **分阶段冻结，在同一样本上减去已冻结模式的输出（选定）** | 用户提出。只有一对子网络占梯度内存；前面模式的噪声在样本上抵消；对第 ℓ 个模式给出的期望梯度与论文 Eq. 5 相同 |
| 论文的 sequential nesting：所有模式同时训练，用掩码 | 所有子网络都要带梯度，GPT-2 副本放不下；成对样本上没有噪声抵消 |
| joint nesting 加随时间变化的权重 | 任意正权重的最优解相同，所以合法；但对独立子网络加 Adam 基本无效，也不去掉第一模式的噪声。不做 |

**子网络用什么**

| 方案 | 结论 |
| --- | --- |
| **每个函数一个从头训练的双向 transformer 加平均池化（选定，`--subnet scratch`，默认）** | 用户提出（修订）。BERT 的结构，输入是 GPT-2 的 token |
| **每个函数一个预训练的 RoBERTa-base 加平均池化（选定，`--subnet roberta`）** | 用户提出预训练的 BERT（修订）；用 RoBERTa 是因为它和 GPT-2 共用 BPE |
| 每个函数一个完整的 GPT-2 副本（`--subnet gpt2`，保留但暂不使用） | 和 exp6 的编码器相同，只改变"是否独立"这一个因素。初版的默认项；用户决定暂时不用 |
| 每个函数一个从头训练的因果 transformer、取最后一个 token | 初版的 `scratch`，已被双向加池化的版本替换 |
| 共用冻结的 GPT-2，每个模式只有小 MLP 头 | exp5 里冻结特征的上限更低（0.35–0.40），不做 |
| 共用冻结的 GPT-2，每个模式有自己的 LoRA 适配器或最后几层 | 没有先例，引入新变量，暂缓 |
| 右侧在固定的锚点 history 后面读 future | 小表上不训练就有 0.60（8 个锚点），但它在锚点上跑语言模型，不是独立子网络，用户决定不做 |

**冻结的时机**

| 方案 | 结论 |
| --- | --- |
| **留出数据上的 energy 不再上升，或到步数上限（选定）** | 我的决定。固定步数在不同模式、不同子网络类型下不可比 |
| 固定步数（例如 1000 步） | 用户最初的提议。exp4 的记录显示 1000 步时第一模式远未收敛。`--patience 0` 加 `--stage-steps N` 可以得到这个行为 |

## 已定的做法

### 数据、目标、评估表

全部沿用 exp5（`experiments.exp5_gpt2_mlp_table.train.prepare_run`、`draw_pairs`，`features.py` 里的 `last_hidden`、`future_only_logits`、`header_tokens`、`pair_targets`）：

- 每一步从 sample_store 抽 `--batch-size` 个独立的 (h, f) 对。
- 每对的目标 `target`（top_k 个数）是 `h + f` 的 logits 减全词表均值，在 f 的 header token 上取值，再减 `g`。`g` 和 header token 由冻结的教师模型对 `f` 单独前向得到。
- 评估表是 exp4 的 `train`、`val` 两张表减 `g`，各自带精确 SVD。exp7 不在这两张表上训练，两张表都是留出的（见 exp5 的说明）。
- **命名**：exp4 的目录名 `train`、`val` 在这里容易误解，所以 exp7 的文档、日志和输出文件里把 exp4 的 `train` 表（2048 × 51200）叫作**选择表**（字段名 `selection`），把 `val` 表（1024 × 25600）叫作**报告表**（字段名 `report`）。exp4 的文件和目录不改。

### 一个模式的两个子网络

```
左 u_ℓ:  h (64 tokens) ──编码器_ℓ^L──> 池化后的向量
                        ──LayerNorm(无仿射) → Linear(·, 1)──> 标量

右 v_ℓ:  f (32 tokens) ──编码器_ℓ^R──> 池化后的向量
                        ──LayerNorm(无仿射) → Linear(·, E)──> w_ℓ(f) ∈ R^E
         v_ℓ(f, z) = ⟨w_ℓ(f), ẽ_z⟩ / sqrt(E)
```

- `E` 是教师模型 unembedding 的维度（GPT-2 Small 是 768）。`ẽ_z` 是教师 unembedding 中 token z 的那一行减去所有行的均值，再除以全部元素的均方根；它是冻结的常数，不训练、不保存。算法同 exp6 `Exp6Network` 里的 `token_vectors`，但那是模块内部的 buffer，无法导入，exp7 里重写这几行。所有模式共用同一份常数，不是每个模式各存一份。
- 右函数取这个形式是因为目标本身满足 `target = ⟨s(h+f) − s(f), e_z − ē⟩`，所以每个右奇异函数都是"一个只依赖 f 的向量"与 `ẽ_z` 的内积。每个模式有自己的网络，直接输出完整的 E 维向量，不需要 exp6 的共享投影 `P`。
- 每个模式的左编码器、右编码器、两个头都是这个模式独有的。L 个模式共 2L 个编码器。

**编码器类型**（`--subnet`，默认 `scratch`）。每个编码器把一段 token 变成一个向量（“池化后的向量”）：

- `scratch`：随机初始化的 BERT 结构 transformer（`transformers.BertModel`，不带它自带的 pooler），双向注意力，默认 6 层、宽 384、6 个注意力头（`--scratch-layers`、`--scratch-width`、`--scratch-heads`），dropout 0，自己的 token embedding 和位置 embedding，词表与教师相同（直接输入 GPT-2 的 token 编号）。池化是对所有位置的最后一层 hidden state 取平均。
- `roberta`：预训练的 RoBERTa-base（`--roberta-path`，默认 `models/roberta-base`）的一个可训练副本，dropout 关闭。输入时把 GPT-2 的 token 编号按 token 字符串映射成 RoBERTa 的编号，前后加上 `<s>` 和 `</s>`。池化是对所有位置（含这两个特殊 token）的最后一层 hidden state 取平均。任何一个 GPT-2 token 在 RoBERTa 的词表里找不到时报错。
- `gpt2`（保留，暂不使用）：`experiments.exp5_gpt2_mlp_table.model.trainable_encoder(inference)`，即教师 transformer 的一个可训练副本，dropout 关闭，因果注意力；“池化”是取最后一个 token 的 hidden state。

平均池化、`<s>`/`</s>`、`scratch` 的大小是我的决定。序列长度固定（history 64、future 32），没有 padding，所以平均不需要掩码。

### 第 ℓ 阶段的一步

1. 抽一批样本，算出 `target (B, top_k)`、`header_ids`。
2. 前 ℓ−1 个模式（已冻结，`eval()`，`torch.no_grad()`）各自算出 `f_i (B,)` 和 `g_i (B, top_k)`。
3. 残差 `r = target − Σ_{i<ℓ} f_i[:, None] · g_i`。
4. 当前模式算出 `f (B,)` 和 `g (B, top_k)`，损失是 1 阶的成对 LoRA 损失：

   ```
   loss = −2 · mean_b [ f_b · mean_j ( r_bj · g_bj ) ] + mean_b(f_b²) · mean_{b,j}(g_bj²)
   ```

   这等于 `experiments.exp5_gpt2_mlp_table.loss.paired_loss` 在 `neigs=1` 时的值。复用它时需要一个 `NestedLoRA(neigs=1)` 实例提供掩码，并把 `f` 变形为 `(B, 1)`、`g` 变形为 `(B·top_k, 1)`。
5. 只更新当前模式的参数。

每个阶段新建优化器（Adam）。预训练的两种（`roberta`、`gpt2`）的编码器用 `--encoder-lr`（默认 2e-5），头用 `--lr`（默认 3e-4）；`scratch` 的全部参数用 `--lr`，带 1000 步线性 warmup（warmup 是我的决定）。`--lr` 的默认值随种类而定：`scratch` 是 1e-4，其余是 3e-4（2026-11-17 修订：在 256 × 1600 的有限表上，`scratch` 用 3e-4 时左编码器在 1500–2000 步后对所有 history 输出同一个向量，三次运行三次如此，加梯度裁剪也一样；1e-4 三次都没有出现）。每次评估检查两个编码器的输出是否还随输入变化，不变时打印 `WARNING`，训练不中断。

### 什么时候冻结

- 每 `--eval-every`（默认 1000）步评估一次。
- **选择指标**：选择表上，前 ℓ 个模式（已冻结的加当前的，不做对齐）在第 ℓ 阶的 captured energy。选择只看选择表，报告表只用于报告。
- 记住本阶段选择指标最高的那次评估时当前模式的权重。
- 阶段结束的条件（满足任一）：本阶段步数达到 `--stage-steps`（默认 30000）；或者本阶段已超过 `--min-stage-steps`（默认 5000）且选择指标连续 `--patience`（默认 5）次评估没有创新高。阶段的最后一步总是做一次评估。
- 阶段结束时：载入本阶段最好的权重，设 `requires_grad_(False)` 和 `eval()`，保存到 `modes/mode-XX.pt`，然后做对齐，再进入下一阶段。
- `--patience 0` 是纯固定步数：训练满 `--stage-steps` 步，冻结最后一步的权重，不做选择。这对应用户最初"训练若干步后冻结"的提议。
- 共训练 `--modes`（默认 8）个模式。

这些默认值是我的决定。

### 对齐

每个阶段结束时做一次（包括第 1 阶段，此时它只是重新定标）：

1. 从 sample_store 新抽 `--align-pairs`（默认 16384）对样本，全部模式只做前向。
2. 估计三个 ℓ×ℓ 矩阵：`Λ_f = E[f fᵀ]`，`Λ_g = E[g gᵀ]`（对所有 (f, z) 列平均），`C_ij = E[ f_i(h) · target(h, f, z) · g_j(f, z) ]`。这里用的是 `target`，不是残差。
3. `M = Λ_f^{-1/2} C Λ_g^{-1/2}`，对 `M` 做 SVD：`M = P S Qᵀ`。
4. 变换矩阵 `T_f = Λ_f^{-1/2} P S^{1/2}`，`T_g = Λ_g^{-1/2} Q S^{1/2}`；对齐后的特征是 `F T_f`、`G T_g`。

对齐后的 ℓ 阶重构是"在这组函数张成的空间里对 `target` 的最小二乘拟合"，各模式正交并按 `S` 从大到小排列。对齐只用 store 的样本估计，不用评估表。它只能重组已学到的函数，不改变任何子网络的权重，也不影响后续阶段的训练（后续阶段的残差仍用未对齐的冻结模式计算）。

### 评估与记录

- 每次评估：把前 ℓ 个模式在两张表上的特征拼成 `(N, ℓ)` 和 `(C, ℓ)`，调用 exp4 的 `compare_with_exact_svd`，指标同 exp5/exp6。已冻结模式在评估表上的特征不再变化，可以缓存。
- 调用时必须传 `ranks=(1, 2, …, ℓ)`。默认的 `RANKS` 只有 1、2、4、8、…，ℓ 为 3、5、6、7 时没有第 ℓ 阶的值，选择指标会读到一个与当前模式无关的数。选择指标取返回的 `captured_energy` 的最后一项。因此 exp7 的记录里阶数是 1 到 ℓ 逐个列出，和 exp5/exp6 的 1、2、4、8 不同；`metadata.json` 里的 `ranks` 相应改写。
- 每个阶段结束：用对齐后的特征再调用一次 `compare_with_exact_svd`，记为 `aligned`。同时记录 `Λ_f`、`Λ_g` 归一化后非对角元的最大绝对值（各模式之间有多不正交）。

### 输出

`data/experiment_results/exp7_independent_subnets/<dataset>__<model>/train-<名字>/` 下。`scratch` 和 `gpt2` 的名字就是种类名；`roberta` 的名字是 `--roberta-path` 的目录名（`train-roberta-base`、`train-minilmv2-l6-h384`），这样不同的预训练模型各有自己的目录（2026-11-17 修订：用户要求再跑一个更小的预训练模型 MiniLMv2 L6-H384，约 3000 万参数，与 `scratch` 同样大小，词表与 RoBERTa-base 相同）：

- `eval_history.jsonl`：每次评估一行，字段 `step`（全局步数）、`stage`、`stage_step`、`train_loss`、`kernel_energy`（`target` 的均方）、`residual_energy`（`r` 的均方）、`seconds_per_step`、`selection`、`report`。
- `stages.jsonl`：每个结束的阶段一行，字段 `stage`、`steps`、`best_step`、结束原因、选择指标的最好值、未对齐与对齐后的 `selection`/`report` 比较、上面的非对角元诊断。
- `modes/mode-XX.pt`：第 XX 个模式冻结时的权重（两个编码器和两个头）。
- `alignment.pt`：最近一次对齐的 `T_f`、`T_g`、`S`。
- `checkpoint.pt`：当前阶段号、阶段内步数、全局步数、当前模式的权重和本阶段最好的权重、优化器、历史、耐心计数、配置。
- `metadata.json`：全部设置和 exp5 的 `run_metadata` 内容。`run_metadata` 需要一个模型来数可训练参数，传第一个模式，所以 `trainable_parameters` 的含义是"每个模式"。

**续跑**：同一命令再次运行时从 `checkpoint.pt` 和 `modes/` 继续。配置不一致时报错，`--overwrite` 重新开始。只有 `--modes` 不参与一致性检查（`--stage-steps`、`--patience` 等都参与）：已完成 k 个模式的目录可以用更大的 `--modes` 接着训练第 k+1 个起的模式；小于已完成数量时报错。已结束的阶段不能延长，只能 `--overwrite` 重来。采样器用 `(seed, 全局步数)` 重新播种。

### 入口与配套

- 模块 `src/experiments/exp7_independent_subnets/`，`pyproject.toml` 加脚本 `exp7`，只有 `train` 一个子命令。
- 参数：`--subnet {scratch,roberta,gpt2}`（默认 `scratch`）、`--roberta-path`（默认 `models/roberta-base`，参与续跑时的一致性检查）、`--modes`、`--stage-steps`、`--min-stage-steps`、`--patience`、`--align-pairs`、`--scratch-layers`、`--scratch-width`、`--scratch-heads`、`--batch-size`（默认 128）、`--eval-every`、`--lr`、`--encoder-lr`、`--seed`、`--store-dir`、`--dataset`、`--model-path`、`--results-root`、`--device`、`--overwrite`，与 exp6 的同名参数含义一致。
- `bluehive_run/exp7-train.sbatch`，照 `exp6-train.sbatch` 写。
- README 加一节 exp7。

### 资源估计（未实测）

- `roberta` 和 `gpt2`：每个编码器约 1.25 亿参数。`gpt2` 一步的梯度内存与 exp6 相同（batch 128 约 16 GB），`roberta` 的大小相同，估计相近。每个已冻结模式多占约 1 GB 权重和两次只做前向的计算，所以第 8 阶段约 23 GB，每步时间大约是第 1 阶段的两到三倍。
- `scratch`（默认大小）：每个编码器约 3000 万参数，其中 token embedding 约 1900 万。
- `models/roberta-base` 需要先下载（`scripts/download_models.py --models roberta`，约 500 MB），本机和集群各一次。

### 测试

用随机初始化的小 GPT-2 和临时的小 store，照 `tests/test_exp6_gpt2_bilinear_table.py` 的方式：

- 两种子网络的输出形状：左 `(B,)`，右 `(B, top_k)`；右特征等于 `⟨w(f), ẽ_z⟩ / sqrt(E)`。
- 不同模式之间没有共享的可训练参数。
- 在一个完整网格的小例子上，分阶段写法对第 ℓ 个模式给出的梯度，等于 `NestedLoRA(sequential=True)` 对同一组特征给出的第 ℓ 个模式的梯度。
- 残差等于 `target` 减已冻结模式输出之和；ℓ=1 时等于 `target`。
- 训练一步后，已冻结模式的全部参数逐位不变，且没有梯度。
- 阶段结束的三种情况：到步数上限；耐心用尽；`--patience 0`。前两种冻结的是本阶段选择指标最好的权重，而不是最后的权重；`--patience 0` 冻结最后的权重。选择指标只读选择表，并且在 ℓ = 3 时确实随当前模式变化。
- 对齐：对一个已知的低秩矩阵，给出一组张成其前 L 维子空间但互不正交、尺度任意的特征，对齐后的 captured energy 等于精确 SVD 的 L 阶值，且对齐后的 Gram 矩阵是对角的。
- 续跑：在阶段中途中断后继续，阶段号、步数、最好权重、耐心计数都恢复；用更大的 `--modes` 续跑时不重训已冻结的模式；配置不同时报错。
- 小规模训练两个阶段后，`residual_energy` 在第 2 阶段低于第 1 阶段开始时的值。
- exp4、exp5、exp6 的现有测试全部通过。

## 留给实现者决定

- 模块内的文件划分；训练循环自己写（exp5 的 `run_training` 是单阶段的，不适用），可复用其中的检查点写法。
- 已冻结模式在评估表上的特征缓存在内存还是磁盘；冻结模式放在哪个设备、用什么精度做前向（只要测试能证明输出与训练时一致到浮点误差）。
- 对齐里 `Λ^{-1/2}` 的数值处理（特征值下限）。
- `scratch` 编码器的位置数上限、初始化细节。
- 本阶段最好权重是放在内存、`checkpoint.pt` 里还是单独的文件。

## 怎么读结果

第一步只跑第 1 阶段，`scratch` 和 `roberta` 各一次：

```bash
uv run exp7 train --subnet scratch --modes 1
uv run exp7 train --subnet roberta --modes 1
```

对照值（报告表，即 exp4 的 `val` 表，减 `g`）：共享网络的第 1 阶 energy 是 0.45–0.47，共享网络在任何阶数下的最好值是 0.51，精确 SVD 的第 1 阶是 0.654。

| 第 1 阶段的结果（报告表第 1 阶 energy） | 含义 | 下一步 |
| --- | --- | --- |
| 高于 0.51 | 一个独立子网络单独学第一模式，就超过了共享网络用 128 个模式得到的总和。独立是有效的 | 用胜出的子网络类型，以更大的 `--modes` 在同一目录续跑 |
| 0.47–0.51 | 独立去掉了干扰，但提升有限 | 仍然续跑到 8 个模式，看总 energy 能否超过 0.51 |
| 0.47 以下 | 瓶颈不在共享上，而在"只读 future 去预测 GPT-2 在没见过的 future 上的反应" | 停下来重新讨论右函数的输入（例如锚点方案） |
| `scratch` 远低于 `roberta` | 预训练是必要的 | 后续只用 `roberta` |
| `scratch` 接近或超过 `roberta` | 小网络足够，可以负担更多模式 | 后续用 `scratch`，并加大 `--modes` |
| 两种都低于 0.47 | 双向加池化没有解决问题 | 用保留的 `--subnet gpt2` 跑一次第 1 阶段作对照，再讨论 |

多阶段的结果另看两点：对齐前后 energy 的差（差大说明后面的模式在修补前面的误差）；`stages.jsonl` 里每个阶段新增的 energy 相对于精确 SVD 该阶新增的 energy。
