## 1. 任务定义与核心指标
本项目旨在基于 Amazon Reviews 2023 (5-Core) 子集，构建一个针对 **Industrial_and_Scientific**、**Musical_Instruments** 和 **CDs_and_Vinyl** 三个类别的 Top-10 序列推荐系统。

### 核心指标：NDCG@10
必须严格按照以下逻辑实现，这是评估的唯一硬性标准：
* **精确计算**：由于测试集每条序列仅有一个 Ground Truth（真实交互商品），理想状态下该商品排在第 1 名。因此，$IDCG@10$ 恒等于 $1.0$。
* **得分规则**：若真实商品在推荐列表（Top-10）中的排名为 $i$（$1 \le i \le 10$），则 $NDCG@10 = \frac{1}{\log_2(i+1)}$。若未进入前 10 名，得分为 0。

## 2. 数据工程深度细节

### 2.1 字段含义与坑点
* **核心 ID 区分**：推荐和预测的目标始终是 **`parent_asin`**（商品组唯一标识），而非 `asin`（具体 SKU），因为一个 parent 可能对应多个 SKU。
* **时序基准**：使用 **`timestamp`**（Unix 毫秒时间戳）进行严格升序排列，这是构建序列特征的唯一依据。
* **辅助特征**：
    * `review.jsonl`：包含 `rating` (1-5)、`text` (评论文本)、`verified_purchase` (是否真实购买) 等关键信息。
    * `meta.jsonl`：包含 `title` (商品标题)、`features` (卖点列表)、`description` (描述)、`categories` (类目层级)、`average_rating` 及 `bought_together` (图关系线索)。

### 2.2 数据划分规则（$N$ 个交互）
| 数据集 | 输入序列 | 预测目标（Ground Truth） |
| :--- | :--- | :--- |
| **Train** | 第 $1$ 到 $N-3$ 个商品 | 第 $N-2$ 个商品 |
| **Validation** | 第 $1$ 到 $N-2$ 个商品 | 第 $N-1$ 个商品 |
| **Test** | 第 $1$ 到 $N-1$ 个商品 | 第 $N$ 个商品 |

## 3. 技术路线：比较与融合方案

### 3.1 技术路线深度对比
| 方案路线 | 核心模型 | 难度等级 | 优劣势分析 |
| :--- | :--- | :--- | :--- |
| **ID 序列 Baseline** | SASRec / GRU4Rec / BERT4Rec | **优**：实现简单，适合作为首个稳定提交的版本。**劣**：忽略了丰富的文本和图结构信息。 |
| **文本/多模态增强** | BERT / SentenceTransformer | **优**：通过编码 Title 和 Review 捕捉语义，缓解冷启动。**劣**：对计算资源要求较高。 |
| **图增强推荐** | LightGCN / GraphSAGE | **优**：利用 `bought_together` 构建关联图，挖掘二阶关系。**劣**：构图复杂，训练较慢。 |
| **LLM 推荐/精排** | LLM Zero-shot / ARAG | **优**：利用大模型做 Zero-shot 排序，语义理解能力最强。**劣**：API 调用或本地推理开销巨大。 |

### 3.2 开发者融合建议：阶梯进化路径
为了追求高分，建议不要孤立地选择某种模型，而是采用以下融合路径：
1.  **初始阶段**：以 **SASRec** 为主体框架，因为它在 5-Core 数据集上表现最稳健，且实现逻辑清晰。
2.  **特征融合阶段**：引入 `meta.jsonl` 中的文本信息。使用 **Sentence-BERT** 将商品标题转化为 Embedding，并与原本的商品 ID Embedding 进行 **Add** 或 **Concat** 融合，作为模型的输入。
3.  **精排阶段（冲刺满分）**：采用 **Two-Stage** 结构。先用融合了文本特征的 SASRec 粗排出 Top-50，再引入 **LLM Zero-shot Ranker** 对这 50 个候选进行重排，输出最终的 Top-10。

## 4. 开发里程碑 (开发者视角)
1. 数据清洗。完成 ID 映射，处理 `null` 价格及异常数据，跑通 Popularity 和 SASRec Baseline，计算初始 NDCG。
2. 特征增强。加入 Title 文本 Embedding，尝试负采样优化，在验证集上进行调参及 Early Stopping。
3。 测试与产出。对三个类别分别生成 `_pred.jsonl` 文件，确保每个用户恰好有 10 个预测 ID。

## 5. 最终交付自检清单
* [ ] **可复现性**：代码是否固定了 Random Seed？是否提供了 Conda/Pip 环境清单？
* [ ] **文件格式**：预测结果是否为 `{"user_id": "...", "predictions": ["B099...", ...], "ground_truth": "..."}`？
* [ ] **报告深度**：是否包含模型架构图？是否对三个类别分别汇报了分数？是否进行了消融实验（证明文本或图特征的有效性）？