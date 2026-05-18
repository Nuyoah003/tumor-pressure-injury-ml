# 模型A：有序逻辑回归与多策略优化（Ordinal Logistic Regression & Multi-Strategy Optimization）

**负责人**：杨旭东

基于有序逻辑回归基线模型，逐步引入SMOTE过采样、代价敏感学习、阈值优化、模型集成等策略，用于预测肿瘤患者的压力性损伤风险等级（4级有序分类：无风险/低风险/中风险/高风险）。

## 目录结构

```
model-a-ordinal-regression/
├── src/                              # Python 源码
│   ├── ml_pipeline.py                # [策略1] 主流程：数据预处理 → 有序逻辑回归基线 → 评估
│   ├── smote_optimization.py         # [策略2] SMOTE 过采样优化实验
│   ├── 03_ensemble_cost_sensitive.py # [策略6] 代价敏感序数集成模型（实验对比脚本）
│   ├── 04_best_model_ensemble.py     # [生产] 综合最优模型：Top3软投票集成 + 阈值优化
│   └── 05_high_risk_model.py         # [生产] 高风险预警模型：XGBoost + 序数代价权重
├── reports/                          # 报告与图表
│   ├── 13_数据清洗质量确认报告.md
│   ├── 14_模型开发技术报告.md
│   └── figures/
└── README.md
```

## 各脚本说明

### `ml_pipeline.py` — 策略1：基线模型

完整的数据预处理流程与有序逻辑回归基线。

- 数据清洗、特征编码、标准化、分层划分（70/30）
- 基于 Proportional Odds 假设的有序逻辑回归（statsmodels OrderedModel）
- OR值 + 95% CI + P值计算
- 特征重要性排序与基线特征表（含 Kruskal-Wallis 检验）
- XGBoost / LightGBM / 有序逻辑回归三模型对比

### `smote_optimization.py` — 策略2：SMOTE过采样优化

在基线基础上引入 SMOTE 过采样解决类别不平衡问题。

- SMOTE / Borderline-SMOTE / SMOTE-ENN 三种过采样变体对比
- XGBoost / LightGBM / 有序逻辑回归 × 3种SMOTE = 9种配置
- 完整的评估指标体系：Cohen's Kappa、Macro/Weighted F1、G-mean、per-class P/R/F1、ROC-AUC

### `03_ensemble_cost_sensitive.py` — 策略6：代价敏感序数集成（实验脚本）

**实验对比脚本**，系统测试12种模型配置，用于论文方法对比。

**5大创新点：**

1. **序数距离代价矩阵**：`cost(i,j) = |i-j|^2`，惩罚与等级间距离成正比（将"高风险误判为无风险"的代价远高于"中风险误判为低风险"）
2. **SMOTE变体优选**：SMOTE / Borderline-SMOTE / SMOTE-ENN 三种过采样策略
3. **代价敏感训练**：基于代价矩阵的样本加权，XGBoost/LightGBM 支持混合权重（`0.5 × 逆频率 + 0.5 × 序数代价`），Logistic Regression 支持纯混合权重
4. **概率偏移阈值优化**：在验证集上搜索最优类别概率边界（非默认argmax），通过概率偏移微调决策
5. **软投票集成**：融合多模型概率预测，降低方差，三种集成策略（全部模型 / Top3+阈值 / XGB+LGB混合）

**实验结果（12个模型，测试集30,152样本）：**

| 配置 | Kappa | Macro F1 | G-mean | 高风险Recall |
|------|-------|----------|--------|-------------|
| LGB+BorderSMOTE+混合 | 0.2751 | **0.3873** | 0.3292 | 0.1525 |
| LR+SMOTE+混合权重 | 0.2635 | 0.3860 | **0.4337** | 0.3446 |
| XGB+SMOTE+序数代价 | 0.2477 | 0.3514 | 0.2457 | **0.4972** |
| XGB+SMOTE-ENN+混合 | 0.2338 | 0.3670 | 0.4191 | 0.1921 |
| **集成: Top3+阈值** | **0.2788** | 0.3859 | 0.3329 | 0.1525 |

**相对策略2的改进：**
- Kappa: 0.210 → 0.279 (+33.0%)
- Macro F1: 0.338 → 0.387 (+9.2%)
- G-mean: 0.398 → 0.434 (+8.9%)
- 中风险Recall: 0.219 → 0.350 (+60.0%)

### `04_best_model_ensemble.py` — 生产代码：综合最优模型

从策略6实验中提取的最优方案，**直接用于生产部署**。

**方案：Top3软投票集成 + 阈值优化**

- 三个基模型：LGB+BorderSMOTE+混合 / LR+SMOTE+混合 / LGB+SMOTE+混合
- 验证集搜索最优概率偏移阈值
- 软投票：三模型概率取平均后应用阈值

**性能：** Kappa=0.2788, Macro F1=0.3859, G-mean=0.3329

**适用场景：** 论文投稿、综合评估，追求整体一致性最高的方案。

### `05_high_risk_model.py` — 生产代码：高风险预警模型

从策略6实验中提取的高风险召回最优方案。

**方案：XGBoost + SMOTE + 纯序数距离代价权重**

- 单模型，结构简单
- 使用纯序数代价权重（不混逆频率），强化对远距离误判的惩罚
- 不调阈值，直接argmax预测

**性能：** Kappa=0.2477, 高风险Recall=0.4972, AUC(macro)=0.9075

**适用场景：** 临床筛查阶段，宁可误报也不漏掉高风险患者。

## 当前进度

- [x] 数据预处理流程（清洗、编码、标准化、分层划分）
- [x] 有序逻辑回归基线模型（策略1）
- [x] SMOTE 过采样优化实验（策略2）
- [x] 代价敏感序数集成模型实验（策略6）
- [x] 生产代码提取（综合最优版 + 高风险预警版）

## 关键结果（跨策略对比）

| 指标 | 策略1 基线 | 策略2 SMOTE最优 | 策略6 最优 | 提升幅度 |
|------|-----------|---------------|-----------|---------|
| Cohen's Kappa | 0.078 | 0.210 | **0.279** | +33.0% |
| Macro F1 | — | 0.338 | **0.387** | +9.2% |
| G-mean | 0.199 | 0.398 | **0.434** | +8.9% |
| 高风险 Recall | 0% | 0.571 | 0.497 | -12.9%* |
| 中风险 Recall | — | 0.219 | **0.350** | +60.0% |

> *高风险Recall在策略6的综合最优方案上为15.3%，但专门的预警方案（`05_high_risk_model.py`）达到49.7%。不同临床场景应选用不同模型。

## 运行方式

```bash
cd model-a-ordinal-regression/src

# 策略1：基线模型
python ml_pipeline.py

# 策略2：SMOTE优化实验
python smote_optimization.py

# 策略6：代价敏感集成实验（12个模型完整对比）
python 03_ensemble_cost_sensitive.py

# 生产部署：综合最优模型
python 04_best_model_ensemble.py

# 生产部署：高风险预警模型
python 05_high_risk_model.py
```

> **数据文件**：`20260511更正后数据.xlsx`（03/04/05）或 `清洗后数据.xlsx`（01/02）需放置在 `src/` 目录下。数据文件因体积较大不纳入版本控制。

## 环境依赖

```
Python 3.8+
xgboost
lightgbm
imbalanced-learn
scikit-learn
pandas, numpy, matplotlib, seaborn
statsmodels（仅 ml_pipeline.py）
```
