# 肿瘤患者压力性损伤风险预测模型

基于肿瘤患者临床特征的压力性损伤（压疮）风险预测模型开发项目。采用有序逻辑回归与MLP神经网络两种算法，对比筛选最优模型，最终输出临床可用的风险预测工具。

## 项目概况

- **样本量**：约 10 万条肿瘤患者数据
- **因变量**：基于 Braden 量表评分的压力性损伤风险等级（有序四分类）
  - 0 = 无风险（≥19分）
  - 1 = 低风险（15~18分）
  - 2 = 中风险（13~14分）
  - 3 = 高风险（≤12分）
- **自变量**：23 个特征（4 连续变量 + 18 二分类变量 + 1 名义变量）

## 目录结构

```
├── model-a-ordinal-regression/   # 模型A：有序逻辑回归（杨旭东）
├── model-b-mlp/                  # 模型B：MLP神经网络（李金城）
├── model-evaluation/             # 模型评估与优化（白伟琪）
├── clinical-tool/                # 临床工具开发（王增翔）
└── README.md
```

## 模块说明

| 模块 | 负责人 | 职责 | 交付节点 |
|------|--------|------|----------|
| [model-a-ordinal-regression](./model-a-ordinal-regression/) | 杨旭东 | 有序逻辑回归建模、OR值计算、特征重要性分析 | 合同生效后第90日 |
| [model-b-mlp](./model-b-mlp/) | 李金城 | MLP神经网络建模、权重可视化 | 合同生效后第90日 |
| [model-evaluation](./model-evaluation/) | 白伟琪 | 两模型统一评估、超参优化、模型文件管理 | 合同生效后第120日 |
| [clinical-tool](./clinical-tool/) | 王增翔 | 临床工具开发（软件界面/API接口） | 合同生效后第120日 |

## 环境要求

- Python >= 3.8
- 依赖：pandas, numpy, scikit-learn, statsmodels, matplotlib, seaborn, scipy
- 模型B额外依赖：PyTorch 或 TensorFlow

## 团队分工

详见 [01_团队分工通知](./docs/01_团队分工通知.pdf)
