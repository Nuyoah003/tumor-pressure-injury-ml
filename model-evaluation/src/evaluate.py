"""
模型评估与优化 —— 统一评估流程
=================================
负责人：白伟琪

职责：
  1. 设计统一评估框架，确保两模型评估口径一致
  2. 评估指标计算：混淆矩阵、多分类 AUC、Cohen's Kappa、ROC 曲线
  3. 对最优模型进行超参数优化（网格搜索 / 随机搜索）
  4. 两模型性能对比报告
  5. 模型文件统一存储与版本管理
  6. 模型推理接口封装

用法：
    cd model-evaluation

    # 完整评估（加载两个模型 + 对比报告）
    python src/evaluate.py \
        --model-a ../model-a-ordinal-regression/saved_models/best_ensemble_model.pkl \
        --model-b ../model-b-mlp/model/model_MLP_v2.pt \
        --model-b-config ../model-b-mlp/model/model_config_v2.json \
        --data ../model-b-mlp/data/

    # 仅评估单个模型
    python src/evaluate.py --model-b ../model-b-mlp/model/model_MLP_v2.pt \
        --model-b-config ../model-b-mlp/model/model_config_v2.json \
        --data ../model-b-mlp/data/

    # 推理模式
    python src/evaluate.py --predict --model-a <pkl路径> --input 患者数据.xlsx
    python src/evaluate.py --predict --model-b <pt路径> --model-b-config <json路径> --input 患者数据.csv
"""

import os, sys, json, time, warnings, pickle, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 尝试导入可选依赖 ──────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.font_manager as fm
    import seaborn as sns
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

try:
    import joblib
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False

try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


# ── MLP 模型类（模块级定义，torch.load 反序列化时需要） ──────────────
if _HAS_TORCH:
    class MLP(nn.Module):
        """多层感知机（与 model-b-mlp/src/train.py 架构一致）"""
        def __init__(self, input_dim, hidden_units=None, num_classes=4, dropout_rate=0.3):
            super().__init__()
            if hidden_units is None:
                hidden_units = [256, 128, 64]
            seq, in_d = [], input_dim
            for u in hidden_units:
                seq += [nn.Linear(in_d, u), nn.BatchNorm1d(u), nn.ReLU(), nn.Dropout(dropout_rate)]
                in_d = u
            seq.append(nn.Linear(in_d, num_classes))
            self.net = nn.Sequential(*seq)

        def forward(self, x):
            return self.net(x)

        @torch.no_grad()
        def predict_proba(self, x):
            return torch.softmax(self.forward(x), dim=1)

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, cohen_kappa_score,
    roc_curve, auc as sk_auc, classification_report,
    f1_score, precision_score, recall_score, accuracy_score,
)


# ══════════════════════════════════════════════════════════════════════
# 全局配置
# ══════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TARGET      = "风险等级"
RISK_LABELS = ["无风险(0)", "低风险(1)", "中风险(2)", "高风险(3)"]
K           = 4

BINARY_VARS = [
    "性别", "民族", "是否入住ICU", "放疗史", "化疗史",
    "营养风险", "血栓风险", "高血压", "糖尿病", "骨转移",
    "疼痛", "肿瘤转移", "激素治疗", "免疫抑制剂", "恶性积液",
    "电解质紊乱", "感染", "导管数目",
]
NOMINAL_VARS    = ["肿瘤类型"]
CONTINUOUS_VARS = ["年龄", "住院时长", "白蛋白", "BMI"]

TUMOR_NAMES = {
    1: "肺癌", 2: "乳腺癌", 3: "结直肠癌", 4: "宫颈癌", 5: "食管癌",
    6: "胃癌", 7: "非霍奇金淋巴瘤", 8: "甲状腺癌", 9: "卵巢癌", 10: "肝癌",
    11: "子宫内膜癌", 12: "前列腺癌", 13: "膀胱癌", 14: "肾癌", 15: "其他",
}


# ══════════════════════════════════════════════════════════════════════
# 中文字体
# ══════════════════════════════════════════════════════════════════════
def _setup_font():
    if not _HAS_MATPLOTLIB:
        return
    preferred = ["SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei",
                 "Noto Sans CJK JP", "Noto Serif CJK JP", "PingFang SC"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in preferred:
        if name in available:
            matplotlib.rcParams["font.family"] = name
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    return None

_setup_font()


# ══════════════════════════════════════════════════════════════════════
# 1. 模型加载
# ══════════════════════════════════════════════════════════════════════

def load_model_a(pkl_path):
    """
    加载 Model A 模型（有序逻辑回归集成模型 / 高风险预警模型）。

    参数:
        pkl_path: .pkl 文件路径（best_ensemble_model.pkl 或 high_risk_model.pkl）

    返回:
        model_pkg: dict 包含 models(或model), scaler, feat_names, threshold_offset, model_type
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Model A 模型文件不存在: {pkl_path}")

    if _HAS_JOBLIB:
        pkg = joblib.load(pkl_path)
    else:
        with open(pkl_path, "rb") as f:
            pkg = pickle.load(f)

    # 判断模型类型：集成模型（包含"models"列表）vs 单模型（包含"model"）
    if "models" in pkg:
        pkg["model_type"] = "ensemble"
        print(f"[Model A] 加载集成模型 ({len(pkg['models'])} 个基模型)")
    else:
        pkg["model_type"] = "single"
        print(f"[Model A] 加载单模型 ({type(pkg.get('model')).__name__})")

    print(f"[Model A] 特征数: {len(pkg['feat_names'])}")
    print(f"[Model A] 阈值偏移: {pkg.get('threshold_offset', 0):.4f}")
    return pkg


def load_model_b(pt_path, config_path=None):
    """
    加载 Model B 模型（MLP 神经网络）。

    参数:
        pt_path:    .pt 模型文件路径
        config_path: model_config JSON 路径（可选，与 pt 同目录时自动查找）

    返回:
        (model, config): PyTorch model + config dict
    """
    if not _HAS_TORCH:
        raise ImportError("需要安装 PyTorch: pip install torch")

    if not os.path.exists(pt_path):
        raise FileNotFoundError(f"Model B 模型文件不存在: {pt_path}")

    # 自动查找配置文件
    if config_path is None:
        dirname = os.path.dirname(pt_path)
        for name in ["model_config_v2.json", "model_config.json"]:
            candidate = os.path.join(dirname, name)
            if os.path.exists(candidate):
                config_path = candidate
                break
        if config_path is None:
            raise FileNotFoundError("未找到 Model B 配置文件，请通过 --model-b-config 指定")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 尝试加载完整模型（.pt 文件可能包含整个 MLP 对象或 state_dict）
    state = torch.load(pt_path, map_location=device)
    if isinstance(state, MLP):
        model = state
    elif isinstance(state, dict) and "net.0.weight" in state:
        model = MLP(config["input_dim"], config["hidden_units"],
                     config.get("num_classes", K), config.get("dropout_rate", 0.3))
        model.load_state_dict(state)
    elif hasattr(state, "state_dict"):
        model = MLP(config["input_dim"], config["hidden_units"],
                     config.get("num_classes", K), config.get("dropout_rate", 0.3))
        model.load_state_dict(state.state_dict())
    else:
        model = MLP(config["input_dim"], config["hidden_units"],
                     config.get("num_classes", K), config.get("dropout_rate", 0.3))
        model.load_state_dict(state)
    model.to(device).eval()

    print(f"[Model B] 加载 MLP (input={config['input_dim']}, hidden={config['hidden_units']})")
    print(f"[Model B] 设备: {device}")
    return model, config


# ══════════════════════════════════════════════════════════════════════
# 2. 数据预处理
# ══════════════════════════════════════════════════════════════════════

def _preprocess_raw(df, feat_names, scaler, ref_category_drop=True):
    """
    对原始 DataFrame 做预处理：One-hot 编码 + 特征对齐 + 标准化。

    参数:
        df:               原始 DataFrame（含原始列名）
        feat_names:       目标特征名列表
        scaler:           StandardScaler 实例
        ref_category_drop: 是否删去最后一个 One-hot 类别（Model A=True, Model B=False）
    """
    df = df.copy()

    # One-hot 编码肿瘤类型
    for col in NOMINAL_VARS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            if ref_category_drop and dummies.shape[1] > 1:
                dummies = dummies.iloc[:, :-1]  # 去掉最后一类作为参照
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)

    # 对齐特征
    for fn in feat_names:
        if fn not in df.columns:
            df[fn] = 0.0
    df = df[feat_names].astype(float)

    # 标准化连续变量
    if scaler is not None:
        cont_present = [c for c in CONTINUOUS_VARS if c in df.columns]
        df[cont_present] = scaler.transform(df[cont_present])

    return df


def load_test_data(data_source, model_a_pkg=None, model_b_config=None):
    """
    加载测试数据，返回两份预处理后的数据（分别适配两模型的特征空间）。

    参数:
        data_source:    数据目录或 test.csv 路径
        model_a_pkg:    Model A 的 pkg（需 feat_names + scaler）
        model_b_config: Model B 的 config（需 feature_names）

    返回:
        dict: {"y_true": array, "X_a": DataFrame or None, "X_b": DataFrame or None}
    """
    # 确定数据路径
    if os.path.isdir(data_source):
        test_path = os.path.join(data_source, "test.csv")
    else:
        test_path = data_source

    if not os.path.exists(test_path):
        print(f"[警告] 测试数据不存在: {test_path}")
        print("  请先运行 model-b-mlp/src/process.py 生成 train.csv / test.csv")
        return None

    df = pd.read_csv(test_path, encoding="utf-8-sig")
    y_true = df[TARGET].values.astype(int) if TARGET in df.columns else None
    print(f"[数据] 测试集: {len(df):,} 条, {df.shape[1]} 列")

    result = {"y_true": y_true}

    # 为 Model A 预处理
    if model_a_pkg is not None:
        result["X_a"] = _preprocess_raw(
            df, model_a_pkg["feat_names"], model_a_pkg.get("scaler"),
            ref_category_drop=True
        )
        print(f"[数据] Model A 特征: {result['X_a'].shape[1]} 维")

    # 为 Model B 预处理
    if model_b_config is not None:
        b_feat = model_b_config.get("feature_names", [])
        # Model B 没有保存 scaler，数据应该在 process.py 阶段已标准化
        # 这里使用 identity scaler
        b_scaler = StandardScaler()
        b_scaler.mean_ = np.zeros(len(CONTINUOUS_VARS))
        b_scaler.scale_ = np.ones(len(CONTINUOUS_VARS))
        result["X_b"] = _preprocess_raw(
            df, b_feat, b_scaler,
            ref_category_drop=False
        )
        print(f"[数据] Model B 特征: {result['X_b'].shape[1]} 维")

    return result


# ══════════════════════════════════════════════════════════════════════
# 3. 模型推理
# ══════════════════════════════════════════════════════════════════════

def predict_model_a(pkg, X):
    """
    Model A 推理。

    参数:
        pkg: load_model_a() 返回的 dict
        X:   (N, F) numpy array 或 DataFrame

    返回:
        y_pred:  (N,) 预测标签
        y_proba: (N, K) 预测概率矩阵
    """
    X_arr = X.values if hasattr(X, "values") else X
    threshold_offset = pkg.get("threshold_offset", 0.0)

    if pkg.get("model_type") == "ensemble":
        # 集成模型：平均各基模型概率
        models = pkg["models"]
        probas = [m.predict_proba(X_arr) for m in models]
        proba = np.mean(probas, axis=0)
    else:
        # 单模型
        clf = pkg["model"]
        proba = clf.predict_proba(X_arr)

    # 应用阈值偏移
    adjusted = proba.copy()
    for i in range(K):
        adjusted[:, i] += threshold_offset * (K - 1 - 2 * i) / (K - 1)
    y_pred = adjusted.argmax(axis=1)

    return y_pred, proba


def predict_model_b(model, X, config):
    """
    Model B 推理。

    参数:
        model:  PyTorch MLP 模型
        X:      (N, F) numpy array
        config: model config dict

    返回:
        y_pred:  (N,) 预测标签
        y_proba: (N, K) 预测概率矩阵
    """
    if not _HAS_TORCH:
        raise ImportError("需要安装 PyTorch: pip install torch")

    device = next(model.parameters()).device
    X_arr = X.values if hasattr(X, "values") else X
    X_tensor = torch.from_numpy(X_arr.astype(np.float32))

    # 批量推理
    batch_size = 1024
    proba_list, pred_list = [], []
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i:i + batch_size].to(device)
            proba = model.predict_proba(xb).cpu().numpy()
            pred_list.append(proba.argmax(axis=1))
            proba_list.append(proba)

    return np.concatenate(pred_list), np.concatenate(proba_list)


def predict_single(pkg_or_model, patient_data, model_type="a"):
    """
    单条患者推理接口。

    参数:
        pkg_or_model: Model A pkg dict（model_type="a"）或 Model B PyTorch model（model_type="b"）
        patient_data:  dict 如 {"性别": 1, "年龄": 68, ...}
        model_type:    "a" | "b"

    返回:
        dict: {"predicted_class": int, "risk_label": str, "probability": {...}}
    """
    row = pd.DataFrame([patient_data])

    if model_type == "a":
        pkg = pkg_or_model
        X = _preprocess_raw(row, pkg["feat_names"], pkg.get("scaler"), ref_category_drop=True)
        y_pred, proba = predict_model_a(pkg, X)
    else:
        model = pkg_or_model
        # 需要一个 config，这里只用于推理示例
        X_arr = row.values.astype(np.float32)
        device = next(model.parameters()).device
        xt = torch.from_numpy(X_arr).to(device)
        with torch.no_grad():
            proba = model.predict_proba(xt).cpu().numpy()
        y_pred = proba.argmax(axis=1)

    idx = int(y_pred[0])
    return {
        "predicted_class": idx,
        "risk_label": RISK_LABELS[idx],
        "probability": {RISK_LABELS[i]: round(float(proba[0, i]), 4) for i in range(K)},
    }


# ══════════════════════════════════════════════════════════════════════
# 4. 统一指标计算
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_proba, model_name=""):
    """计算全部评估指标，返回 dict"""
    y_arr = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    ypb = np.asarray(y_proba, dtype=float)

    metrics = {"model_name": model_name, "n_samples": len(y_arr)}
    cm = confusion_matrix(y_arr, yp, labels=list(range(K)))
    metrics["confusion_matrix"] = cm.tolist()
    metrics["accuracy"] = round(float(accuracy_score(y_arr, yp)), 4)
    metrics["kappa"] = round(float(cohen_kappa_score(y_arr, yp)), 4)
    metrics["macro_f1"] = round(float(f1_score(y_arr, yp, average="macro", zero_division=0)), 4)
    metrics["weighted_f1"] = round(float(f1_score(y_arr, yp, average="weighted", zero_division=0)), 4)

    # Per-class
    per_class = {}
    for i in range(K):
        per_class[i] = {
            "label": RISK_LABELS[i],
            "precision": round(float(precision_score(y_arr == i, yp == i, zero_division=0)), 4),
            "recall": round(float(recall_score(y_arr == i, yp == i, zero_division=0)), 4),
            "f1": round(float(f1_score(y_arr == i, yp == i, zero_division=0)), 4),
            "support": int((y_arr == i).sum()),
        }
    metrics["per_class"] = per_class

    # G-mean
    recalls = [per_class[i]["recall"] for i in range(K)]
    recalls_pos = [r for r in recalls if r > 0]
    metrics["g_mean"] = round(float(np.exp(np.mean(np.log(recalls_pos)))), 4) if recalls_pos else 0.0

    # AUC
    try:
        if ypb.ndim == 2 and ypb.shape[1] == K:
            y_bin = label_binarize(y_arr, classes=list(range(K)))
            metrics["auc_macro"] = round(float(roc_auc_score(y_arr, ypb, multi_class="ovr", average="macro")), 4)
            metrics["auc_weighted"] = round(float(roc_auc_score(y_arr, ypb, multi_class="ovr", average="weighted")), 4)
            per_auc = {}
            for i in range(K):
                if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y_bin):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], ypb[:, i])
                    per_auc[i] = round(float(sk_auc(fpr, tpr)), 4)
                else:
                    per_auc[i] = None
            metrics["per_class_auc"] = per_auc
    except Exception:
        metrics["auc_macro"] = metrics["auc_weighted"] = None
        metrics["per_class_auc"] = {i: None for i in range(K)}

    return metrics


def print_metrics(metrics):
    """打印指标报告"""
    cm = np.array(metrics["confusion_matrix"])
    name = metrics.get("model_name", "")
    print(f"\n{'='*65}")
    print(f"  {name} - 评估报告")
    print(f"{'='*65}")
    print(f"\n  混淆矩阵:")
    print(f"  {'':>14} {'预测0':>8} {'预测1':>8} {'预测2':>8} {'预测3':>8}")
    for i in range(K):
        rs = cm[i].sum()
        rp = cm[i] / rs * 100 if rs > 0 else np.zeros(K)
        s = f"  {RISK_LABELS[i]:>14}"
        for j in range(K):
            s += f" {cm[i,j]:>5}({rp[j]:>4.1f}%)"
        print(s)

    print(f"\n  各类别 Precision / Recall / F1:")
    print(f"  {'类别':>14} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>8}")
    for i in range(K):
        pc = metrics["per_class"][i]
        print(f"  {pc['label']:>14} {pc['precision']:>10.4f} {pc['recall']:>10.4f} {pc['f1']:>10.4f} {pc['support']:>8}")

    print(f"\n  综合指标:")
    print(f"    Accuracy:        {metrics['accuracy']:.4f}")
    print(f"    Cohen's Kappa:   {metrics['kappa']:.4f}")
    if metrics.get("auc_macro") is not None:
        print(f"    AUC (macro):     {metrics['auc_macro']:.4f}")
        print(f"    AUC (weighted):  {metrics['auc_weighted']:.4f}")
    print(f"    Macro F1:        {metrics['macro_f1']:.4f}")
    print(f"    Weighted F1:     {metrics['weighted_f1']:.4f}")
    print(f"    G-mean:          {metrics['g_mean']:.4f}")
    print()


# ══════════════════════════════════════════════════════════════════════
# 5. 可视化
# ══════════════════════════════════════════════════════════════════════

def _plot_confusion_matrix(cm, title, save_path):
    if not _HAS_MATPLOTLIB:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
            annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("预测标签", fontsize=12); ax.set_ylabel("真实标签", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight", dpi=300); plt.close()


def _plot_roc_curves(y_true, y_proba, title, save_path):
    if not _HAS_MATPLOTLIB:
        return
    y_arr = np.asarray(y_true, dtype=int).ravel()
    y_bin = label_binarize(y_arr, classes=list(range(K)))
    colors = ["#2E5496", "#C9541A", "#2E8B57", "#8B2252"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in range(K):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc_val = sk_auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2, label=f"{RISK_LABELS[i]} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("假阳性率 (FPR)", fontsize=12); ax.set_ylabel("真阳性率 (TPR)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight", dpi=300); plt.close()


def _plot_roc_comparison(roc_data, save_path):
    """roc_data: {"Model A": {"y_true":..., "y_proba":...}, "Model B": {...}}"""
    if not _HAS_MATPLOTLIB:
        return
    colors_map = {"Model A": "#E24B4A", "Model B": "#378ADD"}
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for cls_idx in range(K):
        ax = axes[cls_idx // 2][cls_idx % 2]
        for model_name, data in roc_data.items():
            y_arr = np.array(data["y_true"])
            ypb = np.array(data["y_proba"])
            y_bin = label_binarize(y_arr, classes=list(range(K)))
            if y_bin[:, cls_idx].sum() == 0:
                continue
            fpr, tpr, _ = roc_curve(y_bin[:, cls_idx], ypb[:, cls_idx])
            auc_val = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, color=colors_map.get(model_name, "#888"), lw=2,
                    label=f"{model_name} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
        ax.set_xlabel("假阳性率"); ax.set_ylabel("真阳性率")
        ax.set_title(f"ROC - {RISK_LABELS[cls_idx]}", fontweight="bold")
        ax.legend(loc="lower right", fontsize=9); ax.grid(True, alpha=0.3)
    plt.suptitle("ROC曲线对比 - Model A vs Model B", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight", dpi=300); plt.close()


def _plot_metrics_bar(results, save_path):
    """results: {"Model A": metrics_dict, "Model B": metrics_dict}"""
    if not _HAS_MATPLOTLIB:
        return
    keys = ["accuracy", "kappa", "auc_macro", "macro_f1", "weighted_f1", "g_mean"]
    labels = ["Accuracy", "Cohen's Kappa", "AUC (macro)", "Macro F1", "Weighted F1", "G-mean"]
    colors = ["#E24B4A", "#378ADD"]
    names = list(results.keys())
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (key, label) in enumerate(zip(keys, labels)):
        ax = axes[idx // 3][idx % 3]
        vals = [results[m].get(key) or 0 for m in names]
        x = np.arange(len(names))
        bars = ax.bar(x, vals, color=colors[:len(names)], edgecolor="white", width=0.5)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel(label, fontsize=12); ax.set_title(label, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    plt.suptitle("模型性能指标对比", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight", dpi=300); plt.close()


def _plot_confusion_comparison(results, save_path):
    """并排混淆矩阵"""
    if not _HAS_MATPLOTLIB:
        return
    names = list(results.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(7 * len(names), 6))
    if len(names) == 1:
        axes = [axes]
    for idx, name in enumerate(names):
        cm = np.array(results[name]["confusion_matrix"])
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = cm[i, j] / cm[i].sum() * 100 if cm[i].sum() > 0 else 0
                annot[i, j] = f"{cm[i, j]}\n({pct:.1f}%)"
        sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                    xticklabels=RISK_LABELS, yticklabels=RISK_LABELS,
                    linewidths=0.5, ax=axes[idx], cbar_kws={"shrink": 0.8})
        axes[idx].set_xlabel("预测标签", fontsize=10); axes[idx].set_ylabel("真实标签", fontsize=10)
        axes[idx].set_title(f"{name}\nKappa={results[name]['kappa']:.4f}  MF1={results[name]['macro_f1']:.4f}",
                            fontsize=11, fontweight="bold")
    plt.suptitle("混淆矩阵对比", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout(); plt.savefig(save_path, bbox_inches="tight", dpi=300); plt.close()


# ══════════════════════════════════════════════════════════════════════
# 6. 对比报告生成
# ══════════════════════════════════════════════════════════════════════

def generate_comparison_report(metrics_a, metrics_b, out_dir):
    """生成两模型对比报告：CSV + Markdown + 图表"""
    label_a = metrics_a.get("model_name", "Model A")
    label_b = metrics_b.get("model_name", "Model B")
    os.makedirs(out_dir, exist_ok=True)

    # ── CSV 对比表 ──
    rows = []
    for key, cname in [("n_samples","样本量"),("accuracy","Accuracy"),("kappa","Cohen's Kappa"),
                        ("auc_macro","AUC (macro)"),("auc_weighted","AUC (weighted)"),
                        ("macro_f1","Macro F1"),("weighted_f1","Weighted F1"),("g_mean","G-mean")]:
        va, vb = metrics_a.get(key), metrics_b.get(key)
        diff = round(vb - va, 4) if (va is not None and vb is not None) else None
        better = label_a if (diff is not None and diff < 0) else (label_b if (diff is not None and diff > 0) else "持平")
        rows.append({"指标": cname, label_a: va, label_b: vb, "差值(B-A)": diff, "优势": better})

    for i in range(K):
        pc_a = metrics_a.get("per_class", {}).get(str(i)) or metrics_a.get("per_class", {}).get(i, {})
        pc_b = metrics_b.get("per_class", {}).get(str(i)) or metrics_b.get("per_class", {}).get(i, {})
        for sub, cn in [("precision","Precision"),("recall","Recall"),("f1","F1")]:
            va = pc_a.get(sub)
            vb = pc_b.get(sub)
            diff = round(vb - va, 4) if (va is not None and vb is not None) else None
            better = label_a if (diff is not None and diff < 0) else (label_b if (diff is not None and diff > 0) else "持平")
            rows.append({"指标": f"{RISK_LABELS[i]} - {cn}", label_a: va, label_b: vb, "差值(B-A)": diff, "优势": better})

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, "comparison_metrics.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[对比表] -> {csv_path}")

    # ── Markdown 报告 ──
    md = [f"# 模型性能对比报告\n\n**Model A**: {label_a}  \n**Model B**: {label_b}\n\n---\n\n## 综合指标\n"]
    md.append(f"| 指标 | {label_a} | {label_b} | 优势 |")
    md.append(f"|------|{'-'*10}|{'-'*10}|------|")
    for r in rows[:8]:
        va_s = f"{r[label_a]:.4f}" if isinstance(r[label_a], float) else str(r[label_a])
        vb_s = f"{r[label_b]:.4f}" if isinstance(r[label_b], float) else str(r[label_b])
        md.append(f"| {r['指标']} | {va_s} | {vb_s} | {r['优势']} |")
    md.append("\n## 各类别指标\n")
    for i in range(K):
        md.append(f"### {RISK_LABELS[i]}\n")
        md.append(f"| 指标 | {label_a} | {label_b} | 优势 |")
        md.append(f"|------|{'-'*10}|{'-'*10}|------|")
        for r in rows[8:]:
            if RISK_LABELS[i] in r['指标']:
                va_s = f"{r[label_a]:.4f}" if isinstance(r[label_a], float) else str(r[label_a])
                vb_s = f"{r[label_b]:.4f}" if isinstance(r[label_b], float) else str(r[label_b])
                md.append(f"| {r['指标'].split(' - ')[1]} | {va_s} | {vb_s} | {r['优势']} |")
        md.append("")

    # 结论
    ka, kb = metrics_a.get("kappa") or 0, metrics_b.get("kappa") or 0
    fa, fb = metrics_a.get("macro_f1") or 0, metrics_b.get("macro_f1") or 0
    md.append("## 结论与建议\n")
    md.append(f"- 一致性 (Kappa): {label_a if ka > kb else label_b} 更优 ({max(ka,kb):.4f})")
    md.append(f"- 少数类识别 (Macro F1): {label_a if fa > fb else label_b} 更优 ({max(fa,fb):.4f})")
    if ka > kb and fa > fb:
        md.append(f"\n综合最优模型为 **{label_a}**。")
    elif kb > ka and fb > fa:
        md.append(f"\n综合最优模型为 **{label_b}**。")
    else:
        md.append(f"\n两模型各有所长，建议根据临床场景选择。")
    md.append("")

    md_path = os.path.join(out_dir, "comparison_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"[报告] -> {md_path}")

    # ── 图表 ──
    if _HAS_MATPLOTLIB:
        results = {label_a: metrics_a, label_b: metrics_b}
        _plot_confusion_comparison(results, os.path.join(out_dir, "confusion_matrix_comparison.png"))
        _plot_metrics_bar(results, os.path.join(out_dir, "metrics_comparison.png"))
        print(f"[图表] 混淆矩阵对比 + 指标柱状图 -> {out_dir}/")

    return df


# ══════════════════════════════════════════════════════════════════════
# 7. 超参数优化（网格搜索）
# ══════════════════════════════════════════════════════════════════════

def hyperparameter_optimization(model_type, X_train, y_train, X_val, y_val, out_dir="reports"):
    """
    超参数优化框架。支持对选定模型进行网格搜索。

    参数:
        model_type: "mlp" | "xgboost" | "lightgbm"
        X_train, y_train: 训练数据
        X_val, y_val:     验证数据
        out_dir:          输出目录

    返回:
        best_params: dict 最优参数组合
        best_score:  float 最优验证集得分
    """
    os.makedirs(out_dir, exist_ok=True)

    if model_type == "mlp":
        return _grid_search_mlp(X_train, y_train, X_val, y_val, out_dir)
    elif model_type == "xgboost":
        return _grid_search_xgb(X_train, y_train, X_val, y_val, out_dir)
    elif model_type == "lightgbm":
        return _grid_search_lgb(X_train, y_train, X_val, y_val, out_dir)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


def _grid_search_mlp(X_train, y_train, X_val, y_val, out_dir):
    """
    MLP 超参数网格搜索。
    搜索空间: hidden_units, dropout_rate, learning_rate, batch_size
    """
    if not _HAS_TORCH:
        print("[超参优化] 需要安装 PyTorch")
        return None, None

    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.utils.class_weight import compute_class_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[超参优化] MLP 网格搜索 | 设备: {device}")

    # 参数网格
    param_grid = [
        {"hidden": [128, 64], "dropout": 0.3, "lr": 1e-3, "bs": 256},
        {"hidden": [256, 128, 64], "dropout": 0.3, "lr": 1e-3, "bs": 512},
        {"hidden": [256, 128], "dropout": 0.4, "lr": 5e-4, "bs": 256},
        {"hidden": [128, 64, 32], "dropout": 0.2, "lr": 1e-3, "bs": 512},
        {"hidden": [512, 256, 128], "dropout": 0.3, "lr": 1e-3, "bs": 512},
    ]

    input_dim = X_train.shape[1]
    X_tr = torch.from_numpy(X_train.astype(np.float32)) if not isinstance(X_train, torch.Tensor) else X_train
    y_tr = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    X_v = torch.from_numpy(X_val.astype(np.float32)) if not isinstance(X_val, torch.Tensor) else X_val
    y_v = torch.from_numpy(np.asarray(y_val, dtype=np.int64))

    cw = compute_class_weight("balanced", classes=np.unique(y_tr.numpy()), y=y_tr.numpy())
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(device)

    best_score, best_params, best_state = -1.0, None, None
    results = []

    for cfg in param_grid:
        model = MLP(input_dim, cfg["hidden"], K, cfg["dropout"]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(weight=cw_tensor)
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=cfg["bs"], shuffle=True)

        # 训练 30 epochs
        model.train()
        for epoch in range(30):
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        # 验证
        model.eval()
        with torch.no_grad():
            logits = model(X_v.to(device))
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            y_pred = proba.argmax(axis=1)
            score = f1_score(y_v.numpy(), y_pred, average="macro", zero_division=0)

        cfg_label = f"hidden={cfg['hidden']}, dr={cfg['dropout']}, lr={cfg['lr']}, bs={cfg['bs']}"
        print(f"  {cfg_label} → Macro F1={score:.4f}")
        results.append({**cfg, "score": score})

        if score > best_score:
            best_score = score
            best_params = cfg
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(out_dir, "hyperparameter_search.csv"), index=False, encoding="utf-8-sig")
    print(f"\n[超参优化] 最优: {best_params} → Macro F1={best_score:.4f}")
    print(f"[超参优化] 结果 -> {out_dir}/hyperparameter_search.csv")

    return best_params, best_score


def _grid_search_xgb(X_train, y_train, X_val, y_val, out_dir):
    """XGBoost 网格搜索"""
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("[超参优化] 需要安装 xgboost")
        return None, None

    from sklearn.utils.class_weight import compute_class_weight
    cw = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    sample_w = np.array([cw[int(y)] for y in y_train])

    param_grid = [
        {"n_estimators": 200, "max_depth": 4, "lr": 0.1},
        {"n_estimators": 300, "max_depth": 6, "lr": 0.1},
        {"n_estimators": 400, "max_depth": 5, "lr": 0.05},
        {"n_estimators": 300, "max_depth": 4, "lr": 0.05},
    ]
    best_score, best_params = -1.0, None
    results = []

    for cfg in param_grid:
        clf = XGBClassifier(**cfg, objective="multi:softprob", num_class=K,
                            subsample=0.8, colsample_bytree=0.8,
                            random_state=42, n_jobs=-1, verbosity=0)
        clf.fit(X_train, y_train, sample_weight=sample_w)
        proba = clf.predict_proba(X_val)
        y_pred = proba.argmax(axis=1)
        score = f1_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"  {cfg} → Macro F1={score:.4f}")
        results.append({**cfg, "score": score})
        if score > best_score:
            best_score, best_params = score, cfg

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "hyperparameter_search_xgb.csv"),
                                  index=False, encoding="utf-8-sig")
    print(f"\n[超参优化] XGBoost 最优: {best_params} → Macro F1={best_score:.4f}")
    return best_params, best_score


def _grid_search_lgb(X_train, y_train, X_val, y_val, out_dir):
    """LightGBM 网格搜索"""
    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        print("[超参优化] 需要安装 lightgbm")
        return None, None

    param_grid = [
        {"n_estimators": 200, "max_depth": 4, "lr": 0.1},
        {"n_estimators": 300, "max_depth": 6, "lr": 0.1},
        {"n_estimators": 400, "max_depth": 5, "lr": 0.05},
        {"n_estimators": 300, "max_depth": 4, "lr": 0.05},
    ]
    best_score, best_params = -1.0, None
    results = []

    for cfg in param_grid:
        clf = LGBMClassifier(**cfg, objective="multiclass", num_class=K,
                             subsample=0.8, colsample_bytree=0.8,
                             random_state=42, n_jobs=-1, verbose=-1)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)
        y_pred = proba.argmax(axis=1)
        score = f1_score(y_val, y_pred, average="macro", zero_division=0)
        print(f"  {cfg} → Macro F1={score:.4f}")
        results.append({**cfg, "score": score})
        if score > best_score:
            best_score, best_params = score, cfg

    pd.DataFrame(results).to_csv(os.path.join(out_dir, "hyperparameter_search_lgb.csv"),
                                  index=False, encoding="utf-8-sig")
    print(f"\n[超参优化] LightGBM 最优: {best_params} → Macro F1={best_score:.4f}")
    return best_params, best_score


# ══════════════════════════════════════════════════════════════════════
# 8. 主流程
# ══════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="模型评估与优化 —— 统一评估流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整两模型对比评估
  python src/evaluate.py --model-a ../model-a-ordinal-regression/saved_models/best_ensemble_model.pkl \\
                         --model-b ../model-b-mlp/model/model_MLP_v2.pt \\
                         --model-b-config ../model-b-mlp/model/model_config_v2.json \\
                         --data ../model-b-mlp/data/

  # 仅评估 Model B
  python src/evaluate.py --model-b ../model-b-mlp/model/model_MLP_v2.pt \\
                         --model-b-config ../model-b-mlp/model/model_config_v2.json \\
                         --data ../model-b-mlp/data/

  # 推理模式
  python src/evaluate.py --predict --model-a <pkl路径> --input 患者.xlsx
        """
    )
    p.add_argument("--model-a", default=None, help="Model A pkl 模型文件路径")
    p.add_argument("--model-b", default=None, help="Model B .pt 模型文件路径")
    p.add_argument("--model-b-config", default=None, help="Model B model_config JSON 路径")
    p.add_argument("--data", default="../model-b-mlp/data/", help="测试数据目录或 test.csv 路径")
    p.add_argument("--out-dir", default="reports", help="输出目录")
    p.add_argument("--predict", action="store_true", help="推理模式（单条预测）")
    p.add_argument("--input", default=None, help="推理输入 (Excel/CSV)")
    p.add_argument("--hyperopt", default=None, choices=["mlp","xgboost","lightgbm"],
                   help="超参数优化模式")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── 推理模式 ──────────────────────────────────────────
    if args.predict:
        if args.model_a:
            pkg = load_model_a(args.model_a)
            df = pd.read_excel(args.input) if args.input.endswith(".xlsx") else pd.read_csv(args.input)
            for _, row in df.iterrows():
                r = predict_single(pkg, row.to_dict(), "a")
                print(f"  {r['risk_label']}  {r['probability']}")
        elif args.model_b:
            if not args.model_b_config:
                args.model_b_config = os.path.join(os.path.dirname(args.model_b), "model_config_v2.json")
            model, config = load_model_b(args.model_b, args.model_b_config)
            df = pd.read_excel(args.input) if args.input.endswith(".xlsx") else pd.read_csv(args.input)
            for _, row in df.iterrows():
                r = predict_single(model, row.to_dict(), "b")
                print(f"  {r['risk_label']}  {r['probability']}")
        else:
            print("推理模式需要指定 --model-a 或 --model-b")
        return

    # ── 超参数优化模式 ────────────────────────────────────
    if args.hyperopt:
        data = load_test_data(args.data)
        if data is None or data["X_a"] is None:
            print("超参数优化需要数据。请确保 test.csv 存在。")
            return
        # 使用训练集做超参数搜索（需要训练集）
        train_path = args.data if os.path.isfile(args.data) else os.path.join(args.data, "train.csv")
        if not os.path.exists(train_path.replace("test.csv", "train.csv")):
            print(f"超参数优化需要训练集: {train_path}")
            return
        train_df = pd.read_csv(train_path.replace("test.csv", "train.csv"), encoding="utf-8-sig")
        y_tr = train_df[TARGET].values.astype(int)
        X_tr = train_df.drop(columns=[TARGET], errors="ignore").values.astype(np.float32)
        # 用测试集作为验证集
        best_p, best_s = hyperparameter_optimization(
            args.hyperopt, X_tr, y_tr, data["X_b"].values, data["y_true"], args.out_dir
        )
        if best_p:
            print(f"\n最优参数: {best_p}")
            print(f"最优 Macro F1: {best_s:.4f}")
        return

    # ── 正常评估模式 ──────────────────────────────────────
    if not args.model_a and not args.model_b:
        print("请至少指定 --model-a 或 --model-b")
        print("示例: python src/evaluate.py --model-b ../model-b-mlp/model/model_MLP_v2.pt "
              "--model-b-config ../model-b-mlp/model/model_config_v2.json --data ../model-b-mlp/data/")
        return

    all_metrics = {}
    roc_data = {}

    # ── Model A 评估 ──────────────────────────────────────
    if args.model_a:
        print("\n" + "="*65)
        print("  Model A 评估")
        print("="*65)
        pkg_a = load_model_a(args.model_a)
        data = load_test_data(args.data, model_a_pkg=pkg_a)

        if data and data["X_a"] is not None and data["y_true"] is not None:
            y_pred_a, y_proba_a = predict_model_a(pkg_a, data["X_a"])
            metrics_a = compute_metrics(data["y_true"], y_pred_a, y_proba_a, "Model A (Logistic)")
            print_metrics(metrics_a)

            # 保存
            json_path = os.path.join(args.out_dir, "model_a_metrics.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_a, f, ensure_ascii=False, indent=2)
            print(f"[保存] -> {json_path}")

            if _HAS_MATPLOTLIB:
                cm_a = np.array(metrics_a["confusion_matrix"])
                _plot_confusion_matrix(cm_a, "混淆矩阵 - Model A",
                                       os.path.join(args.out_dir, "model_a_confusion_matrix.png"))
                _plot_roc_curves(data["y_true"], y_proba_a, "ROC - Model A",
                                os.path.join(args.out_dir, "model_a_roc.png"))

            all_metrics["Model A (Logistic)"] = metrics_a
            roc_data["Model A"] = {"y_true": data["y_true"].tolist(), "y_proba": y_proba_a.tolist()}
        else:
            print("[Model A] 跳过 — 测试数据不可用")

    # ── Model B 评估 ──────────────────────────────────────
    if args.model_b:
        print("\n" + "="*65)
        print("  Model B 评估")
        print("="*65)
        model_b, config_b = load_model_b(args.model_b, args.model_b_config)
        data = load_test_data(args.data, model_b_config=config_b)

        if data and data["X_b"] is not None and data["y_true"] is not None:
            y_pred_b, y_proba_b = predict_model_b(model_b, data["X_b"], config_b)
            metrics_b = compute_metrics(data["y_true"], y_pred_b, y_proba_b, "Model B (MLP)")
            print_metrics(metrics_b)

            json_path = os.path.join(args.out_dir, "model_b_metrics.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metrics_b, f, ensure_ascii=False, indent=2)
            print(f"[保存] -> {json_path}")

            if _HAS_MATPLOTLIB:
                cm_b = np.array(metrics_b["confusion_matrix"])
                _plot_confusion_matrix(cm_b, "混淆矩阵 - Model B",
                                       os.path.join(args.out_dir, "model_b_confusion_matrix.png"))
                _plot_roc_curves(data["y_true"], y_proba_b, "ROC - Model B",
                                os.path.join(args.out_dir, "model_b_roc.png"))

            all_metrics["Model B (MLP)"] = metrics_b
            roc_data["Model B"] = {"y_true": data["y_true"].tolist(), "y_proba": y_proba_b.tolist()}
        else:
            print("[Model B] 跳过 — 测试数据不可用")

    # ── 对比报告 ──────────────────────────────────────────
    if len(all_metrics) >= 2:
        print("\n" + "="*65)
        print("  两模型对比报告")
        print("="*65)
        names = list(all_metrics.keys())
        generate_comparison_report(all_metrics[names[0]], all_metrics[names[1]], args.out_dir)

        if _HAS_MATPLOTLIB and len(roc_data) == 2:
            _plot_roc_comparison(roc_data, os.path.join(args.out_dir, "roc_comparison.png"))
            print(f"[图表] ROC 对比 -> {args.out_dir}/roc_comparison.png")

    # ── 汇总 ──────────────────────────────────────────────
    print("\n" + "="*65)
    print("  评估完成")
    print(f"  输出目录: {os.path.abspath(args.out_dir)}")
    for fn in sorted(os.listdir(args.out_dir)):
        fp = os.path.join(args.out_dir, fn)
        sz = os.path.getsize(fp) / 1024 if os.path.isfile(fp) else 0
        print(f"    {fn:<50} {sz:>7.1f} KB")
    print("="*65)


if __name__ == "__main__":
    main()
