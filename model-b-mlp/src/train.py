"""
=======================================================
Step 2：MLP 神经网络训练脚本（PyTorch 版）  ── v2 修复版
开发人：李金城（算法开发组）
=======================================================
修复内容（v1 → v2）：
  1. 移除 WeightedRandomSampler，仅保留 CrossEntropyLoss 类别权重，
     避免双重平衡导致训练/验证分布割裂
  2. 验证集损失改用无权重 CrossEntropyLoss，如实反映真实分布表现
  3. EarlyStopping 改为监控验证集 macro-F1（在类别极度不平衡时比 Loss 更稳定）
  4. 修复 SHAP DeepExplainer 多分类形状错误
  5. 新增每轮打印 macro-F1 / per-class accuracy，方便观察少数类学习情况
  6. 新增测试集混淆矩阵、Cohen's Kappa、多分类 AUC 与 ROC 可视化，
     便于和 04_best_model_ensemble.py 使用同一评估口径对比

依赖：
    pip install torch scikit-learn pandas openpyxl shap matplotlib

输入（由 step1_split_data.py 生成）：
    train.csv  test.csv  feature_names.json

输出（output_mlp/ 目录）：
    model_MLP_v2.pt / model_MLP_v2_state_dict.pth
    feature_importance_mlp.png / feature_importance_mlp.csv
    training_curves.png
    test_evaluation_mlp.png
    test_metrics_mlp.json
    test_confusion_matrix_mlp.csv
    model_config_v2.json
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    f1_score, classification_report, confusion_matrix,
    cohen_kappa_score, roc_auc_score, roc_curve, auc as sk_auc,
)
from sklearn.preprocessing import label_binarize

import shap
import tensorflow as tf

# ===========================================================================
# 0. 全局配置
# ===========================================================================
DATA_DIR    = "/Work/ljc/tumor/data"
OUTPUT_DIR  = "model"
TARGET_COL  = "风险等级"
NUM_CLASSES = 4
SEED        = 42

# ── 超参（超参优化阶段由部署组白伟琪调整）──
HIDDEN_UNITS  = [256, 128, 64]
DROPOUT_RATE  = 0.3
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4        # Adam L2 正则
BATCH_SIZE    = 512
EPOCHS        = 20
VAL_RATIO     = 0.15
PATIENCE      = 20          # 监控 macro-F1，连续 PATIENCE 轮不提升则停止

os.makedirs(OUTPUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[设备] 使用：{DEVICE}")

# 中文字体 ── 多策略加载，确保 matplotlib 能正确渲染中文
_FONT_CANDIDATES = [
    # Linux（文泉驿 / Noto CJK）
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/arphic/uming.ttc",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/Library/Fonts/Arial Unicode MS.ttf",
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]

def _setup_chinese_font():
    """
    多策略中文字体注册：
      1. 按已知路径查找字体文件 → addfont() 注册 → 设置 font.family
      2. 路径均不存在时，用 fc-list 在系统中搜索 CJK 字体（Linux）
      3. 再退一步，按字体名称在 matplotlib 缓存中查找
    """
    fp = next((p for p in _FONT_CANDIDATES if os.path.exists(p)), None)

    # 策略 2：Linux fc-list 扩大搜索
    if fp is None:
        try:
            import subprocess
            result = subprocess.run(
                ["fc-list", ":lang=zh", "--format=%{file}\\n"],
                capture_output=True, text=True, timeout=5,
            )
            for line in result.stdout.splitlines():
                line = line.strip()
                if line and os.path.exists(line):
                    fp = line
                    break
        except Exception:
            pass

    if fp:
        # addfont 将字体文件注册进 matplotlib 字体缓存（matplotlib ≥ 3.5）
        fm.fontManager.addfont(fp)
        prop = fm.FontProperties(fname=fp)
        font_name = prop.get_name()
        plt.rcParams["font.family"] = font_name
        print(f"[字体] 已加载中文字体：{font_name}（{fp}）")
    else:
        # 策略 3：字体已在系统注册，按名称查找
        _cjk_names = [
            "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
            "Noto Sans CJK SC", "Noto Sans CJK TC",
            "SimHei", "Microsoft YaHei", "PingFang SC",
            "Arial Unicode MS", "Heiti SC",
        ]
        for name in _cjk_names:
            try:
                found = fm.findfont(
                    fm.FontProperties(family=name), fallback_to_default=False
                )
                # findfont 找不到时会回退到 DejaVu，排除该情况
                if found and "DejaVu" not in found and "dejavu" not in found.lower():
                    plt.rcParams["font.family"] = name
                    print(f"[字体] 已使用系统中文字体：{name}（{found}）")
                    break
            except Exception:
                continue
        else:
            print("[字体] ⚠  未找到可用中文字体，图表中文字符可能显示为方块。")
            print("       建议安装：sudo apt install fonts-wqy-zenhei  或  fonts-noto-cjk")

    plt.rcParams["axes.unicode_minus"] = False   # 防止负号变方块

_setup_chinese_font()

LABEL_MAP = {0: "无风险", 1: "低风险", 2: "中风险", 3: "高风险"}


# ===========================================================================
# 1. 数据加载
# ===========================================================================
def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"), encoding="utf-8-sig")
    test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"),  encoding="utf-8-sig")

    feat_path = os.path.join(DATA_DIR, "feature_names.json")
    if os.path.exists(feat_path):
        with open(feat_path, encoding="utf-8") as f:
            feature_names = json.load(f)
    else:
        feature_names = [c for c in train_df.columns if c != TARGET_COL]

    print(f"[数据] 训练集：{train_df.shape}，测试集：{test_df.shape}")
    print(f"[数据] 特征数：{len(feature_names)}")
    print("[数据] 因变量分布（训练集）：")
    total = len(train_df)
    for k, v in train_df[TARGET_COL].value_counts().sort_index().items():
        print(f"  {k}（{LABEL_MAP[k]}）: {v:>6} 条  ({v/total*100:.1f}%)")
    print()
    return train_df, test_df, feature_names


def to_arrays(df, feature_names):
    X = df[feature_names].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int64)
    return X, y


# ===========================================================================
# 2. Dataset（无采样器，保持真实分布）
# ===========================================================================
class RiskDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# ===========================================================================
# 3. MLP 模型
# ===========================================================================
class MLP(nn.Module):
    """
    多层感知机：Linear -> BatchNorm -> ReLU -> Dropout，循环堆叠。
    输出层不加 Softmax（CrossEntropyLoss 已内含）。
    """
    def __init__(self, input_dim, hidden_units=None,
                 num_classes=NUM_CLASSES, dropout_rate=DROPOUT_RATE):
        super().__init__()
        if hidden_units is None:
            hidden_units = [256, 128, 64]

        seq, in_d = [], input_dim
        for u in hidden_units:
            seq += [nn.Linear(in_d, u), nn.BatchNorm1d(u),
                    nn.ReLU(), nn.Dropout(dropout_rate)]
            in_d = u
        seq.append(nn.Linear(in_d, num_classes))
        self.net = nn.Sequential(*seq)

    def forward(self, x):
        return self.net(x)

    @torch.no_grad()
    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)

    @torch.no_grad()
    def predict(self, x):
        return self.forward(x).argmax(dim=1)


def build_model(input_dim):
    model = MLP(input_dim, HIDDEN_UNITS, NUM_CLASSES, DROPOUT_RATE).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[模型] 输入维度={input_dim}，总参数={n_params:,}\n")
    return model


# ===========================================================================
# 4. 训练
# ===========================================================================
def train(model, train_df, feature_names):
    X_all, y_all = to_arrays(train_df, feature_names)

    # 划验证集（分层）
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=VAL_RATIO, stratify=y_all, random_state=SEED
    )
    print(f"[训练] 训练集：{len(X_tr)} 条，验证集：{len(X_val)} 条")

    tr_loader = DataLoader(
        RiskDataset(X_tr, y_tr),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        RiskDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True
    )

    # 类别权重（仅用于训练损失）
    cw = compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
    cw_tensor = torch.tensor(cw, dtype=torch.float32).to(DEVICE)
    print(f"[训练] 类别权重：{ {LABEL_MAP[i]: round(w, 2) for i, w in enumerate(cw)} }\n")

    # 训练用加权 Loss；验证用无权重 Loss（反映真实分布）  ← 关键修复
    train_criterion = nn.CrossEntropyLoss(weight=cw_tensor)
    val_criterion   = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    # 监控 macro-F1（越大越好），mode="max"
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=8, min_lr=1e-6
    )

    best_f1, best_epoch, no_improve, best_state = -1.0, 0, 0, None
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [], "val_macro_f1": []
    }

    print(f"[训练] 最多 {EPOCHS} 轮，EarlyStopping patience={PATIENCE}（监控 val macro-F1）\n")
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):

        # ── Train ──
        model.train()
        tr_loss, tr_correct, tr_n = 0.0, 0, 0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(Xb)
            loss   = train_criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tr_loss    += loss.item() * len(yb)
            tr_correct += (logits.argmax(1) == yb).sum().item()
            tr_n       += len(yb)

        tr_loss /= tr_n
        tr_acc   = tr_correct / tr_n

        # ── Validation ──
        model.eval()
        val_loss, val_correct, val_n = 0.0, 0, 0
        val_preds, val_labels = [], []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                logits = model(Xb)
                val_loss    += val_criterion(logits, yb).item() * len(yb)
                val_correct += (logits.argmax(1) == yb).sum().item()
                val_n       += len(yb)
                val_preds.extend(logits.argmax(1).cpu().numpy())
                val_labels.extend(yb.cpu().numpy())

        val_loss    /= val_n
        val_acc      = val_correct / val_n
        val_macro_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        scheduler.step(val_macro_f1)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)

        # 每 10 轮打印
        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"  Epoch {epoch:>3}/{EPOCHS} | "
                f"TrainLoss={tr_loss:.4f} TrainAcc={tr_acc:.4f} | "
                f"ValLoss={val_loss:.4f} ValAcc={val_acc:.4f} "
                f"MacroF1={val_macro_f1:.4f} | lr={lr_now:.2e} | "
                f"耗时={time.time()-t0:.0f}s"
            )

        # EarlyStopping（监控 macro-F1）
        if val_macro_f1 > best_f1:
            best_f1    = val_macro_f1
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\n[EarlyStopping] 第 {epoch} 轮停止，最佳在第 {best_epoch} 轮")
                break

    print(f"\n[训练完成] 最佳 MacroF1={best_f1:.4f}（第 {best_epoch} 轮）")
    model.load_state_dict(best_state)

    # 打印验证集各类别详细报告
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for Xb, yb in val_loader:
            val_preds.extend(model.predict(Xb.to(DEVICE)).cpu().numpy())
            val_labels.extend(yb.numpy())
    print("\n[验证集] 各类别详细报告：")
    print(classification_report(
        val_labels, val_preds,
        target_names=[LABEL_MAP[i] for i in range(NUM_CLASSES)],
        zero_division=0
    ))
    return history


# ===========================================================================
# 5. 训练曲线
# ===========================================================================
def plot_training_curves(history):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="valid")
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"],   label="valid")
    axes[1].set_title("Accuracy curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history["val_macro_f1"], color="darkorange", label="Validation Macro-F1")
    axes[2].set_title("Validation set Macro-F1 (monitoring metric for EarlyStopping)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Macro-F1")
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 训练曲线已保存：{path}")


# ===========================================================================
# 6. 特征重要性（SHAP）
# ===========================================================================
def compute_shap_importance(model, X_train, X_test, feature_names,
                             background_n=200, explain_n=300):
    """SHAP DeepExplainer 修复版：统一处理多分类输出的形状差异。"""
    print("\n[SHAP] 开始计算（DeepExplainer）...")
    model.eval()

    bg_idx = np.random.choice(len(X_train), min(background_n, len(X_train)), replace=False)
    background = torch.tensor(X_train[bg_idx], dtype=torch.float32).to(DEVICE)

    n   = min(explain_n, len(X_test))
    X_e = torch.tensor(X_test[:n], dtype=torch.float32).to(DEVICE)

    explainer   = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(X_e)

    # 统一为 list of (n, n_features)
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:           # (n, n_features, num_classes)
            shap_values = [shap_values[:, :, c] for c in range(shap_values.shape[2])]
        else:                               # (n, n_features)
            shap_values = [shap_values]

    # 平均绝对 SHAP（跨类别再平均）
    mean_abs = np.mean(
        np.stack([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0),
        axis=0,
    )

    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": mean_abs})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    csv_path = os.path.join(OUTPUT_DIR, "feature_importance_mlp.csv")
    importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[SHAP] CSV 已保存：{csv_path}")
    print(importance_df.head(10).to_string(index=False))

    _plot_importance(
        importance_df,
        title="MLP 模型特征重要性（SHAP DeepExplainer）",
        xlabel="平均绝对 SHAP 值",
        fname="feature_importance_mlp.png",
    )
    return importance_df


def _plot_importance(df, title, xlabel, fname):
    top_n   = min(20, len(df))
    plot_df = df.head(top_n)
    fig, ax = plt.subplots(figsize=(9, top_n * 0.42 + 1.2))
    colors  = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, top_n))
    bars    = ax.barh(
        plot_df["feature"][::-1].values,
        plot_df["importance"][::-1].values,
        color=colors, edgecolor="white", linewidth=0.5,
    )
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(f"{title}（Top {top_n}）", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    max_val = plot_df["importance"].max()
    for bar, val in zip(bars, plot_df["importance"][::-1].values):
        ax.text(
            bar.get_width() + max_val * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", ha="left", fontsize=8,
        )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[可视化] 图已保存：{path}")


def compute_weight_importance(model, feature_names):
    """SHAP 备选：第一层权重绝对值均值。"""
    print("\n[权重法] 计算第一层权重绝对值均值（SHAP 备选）...")
    first_linear = next(m for m in model.modules() if isinstance(m, nn.Linear))
    W = first_linear.weight.data.cpu().numpy()  # (hidden[0], n_features)
    importance = np.abs(W).mean(axis=0)
    df = (
        pd.DataFrame({"feature": feature_names, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    _plot_importance(
        df,
        title="MLP 特征重要性（权重法，备选）",
        xlabel="输入层权重绝对值均值",
        fname="feature_importance_mlp_weight.png",
    )
    return df


# ===========================================================================
# 7. 测试集评估
# ===========================================================================
def _safe_multiclass_auc(y_true, y_proba):
    """计算多分类 OVR AUC；若测试集中类别缺失，则自动跳过无法计算的项。"""
    auc_macro = auc_weighted = None
    per_class_auc = {}
    y_arr = np.asarray(y_true, dtype=int).ravel()

    try:
        if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == NUM_CLASSES:
            auc_macro = roc_auc_score(
                y_arr, y_proba, multi_class="ovr", average="macro"
            )
            auc_weighted = roc_auc_score(
                y_arr, y_proba, multi_class="ovr", average="weighted"
            )

            y_bin = label_binarize(y_arr, classes=list(range(NUM_CLASSES)))
            for i in range(NUM_CLASSES):
                # 单类别全为 0 或全为 1 时 ROC/AUC 不可定义
                if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y_bin):
                    fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                    per_class_auc[i] = sk_auc(fpr, tpr)
    except Exception as e:
        print(f"  AUC 计算跳过：{e}")

    return auc_macro, auc_weighted, per_class_auc


def _print_confusion_matrix(cm):
    """按 04_best_model_ensemble.py 的口径打印混淆矩阵：数量 + 行百分比。"""
    print(f"\n  混淆矩阵:")
    print(f"  {'':>12} {'预测0':>8} {'预测1':>8} {'预测2':>8} {'预测3':>8}")
    for i in range(NUM_CLASSES):
        row_sum = cm[i].sum()
        row_pct = cm[i] / row_sum * 100 if row_sum > 0 else np.zeros(NUM_CLASSES)
        row_str = f"  {LABEL_MAP[i]:>12}"
        for j in range(NUM_CLASSES):
            row_str += f" {cm[i, j]:>5}({row_pct[j]:>4.1f}%)"
        print(row_str)


def plot_test_evaluation(metrics, y_true, y_proba, filename="test_evaluation_mlp.png"):
    """保存测试集评估图：混淆矩阵 + OVR ROC 曲线。"""
    y_arr = np.asarray(y_true, dtype=int).ravel()
    cm = metrics["confusion_matrix"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1) 混淆矩阵
    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(NUM_CLASSES))
    ax.set_yticks(np.arange(NUM_CLASSES))
    ax.set_xticklabels([LABEL_MAP[i] for i in range(NUM_CLASSES)])
    ax.set_yticklabels([LABEL_MAP[i] for i in range(NUM_CLASSES)])
    ax.set_xlabel("预测标签", fontsize=11)
    ax.set_ylabel("真实标签", fontsize=11)
    ax.set_title("混淆矩阵", fontweight="bold")

    max_val = cm.max() if cm.size else 0
    threshold = max_val / 2 if max_val > 0 else 0
    for i in range(NUM_CLASSES):
        row_sum = cm[i].sum()
        for j in range(NUM_CLASSES):
            pct = cm[i, j] / row_sum * 100 if row_sum > 0 else 0
            color = "white" if cm[i, j] > threshold else "black"
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)",
                    ha="center", va="center", color=color, fontsize=9)

    # 2) ROC 曲线（One-vs-Rest）
    ax = axes[1]
    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == NUM_CLASSES:
        y_bin = label_binarize(y_arr, classes=list(range(NUM_CLASSES)))
        for i in range(NUM_CLASSES):
            if y_bin[:, i].sum() > 0 and y_bin[:, i].sum() < len(y_bin):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                auc_val = sk_auc(fpr, tpr)
                ax.plot(fpr, tpr, lw=2, label=f"{LABEL_MAP[i]} (AUC={auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.legend(fontsize=9)
    else:
        ax.text(0.5, 0.5, "无可用概率，无法绘制 ROC", ha="center", va="center")

    ax.set_xlabel("假阳性率", fontsize=11)
    ax.set_ylabel("真阳性率", fontsize=11)
    ax.set_title("ROC曲线 (OVR)", fontweight="bold")
    ax.grid(True, alpha=0.3)

    auc_text = ""
    if metrics.get("auc_macro") is not None:
        auc_text = f", AUC_macro={metrics['auc_macro']:.4f}"
    plt.suptitle(
        f"MLP测试集评估 (Kappa={metrics['kappa']:.4f}, Macro F1={metrics['macro_f1']:.4f}{auc_text})",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  [图表] 测试集评估图已保存：{path}")


def save_test_metrics(metrics):
    """保存测试集核心指标与混淆矩阵，方便论文/报告汇总。"""
    json_path = os.path.join(OUTPUT_DIR, "test_metrics_mlp.json")
    cm_path = os.path.join(OUTPUT_DIR, "test_confusion_matrix_mlp.csv")

    serializable = {
        "loss": float(metrics["loss"]),
        "accuracy": float(metrics["accuracy"]),
        "kappa": float(metrics["kappa"]),
        "auc_macro": None if metrics["auc_macro"] is None else float(metrics["auc_macro"]),
        "auc_weighted": None if metrics["auc_weighted"] is None else float(metrics["auc_weighted"]),
        "macro_f1": float(metrics["macro_f1"]),
        "weighted_f1": float(metrics["weighted_f1"]),
        "per_class_auc": {
            LABEL_MAP[int(k)]: float(v)
            for k, v in metrics["per_class_auc"].items()
        },
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    cm_df = pd.DataFrame(
        metrics["confusion_matrix"],
        index=[f"真实_{LABEL_MAP[i]}" for i in range(NUM_CLASSES)],
        columns=[f"预测_{LABEL_MAP[i]}" for i in range(NUM_CLASSES)],
    )
    cm_df.to_csv(cm_path, encoding="utf-8-sig")

    print(f"  [指标] 测试集指标已保存：{json_path}")
    print(f"  [指标] 混淆矩阵CSV已保存：{cm_path}")


def evaluate_on_test(model, X_test, y_test):
    model.eval()
    loader = DataLoader(
        RiskDataset(X_test, y_test),
        batch_size=512, shuffle=False, num_workers=4
    )
    criterion = nn.CrossEntropyLoss()   # 无权重，反映真实分布
    total_loss, correct, total = 0.0, 0, 0
    preds, labels, probas = [], [], []

    with torch.no_grad():
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            logits = model(Xb)
            proba = torch.softmax(logits, dim=1)

            total_loss += criterion(logits, yb).item() * len(yb)
            correct    += (logits.argmax(1) == yb).sum().item()
            total      += len(yb)
            preds.extend(logits.argmax(1).cpu().numpy())
            labels.extend(yb.cpu().numpy())
            probas.extend(proba.cpu().numpy())

    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    probas = np.asarray(probas, dtype=float)

    test_loss = total_loss / total
    accuracy = correct / total
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(labels, preds)
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    auc_macro, auc_weighted, per_class_auc = _safe_multiclass_auc(labels, probas)

    print(f"\n{'='*65}")
    print("  MLP模型 - 测试集评估结果")
    print(f"{'='*65}")

    _print_confusion_matrix(cm)

    print(f"\n  综合指标:")
    print(f"    Loss:                {test_loss:.4f}")
    print(f"    Accuracy:            {accuracy:.4f}")
    print(f"    Cohen's Kappa:       {kappa:.4f}")
    if auc_macro is not None:
        print(f"    AUC (macro):          {auc_macro:.4f}")
        print(f"    AUC (weighted):       {auc_weighted:.4f}")
        for i in range(NUM_CLASSES):
            if i in per_class_auc:
                print(f"    AUC - {LABEL_MAP[i]}:       {per_class_auc[i]:.4f}")
    print(f"    Macro F1:             {macro_f1:.4f}")
    print(f"    Weighted F1:          {weighted_f1:.4f}")

    print(f"\n  sklearn classification_report:")
    print(classification_report(
        labels, preds,
        target_names=[LABEL_MAP[i] for i in range(NUM_CLASSES)],
        zero_division=0,
    ))

    metrics = {
        "loss": test_loss,
        "accuracy": accuracy,
        "kappa": kappa,
        "auc_macro": auc_macro,
        "auc_weighted": auc_weighted,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "confusion_matrix": cm,
        "per_class_auc": per_class_auc,
    }
    plot_test_evaluation(metrics, labels, probas)
    save_test_metrics(metrics)
    return metrics


# ===========================================================================
# 8. 保存模型
# ===========================================================================
def convert_to_h5(model, version="2"):
    """Convert PyTorch MLP to TF/Keras and save as .h5."""
    print("\n[转换] PyTorch → TF/Keras .h5 ...")
    tf.config.set_visible_devices([], "GPU")  # force CPU to avoid CUDA version mismatch

    # Rebuild architecture from nn.Sequential children
    layers = []
    for child in model.net:
        if isinstance(child, nn.Linear):
            layers.append(tf.keras.layers.Dense(child.out_features, use_bias=child.bias is not None))
        elif isinstance(child, nn.BatchNorm1d):
            layers.append(tf.keras.layers.BatchNormalization())
        elif isinstance(child, nn.ReLU):
            layers.append(tf.keras.layers.Activation("relu"))
        elif isinstance(child, nn.Dropout):
            layers.append(tf.keras.layers.Dropout(child.p))

    keras_model = tf.keras.Sequential(layers)
    input_dim = next(iter(model.parameters())).shape[1]
    keras_model.build(input_shape=(None, input_dim))

    # Transfer weights layer by layer
    pytorch_layers = [c for c in model.net if not isinstance(c, (nn.ReLU, nn.Dropout))]
    keras_dense_bn = [l for l in keras_model.layers if isinstance(l, (tf.keras.layers.Dense, tf.keras.layers.BatchNormalization))]

    for k_layer, pt_layer in zip(keras_dense_bn, pytorch_layers):
        if isinstance(pt_layer, nn.Linear):
            w = pt_layer.weight.data.cpu().numpy().T  # (in, out) for Keras
            b = pt_layer.bias.data.cpu().numpy()
            k_layer.set_weights([w, b])
        elif isinstance(pt_layer, nn.BatchNorm1d):
            w = pt_layer.weight.data.cpu().numpy()
            b = pt_layer.bias.data.cpu().numpy()
            rm = pt_layer.running_mean.numpy()
            rv = pt_layer.running_var.numpy()
            k_layer.set_weights([w, b, rm, rv])

    h5_path = os.path.join(OUTPUT_DIR, f"model_MLP_v{version}.h5")
    keras_model.save(h5_path)
    print(f"[转换] Keras .h5 已保存：{h5_path}")
    return h5_path


def save_model(model, feature_names, version="2"):
    pt_path  = os.path.join(OUTPUT_DIR, f"model_MLP_v{version}.pt")
    pth_path = os.path.join(OUTPUT_DIR, f"model_MLP_v{version}_state_dict.pth")
    cfg_path = os.path.join(OUTPUT_DIR, f"model_config_v{version}.json")

    torch.save(model, pt_path)
    torch.save(model.state_dict(), pth_path)

    config = {
        "input_dim":     next(iter(model.parameters())).shape[1],
        "hidden_units":  HIDDEN_UNITS,
        "num_classes":   NUM_CLASSES,
        "dropout_rate":  DROPOUT_RATE,
        "feature_names": feature_names,
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n[保存] 完整模型：  {pt_path}")
    print(f"[保存] 权重文件：  {pth_path}")
    print(f"[保存] 配置文件：  {cfg_path}")

    # 转换为 Keras .h5
    try:
        convert_to_h5(model, version)
    except Exception as e:
        print(f"[警告] .h5 转换失败（{e}），请确认已安装 tensorflow：pip install tensorflow")


# ===========================================================================
# 9. 推理接口（供部署组参考）
# ===========================================================================
def predict_risk(model, feature_names, patient_data: dict) -> dict:
    model.eval()
    x  = np.array([[patient_data.get(f, 0.0) for f in feature_names]], dtype=np.float32)
    xt = torch.from_numpy(x).to(DEVICE)
    prob = model.predict_proba(xt)[0].cpu().numpy().tolist()
    cls  = int(np.argmax(prob))
    return {
        "predicted_class": cls,
        "probability":     [round(p, 4) for p in prob],
        "risk_label":      LABEL_MAP[cls],
    }


# ===========================================================================
# 主流程
# ===========================================================================
def main():
    print("=" * 62)
    print("  MLP 神经网络训练 v2（PyTorch）—— 肿瘤患者压力性损伤风险预测")
    print("  开发人：李金城（算法开发组）")
    print("=" * 62 + "\n")

    # Step 1 ── 加载数据
    train_df, test_df, feature_names = load_data()
    X_train, y_train = to_arrays(train_df, feature_names)
    X_test,  y_test  = to_arrays(test_df,  feature_names)

    # Step 2 ── 构建模型
    model = build_model(X_train.shape[1])

    # Step 3 ── 训练
    history = train(model, train_df, feature_names)

    # Step 4 ── 训练曲线
    plot_training_curves(history)

    # Step 5 ── 测试集评估
    evaluate_on_test(model, X_test, y_test)

    # Step 6 ── 特征重要性
    # 6a. 权重法（始终输出）
    compute_weight_importance(model, feature_names)
    # 6b. SHAP（可能失败，不影响后续流程）
    try:
        compute_shap_importance(model, X_train, X_test, feature_names)
    except Exception as e:
        print(f"[警告] SHAP 失败（{e}），已跳过（权重法特征图已输出）")

    # Step 7 ── 保存
    save_model(model, feature_names, version="2")

    # 汇总
    print("\n" + "=" * 62)
    print("  训练完成！交付物清单：")
    for fn in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, fn)
        sz = os.path.getsize(fp) / 1024 if os.path.isfile(fp) else 0
        print(f"  · {fn:<50} {sz:>8.1f} KB" if sz else f"  · {fn}/")
    print("=" * 62)

    # 推理演示
    demo = {fn: 0.0 for fn in feature_names}
    print(f"\n[推理示例] {predict_risk(model, feature_names, demo)}")


if __name__ == "__main__":
    main()