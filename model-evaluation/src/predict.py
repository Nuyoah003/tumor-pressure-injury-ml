"""
模型推理接口
============
提供统一的预测接口，支持 Model A（有序逻辑回归集成/高风险预警）和 Model B（MLP神经网络）。

用法:
    # 命令行
    python src/predict.py --model ensemble --input 患者.xlsx
    python src/predict.py --model high_risk --input 患者.xlsx
    python src/predict.py --model mlp --input 患者.csv --output 结果.csv

    # 代码调用
    from src.predict import Predictor
    p = Predictor("ensemble")
    result = p.predict({"年龄": 68, "白蛋白": 30.5, ...})
    # → {"predicted_class": 1, "risk_label": "低风险", "probability": {...}}

模型类型:
    ensemble:  Model A 综合最优集成模型（Top3软投票+阈值优化，Kappa=0.2856）
    high_risk: Model A 高风险预警模型（XGBoost+序数代价权重，高风险Recall=0.4972）
    mlp:       Model B MLP 神经网络（Kappa=0.1806，高风险Recall=0.6102）
"""

import os, sys, json, warnings, argparse
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)

RISK_LABELS = ["无风险", "低风险", "中风险", "高风险"]
RISK_LABELS_FULL = ["无风险(0)", "低风险(1)", "中风险(2)", "高风险(3)"]
K = 4

BINARY_VARS = [
    "性别", "民族", "是否入住ICU", "放疗史", "化疗史",
    "营养风险", "血栓风险", "高血压", "糖尿病", "骨转移",
    "疼痛", "肿瘤转移", "激素治疗", "免疫抑制剂", "恶性积液",
    "电解质紊乱", "感染", "导管数目",
]
NOMINAL_VARS = ["肿瘤类型"]
CONTINUOUS_VARS = ["年龄", "住院时长", "白蛋白", "BMI"]


# ══════════════════════════════════════════════════════════════════════
# 模型文件路径
# ══════════════════════════════════════════════════════════════════════
MODEL_REGISTRY = {
    "ensemble": {
        "pkl": [
            os.path.join(BASE_DIR, "models", "best_ensemble_model.pkl"),
            os.path.join(BASE_DIR, "..", "model-a-ordinal-regression", "saved_models", "best_ensemble_model.pkl"),
        ],
        "type": "model_a",
    },
    "high_risk": {
        "pkl": [
            os.path.join(BASE_DIR, "models", "high_risk_model.pkl"),
            os.path.join(BASE_DIR, "..", "model-a-ordinal-regression", "saved_models", "high_risk_model.pkl"),
        ],
        "type": "model_a",
    },
    "mlp": {
        "pt": [
            os.path.join(BASE_DIR, "models", "model_MLP_v2.pt"),
            os.path.join(BASE_DIR, "..", "model-b-mlp", "model", "model_MLP_v2.pt"),
        ],
        "config": [
            os.path.join(BASE_DIR, "models", "model_config_v2.json"),
            os.path.join(BASE_DIR, "..", "model-b-mlp", "model", "model_config_v2.json"),
        ],
        "type": "model_b",
    },
}


# ══════════════════════════════════════════════════════════════════════
# 预处理
# ══════════════════════════════════════════════════════════════════════
def _preprocess(df, feat_names, scaler, ref_category_drop=True):
    """One-hot 编码 + 特征对齐 + 标准化"""
    df = df.copy()
    for col in NOMINAL_VARS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col, dtype=int)
            if ref_category_drop and dummies.shape[1] > 1:
                dummies = dummies.iloc[:, :-1]
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    for fn in feat_names:
        if fn not in df.columns:
            df[fn] = 0.0
    df = df[feat_names].astype(float)
    if scaler is not None:
        cont_present = [c for c in CONTINUOUS_VARS if c in df.columns]
        df[cont_present] = scaler.transform(df[cont_present])
    return df


# ══════════════════════════════════════════════════════════════════════
# Predictor
# ══════════════════════════════════════════════════════════════════════
class Predictor:
    """
    统一风险预测器。

    参数:
        model_type: "ensemble" | "high_risk" | "mlp"
        model_path: 自定义模型文件路径（可选，默认从 MODEL_REGISTRY 查找）

    属性:
        model_type: 模型类型字符串
        metadata:    模型元信息（特征名、标签等）

    示例:
        >>> p = Predictor("mlp")
        >>> p.predict({"年龄": 68, "白蛋白": 30.5, "BMI": 20.1, "肿瘤类型": 1})
        {'predicted_class': 1, 'risk_label': '低风险', 'probability': {'无风险': 0.72, ...}}
    """

    def __init__(self, model_type="mlp", model_path=None):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"不支持的模型类型: {model_type}。可选: {list(MODEL_REGISTRY.keys())}")
        self.model_type = model_type
        self._model = None
        self._meta = None
        self._device = None
        self._load(model_path)

    def _find_file(self, candidates):
        for p in candidates:
            if os.path.exists(p):
                return os.path.abspath(p)
        raise FileNotFoundError(f"找不到模型文件，尝试过:\n  " + "\n  ".join(candidates))

    def _load(self, model_path):
        entry = MODEL_REGISTRY[self.model_type]

        if entry["type"] == "model_a":
            pkl_path = model_path if model_path else self._find_file(entry["pkl"])
            self._load_model_a(pkl_path)
        else:
            pt_path = model_path if model_path else self._find_file(entry["pt"])
            cfg_path = self._find_file(entry["config"])
            self._load_model_b(pt_path, cfg_path)

    def _load_model_a(self, pkl_path):
        """加载 Model A .pkl"""
        try:
            import joblib
            pkg = joblib.load(pkl_path)
        except ImportError:
            import pickle
            with open(pkl_path, "rb") as f:
                pkg = pickle.load(f)

        self._meta = {
            "feat_names": pkg["feat_names"],
            "scaler": pkg.get("scaler"),
            "threshold_offset": pkg.get("threshold_offset", 0.0),
            "ref_category_drop": True,
        }
        if "models" in pkg:
            self._model = pkg["models"]
            self._is_ensemble = True
        else:
            self._model = pkg["model"]
            self._is_ensemble = False
        self._model_type_str = "ensemble" if self._is_ensemble else "single"

    def _load_model_b(self, pt_path, cfg_path):
        """加载 Model B PyTorch"""
        import torch
        import torch.nn as nn

        with open(cfg_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_units, num_classes, dropout_rate):
                super().__init__()
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

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = device

        state = torch.load(pt_path, map_location=device)
        if isinstance(state, MLP):
            model = state
        else:
            model = MLP(config["input_dim"], config["hidden_units"],
                         config.get("num_classes", K), config.get("dropout_rate", 0.3))
            if isinstance(state, dict) and "net.0.weight" in state:
                model.load_state_dict(state)
            elif hasattr(state, "state_dict"):
                model.load_state_dict(state.state_dict())
            else:
                model.load_state_dict(state)

        self._model = model.to(device).eval()
        self._meta = {
            "feat_names": config["feature_names"],
            "scaler": None,
            "ref_category_drop": False,
        }
        self._is_ensemble = False
        self._model_type_str = "mlp"

    @property
    def metadata(self):
        """模型元信息"""
        return {
            "model_type": self.model_type,
            "model_class": self._model_type_str,
            "feature_count": len(self._meta["feat_names"]),
            "feature_names": self._meta["feat_names"],
            "risk_labels": RISK_LABELS,
            "device": str(self._device) if self._device else "cpu",
        }

    # ── 推理 ──────────────────────────────────────────────

    def predict_proba(self, data):
        """
        返回概率矩阵。

        参数:
            data: DataFrame | dict | list[dict] | Excel/CSV路径

        返回:
            np.ndarray, shape (N, 4) — 各类别概率
        """
        df = self._to_dataframe(data)
        X = _preprocess(df, self._meta["feat_names"], self._meta["scaler"],
                        ref_category_drop=self._meta["ref_category_drop"])

        if self.model_type in ("ensemble", "high_risk"):
            X_arr = X.values
            if self._is_ensemble:
                probas = [m.predict_proba(X_arr) for m in self._model]
                proba = np.mean(probas, axis=0)
            else:
                proba = self._model.predict_proba(X_arr)
            # 应用阈值偏移
            offset = self._meta["threshold_offset"]
            for i in range(K):
                proba[:, i] += offset * (K - 1 - 2 * i) / (K - 1)
            return proba
        else:
            import torch
            X_arr = X.values.astype(np.float32)
            proba_list = []
            batch_size = 1024
            with torch.no_grad():
                for i in range(0, len(X_arr), batch_size):
                    xb = torch.from_numpy(X_arr[i:i+batch_size]).to(self._device)
                    proba_list.append(self._model.predict_proba(xb).cpu().numpy())
            return np.concatenate(proba_list)

    def predict(self, data):
        """
        预测风险等级。

        参数:
            data: DataFrame | dict | list[dict] | Excel/CSV路径

        返回:
            单条输入 → dict: {"predicted_class": int, "risk_label": str,
                              "probability": {"无风险": float, ...}}
            多条输入 → DataFrame: 含 predicted_class, risk_label, proba_0..3 列
        """
        proba = self.predict_proba(data)
        y_pred = proba.argmax(axis=1)

        if isinstance(data, dict):
            cls = int(y_pred[0])
            return {
                "predicted_class": cls,
                "risk_label": RISK_LABELS[cls],
                "probability": {
                    RISK_LABELS[i]: round(float(proba[0, i]), 4)
                    for i in range(K)
                },
            }

        result = pd.DataFrame({
            "predicted_class": y_pred,
            "risk_label": [RISK_LABELS[p] for p in y_pred],
        })
        for i in range(K):
            result[f"proba_{i}_{RISK_LABELS[i]}"] = proba[:, i].round(4)
        return result

    # ── 辅助 ──────────────────────────────────────────────

    def _to_dataframe(self, data):
        if isinstance(data, str):
            if data.endswith(".xlsx"):
                return pd.read_excel(data)
            else:
                return pd.read_csv(data, encoding="utf-8-sig")
        elif isinstance(data, dict):
            return pd.DataFrame([data])
        elif isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            return data.copy()
        else:
            raise TypeError(f"不支持的数据类型: {type(data)}")

    def __repr__(self):
        return f"Predictor(model_type='{self.model_type}', class={self._model_type_str})"


# ══════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="肿瘤患者压力性损伤风险预测 —— 统一推理接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python src/predict.py --model ensemble --input 患者数据.xlsx
  python src/predict.py --model mlp --input 患者数据.csv --output 结果.csv
  python src/predict.py --model high_risk --input 患者数据.xlsx --output 结果.xlsx
        """
    )
    p.add_argument("--model", required=True, choices=["ensemble", "high_risk", "mlp"],
                   help="模型类型")
    p.add_argument("--input", required=True, help="输入文件 (Excel .xlsx 或 CSV .csv)")
    p.add_argument("--output", default=None, help="输出文件路径（可选，默认打印到控制台）")
    p.add_argument("--model-path", default=None, help="自定义模型文件路径（覆盖默认查找）")
    p.add_argument("--meta", action="store_true", help="仅打印模型元信息")
    return p.parse_args()


def main():
    args = parse_args()
    p = Predictor(args.model, model_path=args.model_path)

    if args.meta:
        import pprint
        pprint.pprint(p.metadata)
        return

    result = p.predict(args.input)

    if isinstance(result, dict):
        print(f"\n预测结果:")
        print(f"  风险等级: {result['risk_label']} (类别 {result['predicted_class']})")
        print(f"  各类别概率:")
        for label, prob in result["probability"].items():
            print(f"    {label}: {prob:.2%}")
        if args.output:
            pd.DataFrame([result]).to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"\n结果已保存到: {args.output}")
    else:
        print(result.to_string(index=False))
        if args.output:
            if args.output.endswith(".xlsx"):
                result.to_excel(args.output, index=False)
            else:
                result.to_csv(args.output, index=False, encoding="utf-8-sig")
            print(f"\n结果已保存到: {args.output} ({len(result)} 条)")


if __name__ == "__main__":
    main()
