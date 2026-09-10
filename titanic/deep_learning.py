"""深度学习模块：PyTorch 多层感知机(MLP)训练与评估。

严格对齐课件《深度学习：神经网络与模型训练》(第3次课)：
    数据划分  : 891 -> test 20%(179)；剩余 712 -> val 20%(143) / train 569
    特征      : Age/SibSp/Parch/Fare + Pclass/Sex/Embarked = 12 维
    网络      : Linear(12,64)->ReLU->Dropout(0.2)->Linear(64,32)->ReLU->Linear(32,2)
    参数量    : 2978
    损失=CrossEntropyLoss  优化器=Adam(lr=0.001, weight_decay=1e-4)
    超参数    : seed=42; batch=32; lr=0.001; dropout=0.2; epochs=80; weight_decay=1e-4

本模块被 app.py 复用(与 train.py 的四个传统模型并列)：
    * run_training()  完整训练一次，产出图/CSV/权重
    * ensure_trained()/get_meta()  供 API 取结果
    * load_model()/predict_one()   供网页表单实时预测

可复现性：固定 python/numpy/torch 随机种子 + DataLoader 生成器种子 +
random_state=42 & stratify=y 的划分，因此每次训练集完全一致、结果可复现。
"""

from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .preprocessing import load_data
from .visualize import _b, _setup_cjk_font

# ---------------------------------------------------------------------------
# 常量的本地别名(方便阅读)
# ---------------------------------------------------------------------------
SEED = config.DL_SEED
BATCH_SIZE = config.DL_BATCH_SIZE
LR = config.DL_LR
DROPOUT = config.DL_DROPOUT
EPOCHS = config.DL_EPOCHS
WEIGHT_DECAY = config.DL_WEIGHT_DECAY
VAL_SIZE = config.DL_VAL_SIZE

MODELS_DIR = config.PROJECT_ROOT / "models"
DL_MODEL_PATH = MODELS_DIR / "best_titanic_mlp.pt"          # 网络权重(state_dict)
DL_PREPROC_PATH = MODELS_DIR / "dl_preprocessor.joblib"     # 训练好的预处理器
DL_HISTORY_CSV = config.CSV_DIR / "dl_training_history.csv"
DL_METRICS_CSV = config.CSV_DIR / "dl_metrics.csv"
DL_PREDS_CSV = config.CSV_DIR / "dl_test_predictions.csv"


# ---------------------------------------------------------------------------
# 1. 可复现性
# ---------------------------------------------------------------------------
def set_seed(seed: int = SEED) -> None:
    """固定 Python / NumPy / PyTorch 随机种子，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 2. 数据准备
# ---------------------------------------------------------------------------
def load_and_split_data():
    """按课件规则读取并划分 train / val / test。

    测试集与第2次课的四个传统模型使用完全相同的划分
    (random_state=42, stratify=y)，因此结果可直接横向比较。
    """
    raw = load_data(config.DATA_PATH)
    X = raw[config.FORM_COLS]           # Pclass/Sex/Age/SibSp/Parch/Fare/Embarked
    y = raw[config.TARGET_COL].astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=VAL_SIZE,
        random_state=SEED, stratify=y_train_full,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, raw


def build_preprocessor() -> ColumnTransformer:
    """按课件 p.53 构建预处理：数值标准化 + 类别独热(与 train.py 一致)。"""
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, config.NUM_COLS),
        ("cat", categorical_pipeline, config.CAT_COLS),
    ], sparse_threshold=0)


def _to_tensor(X_processed, y):
    X_t = torch.tensor(np.asarray(X_processed), dtype=torch.float32)
    y_t = torch.tensor(np.asarray(y), dtype=torch.long)
    return X_t, y_t


# ---------------------------------------------------------------------------
# 3. 模型定义(课件 p.55)
# ---------------------------------------------------------------------------
class TitanicMLP(nn.Module):
    """三层 MLP：12 -> 64 -> 32 -> 2，可训练参数 2978。"""

    def __init__(self, input_dim: int = 12):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """统计可训练参数量。"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# 4. 训练与验证
# ---------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer) -> list[dict]:
    """训练循环(课件 p.56/57)；按验证集 loss 保存最优权重。"""
    history: list[dict] = []
    best_val_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # 训练
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()               # 清空旧梯度
            logits = model(X_batch)             # 前向传播
            loss = criterion(logits, y_batch)   # 计算损失
            loss.backward()                     # 反向传播
            optimizer.step()                    # 更新参数
            tr_loss += loss.item() * X_batch.size(0)
            tr_correct += (logits.argmax(dim=1) == y_batch).sum().item()
            tr_total += X_batch.size(0)
        # 验证(不更新参数、不记录梯度)
        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                va_loss += loss.item() * X_batch.size(0)
                va_correct += (logits.argmax(dim=1) == y_batch).sum().item()
                va_total += X_batch.size(0)

        rec = {
            "epoch": epoch,
            "train_loss": tr_loss / tr_total,
            "train_acc": tr_correct / tr_total,
            "val_loss": va_loss / va_total,
            "val_acc": va_correct / va_total,
        }
        history.append(rec)
        if rec["val_loss"] < best_val_loss:
            best_val_loss = rec["val_loss"]
            torch.save(model.state_dict(), DL_MODEL_PATH)
    model.load_state_dict(torch.load(DL_MODEL_PATH))  # 载入最优权重
    return history


def predict(model, X_tensor):
    """返回 (预测类别数组, 生还概率数组)。"""
    model.eval()
    with torch.no_grad():
        logits = model(X_tensor)
        prob = torch.softmax(logits, dim=1)[:, 1].numpy()
        pred = logits.argmax(dim=1).numpy()
    return pred, prob


# ---------------------------------------------------------------------------
# 5. 可视化
# ---------------------------------------------------------------------------
def make_figures(history, y_test, y_pred, y_prob) -> list[Path]:
    """生成 MLP 的 5 张结果图 + 1 张与传统模型的对比图。"""
    import matplotlib.pyplot as plt
    _setup_cjk_font()
    config.DL_FIG_DIR.mkdir(parents=True, exist_ok=True)

    def _save(fig, name: str) -> Path:
        out = config.DL_FIG_DIR / f"{name}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return out

    saved: list[Path] = []
    epochs = [h["epoch"] for h in history]

    # 图1：Loss 曲线
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(epochs, [h["train_loss"] for h in history],
            label=_b("Train loss", "训练损失"), color="#5fa8d9")
    ax.plot(epochs, [h["val_loss"] for h in history],
            label=_b("Validation loss", "验证损失"), color="#d95f5f")
    ax.set_xlabel(_b("Epoch", "训练轮数")); ax.set_ylabel(_b("Loss", "损失"))
    ax.set_title(_b("Train / Validation loss curve", "训练与验证损失曲线"))
    ax.grid(alpha=0.3); ax.legend()
    saved.append(_save(fig, "01_loss_curve"))

    # 图2：Accuracy 曲线
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.plot(epochs, [h["train_acc"] for h in history],
            label=_b("Train accuracy", "训练准确率"), color="#5fa8d9")
    ax.plot(epochs, [h["val_acc"] for h in history],
            label=_b("Validation accuracy", "验证准确率"), color="#7fb37f")
    ax.set_xlabel(_b("Epoch", "训练轮数")); ax.set_ylabel(_b("Accuracy", "准确率"))
    ax.set_title(_b("Train / Validation accuracy curve", "训练与验证准确率曲线"))
    ax.grid(alpha=0.3); ax.legend()
    saved.append(_save(fig, "02_accuracy_curve"))

    # 图3：混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels([_b("Not survived", "未生还"), _b("Survived", "生还")])
    ax.set_yticklabels([_b("Not survived", "未生还"), _b("Survived", "生还")])
    ax.set_xlabel(_b("Predicted", "预测")); ax.set_ylabel(_b("True", "真实"))
    ax.set_title(_b("Confusion matrix (test set)", "混淆矩阵(测试集)"))
    fig.colorbar(im, ax=ax, shrink=0.8)
    saved.append(_save(fig, "03_confusion_matrix"))

    # 图4：ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val = roc_auc_score(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot(fpr, tpr, color="#5fa8d9", lw=2, label=f"MLP (AUC={auc_val:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label=_b("Chance", "随机基线"))
    ax.set_xlabel(_b("False positive rate", "假正率 FPR"))
    ax.set_ylabel(_b("True positive rate", "真正率 TPR"))
    ax.set_title(_b("ROC curve — MLP", "ROC 曲线 — MLP"))
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    saved.append(_save(fig, "04_roc_curve"))

    # 图5：预测概率分布
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y_test = np.asarray(y_test)
    ax.hist(y_prob[y_test == 0], bins=20, alpha=0.6, color="#d95f5f",
            label=_b("True: not survived", "真实：未生还"))
    ax.hist(y_prob[y_test == 1], bins=20, alpha=0.6, color="#5fa8d9",
            label=_b("True: survived", "真实：生还"))
    ax.set_xlabel(_b("Predicted survival probability", "预测生还概率"))
    ax.set_ylabel(_b("Number of passengers", "人数"))
    ax.set_title(_b("Prediction probability distribution", "预测概率分布(按真实类别)"))
    ax.grid(alpha=0.3); ax.legend()
    saved.append(_save(fig, "05_probability_distribution"))

    # 图6：六模型对比(四个传统模型 + MLP)
    cmp_fig = _draw_all_models_comparison(metrics_mlp={
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    })
    if cmp_fig is not None:
        saved.append(_save(cmp_fig, "06_all_models_comparison"))
    return saved


def _draw_all_models_comparison(metrics_mlp: dict):
    """分组条形图：四个传统模型 + MLP 的五个指标(课件 p.49)。"""
    import matplotlib.pyplot as plt
    trad_path = config.CSV_DIR / "metrics_all_models.csv"
    # 传统模型用同色系渐变(可区分)，MLP 用醒目红色突出
    trad_colors = ["#c3d9ec", "#7fa8c9", "#4a7fae", "#235a86"]
    order, labels, colors, values = [], [], [], {}
    if trad_path.exists():
        df = pd.read_csv(trad_path).set_index("model_id")
        for idx, (mid, label) in enumerate(config.MODELS):
            order.append(mid); labels.append(label)
            colors.append(trad_colors[idx % len(trad_colors)])
            if mid in df.index:
                values[mid] = [float(df.loc[mid, k]) for k in config.METRIC_KEYS]
    # 追加 MLP(用醒目颜色)
    order.append("mlp"); labels.append("MLP (PyTorch)"); colors.append("#e63946")
    values["mlp"] = [float(metrics_mlp[k]) for k in config.METRIC_KEYS]
    if len(order) <= 1:
        return None

    keys = config.METRIC_KEYS
    metric_zh = {"accuracy": "准确率", "precision": "精确率", "recall": "召回率",
                 "f1": "F1 分数", "auc": "AUC"}
    n = len(order)
    # 指标之间留出较大间隙，组内柱子稍微错开，降低重叠感
    x = np.arange(len(keys)) * 1.35
    width = 0.20
    fig, ax = plt.subplots(figsize=(13.0, 5.4))
    for i, mid in enumerate(order):
        offset = (i - (n - 1) / 2) * width
        bars = ax.bar(x + offset, values[mid], width, label=labels[i],
                      color=colors[i], edgecolor="white", linewidth=0.6)
        # 旋转 90° 贴柱顶标注，柱间已错开 -> 互不遮挡
        ax.bar_label(bars, labels=[f"{v:.3f}" for v in values[mid]],
                     fontsize=6.8, padding=1, rotation=90,
                     color="#33475b")
    ax.set_xticks(x)
    # AUC 保持全大写，其余指标首字母大写
    def _mlabel(k):
        en = "AUC" if k == "auc" else ("F1" if k == "f1" else k.capitalize())
        return _b(en, metric_zh[k])
    ax.set_xticklabels([_mlabel(k) for k in keys])
    ax.set_xlabel(_b("Evaluation Metrics", "评估指标"))
    ax.set_ylim(0, 1.10)
    ax.set_ylabel(_b("Score", "得分"))
    ax.set_title(_b("All models on the same test set (n=179): ML (4) vs DL (MLP)",
                    "全部模型在相同测试集上对比(n=179)：传统机器学习(4) vs 深度学习 MLP"))
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=5, fontsize=9, framealpha=0.9, loc="upper center",
              bbox_to_anchor=(0.5, 1.0))
    return fig


# ---------------------------------------------------------------------------
# 6. 主流程(供 app.py 调用)
# ---------------------------------------------------------------------------
_meta: dict = {}
_model: "TitanicMLP | None" = None
_preprocessor = None


def run_training() -> dict:
    """完整训练一次；生成所有产物并返回元信息。"""
    global _model, _preprocessor
    config.ensure_dirs()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 68)
    print("  Deep learning (PyTorch MLP) training — courseware-aligned")
    print("=" * 68)

    set_seed(SEED)
    X_train, X_val, X_test, y_train, y_val, y_test, raw = load_and_split_data()
    print(f"[1/6] Data: train={len(X_train)}  val={len(X_val)}  test={len(X_test)}")

    _preprocessor = build_preprocessor()
    X_train_p = _preprocessor.fit_transform(X_train)   # 只在训练集 fit
    X_val_p = _preprocessor.transform(X_val)
    X_test_p = _preprocessor.transform(X_test)
    input_dim = X_train_p.shape[1]
    print(f"[2/6] Preprocessed input dim = {input_dim}")

    X_train_t, y_train_t = _to_tensor(X_train_p, y_train)
    X_val_t, y_val_t = _to_tensor(X_val_p, y_val)
    X_test_t, _ = _to_tensor(X_test_p, y_test)

    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE)
    print(f"[3/6] DataLoader: batch_size={BATCH_SIZE}, {len(train_loader)} batches/epoch")

    _model = TitanicMLP(input_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    print(f"[4/6] Model: MLP(12->64->32->2), parameters = {count_parameters(_model)}")

    history = train_model(_model, train_loader, val_loader, criterion, optimizer)
    print(f"[5/6] Training done ({EPOCHS} epochs).")

    y_pred, y_prob = predict(_model, X_test_t)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob),
    }
    print(f"      test: " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # 保存产物
    pd.DataFrame(history).to_csv(DL_HISTORY_CSV, index=False)
    pd.DataFrame([{"model_id": "mlp", "model": "MLP (PyTorch)",
                   **{k: round(v, 4) for k, v in metrics.items()}}]
                 ).to_csv(DL_METRICS_CSV, index=False)
    test_ids = raw.loc[X_test.index, "PassengerId"].to_numpy()
    pd.DataFrame({
        "PassengerId": test_ids,
        "Survived_true": np.asarray(y_test),
        "Survived_pred": y_pred,
        "Survived_prob": np.round(y_prob, 4),
    }).to_csv(DL_PREDS_CSV, index=False)
    joblib.dump(_preprocessor, DL_PREPROC_PATH)     # 供 API 实时预测

    fig_paths = make_figures(history, np.asarray(y_test), y_pred, y_prob)
    print(f"[6/6] Figures -> {config.DL_FIG_DIR.name}/  ({len(fig_paths)} files)")

    _meta.update({
        "trained": True,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "input_dim": int(input_dim),
        "n_params": int(count_parameters(_model)),
        "metrics": {k: round(float(v), 4) for k, v in metrics.items()},
        "history": history,
        "test_ids": test_ids.tolist(),
        "y_test": np.asarray(y_test).tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": np.round(y_prob, 4).tolist(),
        "figures": [Path(p).name for p in fig_paths],
        "hyperparams": {
            "seed": SEED, "batch_size": BATCH_SIZE, "lr": LR, "dropout": DROPOUT,
            "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY,
        },
    })
    return _meta


def _load_from_disk() -> bool:
    """尝试从磁盘复用训练产物，避免重新训练。成功返回 True。"""
    global _model, _preprocessor
    if not (DL_METRICS_CSV.exists() and DL_PREDS_CSV.exists()
            and DL_MODEL_PATH.exists() and DL_PREPROC_PATH.exists()
            and DL_HISTORY_CSV.exists()):
        return False
    try:
        _preprocessor = joblib.load(DL_PREPROC_PATH)
        input_dim = len(_preprocessor.get_feature_names_out())
        _model = TitanicMLP(input_dim)
        _model.load_state_dict(torch.load(DL_MODEL_PATH))
        _model.eval()
        m = pd.read_csv(DL_METRICS_CSV).iloc[0]
        preds = pd.read_csv(DL_PREDS_CSV)
        hist = pd.read_csv(DL_HISTORY_CSV).to_dict(orient="records")
        _meta.update({
            "trained": True,
            "n_train": 569, "n_val": 143, "n_test": int(len(preds)),
            "input_dim": input_dim,
            "n_params": count_parameters(_model),
            "metrics": {k: round(float(m[k]), 4) for k in config.METRIC_KEYS},
            "history": hist,
            "test_ids": preds["PassengerId"].tolist(),
            "y_test": preds["Survived_true"].tolist(),
            "y_pred": preds["Survived_pred"].tolist(),
            "y_prob": preds["Survived_prob"].tolist(),
            "figures": sorted(p.name for p in config.DL_FIG_DIR.glob("*.png")),
            "hyperparams": {
                "seed": SEED, "batch_size": BATCH_SIZE, "lr": LR,
                "dropout": DROPOUT, "epochs": EPOCHS, "weight_decay": WEIGHT_DECAY,
            },
        })
        print("[dl] Loaded PyTorch MLP artifacts from disk.")
        return True
    except Exception as exc:  # 产物损坏 -> 退回懒训练
        _model = _preprocessor = None
        print(f"[dl] Artifact reload failed ({exc}); will retrain on demand.")
        return False


def ensure_trained() -> dict:
    """返回缓存的训练结果；缓存为空时先尝试复用磁盘产物，否则训练。"""
    if not _meta.get("trained"):
        if not _load_from_disk():
            run_training()
    return _meta


def get_meta() -> dict:
    """返回最近一次(或磁盘上)的训练元信息。"""
    if not _meta.get("trained"):
        ensure_trained()
    return _meta


def load_model():
    """确保模型与预处理器已就绪，返回 (model, preprocessor)。"""
    ensure_trained()
    return _model, _preprocessor


def predict_one(row: dict) -> dict:
    """用训练好的 MLP 预测单个乘客(供网页表单使用)。

    Args:
        row: 含 Pclass/Sex/Age/SibSp/Parch/Fare/Embarked 的字典。
    Returns:
        dict: 含 survived / survived_probability 等字段。
    """
    model, prep = load_model()
    one = pd.DataFrame([row], columns=config.FORM_COLS)
    x = prep.transform(one)
    x_t = torch.tensor(np.asarray(x), dtype=torch.float32)
    y_pred, y_prob = predict(model, x_t)
    prob = float(y_prob[0]); pred = int(y_pred[0])
    return {
        "model_id": "mlp",
        "model_label": "MLP (PyTorch)",
        "survived": pred,
        "survived_probability": round(prob, 4),
        "not_survived_probability": round(1 - prob, 4),
        "prediction_text": "SURVIVED" if pred == 1 else "NOT SURVIVED",
    }


if __name__ == "__main__":  # 支持: python -m titanic.deep_learning
    run_training()
