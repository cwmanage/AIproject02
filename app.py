"""FastAPI 应用：泰坦尼克号生还预测演示 Web 应用。

本应用展示**两种训练/建模方式**，顶部按钮可切换：
    1. 机器学习  —— 四个传统模型(Logistic/SVM/DT/RF)，由 titanic/train.py 训练
    2. 深度学习  —— PyTorch 多层感知机 MLP，由 titanic/deep_learning.py 训练

应用提供以下接口：
    GET  /                    -> 主页：深度学习(PyTorch MLP)页
    GET  /ml                  -> 传统机器学习页(图表 + 预测表单)
    GET  /deep                -> 兼容旧链接：307 重定向到主页 /
    GET  /api/health          -> 服务健康检查
    GET  /api/summary         -> 四个传统模型指标
    GET  /api/predictions     -> 传统模型：测试集真实 vs 预测
    POST /api/predict         -> 传统模型：预测单个乘客
    GET  /api/predict?..      -> 传统模型：同上(查询参数)
    GET  /api/dl/summary      -> MLP 指标 + 训练历史 + 超参数
    GET  /api/dl/predictions  -> MLP：测试集真实 vs 预测
    GET  /api/dl/predict?..   -> MLP：预测单个乘客
    POST /api/dl/predict      -> MLP：预测单个乘客(JSON)

运行方式:  uvicorn app:app --host 0.0.0.0 --port 8000
"""

from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from titanic import config
from titanic import train as trainer
from titanic import deep_learning as dl
from titanic.i18n import CHART_CAPTIONS, DL_CAPTIONS, UI, pick as i18n_pick
from titanic.preprocessing import build_preprocessor, load_data, make_features

# 四个模型的人类可读名称(id -> label)
MODEL_LABELS = dict(config.MODELS)


# ---------------------------------------------------------------------------
# 应用生命周期：启动时加载训练好的产物一次
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时复用 `python -m titanic.train` 生成的产物。

    如果 outputs/metrics_all_models.csv + models/pipeline_*.joblib 已存在
    (正常跑过一次管道后就是这样)，加载它们瞬间完成，避免重新训练。
    否则第一次 API 调用时退化为懒训练(之后内存缓存)。
    """
    trainer.preprocessor = build_preprocessor()
    pipelines_dir = config.PROJECT_ROOT / "models"
    metrics_path = config.CSV_DIR / "metrics_all_models.csv"
    if metrics_path.exists():
        try:
            # 1. 重新加载四个序列化管道
            for name, _ in config.MODELS:
                p = pipelines_dir / f"pipeline_{name}.joblib"
                if p.exists():
                    trainer._pipelines[name] = joblib.load(p)
            # 2. 从保存的 CSV 重建内存元信息
            metrics_df = pd.read_csv(metrics_path)
            trainer._meta["metrics"] = metrics_df.to_dict(orient="records")
            trainer._meta["figures"] = {
                p.name: str(p) for p in sorted(config.FIG_DIR.glob("*.png"))
            }
            trainer._meta["best_model_id"] = str(
                metrics_df.loc[metrics_df["accuracy"].idxmax(), "model_id"]
            )
            # 3. 从第一个预测 CSV 重建测试集行(每个模型的 id/真实/预测)，
            #    让 /api/predictions 不用重新训练就能工作
            first_pred_csv = sorted(config.CSV_DIR.glob("predictions_*.csv"))
            if first_pred_csv:
                first = pd.read_csv(first_pred_csv[0])
                trainer._meta["n_rows"] = int(first.shape[0])  # 测试集行数
                trainer._meta["test_ids"] = first["PassengerId"].tolist()
                trainer._meta["y_test"] = first["Survived_true"].tolist()
                trainer._meta["y_pred"] = {}
                for csv_file in first_pred_csv:
                    name = csv_file.name.removeprefix("predictions_").removesuffix(".csv")
                    df = pd.read_csv(csv_file)
                    trainer._meta["y_pred"][name] = df["Survived_pred"].tolist()
            print("[startup] Loaded trained pipelines + artifacts from disk.")
        except Exception as exc:  # 产物损坏 -> 按需懒训练
            trainer._pipelines.clear()
            print(f"[startup] Artifact reload failed ({exc}); will retrain on demand.")
    yield


app = FastAPI(
    title="Titanic Survival Prediction",
    description="AI 课程作业：泰坦尼克号生还预测 + 数据可视化。",
    version="1.0.0",
    lifespan=lifespan,
)

# 托管静态资源(如果有 CSS)与生成的图表
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")
app.mount("/figures", StaticFiles(directory=config.FIG_DIR), name="figures")
app.mount("/dl_figures", StaticFiles(directory=config.DL_FIG_DIR), name="dl_figures")

templates = Jinja2Templates(directory=str(config.TEMPLATE_DIR))


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _predict_from_row(row: dict, model_id: str | None = None) -> dict:
    """用训练好的管道预测一个乘客的生还情况。

    Args:
        row: 特征字典，键为 Pclass/Sex/Age/SibSp/Parch/Fare/Embarked。
        model_id: 模型 key；为 None 时使用训练轮次里的最优模型。
    """
    meta = trainer.get_meta()
    if model_id is None or model_id not in trainer._pipelines:
        model_id = meta.get("best_model_id", "random_forest")
    pipe = trainer._pipelines[model_id]

    # 构造单行特征表(列与训练数据一致、顺序一致)
    one = pd.DataFrame([row], columns=config.FORM_COLS)

    # 概率估计对任何能输出概率的模型都稳健；
    # 树模型/线性模型都通过管道暴露 predict_proba。
    proba = pipe.predict_proba(one)[0]      # [P(未生还), P(生还)]
    survived_prob = float(proba[1])
    pred = int(pipe.predict(one)[0])
    return {
        "model_id": model_id,
        "model_label": MODEL_LABELS.get(model_id, model_id),
        "survived": pred,                   # 1 = 生还, 0 = 未生还
        "survived_probability": round(survived_prob, 4),
        "not_survived_probability": round(1 - survived_prob, 4),
        "prediction_text": "SURVIVED" if pred == 1 else "NOT SURVIVED",
    }


def _validate_payload(payload: dict) -> dict:
    """校验/规整原始 JSON 请求体，转成干净的特征行。"""
    try:
        row = {
            "Pclass": int(payload["Pclass"]),
            "Sex": str(payload["Sex"]).strip().lower(),
            "Age": float(payload.get("Age") or 30.0),
            "SibSp": int(payload.get("SibSp", 0)),
            "Parch": int(payload.get("Parch", 0)),
            "Fare": float(payload.get("Fare") or 32.0),
            "Embarked": str(payload.get("Embarked", "S")).strip().upper(),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid payload: {exc}") from exc
    if row["Pclass"] not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Pclass must be 1, 2 or 3.")
    if row["Sex"] not in ("male", "female"):
        raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'.")
    if row["Embarked"] not in ("C", "Q", "S"):
        raise HTTPException(status_code=400, detail="Embarked must be 'C', 'Q' or 'S'.")
    if row["Age"] < 0 or row["Fare"] < 0:
        raise HTTPException(status_code=400, detail="Age/Fare cannot be negative.")
    return row


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, lang: str = Query("en", pattern="^(en|zh)$")):
    """主页：渲染深度学习(PyTorch MLP)页(训练曲线、指标、预测表单)。

    ``?lang=en|zh`` 选择 UI 语言(默认 en)。顶部模式按钮可切到
    机器学习页(/ml)。
    """
    meta = dl.get_meta()
    charts = [
        {
            "file": name,
            "url": f"/dl_figures/{name}",
            "caption": DL_CAPTIONS.get(name.removesuffix(".png"), ("", ""))[
                1 if lang == "zh" else 0
            ],
        }
        for name in sorted(meta.get("figures", []))
    ]

    def td(key: str) -> str:
        return i18n_pick(key, lang)

    return templates.TemplateResponse(
        request,
        "deep.html",
        {
            "lang": lang,
            "t": td,
            "charts": charts,
            "metrics": meta.get("metrics", {}),
            "history": meta.get("history", []),
            "hyperparams": meta.get("hyperparams", {}),
            "n_train": meta.get("n_train"),
            "n_val": meta.get("n_val"),
            "n_test": meta.get("n_test"),
            "input_dim": meta.get("input_dim"),
            "n_params": meta.get("n_params"),
        },
    )


@app.get("/deep", include_in_schema=False)
async def deep_redirect(lang: str = Query("en", pattern="^(en|zh)$")):
    """兼容旧链接：/deep 是旧的深度学习页地址，现 307 重定向到主页 /。"""
    return RedirectResponse(url=f"/?lang={lang}", status_code=307)


@app.get("/ml", response_class=HTMLResponse)
async def ml_page(request: Request, lang: str = Query("en", pattern="^(en|zh)$")):
    """传统机器学习页：图表、四模型指标表、预测表单。

    ``?lang=en|zh`` 选择 UI 语言(默认 en)。图表本身是双语 PNG，
    两种语言都能用。选择结果由页面存在 localStorage，每次跳转带上。
    """
    meta = trainer.get_meta()
    # (file, url, caption) — caption 跟随页面语言
    charts = [
        {
            "file": name,
            "url": f"/figures/{name}",
            "caption": CHART_CAPTIONS.get(name.removesuffix(".png"), ("", ""))[
                1 if lang == "zh" else 0
            ],
        }
        for name in sorted(meta.get("figures", {}))
    ]
    # 暴露全部 UI 文案: 模板调用 t(key) / cap(key)
    def t(key: str) -> str:
        return i18n_pick(key, lang)

    def cap(key: str) -> str:
        en, zh_cn = CHART_CAPTIONS.get(key, ("", ""))
        return zh_cn if lang == "zh" else en

    return templates.TemplateResponse(
        request,  # 新版签名: (request, name, context)
        "index.html",
        {
            "lang": lang,
            "t": t,
            "cap": cap,
            "charts": charts,
            "metrics": meta.get("metrics", []),
            "best_model": meta.get("best_model_id", ""),
            "model_labels": MODEL_LABELS,
        },
    )


@app.get("/api/health")
async def health():
    """简单健康探测(供启动脚本 / 冒烟测试使用)。"""
    return {"status": "ok", "service": "titanic-survival-api"}


@app.get("/api/summary")
async def summary():
    """完整训练总结：每个模型的指标 + 生成的图表清单。"""
    return JSONResponse(trainer.get_meta())


@app.get("/api/predictions")
async def predictions(model: str = Query("best", description="model id or 'best'")):
    """测试集对照表：PassengerId + 真实标签 + 预测标签(+ 是否正确)。"""
    meta = trainer.get_meta()
    best = meta.get("best_model_id", "random_forest")
    model_id = best if model in ("best", "") else model
    if model_id not in trainer._pipelines:
        raise HTTPException(status_code=404, detail=f"Unknown model: {model_id}")
    table = pd.DataFrame(
        {
            "PassengerId": meta["test_ids"],
            "Survived_true": meta["y_test"],
            "Survived_pred": meta["y_pred"][model_id],
        }
    )
    table["Correct"] = (table["Survived_true"] == table["Survived_pred"]).astype(int)
    return JSONResponse(table.to_dict(orient="records"))


@app.post("/api/predict")
async def predict_post(payload: dict):
    """预测一个乘客：JSON 请求体提交。"""
    row = _validate_payload(payload)
    return _predict_from_row(row, payload.get("model"))


@app.get("/api/predict")
async def predict_get(request: Request):
    """预测一个乘客：查询参数提交。

    同时接受 `Pclass`(HTML 表单字段名)和 `pclass`(规范写法)两种拼写——
    FastAPI/HTTP 查询参数区分大小写，而演示表单提交的是大写字段名，
    所以直接从请求里读取，避免大小写不一致的问题。
    """
    q = request.query_params

    # 大小写不敏感地取值；如果两个都给了，规范的小写优先
    def pick(*names, default=None):
        for n in names:
            if n in q:
                return q[n]
        return default

    pclass_raw = pick("pclass", "Pclass")
    sex_raw = pick("sex", "Sex")
    age_raw = pick("age", "Age")
    sibsp_raw = pick("sibsp", "SibSp")
    parch_raw = pick("parch", "Parch")
    fare_raw = pick("fare", "Fare")
    embarked_raw = pick("embarked", "Embarked")
    model_raw = pick("model", "Model")

    # 默认值与 HTML 表单一致
    try:
        pclass = int(pclass_raw) if pclass_raw is not None else 3
        sex = (sex_raw or "male").strip().lower()
        age = float(age_raw) if age_raw is not None else 30.0
        sibsp = int(sibsp_raw) if sibsp_raw is not None else 0
        parch = int(parch_raw) if parch_raw is not None else 0
        fare = float(fare_raw) if fare_raw is not None else 32.0
        embarked = (embarked_raw or "S").strip().upper()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid number: {exc}") from exc

    row = {
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
    }
    # 轻量校验(与 JSON 端点同样的规则)
    if pclass not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Pclass must be 1, 2 or 3.")
    if sex not in ("male", "female"):
        raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'.")
    if embarked not in ("C", "Q", "S"):
        raise HTTPException(status_code=400, detail="Embarked must be 'C', 'Q' or 'S'.")
    return _predict_from_row(row, model_raw)


# ---------------------------------------------------------------------------
# 深度学习(PyTorch MLP)路由 —— 主页 / ；顶部按钮在 /ml 与 / 间切换
# ---------------------------------------------------------------------------

@app.get("/api/dl/summary")
async def dl_summary():
    """MLP 训练总结：指标 + 超参数 + 训练历史。"""
    return JSONResponse(dl.get_meta())


@app.get("/api/dl/predictions")
async def dl_predictions():
    """MLP 在测试集上的真实 vs 预测明细。"""
    meta = dl.get_meta()
    table = pd.DataFrame({
        "PassengerId": meta["test_ids"],
        "Survived_true": meta["y_test"],
        "Survived_pred": meta["y_pred"],
        "Survived_prob": meta["y_prob"],
    })
    table["Correct"] = (table["Survived_true"] == table["Survived_pred"]).astype(int)
    return JSONResponse(table.to_dict(orient="records"))


@app.post("/api/dl/predict")
async def dl_predict_post(payload: dict):
    """用 MLP 预测单个乘客(JSON 请求体)。"""
    row = _validate_payload(payload)
    return dl.predict_one(row)


@app.get("/api/dl/predict")
async def dl_predict_get(request: Request):
    """用 MLP 预测单个乘客(查询参数，与表单字段大小写无关)。"""
    q = request.query_params

    def pick(*names, default=None):
        for n in names:
            if n in q:
                return q[n]
        return default

    try:
        pclass_raw = pick("pclass", "Pclass")
        sex_raw = pick("sex", "Sex")
        age_raw = pick("age", "Age")
        sibsp_raw = pick("sibsp", "SibSp")
        parch_raw = pick("parch", "Parch")
        fare_raw = pick("fare", "Fare")
        embarked_raw = pick("embarked", "Embarked")
        pclass = int(pclass_raw) if pclass_raw is not None else 3
        sex = (sex_raw or "male").strip().lower()
        age = float(age_raw) if age_raw is not None else 30.0
        sibsp = int(sibsp_raw) if sibsp_raw is not None else 0
        parch = int(parch_raw) if parch_raw is not None else 0
        fare = float(fare_raw) if fare_raw is not None else 32.0
        embarked = (embarked_raw or "S").strip().upper()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid number: {exc}") from exc

    row = {"Pclass": pclass, "Sex": sex, "Age": age, "SibSp": sibsp,
           "Parch": parch, "Fare": fare, "Embarked": embarked}
    if pclass not in (1, 2, 3):
        raise HTTPException(status_code=400, detail="Pclass must be 1, 2 or 3.")
    if sex not in ("male", "female"):
        raise HTTPException(status_code=400, detail="Sex must be 'male' or 'female'.")
    if embarked not in ("C", "Q", "S"):
        raise HTTPException(status_code=400, detail="Embarked must be 'C', 'Q' or 'S'.")
    return dl.predict_one(row)


# ---------------------------------------------------------------------------
# 支持: python app.py  (开发时方便)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
