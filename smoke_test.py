# -*- coding: utf-8 -*-
"""对运行中的 FastAPI 服务做冒烟测试(用 .venv 的 python 运行)。

覆盖两种语言：英文页(默认)和中文页(?lang=zh)。
用法: python smoke_test.py [base_url]   (默认 http://127.0.0.1:8000)
"""
import json
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.status, r.read()


# ---- 机器学习页(默认语言) ------------------------------------------------
status, body = get("/ml")
html = body.decode("utf-8")
print("EN /ml (machine learning) status:", status, "len:", len(html))
for kw in [
    "Support Vector Machine",   # 指标表里的模型名
    "Try a Prediction",         # 预测按钮
    "SURVIVED",                 # JS 里的结果模板
    "Data Visualization",       # 导航链接
    "Accuracy",                 # 表格列
    "Service health check",     # API 说明表
    "Scikit-learn",             # 页脚
    "langBtn",                  # 语言切换按钮
    "/?lang=",                  # 顶部模式切换按钮指回主页(深度学习)
]:
    print("EN ml", repr(kw), "->", kw in html)

# ---- 机器学习页(中文) ----------------------------------------------------
status, body = get("/ml?lang=zh")
zh = body.decode("utf-8")
print("ZH /ml status:", status, "len:", len(zh))
for kw in [
    "泰坦尼克号生还预测",   # 页面标题
    "数据可视化",          # 导航链接
    "预测生还",            # 预测按钮
    "准确率",              # 表格列
    "性别与生还率",        # 图表说明
    "最优模型",            # 最优模型提示
    "模型：",              # 结果区模型标签
    "预测中",              # JS 忙态文本
    "服务健康检查",        # API 说明表
    "langBtn",             # 语言切换按钮
]:
    print("ZH", repr(kw), "->", kw in zh)

print("lang button on EN:", "langBtn" in html, "| on ZH:", "langBtn" in zh)

# ---- /deep 旧链接应 307 重定向到主页 --------------------------------------
import urllib.error
req = urllib.request.Request(BASE + "/deep", method="GET")
class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None
try:
    urllib.request.build_opener(_NoRedirect).open(req, timeout=10)
    print("/deep redirect: NO redirect (unexpected)")
except urllib.error.HTTPError as e:
    loc = e.headers.get("Location", "")
    print("/deep redirect:", e.code, "->", loc)

# ---- 图表(8 张双语 PNG) -------------------------------------------------
print("img tags:", zh.count('<img src="/figures/'))
status, body = get("/figures/1_survived_counts.png")
print("png status:", status, "bytes:", len(body))

# ---- 预测结果 API ---------------------------------------------------------
for model in ["logistic", "svm", "decision_tree", "random_forest"]:
    status, body = get(f"/api/predictions?model={model}")
    rows = json.loads(body.decode("utf-8"))
    print(f"predictions[{model}]:", status, "rows:", len(rows),
          "first:", rows[0]["PassengerId"], rows[0]["Survived_true"])

# ---- 预测 API -------------------------------------------------------------
status, body = get("/api/predict?Pclass=1&Sex=female&Age=25&Fare=110&Embarked=C")
j = json.loads(body.decode("utf-8"))
print("predict GET:", status, j["prediction_text"], "prob:", j["survived_probability"])
print("docs status:", get("/docs")[0])

# ==========================================================================
#  深度学习(PyTorch MLP)部分
# ==========================================================================
# ---- 深度学习页(现在是主页 /) -----------------------------------------
status, body = get("/")
dl_en = body.decode("utf-8")
print("EN / (deep learning home) status:", status, "len:", len(dl_en))
for kw in ["Deep Learning", "Training Curves", "PyTorch MLP",
           "Machine Learning (4 models)", "langBtn", "/ml?lang="]:
    print("EN deep-home", repr(kw), "->", kw in dl_en)

status, body = get("/?lang=zh")
dl_zh = body.decode("utf-8")
print("ZH / status:", status, "len:", len(dl_zh))
for kw in ["深度学习", "训练曲线", "机器学习", "测试集", "超参数"]:
    print("ZH deep", repr(kw), "->", kw in dl_zh)

# ---- 深度学习图(6 张) ---------------------------------------------------
print("dl img tags:", dl_zh.count('<img src="/dl_figures/'))
status, body = get("/dl_figures/01_loss_curve.png")
print("dl png status:", status, "bytes:", len(body))

# ---- 深度学习 API -------------------------------------------------------
status, body = get("/api/dl/summary")
meta = json.loads(body.decode("utf-8"))
print("dl summary:", status, "metrics:", meta["metrics"],
      "params:", meta.get("n_params"), "hist_len:", len(meta.get("history", [])))

status, body = get("/api/dl/predictions")
rows = json.loads(body.decode("utf-8"))
print("dl predictions:", status, "rows:", len(rows))

status, body = get("/api/dl/predict?Pclass=1&Sex=female&Age=29&Fare=110&Embarked=C")
j = json.loads(body.decode("utf-8"))
print("dl predict GET:", status, j["prediction_text"], "prob:", j["survived_probability"])

print("SMOKE TEST DONE")
