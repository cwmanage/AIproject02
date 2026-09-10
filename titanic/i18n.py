"""网页 UI 的轻量 i18n 字典(英文 <-> 简体中文)。

单独放在一个模块里，模板上下文保持整洁，每条文案只有一个源头。
图本身是双语的(EN + 中文直接画进 PNG)，所以切换页面语言永远不用
换图。
"""

# ---------------------------------------------------------------------------
# UI 文案: key -> (en, zh)
# ---------------------------------------------------------------------------
UI = {
    # 页面骨架
    "page_title": ("Titanic Survival Prediction — AI Coursework",
                   "泰坦尼克号生还预测 — AI 课程作业"),
    "header_subtitle": ("Titanic Dataset · Classification · Scikit-learn · "
                        "FastAPI · Data Visualization",
                        "泰坦尼克数据集 · 分类 · Scikit-learn · FastAPI · 数据可视化"),
    "lang_btn_en": ("中文", "EN"),          # 按钮显示“另一种”语言
    "lang_hint_en": ("Switch to Chinese", "切换到英文"),
    # 导航
    "nav_charts": ("📊 Data Visualization", "📊 数据可视化"),
    "nav_metrics": ("📈 Model Comparison", "📈 模型对比"),
    "nav_predict": ("🔮 Try a Prediction", "🔮 预测演示"),
    "nav_results": ("📄 Test Results", "📄 测试结果"),
    "nav_api": ("⚡ API", "⚡ 接口文档"),
    # 区块标题
    "sec_charts": ("1 · Data Visualization", "1 · 数据可视化"),
    "sec_metrics": ("2 · Model Comparison", "2 · 四模型对比"),
    "sec_predict": ("3 · Try a Prediction", "3 · 填写信息预测生还"),
    "sec_results": ("4 · Test Set Predictions", "4 · 测试集(20%)预测结果"),
    "sec_api": ("5 · REST API", "5 · 接口说明"),
    # 图表区导语
    "charts_intro": ("8 charts from <code>data/titanic.csv</code> (891 passengers). "
                     "Every figure is bilingual (EN + 中文).",
                     "基于 <code>data/titanic.csv</code>(891 名乘客) 生成的 8 张图。"
                     "每张图均含中英双语标注。"),
    # 指标
    "metrics_intro": ("Same 80/20 split, <code>random_state=42</code>, "
                      "<code>stratify=y</code> for all models (courseware "
                      "reproduction). Metrics are computed on the untouched test set.",
                      "四个模型共用同一划分：80/20、<code>random_state=42</code>、"
                      "<code>stratify=y</code>(课件复现规范)，指标在未经触碰的测试集上计算。"),
    "col_model": ("Model", "模型"),
    "col_accuracy": ("Accuracy", "准确率"),
    "col_precision": ("Precision", "精确率"),
    "col_recall": ("Recall", "召回率"),
    "col_f1": ("F1 Score", "F1 分数"),
    "col_auc": ("AUC", "AUC"),
    "auc_note": ("AUC = area under the ROC curve (threshold-independent "
                  "ranking quality, courseware p.38-40)",
                  "AUC = ROC 曲线下面积(不依赖阈值、衡量概率排序能力，课件 p.38-40)"),
    "best_model_line": ("⭐ Best model (highest test accuracy):", 
                        "⭐ 最优模型(测试集准确率最高)："),
    "default_predict_hint": ("— used by default in the prediction demo below",
                             "—— 默认用于下方预测演示"),
    "metrics_missing": ("Metrics not available yet. Run "
                        "<code>python -m titanic.train</code> first.",
                        "暂无指标数据。请先运行 <code>python -m titanic.train</code>。"),
    # 预测表单
    "form_class": ("Passenger Class", "舱位等级"),
    "form_class_1": ("1st — First", "一等舱"),
    "form_class_2": ("2nd — Second", "二等舱"),
    "form_class_3": ("3rd — Third", "三等舱"),
    "form_sex": ("Sex", "性别"),
    "form_female": ("Female", "女"),
    "form_male": ("Male", "男"),
    "form_age": ("Age", "年龄(岁)"),
    "form_sibsp": ("Siblings / Spouse aboard", "兄弟姐妹/配偶人数"),
    "form_parch": ("Parents / Children aboard", "父母/子女人数"),
    "form_fare": ("Fare (£)", "票价(英镑)"),
    "form_embarked": ("Port of Embarkation", "登船港口"),
    "emb_S": ("S — Southampton", "S — 南安普顿"),
    "emb_C": ("C — Cherbourg", "C — 瑟堡"),
    "emb_Q": ("Q — Queenstown", "Q — 皇后镇"),
    "form_model": ("Model", "预测模型"),
    "model_best": ("Best model (auto)", "最优模型(自动)"),
    "btn_predict": ("Predict Survival", "预测生还"),
    "predicting": ("⏳ Predicting…", "⏳ 预测中…"),
    "res_survived": ("✅ SURVIVED", "✅ 预测生还"),
    "res_not_survived": ("❌ NOT SURVIVED", "❌ 预测未生还"),
    "res_model": ("Model", "模型"),
    "res_surv_prob": ("Survival probability", "生还概率"),
    "res_not_prob": ("Not-survived probability", "未生还概率"),
    "res_error": ("⚠️ Error", "⚠️ 错误"),
    # 结果表
    "results_model_label": ("Model:", "模型："),
    "btn_load": ("Load Table", "加载表格"),
    "loading": ("Loading…", "加载中…"),
    "loading_rows": ("Loading rows…", "加载中…"),
    "hint_best_default": ("Showing best model by default", "默认展示最优模型"),
    "col_pid": ("PassengerId", "乘客编号"),
    "col_true": ("True", "真实"),
    "col_pred": ("Predicted", "预测"),
    "col_correct": ("Correct?", "是否正确"),
    "yes": ("✔ yes", "✔ 是"),
    "no": ("✘ no", "✘ 否"),
    "survived_short": ("✅ survived", "✅ 生还"),
    "not_survived_short": ("💀 not survived", "💀 未生还"),
    "rows_correct": ("{n} test rows · {c} correct ({pct}% accuracy)",
                     "{n} 条测试数据 · 正确 {c} 条(准确率 {pct}%)"),
    # API 区
    "api_desc": ("Description", "说明"),
    "api_health": ("Service health check", "服务健康检查"),
    "api_summary": ("All model metrics + figure list", "四模型全部指标 + 图表清单"),
    "api_predictions": ("Test true-vs-predicted rows", "测试集真实 vs 预测明细"),
    "api_predict_get": ("Predict via query string", "通过查询参数预测"),
    "api_predict_post": ("Predict via JSON body", "通过 JSON 请求体预测"),
    "api_note_title": ("POST example", "POST 请求示例"),
    "api_docs_label": ("Interactive API docs", "交互式 API 文档"),
    # 页脚
    "footer_line": ("AI Coursework · Titanic Survival Prediction & Data "
                    "Visualization · Python + FastAPI + Scikit-learn",
                    "AI 课程作业 · 泰坦尼克号生还预测与数据可视化 · Python + FastAPI + Scikit-learn"),
    "footer_files": ("Data: <code>data/titanic.csv</code> · Results: "
                     "<code>outputs/</code> · Models: <code>models/</code>",
                     "数据：<code>data/titanic.csv</code> · 结果：<code>outputs/</code>"
                     " · 模型：<code>models/</code>"),

    # ------------------------------------------------------------------
    # 顶部导航按钮(切换训练方式)
    # ------------------------------------------------------------------
    "navbtn_ml": ("🧮 Machine Learning (4 models)", "🧮 机器学习(四模型)"),
    "navbtn_dl": ("🧠 Deep Learning (PyTorch MLP)", "🧠 深度学习(PyTorch MLP)"),

    # ------------------------------------------------------------------
    # 深度学习页 (主页 /)
    # ------------------------------------------------------------------
    "dl_page_title": ("Titanic Survival Prediction — Deep Learning (PyTorch MLP)",
                      "泰坦尼克号生还预测 — 深度学习(PyTorch MLP)"),
    "dl_header_subtitle": ("Titanic Dataset · PyTorch MLP · Neural Network Training · "
                           "Courseware-aligned",
                           "泰坦尼克数据集 · PyTorch 多层感知机 · 神经网络训练 · 严格对齐课件"),
    "dl_nav_curves": ("📉 Training Curves", "📉 训练曲线"),
    "dl_nav_metrics": ("📈 Test Metrics", "📈 测试指标"),
    "dl_nav_predict": ("🔮 Try a Prediction", "🔮 预测演示"),
    "dl_nav_results": ("📄 Test Results", "📄 测试结果"),
    "dl_nav_arch": ("🧬 Model & Hyperparameters", "🧬 模型与超参数"),
    "dl_sec_curves": ("1 · Training Curves", "1 · 训练曲线"),
    "dl_sec_metrics": ("2 · Test Set Metrics", "2 · 测试集指标"),
    "dl_sec_predict": ("3 · Try a Prediction", "3 · 填写信息预测生还"),
    "dl_sec_results": ("4 · Test Set Predictions", "4 · 测试集(20%)预测结果"),
    "dl_sec_arch": ("5 · Model & Hyperparameters", "5 · 模型结构与超参数"),
    "dl_curves_intro": ("Training curves and evaluation figures generated by "
                        "<code>titanic/deep_learning.py</code> (PyTorch). Every figure "
                        "is bilingual (EN + 中文).",
                        "由 <code>titanic/deep_learning.py</code>(PyTorch) 生成"
                        "的训练曲线与评估图。每张图均为中英双语标注。"),
    "dl_metrics_intro": ("Same 80/20 test split as the ML models "
                         "(<code>random_state=42</code>, <code>stratify=y</code>); "
                         "train 569 / val 143 / test 179. Metrics are computed on the "
                         "untouched test set.",
                         "与机器学习模型使用完全相同的 20% 测试集划分"
                         "(<code>random_state=42</code>、<code>stratify=y</code>)；"
                         "train 569 / val 143 / test 179。指标在未经触碰的测试集上计算。"),
    "dl_col_metric": ("Metric", "指标"),
    "dl_col_value": ("Value", "数值"),
    "dl_descr": ("Description", "说明"),
    "dl_m_accuracy": ("Accuracy", "准确率"),
    "dl_m_precision": ("Precision", "精确率"),
    "dl_m_recall": ("Recall", "召回率"),
    "dl_m_f1": ("F1 Score", "F1 分数"),
    "dl_m_auc": ("AUC", "AUC"),
    "dl_m_accuracy_d": ("Overall correct rate on the test set", "测试集整体判断正确的比例"),
    "dl_m_precision_d": ("Of predicted survivors, how many truly survived", "预测为生还的人里真正生还的比例"),
    "dl_m_recall_d": ("Of true survivors, how many were found", "真实生还的人里被找出的比例"),
    "dl_m_f1_d": ("Harmonic mean of precision and recall", "精确率与召回率的调和平均"),
    "dl_m_auc_d": ("Threshold-independent ranking quality (courseware p.42)", "不依赖阈值的概率排序能力(课件 p.42)"),
    "dl_arch_intro": ("The network below mirrors the courseware (p.17 / p.55). Overview of "
                      "the data split and the training setup:",
                      "下列网络结构完全对齐课件(p.17 / p.55)。数据划分与训练配置概览："),
    "dl_kv_split": ("Data split (train / val / test)", "数据划分(训练 / 验证 / 测试)"),
    "dl_kv_input": ("Input dimension", "输入维度"),
    "dl_kv_params": ("Trainable parameters", "可训练参数量"),
    "dl_kv_seed": ("Random seed", "随机种子"),
    "dl_kv_batch": ("Batch size", "批大小"),
    "dl_kv_lr": ("Learning rate", "学习率"),
    "dl_kv_dropout": ("Dropout", "Dropout 比例"),
    "dl_kv_epochs": ("Epochs", "训练轮数"),
    "dl_kv_wd": ("Weight decay", "权重衰减"),
    "dl_kv_loss": ("Loss function", "损失函数"),
    "dl_kv_opt": ("Optimizer", "优化器"),
    "dl_repro_note": ("Reproducibility: all random seeds (Python / NumPy / PyTorch) and the "
                      "DataLoader generator are fixed, and the split uses "
                      "<code>random_state=42</code> + <code>stratify=y</code>, so every run "
                      "produces an identical training set and identical results.",
                      "可复现性：已固定 Python / NumPy / PyTorch 全部随机种子与 DataLoader "
                      "生成器种子，划分使用 <code>random_state=42</code> + <code>stratify=y</code>，"
                      "因此每次运行得到的训练集完全一致、结果完全一致。"),
    "dl_back_hint": ("Back to machine learning models", "返回机器学习模型页面"),
    "dl_model_label": ("MLP (PyTorch)", "MLP(PyTorch)"),
    "dl_btn_predict": ("Predict Survival", "预测生还"),
    "dl_btn_load": ("Load Table", "加载表格"),
}

# 深度学习页的图表配文
DL_CAPTIONS = {
    "01_loss_curve": ("Train / validation loss over epochs",
                      "训练与验证损失曲线：两条曲线整体下降，模型在学习"),
    "02_accuracy_curve": ("Train / validation accuracy over epochs",
                          "训练与验证准确率曲线：验证准确率反映泛化能力"),
    "03_confusion_matrix": ("Confusion matrix on the test set",
                            "混淆矩阵：Recall 较低对应较多的 FN(漏判生还)"),
    "04_roc_curve": ("ROC curve of the MLP (AUC in legend)",
                     "MLP 的 ROC 曲线(AUC 见图例)"),
    "05_probability_distribution": ("Predicted probability distribution",
                                    "预测概率分布：两类概率越分开越易区分"),
    "06_all_models_comparison": ("All models: ML (4) vs DL (MLP)",
                                 "全部模型对比：传统机器学习(4) vs 深度学习 MLP"),
}

# 图表配文(每张图下面的中文短注)。
CHART_CAPTIONS = {
    "1_survived_counts": ("Target balance: survived vs not survived",
                          "目标变量分布：生还/未生还人数对比"),
    "2_missing_values": ("Missing values per column", "各字段缺失值数量(Cabin 缺失最多)"),
    "3_sex_survival": ("Survival rate by sex", "性别与生还率：女性远高于男性"),
    "4_pclass_survival": ("Survival rate by passenger class", "舱位等级与生还率：头等舱最高"),
    "5_age_distribution": ("Age distribution by survival", "年龄分布(按是否生还分组)"),
    "6_fare_distribution": ("Fare distribution log(1+Fare)", "票价分布(log(1+Fare) 变换)"),
    "7_model_comparison": ("Model comparison: five metrics per model",
                           "四模型对比条形图：每模型五指标(含 AUC)"),
    "8_roc_curves": ("ROC curves of the four models (AUC in legend)",
                     "四模型 ROC 曲线(AUC 见图例)"),
}


def pick(key: str, lang: str) -> str:
    """按 ``lang``('en' | 'zh') 返回 ``key`` 对应的 UI 文案。"""
    en, zh = UI[key]
    return zh if lang == "zh" else en
