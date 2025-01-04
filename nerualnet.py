import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 加载数据
file_path = 'Minlost.csv'  # 请确保文件路径正确
df = pd.read_csv(file_path)

# 初始化结果存储
r2_scores = {"DecisionTree": [], "RandomForest": [], "XGBoost": [], "LightGBM": []}

# 遍历每个站点进行训练和预测
grouped = df.groupby("PortID")
for port_id, group in grouped:
    print(f"Training for PortID {port_id}...")

    # 准备数据
    X = group[['Available bikes']].values
    y = group['機会損失'].values

    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 决策树模型
    dt_model = DecisionTreeRegressor(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_y_pred = dt_model.predict(X_test)
    dt_r2 = r2_score(y_test, dt_y_pred)
    r2_scores["DecisionTree"].append((port_id, dt_r2))

    # 随机森林模型
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_y_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_y_pred)
    r2_scores["RandomForest"].append((port_id, rf_r2))

    # XGBoost 模型
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_y_pred = xgb_model.predict(X_test)
    xgb_r2 = r2_score(y_test, xgb_y_pred)
    r2_scores["XGBoost"].append((port_id, xgb_r2))

    # LightGBM 模型
    lgb_model = lgb.LGBMRegressor(n_estimators=100, random_state=42)
    lgb_model.fit(X_train, y_train)
    lgb_y_pred = lgb_model.predict(X_test)
    lgb_r2 = r2_score(y_test, lgb_y_pred)
    r2_scores["LightGBM"].append((port_id, lgb_r2))

    print(f"PortID {port_id} R² (DecisionTree): {dt_r2:.2f}")
    print(f"PortID {port_id} R² (RandomForest): {rf_r2:.2f}")
    print(f"PortID {port_id} R² (XGBoost): {xgb_r2:.2f}")
    print(f"PortID {port_id} R² (LightGBM): {lgb_r2:.2f}")

# 转换 R² 结果为 DataFrame
dt_scores_df = pd.DataFrame(r2_scores["DecisionTree"], columns=["PortID", "R2"])
rf_scores_df = pd.DataFrame(r2_scores["RandomForest"], columns=["PortID", "R2"])
xgb_scores_df = pd.DataFrame(r2_scores["XGBoost"], columns=["PortID", "R2"])
lgb_scores_df = pd.DataFrame(r2_scores["LightGBM"], columns=["PortID", "R2"])

# 可视化 R² 分数
plt.figure(figsize=(15, 10))
models = ["DecisionTree", "RandomForest", "XGBoost", "LightGBM"]
colors = ["skyblue", "orange", "green", "purple"]
for i, (model, color) in enumerate(zip(models, colors)):
    scores_df = pd.DataFrame(r2_scores[model], columns=["PortID", "R2"])
    plt.bar(
        [x + i * 0.2 for x in range(len(scores_df))],  # 设置偏移量以避免柱状图重叠
        scores_df["R2"],
        width=0.2,
        label=model,
        color=color,
        align="center"
    )

plt.xlabel("PortID")
plt.ylabel("R² Score")
plt.title("R² Scores for Each PortID by Model")
plt.legend()
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()