import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 删除时间列
train_data = train_data.drop(columns=['利用開始日', '年度', '月', 'PortID'], errors='ignore')
test_data = test_data.drop(columns=['利用開始日', '年度', '月', 'PortID'], errors='ignore')

# 计算每个站点的历史平均利用次数
station_mean_count = train_data.groupby('利用ステーション')['count'].mean()
station_var_count = train_data.groupby('利用ステーション')['count'].var()

# 将统计特征加入训练集
train_data['利用ステーション_平均利用次数'] = train_data['利用ステーション'].map(station_mean_count)
train_data['利用ステーション_利用方差'] = train_data['利用ステーション'].map(station_var_count)

# 对测试集进行同样的处理
global_mean_count = train_data['count'].mean()  # 全局均值
global_var_count = train_data['count'].var()  # 全局方差

test_data['利用ステーション_平均利用次数'] = test_data['利用ステーション'].map(station_mean_count).fillna(
    global_mean_count)
test_data['利用ステーション_利用方差'] = test_data['利用ステーション'].map(station_var_count).fillna(global_var_count)

# 删除原始的站点列
train_data = train_data.drop(columns=['利用ステーション'])
test_data = test_data.drop(columns=['利用ステーション'])

# 定义特征和目标
target_column = 'count'
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# 确保测试集有真实值
if target_column not in test_data:
    raise ValueError(f"测试集中必须包含真实值列 '{target_column}' 以进行对比分析")

X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# 独热编码和对齐测试集列
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 设置 XGBoost 参数
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
}

# 5折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []
cv_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    # 划分训练集和验证集
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 创建 DMatrix 对象
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 训练模型
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # 保存每折的模型
    cv_models.append(model)

    # 验证集预测
    y_val_pred = model.predict(dval)

    # 计算 RMSE
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    cv_rmse_scores.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse}")

# 打印交叉验证结果
mean_rmse = np.mean(cv_rmse_scores)
std_rmse = np.std(cv_rmse_scores)
print(f"Cross-Validation Mean RMSE: {mean_rmse}")
print(f"Cross-Validation Std RMSE: {std_rmse}")

# 使用最后一折模型对测试集进行预测
dtest = xgb.DMatrix(X_test)
y_test_pred = cv_models[-1].predict(dtest)

# 计算测试集指标
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mse_test ** 0.5
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test MSE: {mse_test}")
print(f"Test RMSE: {rmse_test}")
print(f"Test MAE: {mae_test}")
print(f"Test R²: {r2_test}")

# 打印真实值与预测值对比
comparison = pd.DataFrame({
    'Real': y_test,
    'Predicted': y_test_pred
})
print("Test Set Comparison (Real vs Predicted):")
print(comparison.head(10))  # 打印前10行对比

# 绘制实际值和预测值的对比图
plt.figure()
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()

# 分布差距分析
plt.figure(figsize=(12, 6))
plt.hist(comparison['Real'], bins=20, alpha=0.6, label='Real Values', color='blue', edgecolor='black')
plt.hist(comparison['Predicted'], bins=20, alpha=0.6, label='Predicted Values', color='orange', edgecolor='black')
plt.title("Distribution of Real vs Predicted Values")
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('distribution_real_vs_predicted.png')
plt.show()

# 保存对比结果
comparison.to_excel('test_real_vs_predicted.xlsx', index=False)
print("测试集真实值与预测值对比已保存至 test_real_vs_predicted.xlsx")
