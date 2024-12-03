import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

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

test_data['利用ステーション_平均利用次数'] = test_data['利用ステーション'].map(station_mean_count).fillna(global_mean_count)
test_data['利用ステーション_利用方差'] = test_data['利用ステーション'].map(station_var_count).fillna(global_var_count)

# 删除原始的站点列
train_data = train_data.drop(columns=['利用ステーション'])
test_data = test_data.drop(columns=['利用ステーション'])

# 定义特征和目标
target_column = 'count'
X = train_data.drop(columns=[target_column])
y = train_data[target_column]
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# 检查目标变量中 0 值的比例
zero_count = (y == 0).sum()
total_count = len(y)
zero_ratio = zero_count / total_count

print(f"Total Samples: {total_count}")
print(f"Zero Count: {zero_count}")
print(f"Zero Ratio: {zero_ratio:.2%}")

# 分析高值样本的分布
high_value_threshold = y.quantile(0.95)  # 定义高值样本的阈值为 95% 分位数
high_value_count = (y > high_value_threshold).sum()
high_value_ratio = high_value_count / total_count

print(f"High Value Threshold (95% Quantile): {high_value_threshold}")
print(f"High Value Count: {high_value_count}")
print(f"High Value Ratio: {high_value_ratio:.2%}")

# 绘制目标变量分布
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, color='blue', alpha=0.7, edgecolor='black')
plt.axvline(high_value_threshold, color='red', linestyle='--', label=f'High Value Threshold ({high_value_threshold:.2f})')
plt.title('Distribution of Target Variable (Count)')
plt.xlabel('Count')
plt.ylabel('Frequency')
plt.legend()
plt.grid(axis='y')
plt.show()

# 对目标变量进行对数变换
train_data['log_count'] = np.log1p(train_data['count'])
test_data['log_count'] = np.log1p(test_data['count'])

# 替换原目标变量
y = train_data['log_count']
y_test = test_data['log_count']

# 独热编码和对齐测试集列
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 重新训练模型
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

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    y_val_pred = model.predict(dval)
    rmse = mean_squared_error(np.expm1(y_val), np.expm1(y_val_pred), squared=False)  # 反变换评估
    cv_rmse_scores.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse}")

mean_rmse = np.mean(cv_rmse_scores)
print(f"Cross-Validation Mean RMSE: {mean_rmse}")

# 测试集预测
dtest = xgb.DMatrix(X_test)
y_test_pred = model.predict(dtest)

# 反变换预测值
y_test_pred = np.expm1(y_test_pred)
y_test_true = np.expm1(y_test)

# 测试集评估
mse_test = mean_squared_error(y_test_true, y_test_pred)
rmse_test = mse_test ** 0.5
mae_test = mean_absolute_error(y_test_true, y_test_pred)
r2_test = r2_score(y_test_true, y_test_pred)

print(f"Test MSE: {mse_test}")
print(f"Test RMSE: {rmse_test}")
print(f"Test MAE: {mae_test}")
print(f"Test R²: {r2_test}")

# 打印测试集预测结果对比
comparison = pd.DataFrame({
    'Real': y_test_true,
    'Predicted': y_test_pred
})
print("Test Set Comparison (Real vs Predicted):")
print(comparison.head(10))

# 保存对比结果
comparison.to_excel('test_real_vs_predicted.xlsx', index=False)
