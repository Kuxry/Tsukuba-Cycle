import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 删除时间列
train_data = train_data.drop(columns=['利用開始日', '年度', '月', '曜日', 'PortID'], errors='ignore')
test_data = test_data.drop(columns=['利用開始日', '年度', '月', '曜日', 'PortID'], errors='ignore')

# 1. 计算每个站点的历史平均利用次数
station_mean_count = train_data.groupby('利用ステーション')['count'].mean()
station_var_count = train_data.groupby('利用ステーション')['count'].var()

# 将这些统计特征加入训练集
train_data['利用ステーション_平均利用次数'] = train_data['利用ステーション'].map(station_mean_count)
train_data['利用ステーション_利用方差'] = train_data['利用ステーション'].map(station_var_count)

# 对测试集进行同样的处理，填补新站点的缺失值
global_mean_count = train_data['count'].mean()  # 全局均值
global_var_count = train_data['count'].var()    # 全局方差

test_data['利用ステーション_平均利用次数'] = test_data['利用ステーション'].map(station_mean_count).fillna(global_mean_count)
test_data['利用ステーション_利用方差'] = test_data['利用ステーション'].map(station_var_count).fillna(global_var_count)

# 删除原始的站点列
train_data = train_data.drop(columns=['利用ステーション'])
test_data = test_data.drop(columns=['利用ステーション'])

# 定义特征和目标
target_column = 'count'  # 替换为目标列名
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# 确保测试集有真实值
if target_column not in test_data:
    raise ValueError(f"测试集中必须包含真实值列 '{target_column}' 以进行对比分析")

X_test = test_data.drop(columns=[target_column])  # 特征
y_test = test_data[target_column]  # 真实值

# 数据拆分
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用 CatBoostRegressor
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    early_stopping_rounds=10,
    verbose=100
)

# 训练模型
train_pool = Pool(X_train, y_train)
val_pool = Pool(X_val, y_val)
test_pool = Pool(X_test)

model.fit(train_pool, eval_set=val_pool)

# 测试集预测
y_test_pred = model.predict(test_pool)

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

# 绘制实际值 vs 预测值
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
