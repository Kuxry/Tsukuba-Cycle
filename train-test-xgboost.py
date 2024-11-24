import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 查看数据结构
print("Train Data Sample:")
print(train_data.head())

print("Test Data Sample:")
print(test_data.head())

# 特征和目标变量定义（假设目标列为 'target'，根据实际数据替换）
target_column = 'count'  # 替换为实际目标列名
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# 预处理（视数据需要处理缺失值和编码）
X = pd.get_dummies(X)
test_data = pd.get_dummies(test_data)

# 确保测试集和训练集的列对齐
test_data = test_data.reindex(columns=X.columns, fill_value=0)

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建DMatrix对象
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(test_data)

# 设置XGBoost参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'eval_metric': 'rmse',            # 使用RMSE作为评价指标
    'max_depth': 6,
    'eta': 0.3,
    'seed': 42
}

# 训练模型
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)

# 验证集预测
y_val_pred = model.predict(dval)

# 计算回归分数
mse = mean_squared_error(y_val, y_val_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_val, y_val_pred)
r2 = r2_score(y_val, y_val_pred)

print(f"Validation MSE: {mse}")
print(f"Validation RMSE: {rmse}")
print(f"Validation MAE: {mae}")
print(f"Validation R²: {r2}")

# 测试集预测
y_test_pred = model.predict(dtest)

# 保存预测结果
test_data['Prediction'] = y_test_pred
test_data.to_excel('/mnt/data/test_predictions_with_scores.xlsx', index=False)
print("预测结果已保存至 test_predictions_with_scores.xlsx")
