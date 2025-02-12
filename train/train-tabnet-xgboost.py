import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, hp, Trials
import joblib

# 自定义 MAPE 计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # 只计算非零值
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# Step 1: 读取和处理数据
data = pd.read_excel('data2.xlsx')  # 替换为你的 Excel 文件路径
data.replace({'#DIV/0!': 0}, inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 类别编码 for 利用ステーション and 立地タイプ (Type of location)
label_encoder_station = LabelEncoder()
data['利用ステーション'] = label_encoder_station.fit_transform(data['利用ステーション'])
label_encoder_type = LabelEncoder()
data['立地タイプ'] = label_encoder_type.fit_transform(data['立地タイプ'])
label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# Step 2: 标准化数值特征
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']].values
y = data['利用回数'].values

# Step 3: XGBoost参数空间定义和目标函数
space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
    'gamma': hp.uniform('gamma', 0, 0.5)
}

def objective(params):
    params['max_depth'] = int(params['max_depth'])
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        **params
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {'loss': mse, 'status': 'ok'}

# Step 4: 循环3次训练XGBoost模型
num_trials = 1
xgb_preds_test_all = []

for i in range(num_trials):
    print(f"第 {i+1} 次 XGBoost 模型训练")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # 贝叶斯优化
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)

    best_params = {
        'n_estimators': [100, 200, 300, 400][best['n_estimators']],
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma']
    }

    # 使用最优参数训练XGBoost模型
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        **best_params
    )
    best_model.fit(X_train, y_train)

    # 保存每次测试集上的预测结果
    xgb_preds_test_all.append(best_model.predict(X_test))

# 将3次XGBoost模型的预测平均
xgb_preds_test = np.mean(xgb_preds_test_all, axis=0)

# Step 5: 评估最终XGBoost模型表现
mse = mean_squared_error(y_test, xgb_preds_test)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, xgb_preds_test)
mape = mean_absolute_percentage_error(y_test, xgb_preds_test)
r2 = r2_score(y_test, xgb_preds_test)

print(f"XGBoost模型的均方误差 (MSE): {mse}")
print(f"XGBoost模型的根均方误差 (RMSE): {rmse}")
print(f"XGBoost模型的平均绝对误差 (MAE): {mae}")
print(f"XGBoost模型的平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"XGBoost模型的 R² 分数为: {r2:.2f}")
