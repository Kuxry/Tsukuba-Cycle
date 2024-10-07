import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 自定义 MAPE 计算函数，避免除零问题
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # 只计算非零值
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# Step 1: 读取和处理数据
data = pd.read_excel('data2.xlsx')  # 替换为你的 Excel 文件路径

# 处理缺失值，例如 '#DIV/0!'
data.replace({'#DIV/0!': 0}, inplace=True)

# 类别编码 for 立地タイプ (Type of location)
label_encoder = LabelEncoder()
data['立地タイプ'] = label_encoder.fit_transform(data['立地タイプ'])

# 对 曜日 (Day of the week) 进行类别编码，将文字转换为数值
label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# Step 2: 加入"駅との距離"特征并进行标准化
# 假设原数据集有"駅との距離"列
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 保存 scaler 和 LabelEncoder 对象，供以后预测时使用
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']]
y = data['利用回数']  # 目标变量

# Step 3: 定义贝叶斯优化的搜索空间
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
    'gamma': hp.uniform('gamma', 0, 0.5)
}

# Step 4: 定义目标函数
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

    return {'loss': mse, 'status': STATUS_OK}

# Step 5: 循环多次训练模型
num_trials = 3  # 定义训练次数
best_models = []

for i in range(num_trials):
    print(f"第 {i+1} 次模型训练")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # 贝叶斯优化
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=trials
    )

    best_params = {
        'n_estimators': [100, 200, 300, 400][best['n_estimators']],
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma']
    }

    # 使用最优参数重新训练模型
    best_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        **best_params
    )
    best_model.fit(X_train, y_train)

    # 预测并评估最终模型
    y_pred = best_model.predict(X_test)

    # 计算 RMSE, MAE, MAPE
    final_mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(final_mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"第 {i+1} 次模型的均方误差 (MSE): {final_mse}")
    print(f"第 {i+1} 次模型的根均方误差 (RMSE): {rmse}")
    print(f"第 {i+1} 次模型的平均绝对误差 (MAE): {mae}")
    print(f"第 {i+1} 次模型的平均绝对百分比误差 (MAPE): {mape:.2f}%")
    print(f"第 {i+1} 次模型的 R² 分数为: {r2:.2f}")

    # 保存模型
    model_filename = f'best_xgb_model_trial_{i+1}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"第 {i+1} 次模型已保存为 {model_filename}")

    # 将每次训练的模型添加到列表中
    best_models.append((best_model, final_mse, rmse, mae, mape, r2))

# 输出所有模型的评估结果
for idx, (model, mse, rmse, mae, mape, r2) in enumerate(best_models):
    print(f"第 {idx+1} 结果: MSE={mse}, RMSE={rmse}, MAE={mae}, MAPE={mape:.2f}%, R²={r2:.2f}")
