import torch
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 自定义 MAPE 计算函数，避免除零问题
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

# 保存 scaler 和 LabelEncoder 对象，供以后预测时使用
joblib.dump(scaler, '../scaler.joblib')
joblib.dump(label_encoder_station, 'label_encoder_station.joblib')
joblib.dump(label_encoder_type, 'label_encoder_type.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']].values
y = data['利用回数'].values

# Step 3: 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Step 4: 定义TabNet参数搜索空间
tabnet_space = {
    'n_d': hp.quniform('n_d', 8, 32, 4),
    'n_a': hp.quniform('n_a', 8, 32, 4),
    'n_steps': hp.quniform('n_steps', 3, 10, 1),
    'gamma': hp.uniform('gamma', 1.0, 2.0),
    'lambda_sparse': hp.loguniform('lambda_sparse', np.log(1e-5), np.log(1e-2))
}

# 定义TabNet的目标函数
def tabnet_objective(params):
    params['n_d'] = int(params['n_d'])
    params['n_a'] = int(params['n_a'])
    params['n_steps'] = int(params['n_steps'])

    model = TabNetRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric=['rmse'],
        max_epochs=500,
        patience=300,
        batch_size=1024,
        virtual_batch_size=128
    )

    preds = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, preds)
    return mse

# 贝叶斯优化TabNet参数
tabnet_trials = Trials()
best_tabnet_params = fmin(
    fn=tabnet_objective,
    space=tabnet_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=tabnet_trials
)

# 使用最佳参数训练TabNet
best_tabnet_params['n_d'] = int(best_tabnet_params['n_d'])
best_tabnet_params['n_a'] = int(best_tabnet_params['n_a'])
best_tabnet_params['n_steps'] = int(best_tabnet_params['n_steps'])

optimized_tabnet_model = TabNetRegressor(**best_tabnet_params)
optimized_tabnet_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],
    max_epochs=500,
    patience=300,
    batch_size=1024,
    virtual_batch_size=128
)
tabnet_preds = optimized_tabnet_model.predict(X_test).flatten()

# Step 5: 训练XGBoost回归模型
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=6,
    learning_rate=0.1,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
xgb_model.fit(X_train, y_train.ravel())
xgb_preds = xgb_model.predict(X_test)

# Step 6: 加权平均组合XGBoost和优化后的TabNet预测
xgb_weight = 0.7
tabnet_weight = 0.3
ensemble_preds = xgb_weight * xgb_preds + tabnet_weight * tabnet_preds

# Step 7: 评估模型表现
y_test_flat = y_test.flatten()
mse = mean_squared_error(y_test_flat, ensemble_preds)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_flat, ensemble_preds)
mape = mean_absolute_percentage_error(y_test_flat, ensemble_preds)
r2 = r2_score(y_test_flat, ensemble_preds)

print(f"集成模型的均方误差 (MSE): {mse}")
print(f"集成模型的根均方误差 (RMSE): {rmse}")
print(f"集成模型的平均绝对误差 (MAE): {mae}")
print(f"集成模型的平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"集成模型的 R² 分数为: {r2:.2f}")
