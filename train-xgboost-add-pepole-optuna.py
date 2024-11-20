import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from xgboost.callback import EarlyStopping
from sklearn.model_selection import KFold

# 自定义 MAPE 计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# Step 1: 数据加载和处理
data = pd.read_excel('data4.xlsx')
# 修改：用中位数代替缺失值
data.replace({'#DIV/0!': np.nan}, inplace=True)
data.fillna(data.median(numeric_only=True), inplace=True)

# 确保必要的列存在
if '人流__1時間平均' not in data.columns:
    raise ValueError("数据中缺少 '人流__1時間平均' 列，请检查数据来源。")

# 日期解析及特征提取
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
data['年'] = data['date'].dt.year
data['月'] = data['date'].dt.month
data['日'] = data['date'].dt.day
data['星期'] = data['date'].dt.weekday
data['月_sin'] = np.sin(2 * np.pi * data['月'] / 12)
data['月_cos'] = np.cos(2 * np.pi * data['月'] / 12)
data['星期_sin'] = np.sin(2 * np.pi * data['星期'] / 7)
data['星期_cos'] = np.cos(2 * np.pi * data['星期'] / 7)

# 类别编码
label_encoder_portid = LabelEncoder()
data['PortID'] = label_encoder_portid.fit_transform(data['PortID'])
label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# 交互特征
data['人口_就业交互'] = data['人口_総数_300m以内'] * data['就業者_通学者割合']
data['距离交互'] = data['バスとの距離'] * data['駅との距離']

# 新增特征处理
new_numeric_cols = ['ポート数_300mBuffer', 'ポート数_500mBuffer', 'ポート数_1000mBuffer', '平均気温', '降水量の合計（mm）']
missing_cols = [col for col in new_numeric_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"以下新增特征缺失: {missing_cols}")
data[new_numeric_cols] = data[new_numeric_cols].fillna(data[new_numeric_cols].median())

# 数值特征标准化
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '@15_64人口割合', '就業者_通学者割合',
                '就業者_通学者利用交通手段_自転車割合', '月_sin', '月_cos', '星期_sin', '星期_cos',
                '人口_就业交互', '距离交互', '人流__1時間平均'] + new_numeric_cols
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 保存编码器和标准化器
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')
joblib.dump(label_encoder_portid, 'label_encoder_portid.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '曜日', 'PortID', '年', '月', '日', '月_sin', '月_cos',
          '星期_sin', '星期_cos', '人口_総数_300m以内', '男性割合', '@15_64人口割合',
          '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合', '人口_就业交互',
          '距离交互', '人流__1時間平均'] + new_numeric_cols]
y = data['利用回数']

# Step 3: Optuna 优化
def objective(trial):
    # 定义超参数空间
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [100, 200, 300, 400,500]),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 0.5),
    }

    # 5 折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_list = []

    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda', **params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # 在验证集上评估
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_list.append(mse)

    # 返回平均 MSE
    return np.mean(mse_list)

# 开始优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=500)

# 输出最佳超参数
print("最佳超参数:", study.best_params)
# 使用最佳参数训练最终模型
best_params = study.best_params  # 从 Optuna Study 中获取最佳超参数
final_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda', **best_params)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
final_model.fit(X_train, y_train)
y_pred = final_model.predict(X_test)

# 模型评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape:.2f}%")
print(f"R²: {r2:.2f}")

# 保存模型
joblib.dump(final_model, 'final_optuna_xgb_model.joblib')
print("模型已保存为 final_optuna_xgb_model.joblib")
