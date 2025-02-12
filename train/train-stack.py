# 导入所需库
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import torch
import torch.nn as nn
import torch.optim as optim
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from tqdm import tqdm


# 固定 random_state 值
random_state_value = 42  # 选择任意整数值来保持一致性

# 自定义 MAPE 计算函数，避免除零问题
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100


# 数据预处理
data = pd.read_excel('data2.xlsx')
data.replace({'#DIV/0!': 0}, inplace=True)

# 填充数值列中的 NaN 值
numeric_cols = data.select_dtypes(include=[np.number]).columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

label_encoder = LabelEncoder()
data['立地タイプ'] = label_encoder.fit_transform(data['立地タイプ'])
label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合',
                '就業者_通学者利用交通手段_自転車割合']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
joblib.dump(scaler, '../scaler.joblib')
joblib.dump(label_encoder, 'label_encoder.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', '人口_総数_300m以内', '男性割合', '15_64人口割合',
          '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']]
y = data['利用回数']

# XGBoost 贝叶斯优化搜索空间
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 500, 50),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
    'gamma': hp.uniform('gamma', 0, 0.5),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'reg_alpha': hp.uniform('reg_alpha', 0, 0.1),
    'reg_lambda': hp.uniform('reg_lambda', 0.8, 1.2)
}



# 定义XGBoost目标函数
# 定义XGBoost目标函数
def objective(params):
    params['max_depth'] = int(params['max_depth'])
    params['n_estimators'] = int(params['n_estimators'])  # 确保 n_estimators 是整数
    model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda', **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {'loss': mse, 'status': STATUS_OK}


# RNN模型定义
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        return self.fc(out)


# 模型训练与堆叠
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_trials = 3
stacked_results = []

for i in range(num_trials):
    print(f"第 {i + 1} 次模型训练")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state_value)

    # XGBoost模型训练
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=200, trials=trials)
    best_params = {
        'n_estimators': int(best['n_estimators']),  # 使用整数
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma']
    }
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda', **best_params)
    xgb_model.fit(X_train, y_train)
    xgb_pred_train = xgb_model.predict(X_train)
    xgb_pred_test = xgb_model.predict(X_test)

    # KNN模型训练
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    knn_pred_train = knn_model.predict(X_train)
    knn_pred_test = knn_model.predict(X_test)


    # RNN模型训练
    X_train_rnn = torch.tensor(X_train.values, dtype=torch.float32).unsqueeze(2).to(device)
    y_train_rnn = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    X_test_rnn = torch.tensor(X_test.values, dtype=torch.float32).unsqueeze(2).to(device)

    rnn_model = RNNModel(input_size=1, hidden_size=64, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

    # 使用 tqdm 显示进度条
    epochs = 100
    for epoch in tqdm(range(epochs), desc="训练 RNN 模型"):
        rnn_model.train()
        optimizer.zero_grad()
        outputs = rnn_model(X_train_rnn)
        loss = criterion(outputs, y_train_rnn)
        loss.backward()
        optimizer.step()

    rnn_model.eval()
    with torch.no_grad():
        rnn_pred_train = rnn_model(X_train_rnn).cpu().numpy().flatten()
        rnn_pred_test = rnn_model(X_test_rnn).cpu().numpy().flatten()

    # 合并模型预测结果
    stacked_train = np.column_stack((xgb_pred_train, knn_pred_train, rnn_pred_train))
    stacked_test = np.column_stack((xgb_pred_test, knn_pred_test, rnn_pred_test))

    # 使用线性回归作为元学习器
    stacked_model = LinearRegression()
    stacked_model.fit(stacked_train, y_train)
    final_predictions = stacked_model.predict(stacked_test)

    # 计算评估指标
    mse = mean_squared_error(y_test, final_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, final_predictions)
    mape = mean_absolute_percentage_error(y_test, final_predictions)
    r2 = r2_score(y_test, final_predictions)

    print(f"第 {i + 1} 次集成模型的 MSE: {mse}")
    print(f"第 {i + 1} 次集成模型的 RMSE: {rmse}")
    print(f"第 {i + 1} 次集成模型的 MAE: {mae}")
    print(f"第 {i + 1} 次集成模型的 MAPE: {mape:.2f}%")
    print(f"第 {i + 1} 次集成模型的 R²: {r2}")

    stacked_results.append((final_predictions, mse, rmse, mae, mape, r2))
