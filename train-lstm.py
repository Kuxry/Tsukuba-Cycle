import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import joblib

# Step 1: 读取和处理数据
data = pd.read_excel('data2.xlsx')
data.replace({'#DIV/0!': 0}, inplace=True)
data['利用開始日'] = pd.to_datetime(data['利用開始日'], format='%Y-%m-%d')

# 提取日期相关特征：年、月、日、星期
data['年'] = data['利用開始日'].dt.year
data['月'] = data['利用開始日'].dt.month
data['日'] = data['利用開始日'].dt.day
data['星期'] = data['利用開始日'].dt.weekday

# 创建周期性特征
data['月_sin'] = np.sin(2 * np.pi * data['月'] / 12)
data['月_cos'] = np.cos(2 * np.pi * data['月'] / 12)
data['星期_sin'] = np.sin(2 * np.pi * data['星期'] / 7)
data['星期_cos'] = np.cos(2 * np.pi * data['星期'] / 7)

# 类别编码
label_encoder_portid = LabelEncoder()
data['PortID'] = label_encoder_portid.fit_transform(data['PortID'])

label_encoder_station = LabelEncoder()
data['利用ステーション'] = label_encoder_station.fit_transform(data['利用ステーション'])

label_encoder_type = LabelEncoder()
data['立地タイプ'] = label_encoder_type.fit_transform(data['立地タイプ'])

label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# 特征工程：创建交互特征
data['人口_就业交互'] = data['人口_総数_300m以内'] * data['就業者_通学者割合']
data['距离交互'] = data['バスとの距離'] * data['駅との距離']

# 数值特征标准化
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合',
                '就業者_通学者利用交通手段_自転車割合', '月_sin', '月_cos', '星期_sin', '星期_cos',
                '人口_就业交互', '距离交互']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 保存 scaler 和 LabelEncoder 对象
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_station, 'label_encoder_station.joblib')
joblib.dump(label_encoder_type, 'label_encoder_type.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')
joblib.dump(label_encoder_portid, 'label_encoder_portid.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', 'PortID', '利用ステーション',
          '年', '月', '日', '月_sin', '月_cos', '星期_sin', '星期_cos',
          '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合',
          '就業者_通学者利用交通手段_自転車割合',
          '人口_就业交互', '距离交互']]
y = data['利用回数']

# 创建时间序列数据
sequence_length = 5
X_seq, y_seq = [], []
for i in range(len(X) - sequence_length):
    X_seq.append(X.iloc[i:i + sequence_length].values)
    y_seq.append(y.iloc[i + sequence_length])
X_seq, y_seq = np.array(X_seq), np.array(y_seq)

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)


# Step 2: 定义LSTM模型
class LSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size=1):
        super(LSTMPredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c_0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        _, (h_out, _) = self.lstm(x, (h_0, c_0))
        h_out = h_out.view(-1, self.hidden_size)
        return self.fc(h_out)


# Step 3: 使用贝叶斯优化确定最佳超参数
space = {
    'hidden_size': hp.choice('hidden_size', [32, 64, 128]),
    'num_layers': hp.choice('num_layers', [1, 2, 3]),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01)),
    'batch_size': hp.choice('batch_size', [16, 32, 64])
}


def objective(params):
    hidden_size = params['hidden_size']
    num_layers = params['num_layers']
    learning_rate = params['learning_rate']
    batch_size = params['batch_size']

    model = LSTMPredictor(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(10):
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i:i + batch_size]
            y_batch = y_train[i:i + batch_size]
            if X_batch.size(0) < batch_size:
                continue
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = mean_squared_error(y_test.numpy(), predictions.numpy())

    return {'loss': mse, 'status': STATUS_OK}


# 贝叶斯优化
trials = Trials()
best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials)
print("最佳参数：", best_params)

# Step 4: 使用最佳参数训练模型
hidden_size = [32, 64, 128][best_params['hidden_size']]
num_layers = [1, 2, 3][best_params['num_layers']]
learning_rate = best_params['learning_rate']
batch_size = [16, 32, 64][best_params['batch_size']]

model = LSTMPredictor(input_size=X_train.shape[2], hidden_size=hidden_size, num_layers=num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        if X_batch.size(0) < batch_size:
            continue
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 评估模型
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    mse = mean_squared_error(y_test.numpy(), predictions.numpy())
    rmse = np.sqrt(mse)
    print(f"LSTM模型的最终RMSE: {rmse:.2f}")
