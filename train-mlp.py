import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
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

# 处理缺失值，例如 '#DIV/0!'
data.replace({'#DIV/0!': 0}, inplace=True)

# 替换可能存在的NaN值以及无穷大值
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 类别编码 for 利用ステーション and 立地タイプ (Type of location)
label_encoder_station = LabelEncoder()
data['利用ステーション'] = label_encoder_station.fit_transform(data['利用ステーション'])

label_encoder_type = LabelEncoder()
data['立地タイプ'] = label_encoder_type.fit_transform(data['立地タイプ'])

# 对 曜日 (Day of the week) 进行类别编码
label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# Step 2: 加入"駅との距離"特征并进行标准化
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合',
                '就業者_通学者利用交通手段_自転車割合']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 保存 scaler 和 LabelEncoder 对象，供以后预测时使用
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_station, 'label_encoder_station.joblib')
joblib.dump(label_encoder_type, 'label_encoder_type.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', '人口_総数_300m以内', '男性割合', '15_64人口割合',
          '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']]
y = data['利用回数']  # 目标变量

# Step 3: 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train.values, dtype=torch.float32)
X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Step 4: 定义神经网络模型
class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

# 初始化模型和优化器
input_dim = X_train.shape[1]
model = MLPRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Step 5: 训练模型
epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Step 6: 评估模型
# 评估模型
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred = y_pred.numpy().flatten()

# 确保 y_test 也是一维数组
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test.numpy().flatten(), y_pred)
r2 = r2_score(y_test, y_pred)

print(f"神经网络模型的均方误差 (MSE): {mse}")
print(f"神经网络模型的根均方误差 (RMSE): {rmse}")
print(f"神经网络模型的平均绝对误差 (MAE): {mae}")
print(f"神经网络模型的平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"神经网络模型的 R² 分数为: {r2:.2f}")

# 保存模型
torch.save(model.state_dict(), 'best_mlp_model.pth')
print("模型已保存为 best_mlp_model.pth")
