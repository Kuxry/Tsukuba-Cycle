import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import r2_score


# Step 1: 加载数据
train_file = '../train.xlsx'  # 替换为你的训练集文件路径
test_file = '../test.xlsx'  # 替换为你的测试集文件路径

train_df = pd.read_excel(train_file)
test_df = pd.read_excel(test_file)

# Step 2: 数据预处理
# 确保按时间排序（假设时间列名为 '利用開始'）
train_df['利用開始日'] = pd.to_datetime(train_df['利用開始日'])
test_df['利用開始日'] = pd.to_datetime(test_df['利用開始日'])

# 删除 PortID 列
train_df = train_df.drop(columns=['PortID','利用ステーション'], errors='ignore')
test_df = test_df.drop(columns=['PortID','利用ステーション'], errors='ignore')

train_df = train_df.sort_values(by='利用開始日')
test_df = test_df.sort_values(by='利用開始日')

# 仅提取目标变量（count）
train_series = train_df['count'].values.reshape(-1, 1)  # 替换 'count' 为实际目标列名
test_series = test_df['count'].values.reshape(-1, 1)

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_series)
test_scaled = scaler.transform(test_series)

# 创建时间序列数据
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

# 设置时间步
time_steps = 10
X_train, y_train = create_sequences(train_scaled, time_steps)
X_test, y_test = create_sequences(test_scaled, time_steps)

# Step 3: 构建 LSTM 模型
model = Sequential([
    LSTM(128, activation='relu', return_sequences=True, input_shape=(time_steps, 1)),
    LSTM(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Step 4: 训练模型
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Step 5: 测试集预测
predictions = model.predict(X_test)

# 反归一化预测值和实际值
predictions_rescaled = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test)

# Step 6: 计算误差
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
print(f"测试集的均方误差 (MSE): {mse:.4f}")
r2 = r2_score(y_test_rescaled, predictions_rescaled)
print(f"测试集的 R² 分数: {r2:.4f}")
# Step 7: 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label='实际值', color='blue')
plt.plot(predictions_rescaled, label='预测值', color='red')
plt.title('LSTM 测试集预测 vs 实际值')
plt.legend()
plt.show()
