import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Step 1: 加载数据
train_file = 'train.xlsx'  # 训练集文件
test_file = 'test.xlsx'    # 测试集文件

# 加载数据
train_df = pd.read_excel(train_file)
test_df = pd.read_excel(test_file)

# Step 2: 准备训练数据
# 假设目标变量是 'count'，时间特征为 'date'（根据你的数据调整）
# 且 PortID 不用于预测
if 'PortID' in train_df.columns:
    train_df = train_df.drop(columns=['PortID','月','年度'], errors='ignore')
    test_df = test_df.drop(columns=['PortID','月','年度'], errors='ignore')

# 确保时间列被正确解析
train_df['利用開始日'] = pd.to_datetime(train_df['利用開始日'])  # 转换为日期时间格式
test_df['利用開始日'] = pd.to_datetime(test_df['利用開始日'])

# 按时间排序
train_df = train_df.sort_values(by='利用開始日')
test_df = test_df.sort_values(by='利用開始日')
# 提取时间序列目标变量
train_series = train_df['count']  # 替换 'count' 为实际目标列名
test_series = test_df['count']    # 测试集目标列（真实值，用于验证）



from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 自动选择 ARIMA 参数
auto_model = auto_arima(
    train_series,
    seasonal=True,  # 如果有显著季节性可以设为 True
    stepwise=True,
    trace=True
)

# 使用自动选择的参数进行拟合
model_fit = auto_model.fit(train_series)

# 训练集预测
train_predictions = model_fit.predict_in_sample()

# 测试集预测
forecast_steps = len(test_series)
forecast = model_fit.predict(n_periods=forecast_steps)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(train_series, label='训练集实际值')
plt.plot(train_predictions, label='训练集预测值', color='orange')
plt.plot(test_series.values, label='测试集实际值', color='blue')
plt.plot(forecast, label='测试集预测值', color='red')
plt.legend()
plt.title("ARIMA 自动优化预测")
plt.show()

# 计算测试集误差
mse = mean_squared_error(test_series, forecast)
print(f"测试集的均方误差 (MSE): {mse:.4f}")
