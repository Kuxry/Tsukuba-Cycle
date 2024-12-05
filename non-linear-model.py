import numpy as np
from scipy.optimize import curve_fit

# 定义二次模型
def quad_loss(x, a, b, c):
    return a * x**2 + b * x + c

# 站点的历史数据示例
bikes_history = np.array([0, 1, 2, 3, 4, 5])  # 自行车数量
loss_history = np.array([10, 8, 6, 5, 5.5, 6])  # 对应机会损失

# 拟合二次曲线
params, _ = curve_fit(quad_loss, bikes_history, loss_history)

# 输出拟合参数 a, b, c
a, b, c = params
print(f"拟合结果: a = {a}, b = {b}, c = {c}")
