import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# 读取 CSV 文件
file_path = 'Minlost.csv'  # 请确保文件路径正确
df = pd.read_csv(file_path)

# 按站点 (PortID) 分组
grouped = df.groupby("PortID")

# 初始化字典存储拟合参数和 R² 值
fit_results = {}

# 定义二次函数
def quadratic_function(x, a, b, c):
    return a * x**2 + b * x + c

# 定义 R² 计算函数
def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)  # 残差平方和
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # 总体平方和
    return 1 - (ss_res / ss_tot)

# 绘图准备
plt.figure(figsize=(15, 10))

# 遍历每个站点进行拟合
r2_values = []  # 存储每个站点的 R² 值
for idx, (port_id, group) in enumerate(grouped):
    # 按 Available bikes 排序
    group = group.sort_values("Available bikes")

    # 提取 x (Available bikes) 和 y (機会損失)
    x = group["Available bikes"].values
    y = group["機会損失"].values

    # 拟合二次函数
    try:
        params, _ = curve_fit(quadratic_function, x, y)
        fit_results[port_id] = params  # 存储拟合参数
        a, b, c = params

        # 生成预测值和拟合曲线
        y_pred = quadratic_function(x, a, b, c)  # 用拟合模型计算的预测值
        x_fit = np.linspace(min(x), max(x), 100)  # 用于绘制平滑拟合曲线
        y_fit = quadratic_function(x_fit, a, b, c)

        # 计算 R² 值
        r2 = calculate_r2(y, y_pred)
        r2_values.append((port_id, r2))

        # 绘制原始数据点和拟合曲线
        plt.subplot(6, 8, idx + 1)
        plt.scatter(x, y, label=f"Data (PortID {port_id})", color='red')
        plt.plot(x_fit, y_fit, label=f"Fit (R²={r2:.2f})", color='blue')
        plt.title(f"PortID {port_id}")
        plt.xlabel("Available Bikes")
        plt.ylabel("Opportunity Loss")
        plt.legend(fontsize=8)

    except Exception as e:
        print(f"Failed to fit PortID {port_id}: {e}")

# 调整布局并显示绘图
plt.tight_layout()
plt.show()

# 将拟合结果和 R² 值保存为 DataFrame
fit_results_df = pd.DataFrame.from_dict(fit_results, orient='index', columns=["a", "b", "c"])
fit_results_df.index.name = "PortID"
fit_results_df.reset_index(inplace=True)

r2_df = pd.DataFrame(r2_values, columns=["PortID", "R2"])
final_results_df = pd.merge(fit_results_df, r2_df, on="PortID")

# 添加计算最优空车数量的函数
# 添加计算最优空车数量的函数
def calculate_optimal_bikes(row):
    x_min = 0  # 假设 Available bikes 的最小值
    x_max = 83  # 假设 Available bikes 的最大值（需要根据实际数据调整）

    # 计算顶点
    x_vertex = -row["b"] / (2 * row["a"])

    # 如果开口向上，顶点是最小值
    if row["a"] > 0:
        return x_vertex

    # 如果开口向下，比较区间边界和顶点值的最小值
    else:
        f_x_min = row["a"] * x_min**2 + row["b"] * x_min + row["c"]
        f_x_max = row["a"] * x_max**2 + row["b"] * x_max + row["c"]
        f_x_vertex = row["a"] * x_vertex**2 + row["b"] * x_vertex + row["c"]

        # 检查顶点是否在区间内
        if x_min <= x_vertex <= x_max:
            return x_vertex if f_x_vertex <= min(f_x_min, f_x_max) else (x_min if f_x_min < f_x_max else x_max)
        else:
            return x_min if f_x_min < f_x_max else x_max

# 计算最优空车数量并添加到结果 DataFrame
final_results_df["Optimal Bikes"] = final_results_df.apply(calculate_optimal_bikes, axis=1)
# 输出最终表格
print(final_results_df)

# 保存为 CSV 文件（可选）
final_results_df.to_csv("Final_Results_With_Optimal_Bikes.csv", index=False)