import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 读取 CSV 文件 (损失和可用自行车数据)
file_path = 'Minlost.csv'  # 请确保文件路径正确
df = pd.read_csv(file_path)

# 读取容量数据
capacity_file_path = 'ポートの容量.csv'  # 容量文件路径
port_capacity_df = pd.read_csv(capacity_file_path)

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

# 将容量数据合并到最终结果
final_results_df = pd.merge(final_results_df, port_capacity_df, on="PortID", how="left")

# 定义全局优化函数，满足总车数限制
def global_objective(bike_distribution):
    total_loss = 0
    for i, bikes in enumerate(bike_distribution):
        row = final_results_df.iloc[i]
        a, b, c = row["a"], row["b"], row["c"]
        total_loss += a * bikes**2 + b * bikes + c
    return total_loss



# 定义变量边界：每个站点车数范围 [1, 容量] （假设下界为1）
bounds = [(1, row["容量"]) for _, row in final_results_df.iterrows()]

# 初始猜测值：平分总车数
initial_guess = [83 / len(final_results_df)] * len(final_results_df)

# 执行优化
result = minimize(
    global_objective,
    initial_guess,
    bounds=bounds,

    method="SLSQP",
)

# 对连续解进行取整
optimized_bikes = np.round(result.x)
total_bikes = np.sum(optimized_bikes)

final_results_df["Optimized Bikes"] = optimized_bikes

# 计算当前机会损失值，并确保不为负
final_results_df["Current Opportunity Loss"] = final_results_df.apply(
    lambda row: max(0, row["a"] * row["Optimized Bikes"]**2 + row["b"] * row["Optimized Bikes"] + row["c"]),
    axis=1,
)
# 对当前所有损失求和
total_loss = final_results_df["Current Opportunity Loss"].sum()
print("Total Current Opportunity Loss:", total_loss)

# 输出最终结果表格
print(final_results_df)

# 保存为 CSV 文件（可选）
final_results_df.to_csv("Final_Results_With_Current_Opportunity_Loss.csv", index=False)
