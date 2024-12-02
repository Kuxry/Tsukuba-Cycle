import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import shap
from matplotlib import font_manager

# 手动加载 MS Gothic 字体
font_path = "MS Gothic.ttf"  # 确保路径是正确的
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'MS Gothic'

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 删除时间列
train_data = train_data.drop(columns=['利用開始日', '年度'], errors='ignore')
test_data = test_data.drop(columns=['利用開始日', '年度'], errors='ignore')

# 删除原始的站点列
train_data = train_data.drop(columns=['利用ステーション'])
test_data = test_data.drop(columns=['利用ステーション'])

# 提取测试集真实值
target_column = 'count'  # 替换为目标列名
y_test = test_data[target_column]  # 测试集真实值
X_test = test_data.drop(columns=[target_column])

# 特征和目标变量定义
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# 预处理（独热编码）
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 分割训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建DMatrix对象
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
params = {
    'objective': 'reg:squarederror',  # 回归任务
    'eval_metric': 'rmse',            # 使用RMSE作为评价指标
    'max_depth': 6,
    'eta': 0.3,
    'seed': 42
}

# 训练模型
evals = [(dtrain, 'train'), (dval, 'eval')]
model = xgb.train(params, dtrain, num_boost_round=100, evals=evals, early_stopping_rounds=10)



# 加入SHAP解释
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# 计算 SHAP 值
shap_values = explainer.shap_values(X)

# 可视化全局特征重要性
shap.summary_plot(shap_values, X, plot_type="bar")

# 可视化特征对预测的影响（汇总图）
shap.summary_plot(shap_values, X)

# 单个样本的特征解释（以测试集第一条数据为例）
# 注意这里 shap_values 是一个二维数组
sample_index = 0  # 选择样本索引
shap.waterfall_plot(
    shap.Explanation(
        values=shap_values[sample_index],
        base_values=explainer.expected_value,  # 使用模型的期望值
        data=X.iloc[sample_index].values,
        feature_names=X.columns,
    )
)

# 保存 SHAP 解释结果为文件
shap_values_df = pd.DataFrame(shap_values, columns=X.columns)
shap_values_df.to_excel("shap_feature_importance.xlsx", index=False)
print("SHAP 特征解释结果已保存至 shap_feature_importance.xlsx")


# 测试集预测
y_test_pred = model.predict(dtest)

# 计算测试集指标
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mse_test ** 0.5
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test MSE: {mse_test}")
print(f"Test RMSE: {rmse_test}")
print(f"Test MAE: {mae_test}")
print(f"Test R²: {r2_test}")

# 绘制图1：测试集真实值 vs 预测值的散点图
plt.figure()
plt.scatter(y_test, y_test_pred, alpha=0.5, label='Predicted vs Real')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted (Test Set)')
plt.legend()
plt.grid(True)
plt.show()

# 绘制图2：测试集真实值与预测值的分布对比
plt.figure(figsize=(12, 6))
plt.hist(y_test, bins=20, alpha=0.6, label='Real Values (Test Set)', color='blue', edgecolor='black')
plt.hist(y_test_pred, bins=20, alpha=0.6, label='Predicted Values (Test Set)', color='orange', edgecolor='black')
plt.title("Distribution of Real vs Predicted Values (Test Set)")
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('distribution_real_vs_predicted_test.png')
plt.show()

# 保存预测结果
output_data = pd.DataFrame({'Real': y_test, 'Predicted': y_test_pred})
output_data.to_excel('test_predictions_with_scores.xlsx', index=False)
print("测试集真实值与预测值已保存至 test_predictions_with_scores.xlsx")

# 加载数据并预处理（同前）

# 创建新的测试数据框，按天分组
test_data['曜日'] = test_data['曜日'].astype(str)  # 确保曜日是字符串类型
days_of_week = ['月曜日', '火曜日', '水曜日', '木曜日', '金曜日', '土曜日', '日曜日']
daywise_results = {}
for day in days_of_week:
    # 筛选当天的数据
    day_data = test_data[test_data['曜日'] == day]
    if day_data.empty:
        print(f"No data available for {day}.")
        continue

    # 提取特征和目标
    X_day = day_data.drop(columns=[target_column, '曜日'])  # 删除目标列和曜日列
    y_day = day_data[target_column]

    # 确保特征列对齐
    X_day = X_day.reindex(columns=X.columns, fill_value=0)

    # 创建 DMatrix 并预测
    dtest_day = xgb.DMatrix(X_day)
    y_day_pred = model.predict(dtest_day)

    # 计算 R²
    r2_day = r2_score(y_day, y_day_pred)
    daywise_results[day] = r2_day

    # 绘制真实值 vs 预测值
    plt.figure()
    plt.scatter(y_day, y_day_pred, alpha=0.5, label='Predicted vs Real')
    plt.plot([min(y_day), max(y_day)], [min(y_day), max(y_day)], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{day}: Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

# 绘制7天的 R² 图
plt.figure(figsize=(10, 6))
plt.bar(daywise_results.keys(), daywise_results.values(), color='skyblue', edgecolor='black')
plt.title("R² Score for Each Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("R² Score")
plt.ylim(0, 1)  # R² 范围
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('r2_scores_per_day.png')
plt.show()
plt.show()