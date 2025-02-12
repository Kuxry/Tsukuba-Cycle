import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# 手动加载 MS Gothic 字体
font_path = "MS Gothic.ttf"  # 确保路径是正确的
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'MS Gothic'

# 定义曜日映射函数
day_mapping = {
    '日曜日': 7, '月曜日': 1, '火曜日': 2, '水曜日': 3,
    '木曜日': 4, '金曜日': 5, '土曜日': 6
}

# Step 1: 加载数据
file_path = 'data/train.xlsx'  # 替换为你的文件路径
data = pd.ExcelFile(file_path)
df = data.parse('Sheet1')  # 假设数据在第一个工作表中

# 将曜日映射为数值
if '曜日' in df.columns:
    df['曜日'] = df['曜日'].map(day_mapping)

# Step 2: 提取数值型数据并删除目标变量
numeric_data = df.select_dtypes(include=['number']).drop(columns=['count'], errors='ignore')

# Step 3: 标准化数据
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Step 4: 执行 PCA
pca = PCA()
pca_result = pca.fit_transform(scaled_data)

# Step 5: 提取主成分的特征载荷矩阵
feature_names = numeric_data.columns
components_df = pd.DataFrame(pca.components_, columns=feature_names, index=[f"PC{i+1}" for i in range(pca.components_.shape[0])])

# Step 6: 计算特征贡献度（有符号）
feature_contribution = components_df.sum(axis=0)
sorted_features_signed = feature_contribution.sort_values(ascending=False)

# 绘制有符号贡献度条形图
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_features_signed.index, y=sorted_features_signed.values, palette="coolwarm")
plt.title("特征总贡献排名（有符号）")
plt.xlabel("特征")
plt.ylabel("贡献值（带符号）")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Step 7: 计算特征绝对值贡献度
absolute_contribution = components_df.abs().sum(axis=0)
sorted_features_absolute = absolute_contribution.sort_values(ascending=False)

# 绘制绝对值贡献度条形图
plt.figure(figsize=(12, 6))
sns.barplot(x=sorted_features_absolute.index, y=sorted_features_absolute.values, palette="viridis")
plt.title("特征绝对值贡献排名")
plt.xlabel("特征")
plt.ylabel("绝对贡献值")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# Step 8: 验证不同特征组合的效果
X_positive = numeric_data[sorted_features_signed[sorted_features_signed > 0].index]  # 正贡献特征
X_negative = numeric_data[sorted_features_signed[sorted_features_signed < 0].index]  # 负贡献特征
X_all = numeric_data[absolute_contribution.index]  # 全部特征
y = df['count']

# 定义 XGBoost 模型
model = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)

# 验证不同特征组合的效果
print("\n验证不同特征组合的效果：")
positive_score = cross_val_score(model, X_positive, y, scoring='r2', cv=5).mean()
negative_score = cross_val_score(model, X_negative, y, scoring='r2', cv=5).mean()
all_features_score = cross_val_score(model, X_all, y, scoring='r2', cv=5).mean()

print(f"仅正贡献特征：R² = {positive_score:.4f}")
print(f"仅负贡献特征：R² = {negative_score:.4f}")
print(f"正负贡献特征混用：R² = {all_features_score:.4f}")
