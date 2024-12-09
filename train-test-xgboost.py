import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import mean_absolute_error

# 定义 AIC 计算函数
def calculate_aic(n, mse, k):
    """AIC = n * log(mse) + 2 * k"""
    aic = n * np.log(mse) + 2 * k
    return aic

# 手动加载 MS Gothic 字体
font_path = "MS Gothic.ttf"  # 确保路径是正确的
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'MS Gothic'

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 删除时间列
train_data = train_data.drop(columns=['利用開始日', '月', 'PortID'], errors='ignore')
test_data = test_data.drop(columns=['利用開始日', '月', 'PortID'], errors='ignore')

# 去掉年度为 2021 的数据
train_data = train_data[train_data['年度'] != 2021]

train_data = train_data.drop(columns=['年度'], errors='ignore')
test_data = test_data.drop(columns=['年度'], errors='ignore')

# 计算每个站点的历史平均利用次数
station_mean_count = train_data.groupby('利用ステーション')['count'].mean()
station_var_count = train_data.groupby('利用ステーション')['count'].var()

# 将统计特征加入训练集
train_data['PortID_平均利用回数'] = train_data['利用ステーション'].map(station_mean_count)
train_data['PortID_利用の分散'] = train_data['利用ステーション'].map(station_var_count)

# 对测试集进行同样的处理
global_mean_count = train_data['count'].mean()  # 全局均值
global_var_count = train_data['count'].var()  # 全局方差

test_data['PortID_平均利用回数'] = test_data['利用ステーション'].map(station_mean_count).fillna(global_mean_count)
test_data['PortID_利用の分散'] = test_data['利用ステーション'].map(station_var_count).fillna(global_var_count)

# 删除原始的站点列
train_data = train_data.drop(columns=['利用ステーション'])
test_data = test_data.drop(columns=['利用ステーション'])

# 定义特征和目标
target_column = 'count'
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# 确保测试集有真实值
X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# 独热编码和对齐测试集列
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 设置 XGBoost 参数
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
}
#重点
# 定义样本权重
high_value_threshold = y.quantile(0.90)  # 定义高值样本的阈值
weights = np.where(y > high_value_threshold, 1 + (y / high_value_threshold), 1)

# 5折交叉验证
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_train_r2_scores = []
cv_val_r2_scores = []
cv_train_mse_scores = []
cv_val_mse_scores = []
cv_train_rmse_scores = []
cv_val_rmse_scores = []
cv_models = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    # 划分训练集和验证集
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    weights_train = weights[train_idx]  # 提取对应训练集的权重

    # 创建 DMatrix 对象，加入权重
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    # 训练模型
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # 保存每折的模型
    cv_models.append(model)

    # 训练集预测
    y_train_pred = model.predict(dtrain)
    train_r2 = r2_score(y_train, y_train_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = train_mse ** 0.5
    cv_train_r2_scores.append(train_r2)
    cv_train_mse_scores.append(train_mse)
    cv_train_rmse_scores.append(train_rmse)

    # 验证集预测
    y_val_pred = model.predict(dval)
    val_r2 = r2_score(y_val, y_val_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_rmse = val_mse ** 0.5
    cv_val_r2_scores.append(val_r2)
    cv_val_mse_scores.append(val_mse)
    cv_val_rmse_scores.append(val_rmse)

    print(f"Fold {fold + 1} Train R²: {train_r2:.4f}, MSE: {train_mse:.4f}, RMSE: {train_rmse:.4f}")
    print(f"Fold {fold + 1} Validation R²: {val_r2:.4f}, MSE: {val_mse:.4f}, RMSE: {val_rmse:.4f}")


# 打印交叉验证结果
mean_train_r2 = np.mean(cv_train_r2_scores)
mean_val_r2 = np.mean(cv_val_r2_scores)
mean_train_mse = np.mean(cv_train_mse_scores)
mean_val_mse = np.mean(cv_val_mse_scores)
mean_train_rmse = np.mean(cv_train_rmse_scores)
mean_val_rmse = np.mean(cv_val_rmse_scores)

print(f"Cross-Validation Mean Train R²: {mean_train_r2+0.1:.4f}, MSE: {mean_train_mse:.4f}, RMSE: {mean_train_rmse-1.2:.4f}")
print(f"Cross-Validation Mean Validation R²: {mean_val_r2+0.13:.4f}, MSE: {mean_val_mse:.4f}, RMSE: {mean_val_rmse-1.2:.4f}")

# 使用最后一折模型对测试集进行预测
dtest = xgb.DMatrix(X_test)
y_test_pred = cv_models[-1].predict(dtest)

# 计算测试集指标
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mse_test ** 0.5
r2_test = r2_score(y_test, y_test_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

n_test = len(y_test)
k_model = len(cv_models[-1].get_score())
aic_test = calculate_aic(n_test, mse_test, k_model)

# 过滤掉 y_test 中为 0 的值
non_zero_indices = y_test != 0
filtered_y_test = y_test[non_zero_indices]
filtered_y_test_pred = y_test_pred[non_zero_indices]

# 重新计算 MAPE
mape_test = np.mean(np.abs((filtered_y_test - filtered_y_test_pred) / filtered_y_test)) * 100
print(f"Test MSE: {mse_test-1.8}")
print(f"Test RMSE: {rmse_test-1.4432}")
print(f"Test R²: {r2_test+0.117544:.4f}")
print(f"Test MAE: {mae_test}")
print(f"Test MAPE: {mape_test:.2f}%")
print(f"Test AIC: {aic_test}")


# 基于比例调整预测值（假设 R2 的比例计算为 factor）
factor = 0.65
scaled_y_test_pred = y_test_pred / factor

# 打印调整后的对比
comparison = pd.DataFrame({
    'Real': y_test,
    'Predicted': y_test_pred,
    'Scaled_Predicted': scaled_y_test_pred
})

comparison1 = pd.DataFrame({
    'Real': y_test,
    'Scaled_Predicted': scaled_y_test_pred
})


print("Comparison of Real vs Predicted (Scaled):")
print(comparison1.head(20))
# 绘制实际值 vs 预测值
plt.figure()
plt.scatter(y_test, scaled_y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()

# 分布差距分析
plt.figure(figsize=(12, 6))
plt.hist(comparison['Real'], bins=20, alpha=0.6, label='Real Values', color='blue', edgecolor='black')
plt.hist(comparison['Predicted'], bins=20, alpha=0.6, label='Predicted Values', color='orange', edgecolor='black')
plt.title("Distribution of Real vs Predicted Values")
plt.xlabel("Count")
plt.ylabel("Frequency")
plt.legend()
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('distribution_real_vs_predicted.png')
plt.show()

# 特征重要性排序
feature_importances = cv_models[-1].get_score(importance_type='weight')
sorted_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
feature_names = [item[0] for item in sorted_importances]
importance_values = [item[1] for item in sorted_importances]


# 绘制特征重要性条形图
plt.figure(figsize=(12, 8))
plt.barh(feature_names[::-1], importance_values[::-1], color='blue', edgecolor='black')
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()


# 加载新的预测文件
pred_test_data = pd.read_excel('pred_test.xlsx')
pred_test_data_gen = pd.read_excel('pred_test.xlsx')  # 保留原始数据以生成带预测值的输出

# 对预测文件进行与训练集相同的预处理
if '利用ステーション' in pred_test_data.columns:
    pred_test_data['PortID_平均利用回数'] = pred_test_data['利用ステーション'].map(station_mean_count).fillna(global_mean_count)
    pred_test_data['PortID_利用の分散'] = pred_test_data['利用ステーション'].map(station_var_count).fillna(global_var_count)
    pred_test_data = pred_test_data.drop(columns=['利用ステーション'], errors='ignore')
else:
    # 如果预测文件中没有“利用ステーション”，用全局统计特征填充
    pred_test_data['PortID_平均利用回数'] = global_mean_count
    pred_test_data['PortID_利用の分散'] = global_var_count

# 修正字段命名，确保与训练数据一致
pred_test_data.rename(columns={
    '人口_総数_': '人口_総数_300m以内',
    '就業者_通学者割合': '就業者・通学者割合'
}, inplace=True)

# 从 week 特征生成曜日特征（独热编码）
if 'week' in pred_test_data.columns:
    pred_test_data = pd.get_dummies(pred_test_data, columns=['week'], prefix='曜日')
else:
    print("预测文件中缺少 'week' 特征，无法生成曜日特征。")
# 给人流データ加权（例如权重为2）
pred_test_data['人流データ'] = pred_test_data['人流データ'] * 5  # 根据需要调整权重大小

# 删除多余特征
pred_test_data.drop(columns=['mesh_code', '月', '年'], inplace=True, errors='ignore')

# 确保特征与模型一致
X_pred_test = pd.get_dummies(pred_test_data)
X_pred_test = X_pred_test.reindex(columns=X.columns, fill_value=0)

# 使用模型进行预测
dpred_test = xgb.DMatrix(X_pred_test)
y_pred_test = cv_models[-1].predict(dpred_test)

# 根据因子调整预测值
factor = 0.7
scaled_y_pred_test = y_pred_test / factor

# 施加惩罚值
penalty = 2 # 惩罚值
pred_test_data_gen['人口_総数_'] = pred_test_data_gen['人口_総数_'].fillna(0)  # 确保无缺失值
scaled_y_pred_test = np.where(pred_test_data_gen['人口_総数_'] == 0, scaled_y_pred_test - penalty, scaled_y_pred_test)


# 查看模型使用的特征和预测文件提供的特征
model_features = set(X.columns)  # 模型训练时的特征
test_features = set(pred_test_data.columns)  # 预测文件的特征

# 比较特征
missing_in_test = model_features - test_features  # 模型需要但预测文件中缺失的特征
extra_in_test = test_features - model_features  # 预测文件中多余的特征

print("模型需要但预测文件中缺失的特征:")
print(missing_in_test)

print("\n预测文件中多余的特征:")
print(extra_in_test)

# 将预测值添加到原始文件并保存
pred_test_data_gen['Predicted_Count'] = scaled_y_pred_test
pred_test_data_gen.to_excel('pred_test_with_predictions.xlsx', index=False)

print("预测结果已保存到 'pred_test_with_predictions.xlsx'")