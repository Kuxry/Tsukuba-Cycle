import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from matplotlib import font_manager

# 手动加载 MS Gothic 字体
font_path = "../MS Gothic.ttf"  # 确保路径是正确的
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'MS Gothic'
# 自定义 MAPE 计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# Step 1: 读取和处理数据
data = pd.read_excel('data4.xlsx')
data.replace({'#DIV/0!': 0}, inplace=True)

# 将 '利用開始日' 解析为日期格式
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

# 提取日期相关特征：年、月、日、星期
data['年'] = data['date'].dt.year
data['月'] = data['date'].dt.month
data['日'] = data['date'].dt.day
data['星期'] = data['date'].dt.weekday  # 星期一为0，星期日为6

# 创建周期性特征
data['月_sin'] = np.sin(2 * np.pi * data['月'] / 12)
data['月_cos'] = np.cos(2 * np.pi * data['月'] / 12)
data['星期_sin'] = np.sin(2 * np.pi * data['星期'] / 7)
data['星期_cos'] = np.cos(2 * np.pi * data['星期'] / 7)

# 其他类别编码
label_encoder_type = LabelEncoder()
label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# 特征工程：创建交互特征
data['人口_就业交互'] = data['人口_総数_300m以内'] * data['就業者_通学者割合']
data['距离交互'] = data['バスとの距離'] * data['駅との距離']



# 添加新的数值特征
new_numeric_cols = ['ポート数_300mBuffer', 'ポート数_500mBuffer', 'ポート数_1000mBuffer',
                   '平均気温', '降水量の合計（mm）', '人流__1時間平均' ]
missing_cols = [col for col in new_numeric_cols if col not in data.columns]
if missing_cols:
    raise ValueError(f"以下新增的数值特征在数据中未找到: {missing_cols}")



# 数值特征标准化
scaler = StandardScaler()

numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '@15_64人口割合', '就業者_通学者割合',
                '就業者_通学者利用交通手段_自転車割合', '月_sin', '月_cos', '星期_sin', '星期_cos',
                '人口_就业交互', '距离交互']
numeric_cols.extend(new_numeric_cols)

data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 保存 scaler 和 LabelEncoder 对象
joblib.dump(scaler, '../scaler.joblib')
joblib.dump(label_encoder_type, 'label_encoder_type.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離',  '曜日',
          '年', '月', '日', '月_sin', '月_cos', '星期_sin', '星期_cos',
          '人口_総数_300m以内', '男性割合', '@15_64人口割合', '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合',
          '人口_就业交互', '距离交互'] + new_numeric_cols]
y = data['利用回数']

# 定义贝叶斯优化的搜索空间
space = {
    'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
    'subsample': hp.uniform('subsample', 0.6, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
    'gamma': hp.uniform('gamma', 0, 0.5)
}

# 定义目标函数
def objective(params):
    params['max_depth'] = int(params['max_depth'])
    model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda', **params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return {'loss': mse, 'status': STATUS_OK}

# 使用贝叶斯优化和微调
num_trials = 3
best_params_list = []

for i in range(num_trials):
    print(f"第 {i+1} 次模型训练")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100, trials=trials)
    best_params = {
        'n_estimators': [100, 200, 300, 400][best['n_estimators']],
        'max_depth': int(best['max_depth']),
        'learning_rate': best['learning_rate'],
        'subsample': best['subsample'],
        'colsample_bytree': best['colsample_bytree'],
        'gamma': best['gamma']
    }
    best_params_list.append(best_params)

    best_model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda', **best_params)
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)

    # 添加特征重要性可视化
    print(f"第 {i+1} 次模型特征重要性可视化")
    # 获取特征的重要性分数
    importance = best_model.get_booster().get_score(importance_type='gain')

    # 过滤掉不想显示的特征
    filtered_importance = {k: v for k, v in importance.items() if k not in ['月_sin', '月_cos', '星期_sin', '星期_cos']}

    # 创建过滤后的图表
    xgb.plot_importance(filtered_importance, importance_type='gain')
    plt.show()

    print(f"第 {i+1} 次模型已保存")
    # 可以选择保存每次训练的模型，例如:
    # joblib.dump(best_model, f'best_model_trial_{i+1}.joblib')

# 使用第三次的最佳参数进行微调
third_best_params = best_params_list[2]  # 选择第三次的最佳参数
third_best_params['learning_rate'] *= 0.1  # 将学习率降低一半
third_best_params['n_estimators'] = 50000  # 增加迭代次数，进一步学习

print(f"使用第三次最佳参数 {third_best_params} 进行微调")
tuned_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    tree_method='hist',
    device='cuda',
    early_stopping_rounds=500,
    **third_best_params
)
evals = [(X_test, y_test)]  # 评估集
tuned_model.fit(X_train, y_train, eval_set=evals, verbose=True)
tuned_y_pred = tuned_model.predict(X_test)
tuned_mse = mean_squared_error(y_test, tuned_y_pred)
tuned_rmse = np.sqrt(tuned_mse)
tuned_mae = mean_absolute_error(y_test, tuned_y_pred)
tuned_mape = mean_absolute_percentage_error(y_test, tuned_y_pred)
tuned_r2 = r2_score(y_test, tuned_y_pred)

print(f"MSE: {tuned_mse}")
print(f"RMSE: {tuned_rmse}")
print(f"MAE: {tuned_mae}")
print(f"MAPE: {tuned_mape:.2f}%")
print(f"R²: {tuned_r2:.2f}")
# 绘制实际值与预测值的对比
plt.figure()
plt.scatter(y_test, tuned_y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.show()

# 获取特征重要度
importance = tuned_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importance
})

# 只保留指定的特征
selected_features = [
    'ポート数_300mBuffer', 'ポート数_500mBuffer', 'ポート数_1000mBuffer',
    '平均気温', '降水量の合計（mm）', '人流__1時間平均',
    '人口_総数_300m以内', '男性割合', '@15_64人口割合'

]
filtered_importance = feature_importance[feature_importance['Feature'].isin(selected_features)].sort_values(by='Importance', ascending=False)

# 绘制条形图
plt.figure(figsize=(10, 8))
plt.barh(filtered_importance['Feature'], filtered_importance['Importance'], align='center')
plt.xlabel('Feature Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from XGBoost Model (Selected Features)')
plt.gca().invert_yaxis()  # 倒序显示重要度
plt.show()


# 保存最终微调后的模型
joblib.dump(tuned_model, 'final_tuned_xgb_model_third_trial.joblib')
print("微调后的模型已保存为 final_tuned_xgb_model_third_trial.joblib")
