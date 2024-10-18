import joblib
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 自定义 MAPE 计算函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# Step 1: 读取和处理数据
data = pd.read_excel('data2.xlsx')
data.replace({'#DIV/0!': 0}, inplace=True)

# 将 '利用開始日' 解析为日期格式
data['利用開始日'] = pd.to_datetime(data['利用開始日'], format='%Y-%m-%d')

# 提取日期相关特征：年、月、日、星期
data['年'] = data['利用開始日'].dt.year
data['月'] = data['利用開始日'].dt.month
data['日'] = data['利用開始日'].dt.day
data['星期'] = data['利用開始日'].dt.weekday  # 星期一为0，星期日为6

# 创建周期性特征
data['月_sin'] = np.sin(2 * np.pi * data['月'] / 12)
data['月_cos'] = np.cos(2 * np.pi * data['月'] / 12)
data['星期_sin'] = np.sin(2 * np.pi * data['星期'] / 7)
data['星期_cos'] = np.cos(2 * np.pi * data['星期'] / 7)

# 类别编码：将 'PortID' 作为类别特征处理
label_encoder_portid = LabelEncoder()
data['PortID'] = label_encoder_portid.fit_transform(data['PortID'])

# 对 '立地タイプ' 和 '曜日' 进行编码
label_encoder_type = LabelEncoder()
data['立地タイプ'] = label_encoder_type.fit_transform(data['立地タイプ'])

label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# 特征工程：创建交互特征
data['人口_就业交互'] = data['人口_総数_300m以内'] * data['就業者_通学者割合']
data['距离交互'] = data['バスとの距離'] * data['駅との距離']

# 数值特征标准化
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合',
                '就業者_通学者利用交通手段_自転車割合', '月_sin', '月_cos', '星期_sin', '星期_cos',
                '人口_就业交互', '距离交互']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 保存 scaler 和 LabelEncoder 对象
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_portid, 'label_encoder_portid.joblib')
joblib.dump(label_encoder_type, 'label_encoder_type.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', 'PortID',
          '年', '月', '日', '月_sin', '月_cos', '星期_sin', '星期_cos',
          '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合',
          '人口_就业交互', '距离交互']]
y = data['利用回数']

# 确保 X 是浮点型 NumPy 数组格式
X = X.astype(float)

# Step 2: 定义泊松回归的对数似然函数
def log_likelihood(beta, X, y, alpha=0.1):
    lambda_n = np.exp(np.clip(np.dot(X, beta), -100, 100))
    ll = np.sum(y * np.log(lambda_n) - lambda_n - gammaln(y + 1))
    # 添加正则化项以控制参数大小
    regularization = alpha * np.sum(beta ** 2)
    return -ll + regularization


# 初始化 beta 参数
beta_init = np.zeros(X.shape[1])

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用最小化负对数似然进行参数优化
result = minimize(log_likelihood, beta_init, args=(X_train, y_train), method='L-BFGS-B')
beta_estimated = result.x

# 预测函数
def predict_rental_count(X, beta):
    # 计算预测值并防止 NaN 和 Inf
    prediction = np.exp(np.clip(np.dot(X, beta), -100, 100))
    prediction[np.isnan(prediction)] = 0  # 将 NaN 替换为0，或你可以选择替换为其他值
    prediction[np.isinf(prediction)] = 1e10  # 将无穷大值替换为一个合理的数值上限
    return prediction

# 计算模型的对数似然值
log_likelihood_model = -log_likelihood(beta_estimated, X_test, y_test)

# 计算空模型的对数似然值
y_mean = np.mean(y_train)
log_likelihood_null = np.sum(y_test * np.log(y_mean) - y_mean - gammaln(y_test + 1))

# 计算泊松伪 R²
poisson_r_squared = 1 - (log_likelihood_model / log_likelihood_null)

# 打印泊松伪 R²
print(f"泊松伪 R²: {poisson_r_squared:.4f}")

# 进行预测和评估
y_pred = predict_rental_count(X_test, beta_estimated)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("泊松回归模型评估：")
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"MAPE: {mape:.2f}%")