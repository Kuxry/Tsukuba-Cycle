import torch
import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# 自定义 MAPE 计算函数，避免除零问题
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0  # 只计算非零值
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100

# Step 1: 读取和处理数据
data = pd.read_excel('data2.xlsx')  # 替换为你的 Excel 文件路径

# 处理缺失值，例如 '#DIV/0!'
data.replace({'#DIV/0!': 0}, inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# 类别编码 for 利用ステーション and 立地タイプ (Type of location)
label_encoder_station = LabelEncoder()
data['利用ステーション'] = label_encoder_station.fit_transform(data['利用ステーション'])

label_encoder_type = LabelEncoder()
data['立地タイプ'] = label_encoder_type.fit_transform(data['立地タイプ'])

label_encoder_day_type = LabelEncoder()
data['曜日'] = label_encoder_day_type.fit_transform(data['曜日'])

# Step 2: 加入"駅との距離"特征并进行标准化
scaler = StandardScaler()
numeric_cols = ['バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合',
                '就業者_通学者利用交通手段_自転車割合']
data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

# 保存 scaler 和 LabelEncoder 对象，供以后预测时使用
joblib.dump(scaler, 'scaler.joblib')
joblib.dump(label_encoder_station, 'label_encoder_station.joblib')
joblib.dump(label_encoder_type, 'label_encoder_type.joblib')
joblib.dump(label_encoder_day_type, 'label_encoder_day_type.joblib')

# 定义特征和目标
X = data[['バスとの距離', '駅との距離', '立地タイプ', '曜日', '人口_総数_300m以内', '男性割合', '15_64人口割合',
          '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合']].values
y = data['利用回数'].values  # 目标变量

# Step 3: 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: 定义TabNet回归模型
model = TabNetRegressor()

# Step 5: 训练模型
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric=['rmse'],
    max_epochs=500,
    patience=50,
    batch_size=1024,
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False
)

# Step 6: 评估模型
y_pred = model.predict(X_test).flatten()

# 计算评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"TabNet模型的均方误差 (MSE): {mse}")
print(f"TabNet模型的根均方误差 (RMSE): {rmse}")
print(f"TabNet模型的平均绝对误差 (MAE): {mae}")
print(f"TabNet模型的平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"TabNet模型的 R² 分数为: {r2:.2f}")

# 保存模型
model.save_model('best_tabnet_model')
