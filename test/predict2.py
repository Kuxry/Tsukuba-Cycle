import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from math import radians, sin, cos, sqrt, atan2

# Step 1: 加载保存好的模型和预处理对象
loaded_model = joblib.load('best_xgb_model.joblib')  # 加载模型
scaler = joblib.load('../scaler.joblib')  # 加载保存的 scaler
label_encoder = joblib.load('label_encoder.joblib')  # 加载保存的 タイプ 编码器
label_encoder_day_type = joblib.load('label_encoder_day_type.joblib')  # 加载保存的 day_type 编码器

# 定义地球半径（千米）
R = 6371.0

# 使用 haversine 公式计算两点之间的距离
def haversine(lon1, lat1, lon2, lat2):
    # 将经纬度转换为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # 计算经纬度差
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    # 计算公式中的 a 和 c
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    # 计算距离
    distance = R * c
    return distance

# Step 2: 生成更多的预测数据
# 假设这些数据表示站点的经纬度和类型
longitude_samples = np.random.uniform(139.6, 140.0, 50)  # 生成 50 个在 139.6 到 140.0 之间的随机经度
latitude_samples = np.random.uniform(35.6, 36.0, 50)  # 生成 50 个在 35.6 到 36.0 之间的随机纬度
types_samples = np.random.choice(['駅', '商業施設', '公園'], 50)  # 随机生成 50 个 'タイプ' (駅, 商業施設, 公園)

# 创建 DataFrame 包含新数据
new_data = pd.DataFrame({
    'Longitude': longitude_samples,  # 经度
    'Latitude': latitude_samples,  # 纬度
    'タイプ': types_samples,
    'day_type': ['月曜日'] * 50  # 给所有样本添加默认的 day_type（这里全部设为 '月曜日'）
})

# 设定一个参考点（例如城市中心的经纬度）
reference_point = (139.7, 35.7)  # 假设这是参考点（例如东京的某个地点）

# 计算每个站点到参考点的距离，并作为 Total_Length
new_data['Total_Length'] = new_data.apply(
    lambda row: haversine(reference_point[0], reference_point[1], row['Longitude'], row['Latitude']),
    axis=1
)

# Step 3: 对新数据进行处理
new_data['タイプ'] = label_encoder.transform(new_data['タイプ'])
new_data['day_type'] = label_encoder_day_type.transform(new_data['day_type'])

# 只保留模型需要的特征
input_data = new_data[['Total_Length', 'タイプ', 'day_type']]

# 使用加载的 scaler 对 'Total_Length' 进行标准化
input_data['Total_Length'] = scaler.transform(input_data[['Total_Length']])

# Step 4: 使用模型进行预测
predicted_counts = loaded_model.predict(input_data)
new_data['predicted_count'] = predicted_counts  # 保存预测的 count 数量

# Step 5: 聚类分析
# 使用经纬度和预测的 count 聚类
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(new_data[['Longitude', 'Latitude']])  # 使用经纬度进行聚类

# 获取聚类标签
clusters = kmeans.labels_
new_data['Cluster'] = clusters

# Step 6: 可视化结果
# 绘制聚类图：横轴为经度，纵轴为纬度，点的大小代表 count 数量，颜色代表密集度
plt.figure(figsize=(10, 6))
plt.scatter(
    new_data['Longitude'],
    new_data['Latitude'],
    c=new_data['predicted_count'],  # 颜色表示 count 的数量
    cmap='viridis',
    s=new_data['predicted_count'] * 10,  # 点的大小表示 count 数量
    alpha=0.6  # 设置点的透明度，便于看清密集区域
)
plt.title('Predicted Counts and Clusters on Map')
plt.xlabel('Longitude (经度)')
plt.ylabel('Latitude (纬度)')
plt.colorbar(label='Predicted Count')
plt.grid(True)
plt.show()

# 绘制聚类图，展示不同聚类的分布
plt.figure(figsize=(10, 6))
plt.scatter(
    new_data['Longitude'],
    new_data['Latitude'],
    c=new_data['Cluster'],  # 颜色表示聚类结果
    cmap='viridis',
    s=new_data['predicted_count'] * 10,  # 点的大小表示 count 数量
    alpha=0.6
)
plt.title('Clustered Locations on Map')
plt.xlabel('Longitude (经度)')
plt.ylabel('Latitude (纬度)')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
