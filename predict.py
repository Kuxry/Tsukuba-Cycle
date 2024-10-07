import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 手动加载 MS Gothic 字体
font_path = "MS Gothic.ttf"  # 确保路径是正确的
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'MS Gothic'

# Step 1: 加载保存好的模型和预处理对象
loaded_model = joblib.load('best_xgb_model_trial_3.joblib')  # 加载模型
scaler = joblib.load('scaler.joblib')  # 加载保存的 scaler
label_encoder = joblib.load('label_encoder.joblib')  # 加载保存的 立地タイプ 编码器
label_encoder_day_type = joblib.load('label_encoder_day_type.joblib')  # 加载保存的 曜日 编码器

# Step 2: 生成更多的预测数据
# 随机生成所有模型所需的特征，包括新特征 "駅との距離"
bus_distance_samples = np.random.uniform(0, 3, 20)  # 生成 20 个 0 到 3 之间的随机数，作为 'バスとの距離'
station_distance_samples = np.random.uniform(0, 5, 20)  # 生成 20 个 0 到 5 之间的随机数，作为 '駅との距離'
population_samples = np.random.uniform(500, 5000, 20)  # 生成 20 个 500 到 5000 之间的随机人口数量 '人口_総数_300m以内'
male_ratio_samples = np.random.uniform(0.45, 0.55, 20)  # 生成 20 个 0.45 到 0.55 之间的随机男性比例 '男性割合'
age_15_64_ratio_samples = np.random.uniform(0.5, 0.7, 20)  # 生成 20 个 0.5 到 0.7 之间的 15-64 岁人口比例 '15_64人口割合'
commuter_ratio_samples = np.random.uniform(0.5, 0.8, 20)  # 生成 20 个 0.5 到 0.8 之间的随机值 '就業者_通学者割合'
bicycle_ratio_samples = np.random.uniform(0, 0.3, 20)  # 生成 20 个 0 到 0.3 之间的随机值 '就業者_通学者利用交通手段_自転車割合'
types_samples = np.random.choice(['駅', '商業施設', '公園'], 20)  # 随机生成 20 个 '立地タイプ' (駅, 商業施設, 公園)
day_type_samples = np.random.choice(['月曜日', '火曜日', '水曜日'], 20)  # 随机生成 20 个 '曜日' (月曜日, 火曜日, 水曜日)

# 创建 DataFrame 包含新数据
new_data = pd.DataFrame({
    'バスとの距離': bus_distance_samples,  # 随机生成的 'バスとの距離'
    '駅との距離': station_distance_samples,  # 随机生成的 '駅との距離'
    '人口_総数_300m以内': population_samples,  # 随机生成的 '人口_総数_300m以内'
    '男性割合': male_ratio_samples,  # 随机生成的 '男性割合'
    '15_64人口割合': age_15_64_ratio_samples,  # 随机生成的 '15_64人口割合'
    '就業者_通学者割合': commuter_ratio_samples,  # 随机生成的 '就業者_通学者割合'
    '就業者_通学者利用交通手段_自転車割合': bicycle_ratio_samples,  # 随机生成的 '就業者_通学者利用交通手段_自転車割合'
    '立地タイプ': types_samples,  # 随机生成的 '立地タイプ'
    '曜日': day_type_samples  # 随机生成的 '曜日'
})

# Step 3: 对新数据进行处理
# 类别编码 for 立地タイプ and 曜日
new_data['立地タイプ'] = label_encoder.transform(new_data['立地タイプ'])
new_data['曜日'] = label_encoder_day_type.transform(new_data['曜日'])

# Step 4: 使用模型进行预测
# 确保特征顺序与训练时保持一致
# 从模型中获取训练时的特征顺序
model_feature_names = loaded_model.get_booster().feature_names

# 重新排列 new_data 的列顺序以匹配模型的特征顺序
input_data = new_data[model_feature_names]

predicted_counts = loaded_model.predict(input_data)
new_data['predicted_count'] = predicted_counts  # 保存预测的 count 数量

# Step 5: 获取特征重要性
feature_importances = loaded_model.feature_importances_

# 将特征重要性保存为 DataFrame
importance_df = pd.DataFrame({
    'Feature': model_feature_names,
    'Importance': feature_importances
})

# 将结果保存到 Excel 文件中
output_file = 'generated_predictions_and_importance_with_station_distance.xlsx'
with pd.ExcelWriter(output_file) as writer:
    new_data.to_excel(writer, sheet_name='Predictions', index=False)
    importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)

# 输出保存文件的路径
print(f"预测结果和特征重要性已保存到 {output_file}")

# Step 6: 可视化特征重要性（可选）
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("重要度")
plt.title("変数重要度")
plt.show()
