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
loaded_model = joblib.load('final_tuned_xgb_model_third_trial.joblib')  # 加载最终微调后的模型
scaler = joblib.load('scaler.joblib')  # 加载保存的 scaler
label_encoder_type = joblib.load('label_encoder_type.joblib')
label_encoder_day_type = joblib.load('label_encoder_day_type.joblib')


# Step 2: 创建预测数据集
# 示例数据创建（根据您的需求进行修改）
new_data = pd.DataFrame({
    'バスとの距離': np.random.uniform(0, 3, 20),
    '駅との距離': np.random.uniform(0, 5, 20),
    '人口_総数_300m以内': np.random.uniform(500, 5000, 20),
    '男性割合': np.random.uniform(0.45, 0.55, 20),
    '15_64人口割合': np.random.uniform(0.5, 0.7, 20),
    '就業者_通学者割合': np.random.uniform(0.5, 0.8, 20),
    '就業者_通学者利用交通手段_自転車割合': np.random.uniform(0, 0.3, 20),
    '立地タイプ': np.random.choice(['駅', '商業施設', '公園'], 20),
    '曜日': np.random.choice(['月曜日', '火曜日', '水曜日'], 20),
    '年': np.random.choice([2021, 2022], 20),
    '月': np.random.choice(list(range(1, 13)), 20),
    '日': np.random.choice(list(range(1, 29)), 20)
})

# Step 3: 对新数据进行特征处理
# Step 3: 处理类别编码，忽略未知类别
def safe_transform(encoder, values, default_class):
    known_classes = encoder.classes_
    valid_values = [value if value in known_classes else default_class for value in values]
    return encoder.transform(valid_values)


# 为未知类别指定默认值

default_type = label_encoder_type.classes_[0]
default_day_type = label_encoder_day_type.classes_[0]


# 转换类别

new_data['立地タイプ'] = safe_transform(label_encoder_type, new_data['立地タイプ'], default_type)
new_data['曜日'] = safe_transform(label_encoder_day_type, new_data['曜日'], default_day_type)


# 日期周期性特征
new_data['月_sin'] = np.sin(2 * np.pi * new_data['月'] / 12)
new_data['月_cos'] = np.cos(2 * np.pi * new_data['月'] / 12)
new_data['星期'] = new_data['日'] % 7
new_data['星期_sin'] = np.sin(2 * np.pi * new_data['星期'] / 7)
new_data['星期_cos'] = np.cos(2 * np.pi * new_data['星期'] / 7)

# 特征工程：交互特征
new_data['人口_就业交互'] = new_data['人口_総数_300m以内'] * new_data['就業者_通学者割合']
new_data['距离交互'] = new_data['バスとの距離'] * new_data['駅との距離']

# 标准化数值特征
numeric_cols = [
    'バスとの距離', '駅との距離', '人口_総数_300m以内', '男性割合', '15_64人口割合',
    '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合', '月_sin', '月_cos',
    '星期_sin', '星期_cos', '人口_就业交互', '距离交互'
]
new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])
# Step 4: 使用模型进行预测
# 确保特征顺序与模型训练时一致
input_data = new_data[[
    'バスとの距離', '駅との距離', '立地タイプ', '曜日',
    '年', '月', '日', '月_sin', '月_cos', '星期_sin', '星期_cos',
    '人口_総数_300m以内', '男性割合', '15_64人口割合', '就業者_通学者割合',
    '就業者_通学者利用交通手段_自転車割合', '人口_就业交互', '距离交互'
]]

predicted_counts = loaded_model.predict(input_data)
new_data['predicted_count'] = predicted_counts

# Step 5: 获取特征重要性
feature_importances = loaded_model.feature_importances_
importance_df = pd.DataFrame({'Feature': input_data.columns, 'Importance': feature_importances})

# Step 6: 保存预测结果和特征重要性到 Excel 文件
output_file = 'predictions_and_importance.xlsx'
with pd.ExcelWriter(output_file) as writer:
    new_data.to_excel(writer, sheet_name='Predictions', index=False)
    importance_df.to_excel(writer, sheet_name='Feature Importance', index=False)
print(f"预测结果和特征重要性已保存到 {output_file}")

# Step 7: 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel("重要度")
plt.title("変数重要度")
plt.show()
