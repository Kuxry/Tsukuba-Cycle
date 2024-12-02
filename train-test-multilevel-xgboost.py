import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
import statsmodels.formula.api as smf
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns


# 手动加载 MS Gothic 字体
font_path = "MS Gothic.ttf"  # 确保路径是正确的
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'MS Gothic'

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 定义列名映射函数
def rename_columns(df):
    return df.rename(columns={
        '駅との距離': 'StationDistance',
        'バスとの距離': 'BusDistance',
        '人口_総数_300m以内': 'Population300m',
        '男性割合': 'MaleRatio',
        '人口15_64割合': 'Population15_64Ratio',
        '就業者・通学者割合': 'WorkerStudentRatio',
        '就業者_通学者利用交通手段_自転車割合': 'BikeUsageRatio',
        '人流データ': 'FlowData',
        '平均気温(℃)': 'AvgTemperature',
        '降水量の合計(mm)': 'TotalRainfall',
        '平均風速(m/s)': 'AvgWindSpeed',
        '日照時間(時間)': 'SunlightHours',
        'ポート数_300mBuffer': 'Port300m',
        'ポート数_500mBuffer': 'Port500m',
        'ポート数_1000mBuffer': 'Port1000m'
    }, inplace=False)

# 定义曜日映射函数
day_mapping = {
    '日曜日': 7, '月曜日': 1, '火曜日': 2, '水曜日': 3,
    '木曜日': 4, '金曜日': 5, '土曜日': 6
}

def preprocess_data(df):
    df['曜日_数字'] = df['曜日'].map(day_mapping)
    df = df.drop(columns=['曜日', '利用開始日'], errors='ignore')  # 删除无关列
    df = rename_columns(df)
    return df

# Step 1: 数据清理和预处理
data_cleaned = preprocess_data(train_data)

# Step 2: 构建多层次模型
formula = '''
count ~ 月 + StationDistance + BusDistance + Population300m + MaleRatio + 
          Population15_64Ratio + WorkerStudentRatio + BikeUsageRatio + 
          FlowData + AvgTemperature + TotalRainfall + AvgWindSpeed + 
          SunlightHours + Port300m + Port500m + Port1000m + (1|PortID)
'''

multi_level_model = smf.mixedlm(formula, data_cleaned, groups=data_cleaned['PortID']).fit()

# 提取随机效应
random_effects = multi_level_model.random_effects
if isinstance(next(iter(random_effects.values())), dict):
    data_cleaned['hierarchical_effect'] = data_cleaned['PortID'].map(
        lambda x: random_effects.get(x, {}).get('Intercept', 0))
else:
    data_cleaned['hierarchical_effect'] = data_cleaned['PortID'].map(random_effects)

data_cleaned['hierarchical_effect'] = data_cleaned['hierarchical_effect'].fillna(0).astype(float)

# Step 3: 准备训练数据
X = data_cleaned.drop(columns=['count', 'PortID', '利用ステーション'], errors='ignore')
y = data_cleaned['count']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: 训练 XGBoost 模型
xgb_model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)
xgb_model.fit(X_train, y_train)


# 测试模型性能
predictions = xgb_model.predict(X_test)
print("Training Data R² Score:", r2_score(y_test, predictions))

# Step 5: 处理测试集并预测
test_data = preprocess_data(test_data)
if isinstance(next(iter(random_effects.values())), dict):
    test_data['hierarchical_effect'] = test_data['PortID'].map(
        lambda x: random_effects.get(x, {}).get('Intercept', 0))
else:
    test_data['hierarchical_effect'] = test_data['PortID'].map(random_effects)

test_data['hierarchical_effect'] = test_data['hierarchical_effect'].fillna(0).astype(float)

X_test_final = test_data.drop(columns=['count', 'PortID', '利用ステーション'], errors='ignore')
y_test_final = test_data['count']

# 测试集预测和评估
test_predictions = xgb_model.predict(X_test_final)
test_r2 = r2_score(y_test_final, test_predictions)

# 输出测试集 R² 分数
print("Test Data R² Score:", test_r2)
