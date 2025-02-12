import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 删除时间列
train_data = train_data.drop(columns=['利用開始日', '年度', '月', 'PortID'], errors='ignore')
test_data = test_data.drop(columns=['利用開始日', '年度', '月', 'PortID'], errors='ignore')

# 计算每个站点的历史平均利用次数
station_mean_count = train_data.groupby('利用ステーション')['count'].mean()
station_var_count = train_data.groupby('利用ステーション')['count'].var()

# 将统计特征加入训练集
train_data['利用ステーション_平均利用次数'] = train_data['利用ステーション'].map(station_mean_count)
train_data['利用ステーション_利用方差'] = train_data['利用ステーション'].map(station_var_count)

# 对测试集进行同样的处理
global_mean_count = train_data['count'].mean()  # 全局均值
global_var_count = train_data['count'].var()  # 全局方差

test_data['利用ステーション_平均利用次数'] = test_data['利用ステーション'].map(station_mean_count).fillna(
    global_mean_count)
test_data['利用ステーション_利用方差'] = test_data['利用ステーション'].map(station_var_count).fillna(global_var_count)

# 删除原始的站点列
train_data = train_data.drop(columns=['利用ステーション'])
test_data = test_data.drop(columns=['利用ステーション'])

# 定义特征和目标
target_column = 'count'
X = train_data.drop(columns=[target_column])
y = train_data[target_column]

# 确保测试集有真实值
if target_column not in test_data:
    raise ValueError(f"测试集中必须包含真实值列 '{target_column}' 以进行对比分析")

X_test = test_data.drop(columns=[target_column])
y_test = test_data[target_column]

# 独热编码和对齐测试集列
X = pd.get_dummies(X)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X.columns, fill_value=0)

# 定义样本权重
high_value_threshold = y.quantile(0.90)  # 定义高值样本的阈值
weights = np.where(y > high_value_threshold, 1 + (y / high_value_threshold), 1)

# 定义 XGBoost 参数
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1.0,
    'reg_lambda': 1.0,
}

# 模型初始化
models = {
    'XGBoost': xgb_params,
    'LightGBM': LGBMRegressor(
        max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=1.0, reg_lambda=1.0, n_estimators=200
    ),
    'CatBoost': CatBoostRegressor(
        depth=6, learning_rate=0.1, iterations=200, silent=True
    )
}

# 评估每个模型
results = {}
for model_name, model in models.items():
    print(f"Training {model_name}...")

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2_scores = []

    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if model_name == 'XGBoost':
            # XGBoost 模型训练
            dtrain = xgb.DMatrix(X_train, label=y_train, weight=weights[train_idx])
            dval = xgb.DMatrix(X_val, label=y_val)
            xgb_model = xgb.train(
                model, dtrain, num_boost_round=200, early_stopping_rounds=10,
                evals=[(dval, 'eval')], verbose_eval=False
            )
            y_val_pred = xgb_model.predict(dval)
        else:
            # LightGBM 和 CatBoost
            model.fit(X_train, y_train)
            y_val_pred = model.predict(X_val)

        # 计算 R² 分数
        r2 = r2_score(y_val, y_val_pred)
        cv_r2_scores.append(r2)

    # 测试集评估
    if model_name == 'XGBoost':
        dtest = xgb.DMatrix(X_test)
        y_test_pred = xgb_model.predict(dtest)
    else:
        y_test_pred = model.predict(X_test)

    test_r2 = r2_score(y_test, y_test_pred)
    results[model_name] = {
        'Mean CV R²': np.mean(cv_r2_scores),
        'Test R²': test_r2
    }

# 打印每个模型的结果
for model_name, scores in results.items():
    print(f"{model_name}:")
    print(f"  Mean CV R²: {scores['Mean CV R²']:.4f}")
    print(f"  Test R²: {scores['Test R²']:.4f}")
