import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.regression.mixed_linear_model import MixedLM
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
train_data = pd.read_excel('train.xlsx')
test_data = pd.read_excel('test.xlsx')

# 删除时间列
train_data = train_data.drop(columns=['利用開始日', '年度', '月', 'PortID'], errors='ignore')
test_data = test_data.drop(columns=['利用開始日', '年度', '月', 'PortID'], errors='ignore')

# 添加层级特征：StationID
train_data['StationID'] = train_data['利用ステーション'].astype('category').cat.codes
test_data['StationID'] = test_data['利用ステーション'].astype('category').cat.codes

# 定义多层次特征
hierarchical_features = [
    'ポート数_300mBuffer', 'ポート数_500mBuffer', 'ポート数_1000mBuffer',
    'バスとの距離', '駅との距離', '人口_総数_300m以内'
]

# 确保测试集中的特征与训练集一致
for feature in hierarchical_features:
    if feature not in test_data.columns:
        raise ValueError(f"Feature {feature} not found in test data.")

# 设置 5 折交叉验证
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
    print(f"Fold {fold + 1}")

    # 分割数据
    train_fold = train_data.iloc[train_idx]
    val_fold = train_data.iloc[val_idx]

    # 构建多层次模型
    formula = f"count ~ {' + '.join(hierarchical_features)}"
    md = MixedLM.from_formula(formula, groups=train_fold["StationID"], data=train_fold)
    mixed_model = md.fit()

    # 提取随机效应并加入到训练集和验证集
    random_effects = mixed_model.random_effects

    for feature in hierarchical_features:
        train_fold[f"{feature}_RandomEffect"] = train_fold['StationID'].map(
            lambda x: random_effects[x][0] if x in random_effects else 0
        )
        val_fold[f"{feature}_RandomEffect"] = val_fold['StationID'].map(
            lambda x: random_effects.get(x, {0: 0})[0]
        )

    # 定义特征和目标
    X_train = train_fold.drop(columns=['count', '利用ステーション', 'StationID'])
    y_train = train_fold['count']
    X_val = val_fold.drop(columns=['count', '利用ステーション', 'StationID'])
    y_val = val_fold['count']

    # 独热编码和对齐验证集列
    X_train = pd.get_dummies(X_train)
    X_val = pd.get_dummies(X_val)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    # 创建 DMatrix 对象
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

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

    # 训练 XGBoost 模型
    evals = [(dtrain, 'train'), (dval, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=False
    )

    # 验证集预测
    y_val_pred = model.predict(dval)

    # 计算验证集性能指标
    mse_val = mean_squared_error(y_val, y_val_pred)
    rmse_val = mse_val ** 0.5
    mae_val = mean_absolute_error(y_val, y_val_pred)
    r2_val = r2_score(y_val, y_val_pred)

    print(f"Fold {fold + 1} RMSE: {rmse_val}, MAE: {mae_val}, R²: {r2_val}")
    cv_results.append({
        'fold': fold + 1,
        'rmse': rmse_val,
        'mae': mae_val,
        'r2': r2_val
    })

# 交叉验证结果统计
cv_rmse = np.mean([result['rmse'] for result in cv_results])
cv_mae = np.mean([result['mae'] for result in cv_results])
cv_r2 = np.mean([result['r2'] for result in cv_results])

print(f"Cross-Validation Mean RMSE: {cv_rmse}")
print(f"Cross-Validation Mean MAE: {cv_mae}")
print(f"Cross-Validation Mean R²: {cv_r2}")

# 使用最后一折模型对测试集进行预测
test_data = test_data.copy()
for feature in hierarchical_features:
    test_data[f"{feature}_RandomEffect"] = test_data['StationID'].map(
        lambda x: random_effects.get(x, {0: 0})[0]
    )

X_test = test_data.drop(columns=['count', '利用ステーション', 'StationID'])
y_test = test_data['count']

X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

dtest = xgb.DMatrix(X_test)
y_test_pred = model.predict(dtest)

# 测试集性能指标
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = mse_test ** 0.5
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Test MSE: {mse_test}")
print(f"Test RMSE: {rmse_test}")
print(f"Test MAE: {mae_test}")
print(f"Test R²: {r2_test}")
