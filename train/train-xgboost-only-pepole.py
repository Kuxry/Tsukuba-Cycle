import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import logging
from matplotlib import font_manager

# 手动加载 MS Gothic 字体
font_path = "../MS Gothic.ttf"  # 确保路径是正确的
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'MS Gothic'
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# 自定义 MAPE 函数
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_indices = y_true != 0
    return np.mean(np.abs((y_true[non_zero_indices] - y_pred[non_zero_indices]) / y_true[non_zero_indices])) * 100


# 数据预处理函数
def preprocess_data(data_path):
    data = pd.read_excel(data_path)
    data.replace({'#DIV/0!': 0}, inplace=True)

    # 保留指定的三个特征
    selected_features = ['平均気温', '降水量の合計（mm）', '人流__1時間平均']
    missing_cols = [col for col in selected_features if col not in data.columns]
    if missing_cols:
        raise ValueError(f"以下必要特征未找到: {missing_cols}")

    # 删除包含 0 的行
    data = data[(data[selected_features] != 0).all(axis=1)]
    logging.info(f"数据清洗完成，共删除了包含 0 值的行，剩余 {data.shape[0]} 条数据")

    scaler = StandardScaler()
    data[selected_features] = scaler.fit_transform(data[selected_features])

    joblib.dump(scaler, '../scaler.joblib')

    X = data[selected_features]
    y = data['利用回数']
    return X, y


# 5 折交叉验证函数
def cross_validate_model(X, y, num_folds=5):
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_results = []

    space = {
        'n_estimators': hp.choice('n_estimators', [100, 200, 300, 400]),
        'max_depth': hp.quniform('max_depth', 3, 15, 1),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'subsample': hp.uniform('subsample', 0.6, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1),
        'gamma': hp.uniform('gamma', 0, 0.5)
    }

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        logging.info(f"正在训练第 {fold + 1}/{num_folds} 折...")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        def objective(params):
            params['max_depth'] = int(params['max_depth'])
            model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='hist', device='cuda', **params)
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            return {'loss': mse, 'status': STATUS_OK}

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

        # 计算评估指标
        fold_results.append({
            'fold': fold + 1,
            'mse': trials.best_trial['result']['loss'],
            'best_params': best_params
        })

    return pd.DataFrame(fold_results)


# 使用最佳参数重新训练最终模型，并绘制特征重要性图
def train_final_model(X, y, best_params):
    logging.info("使用最佳参数重新训练最终模型...")
    final_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        device='cuda',
        **best_params
    )
    final_model.fit(X, y)
    joblib.dump(final_model, 'final_xgb_model.joblib')
    logging.info("最终模型已保存为 final_xgb_model.joblib")

    # 绘制特征重要性图
    importance = final_model.get_booster().get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': [importance.get(f'f{i}', 0) for i in range(len(X.columns))]
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], align='center')
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance from XGBoost Model')
    plt.gca().invert_yaxis()
    plt.show()

    return final_model


# 调用主要流程
if __name__ == "__main__":
    X, y = preprocess_data('../data4.xlsx')

    # 进行 5 折交叉验证
    fold_results = cross_validate_model(X, y, num_folds=5)
    logging.info("5 折交叉验证完成！")

    # 选取最佳参数（基于交叉验证结果的平均性能）
    best_params = fold_results.loc[fold_results['mse'].idxmin(), 'best_params']
    logging.info(f"选取的最佳参数为：{best_params}")

    # 使用最佳参数训练最终模型并绘制特征重要性图
    final_model = train_final_model(X, y, best_params)
