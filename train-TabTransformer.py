import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tab_transformer_pytorch import TabTransformer
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import shap
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
import numpy as np

# 1. 定义设备（确保在所有函数之前定义）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 1. 加载数据
data = pd.read_excel('data3.xlsx')  # 替换为您的文件路径

# 2. 定义特征和目标
target = '利用回数'
features = [
    '年度', '月', '曜日', 'バスとの距離', '駅との距離',
    '人口_総数_300m以内', '男性割合', '15_64人口割合',
    '就業者_通学者割合', '就業者_通学者利用交通手段_自転車割合',
    'ポート数_300mBuffer', 'ポート数_500mBuffer', 'ポート数_1000mBuffer',
    '平均気温', '降水量の合計（mm）'
]

X = data[features].copy()  # 使用 .copy() 创建独立副本
y = data[target].copy()

categorical_features = ['年度', '月', '曜日']
numerical_features = [feat for feat in features if feat not in categorical_features]

# 处理分类特征
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le  # 保存编码器以便后续使用

# 划分数据为训练集、验证集和测试集
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2
)

# 标准化数值特征
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_val[numerical_features] = scaler.transform(X_val[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])


# 3. 定义 Dataset 和 DataLoader
class TabularDataset(Dataset):
    def __init__(self, X, y, categorical_features, numerical_features):
        self.X_categ = X[categorical_features].values.astype(int)
        self.X_cont = X[numerical_features].values.astype(float)
        self.y = y.values.astype(float)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'categorical': torch.tensor(self.X_categ[idx], dtype=torch.long),
            'continuous': torch.tensor(self.X_cont[idx], dtype=torch.float),
            'target': torch.tensor(self.y[idx], dtype=torch.float)
        }


train_dataset = TabularDataset(X_train, y_train, categorical_features, numerical_features)
val_dataset = TabularDataset(X_val, y_val, categorical_features, numerical_features)
test_dataset = TabularDataset(X_test, y_test, categorical_features, numerical_features)

batch_size = 1024
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# 4. 定义模型的初始化函数
def create_model(trial):
    # 定义超参数搜索空间
    dim = trial.suggest_int('dim', 16, 64)
    depth = trial.suggest_int('depth', 4, 12)
    heads = trial.suggest_int('heads', 4, 16)
    mlp_hidden_mults = trial.suggest_categorical('mlp_hidden_mults', [(4, 2), (8, 4), (16, 8)])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)

    model = TabTransformer(
        categories=tuple([X_train[c].nunique() for c in categorical_features]),
        num_continuous=len(numerical_features),
        dim=dim,
        dim_out=1,
        depth=depth,
        heads=heads,
        attn_dropout=dropout,
        ff_dropout=dropout,
        mlp_hidden_mults=mlp_hidden_mults,
        mlp_act=nn.ReLU(),
    )

    return model, learning_rate


# 5. 定义训练和验证函数
def train_model(model, optimizer, criterion, train_loader, device):
    model.train()
    epoch_losses = []
    for batch in train_loader:
        categ = batch['categorical'].to(device)
        cont = batch['continuous'].to(device)
        target_batch = batch['target'].to(device).unsqueeze(1)

        # 前向传播
        output = model(categ, cont)

        # 计算损失
        loss = criterion(output, target_batch)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

    return np.mean(epoch_losses)


def evaluate_model(model, criterion, val_loader, device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch in val_loader:
            categ = batch['categorical'].to(device)
            cont = batch['continuous'].to(device)
            target_batch = batch['target'].to(device).unsqueeze(1)

            output = model(categ, cont)
            loss = criterion(output, target_batch)
            val_losses.append(loss.item())

    return np.mean(val_losses)


# 6. 定义Optuna的目标函数
def objective(trial):
    model, learning_rate = create_model(trial)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    epochs = 100  # 为了加快调参过程，可以减少训练轮数
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_model(model, optimizer, criterion, train_loader, device)
        val_loss = evaluate_model(model, criterion, val_loader, device)

        # 简单的早停法
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        else:
            pass  # 可以添加更复杂的早停逻辑

        # 每10个 epoch 打印一次损失
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Trial {trial.number}, Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return best_val_loss


# 7. 运行Optuna进行超参数优化
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, timeout=3600)  # 设置试验次数和超时时间

print("Number of finished trials: ", len(study.trials))
print("Best trial:")
trial = study.best_trial

print(f"  Value: {trial.value}")
print("  Params: ")
for key, value in trial.params.items():
    print(f"    {key}: {value}")

# 8. 使用最佳超参数训练最终模型
best_params = trial.params
best_model, best_learning_rate = create_model(trial)
best_model.to(device)

optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)
criterion = nn.MSELoss()

epochs = 500  # 使用您之前的训练轮数

for epoch in range(epochs):
    train_loss = train_model(best_model, optimizer, criterion, train_loader, device)
    val_loss = evaluate_model(best_model, criterion, val_loader, device)

    # 每10个 epoch 打印一次损失
    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(f"Final Model Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# 9. 在测试集上进行预测和评估
best_model.eval()
predictions = []
actuals = []

with torch.no_grad():
    for batch in test_loader:
        categ = batch['categorical'].to(device)
        cont = batch['continuous'].to(device)
        target_batch = batch['target'].to(device).unsqueeze(1)

        output = best_model(categ, cont)
        predictions.extend(output.cpu().numpy())
        actuals.extend(target_batch.cpu().numpy())

# 转换为一维数组
predictions = [pred[0] for pred in predictions]
actuals = [act[0] for act in actuals]

# 计算评估指标
mae = mean_absolute_error(actuals, predictions)
mse = mean_squared_error(actuals, predictions)
rmse = mse ** 0.5
mape = mean_absolute_percentage_error(actuals, predictions) * 100
r2 = r2_score(actuals, predictions)

print(f"\n最终 TabTransformer 模型的均方误差 (MSE): {mse:.4f}")
print(f"最终 TabTransformer 模型的根均方误差 (RMSE): {rmse:.4f}")
print(f"最终 TabTransformer 模型的平均绝对误差 (MAE): {mae:.4f}")
print(f"最终 TabTransformer 模型的平均绝对百分比误差 (MAPE): {mape:.2f}%")
print(f"最终 TabTransformer 模型的 R² 分数为: {r2:.2f}")


# 10. 展示特征的重要性（使用 SHAP）
# 注意：由于 TabTransformer 是深度学习模型，SHAP 的解释可能会比较慢

# 10.1 定义一个包装函数，使其接受两个独立的参数
def model_forward(categ, cont):
    return best_model(categ, cont)


# 10.2 准备背景数据
# 选择前 100 个训练样本作为背景数据
categ_background = torch.tensor(X_train[categorical_features].values[:100], dtype=torch.long).to(device)
cont_background = torch.tensor(X_train[numerical_features].values[:100], dtype=torch.float).to(device)

# 10.3 初始化 SHAP 的 DeepExplainer
explainer = shap.DeepExplainer(model_forward, (categ_background, cont_background))

# 10.4 准备要解释的数据
categ_test = torch.tensor(X_test[categorical_features].values, dtype=torch.long).to(device)
cont_test = torch.tensor(X_test[numerical_features].values, dtype=torch.float).to(device)

# 10.5 计算 SHAP 值
shap_values = explainer.shap_values((categ_test, cont_test))[0].cpu().numpy()

# 10.6 将 SHAP 值转换为 Pandas DataFrame
shap_df = pd.DataFrame(shap_values, columns=features)

# 10.7 计算平均绝对 SHAP 值作为特征重要性
feature_importance = shap_df.abs().mean().sort_values(ascending=False)

print("\n特征重要性（基于 SHAP 的平均绝对值）:")
print(feature_importance)

# 10.8 可视化特征重要性
plt.figure(figsize=(10, 8))
feature_importance.plot(kind='barh')
plt.title('Feature Importance based on SHAP values')
plt.xlabel('Mean |SHAP value|')
plt.gca().invert_yaxis()  # 使最重要的特征在上方
plt.show()

# 10.9 可选：绘制 SHAP 摘要图
shap.summary_plot(shap_values, X_test, feature_names=features, plot_type="bar")
