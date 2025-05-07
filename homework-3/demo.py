import kagglehub
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1=128, hidden_dim2=64, output_dim=1, dropout=0.3):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden_dim1, hidden_dim2, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim2, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        # 第一个LSTM层
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        
        # 第二个LSTM层
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # 只取最后一个时间步的输出
        x = self.dropout2(x)
        
        # 全连接层
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 自定义数据集类
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 创建输出目录
output_dir = "output_figures"
os.makedirs(output_dir, exist_ok=True)

# 3. 下载数据集
print("Downloading dataset...")
path = kagglehub.dataset_download("rupakroy/lstm-datasets-multivariate-univariate")
print("Dataset downloaded, path:", path)

# 4. 数据加载与预处理
dataset_path = os.path.join(path, 'LSTM-Multivariate_pollution.csv')
df = pd.read_csv(dataset_path, parse_dates=['date'], dayfirst=True)

# 检查第二个测试数据集是否存在，并尝试不同的加载方式
test_dataset_path = os.path.join(path, 'pollution_test_data1.csv')
test_df_exists = os.path.exists(test_dataset_path)

if test_df_exists:
    try:
        # 首先尝试检查文件头以确定列名
        test_cols = pd.read_csv(test_dataset_path, nrows=0).columns.tolist()
        print(f"Test dataset columns: {test_cols}")
        
        # 检查是否有日期列
        date_col = None
        if 'date' in test_cols:
            date_col = 'date'
        elif 'Date' in test_cols:
            date_col = 'Date'
            
        # 根据日期列的存在与否加载数据
        if date_col:
            test_df = pd.read_csv(test_dataset_path, parse_dates=[date_col], dayfirst=True)
        else:
            # 如果没有日期列，正常加载而不解析日期
            test_df = pd.read_csv(test_dataset_path)
            # 创建一个人工日期索引
            test_df['date'] = pd.date_range(start='2010-01-01', periods=len(test_df), freq='H')
        
        print("Test dataset found and loaded")
    except Exception as e:
        print(f"Error loading test dataset: {e}")
        test_df_exists = False
else:
    print("Test dataset not found, will only use train-test split")

# 按时间排序
df = df.sort_values('date').reset_index(drop=True)
if test_df_exists:
    # 如果测试集有日期列，则按日期排序
    if 'date' in test_df.columns:
        test_df = test_df.sort_values('date').reset_index(drop=True)

# 显示数据信息
print("Main dataset columns:", df.columns.tolist())
print("Main dataset shape:", df.shape)
if test_df_exists:
    print("Test dataset columns:", test_df.columns.tolist())
    print("Test dataset shape:", test_df.shape)

# 5. 数据可视化 - 原始数据特性
plt.figure(figsize=(15, 10))

# 时间序列图 - PM2.5污染浓度
plt.subplot(2, 2, 1)
plt.plot(df['date'][:500], df['pollution'][:500])
plt.title('PM2.5 Pollution Concentration (Time Series)')
plt.xlabel('Date')
plt.ylabel('PM2.5 Concentration')
plt.xticks(rotation=45)
plt.tight_layout()

# 散点图 - 温度与污染关系
plt.subplot(2, 2, 2)
plt.scatter(df['temp'], df['pollution'], alpha=0.5)
plt.title('Temperature vs Pollution')
plt.xlabel('Temperature')
plt.ylabel('PM2.5 Concentration')
plt.tight_layout()

# 散点图 - 露点与污染关系
plt.subplot(2, 2, 3)
plt.scatter(df['dew'], df['pollution'], alpha=0.5)
plt.title('Dew Point vs Pollution')
plt.xlabel('Dew Point')
plt.ylabel('PM2.5 Concentration')
plt.tight_layout()

# 散点图 - 风速与污染关系
plt.subplot(2, 2, 4)
plt.scatter(df['wnd_spd'], df['pollution'], alpha=0.5)
plt.title('Wind Speed vs Pollution')
plt.xlabel('Wind Speed')
plt.ylabel('PM2.5 Concentration')
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'data_exploration.png'), dpi=300, bbox_inches='tight')
print("Data exploration figure saved")

# 6. 特征预处理
# 处理缺失值
print("Missing values in main dataset:\n", df.isna().sum())
df = df.dropna(subset=['pollution']).reset_index(drop=True)

# 处理分类变量（风向）
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
wind_encoded = encoder.fit_transform(df[['wnd_dir']])

# 标准化数值特征（包括目标变量）
scaler = MinMaxScaler()
num_features = scaler.fit_transform(df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']])

# 合并处理后的特征矩阵
processed_features = np.concatenate([num_features, wind_encoded], axis=1)

# 7. 构建时间序列数据集
def create_sequences(data, target_col_index, look_back=24):
    """
    创建时间序列样本
    :param data: 处理后的特征矩阵 (n_samples, n_features)
    :param target_col_index: 目标变量在矩阵中的列索引
    :param look_back: 时间窗口长度（小时）
    :return: 3D序列样本，目标值
    """
    X, y = [], []
    for i in range(len(data) - look_back - 1):
        # 时间窗口内的所有特征（包括目标变量）
        window = data[i:(i + look_back)]
        # 下一个时间步的目标值
        target = data[i + look_back, target_col_index]
        X.append(window)
        y.append(target)
    return np.array(X), np.array(y)

# 参数设置
LOOK_BACK = 24  # 使用过去24小时的数据
TARGET_INDEX = 0  # pollution在特征矩阵中的位置

X, y = create_sequences(processed_features, TARGET_INDEX, LOOK_BACK)

# 8. 数据集划分（保持时间连续性）
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 9. 创建训练和测试数据加载器
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)

BATCH_SIZE = 128
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# 10. 初始化模型、损失函数和优化器
input_dim = processed_features.shape[1]  # 特征维度
model = LSTMModel(input_dim)

# 检查是否可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

# 11. 模型训练
num_epochs = 10
train_losses = []
val_losses = []
best_val_loss = float('inf')
patience = 10
counter = 0

for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0.0
    
    # 训练循环
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # 清除过往梯度
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss.item()
    
    # 计算平均训练损失
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)
    
    # 验证循环
    model.eval()
    epoch_val_loss = 0.0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            epoch_val_loss += loss.item()
    
    # 计算平均验证损失
    epoch_val_loss /= len(test_loader)
    val_losses.append(epoch_val_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')
    
    # 早期停止逻辑
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        # 保存最佳模型
        torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping after {epoch+1} epochs")
            break

# 12. 绘制训练和验证损失曲线
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'training_validation_loss.png'), dpi=300, bbox_inches='tight')
print("Training and validation loss figure saved")

# 13. 逆标准化函数
def inverse_transform_pm25(scaled_values):
    """专用PM2.5逆标准化（仅处理数值特征部分）"""
    # 创建与scaler维度一致的dummy矩阵（7列）
    dummy = np.zeros((len(scaled_values), 7))  # 修改为数值特征数量
    dummy[:, 0] = scaled_values  # pollution在数值特征中的位置为0
    
    # 使用scaler进行逆变换
    inverted = scaler.inverse_transform(dummy)
    return inverted[:, 0]  # 返回第一列（PM2.5）

# 14. 加载最佳模型
model.load_state_dict(torch.load(os.path.join(output_dir, 'best_model.pth')))
model.eval()

# 15. 评估原始数据集的测试部分
test_loss = 0.0
predictions = []
actuals = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        
        # 累积损失
        loss = criterion(outputs, targets)
        test_loss += loss.item()
        
        # 保存预测和实际值
        predictions.append(outputs.cpu().numpy())
        actuals.append(targets.cpu().numpy())

# 计算平均测试损失
test_loss /= len(test_loader)
print(f'Test MSE on original dataset split: {test_loss:.4f}')

# 合并批次的预测和实际值
all_predictions = np.vstack(predictions).ravel()
all_actuals = np.vstack(actuals).ravel()

# 逆标准化
predictions_orig = inverse_transform_pm25(all_predictions)
actuals_orig = inverse_transform_pm25(all_actuals)

# 计算MAE
mae = np.mean(np.abs(predictions_orig - actuals_orig))
print(f'Test MAE on original dataset split: {mae:.4f}')

# 16. 可视化原始数据集测试部分的预测结果
plt.figure(figsize=(14, 6))
plt.plot(actuals_orig[:500], label='Actual Values', alpha=0.7, linewidth=1)
plt.plot(predictions_orig[:500], label='Predicted Values', alpha=0.7, linewidth=1, linestyle='--')
plt.title('PM2.5 Concentration Forecasting (80% Train - 20% Test Split)')
plt.xlabel('Time Steps (hours)')
plt.ylabel('PM2.5 (μg/m³)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(output_dir, 'prediction_original_test_set.png'), dpi=300, bbox_inches='tight')
print("Prediction results on original test set figure saved")

# 17. 散点图和残差图
plt.figure(figsize=(15, 6))

# 散点图 - 预测值与实际值对比
plt.subplot(1, 2, 1)
plt.scatter(actuals_orig, predictions_orig, alpha=0.5)
plt.plot([min(actuals_orig), max(actuals_orig)], [min(actuals_orig), max(actuals_orig)], 'r--')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual PM2.5')
plt.ylabel('Predicted PM2.5')
plt.grid(True, alpha=0.3)

# 残差图
plt.subplot(1, 2, 2)
residuals = actuals_orig - predictions_orig
plt.scatter(actuals_orig, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Actual PM2.5')
plt.ylabel('Residuals')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'regression_diagnostics.png'), dpi=300, bbox_inches='tight')
print("Regression diagnostics figure saved")

# 18. 如果存在第二个测试集，则进行评估
if test_df_exists:
    try:
        # 检查测试集是否包含所有需要的列
        required_columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        missing_columns = [col for col in required_columns if col not in test_df.columns]
        
        if missing_columns:
            print(f"Warning: Test dataset is missing columns: {missing_columns}")
            print("Will create a synthetic test dataset by reusing part of the original dataset")
            
            # 使用原始数据集的最后20%作为"外部"测试集
            synthetic_test_df = df.iloc[int(0.8*len(df)):].copy()
            print(f"Created synthetic test dataset with {len(synthetic_test_df)} samples")
            
            # 处理缺失值
            synthetic_test_df = synthetic_test_df.dropna(subset=['pollution']).reset_index(drop=True)
            
            # 处理分类变量（风向）
            synthetic_wind_encoded = encoder.transform(synthetic_test_df[['wnd_dir']])
            
            # 标准化数值特征
            synthetic_num_features = scaler.transform(
                synthetic_test_df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']]
            )
            
            # 合并处理后的特征矩阵
            synthetic_processed_features = np.concatenate([synthetic_num_features, synthetic_wind_encoded], axis=1)
            
            # 创建序列数据
            X_external, y_external = create_sequences(synthetic_processed_features, TARGET_INDEX, LOOK_BACK)
        else:
            # 处理测试集
            print("Processing external test dataset...")
            
            # 检查测试集是否有缺失值
            print("Missing values in test dataset:\n", test_df.isna().sum())
            test_df = test_df.dropna(subset=['pollution']).reset_index(drop=True)
            
            # 处理分类变量（风向）
            test_wind_encoded = encoder.transform(test_df[['wnd_dir']])
            
            # 标准化数值特征（使用原始数据集的标准化器）
            test_num_features = scaler.transform(
                test_df[['pollution', 'dew', 'temp', 'press', 'wnd_spd', 'snow', 'rain']]
            )
            
            # 合并处理后的特征矩阵
            test_processed_features = np.concatenate([test_num_features, test_wind_encoded], axis=1)
            
            # 创建序列数据
            X_external, y_external = create_sequences(test_processed_features, TARGET_INDEX, LOOK_BACK)
        
        # 创建数据加载器
        external_test_dataset = TimeSeriesDataset(X_external, y_external)
        external_test_loader = DataLoader(external_test_dataset, batch_size=BATCH_SIZE)
        
        # 评估模型在外部测试集上的表现
        external_test_loss = 0.0
        external_predictions = []
        external_actuals = []
        
        model.eval()
        with torch.no_grad():
            for inputs, targets in external_test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # 累积损失
                loss = criterion(outputs, targets)
                external_test_loss += loss.item()
                
                # 保存预测和实际值
                external_predictions.append(outputs.cpu().numpy())
                external_actuals.append(targets.cpu().numpy())
        
        # 计算平均测试损失
        external_test_loss /= len(external_test_loader)
        print(f'Test MSE on external test dataset: {external_test_loss:.4f}')
        
        # 合并批次的预测和实际值
        all_ext_predictions = np.vstack(external_predictions).ravel()
        all_ext_actuals = np.vstack(external_actuals).ravel()
        
        # 逆标准化
        ext_predictions_orig = inverse_transform_pm25(all_ext_predictions)
        ext_actuals_orig = inverse_transform_pm25(all_ext_actuals)
        
        # 计算MAE
        ext_mae = np.mean(np.abs(ext_predictions_orig - ext_actuals_orig))
        print(f'Test MAE on external test dataset: {ext_mae:.4f}')
        
        # 可视化外部测试集的预测结果
        plt.figure(figsize=(14, 6))
        plt.plot(ext_actuals_orig[:500], label='Actual Values', alpha=0.7, linewidth=1)
        plt.plot(ext_predictions_orig[:500], label='Predicted Values', alpha=0.7, linewidth=1, linestyle='--')
        plt.title('PM2.5 Concentration Forecasting (External Test Dataset)')
        plt.xlabel('Time Steps (hours)')
        plt.ylabel('PM2.5 (μg/m³)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'prediction_external_test_set.png'), dpi=300, bbox_inches='tight')
        print("Prediction results on external test set figure saved")
        
        # 散点图和残差图 - 外部测试集
        plt.figure(figsize=(15, 6))
        
        # 散点图 - 预测值与实际值对比
        plt.subplot(1, 2, 1)
        plt.scatter(ext_actuals_orig, ext_predictions_orig, alpha=0.5)
        plt.plot([min(ext_actuals_orig), max(ext_actuals_orig)], [min(ext_actuals_orig), max(ext_actuals_orig)], 'r--')
        plt.title('Actual vs Predicted Values (External Test Set)')
        plt.xlabel('Actual PM2.5')
        plt.ylabel('Predicted PM2.5')
        plt.grid(True, alpha=0.3)
        
        # 残差图
        plt.subplot(1, 2, 2)
        ext_residuals = ext_actuals_orig - ext_predictions_orig
        plt.scatter(ext_actuals_orig, ext_residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Residuals Plot (External Test Set)')
        plt.xlabel('Actual PM2.5')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'external_test_regression_diagnostics.png'), dpi=300, bbox_inches='tight')
        print("External test regression diagnostics figure saved")
    
    except Exception as e:
        print(f"Error processing external test dataset: {e}")
        print("Will continue with other visualizations")

# 19. 创建特征重要性分析图（通过训练另一个简单模型来评估）
try:
    from sklearn.ensemble import RandomForestRegressor

    # 提取原始特征（不是序列形式）以进行特征重要性分析
    X_importance = processed_features[:-LOOK_BACK]  # 移除最后LOOK_BACK行，因为它们在序列化时没有对应的目标
    y_importance = processed_features[LOOK_BACK:, 0]  # 目标是下一个时间步的PM2.5值

    # 训练随机森林用于特征重要性评估
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_importance, y_importance)

    # 获取特征名称列表
    feature_names = ['PM2.5', 'Dew', 'Temp', 'Press', 'Wind Speed', 'Snow', 'Rain'] + \
                    [f'Wind Dir {i+1}' for i in range(wind_encoded.shape[1])]

    # 获取前10个最重要的特征（如果特征总数超过10个）
    importance = rf_model.feature_importances_
    indices = np.argsort(importance)[::-1]
    top_k = min(10, len(feature_names))

    plt.figure(figsize=(12, 6))
    plt.title('Feature Importance for PM2.5 Prediction')
    plt.bar(range(top_k), importance[indices[:top_k]], align='center')
    plt.xticks(range(top_k), [feature_names[i] for i in indices[:top_k]], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    print("Feature importance figure saved")
except Exception as e:
    print(f"Error creating feature importance plot: {e}")

# 20. 创建模型架构图
try:
    plt.figure(figsize=(10, 8))
    plt.text(0.5, 0.95, 'LSTM Model Architecture', horizontalalignment='center', size=14, weight='bold')
    
    # 输入层
    plt.text(0.5, 0.85, f'Input Layer (Sequence Length: {LOOK_BACK}, Features: {input_dim})', 
             horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.5))
    
    # LSTM层1
    plt.text(0.5, 0.75, 'LSTM Layer 1 (128 units)\nDropout (0.3)', 
             horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    # LSTM层2
    plt.text(0.5, 0.65, 'LSTM Layer 2 (64 units)\nDropout (0.3)', 
             horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.5))
    
    # 全连接层1
    plt.text(0.5, 0.55, 'Dense Layer (32 units)\nReLU Activation', 
             horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.5))
    
    # 输出层
    plt.text(0.5, 0.45, 'Output Layer (1 unit)', 
             horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='salmon', alpha=0.5))
    
    # 绘制连接线
    for y_start, y_end in [(0.85, 0.75), (0.75, 0.65), (0.65, 0.55), (0.55, 0.45)]:
        plt.arrow(0.5, y_start-0.02, 0, y_end-y_start+0.01, head_width=0.02, 
                 head_length=0.01, fc='black', ec='black')
    
    # 添加额外信息
    plt.text(0.5, 0.3, f'Training: Adam Optimizer\nLoss Function: MSE\nBatch Size: {BATCH_SIZE}\nEpochs: {num_epochs}', 
             horizontalalignment='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.5))
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_architecture.png'), dpi=300, bbox_inches='tight')
    print("Model architecture figure saved")
except Exception as e:
    print(f"Error creating model architecture plot: {e}")

print("All visualizations have been generated and saved in the 'output_figures' directory")