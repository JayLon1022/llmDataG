import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 自定义数据集类
class IntrusionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 定义GRU+LSTM
class GRU_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size_gru=128, hidden_size_lstm=64, dropout=0.3):
        super(GRU_LSTM, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size_gru, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size_gru*2, hidden_size_lstm, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size_lstm*2, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        gru_out = self.dropout1(gru_out)
        
        lstm_out, _ = self.lstm(gru_out)
        lstm_out = self.dropout2(lstm_out[:, -1, :])
        
        out = self.relu(self.fc1(lstm_out))
        out = self.fc2(out)
        return out

# 数据加载
def load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path,unknown_path):
    train_nonself = pd.read_csv(train_nonself_path)
    train_self = pd.read_csv(train_self_path)
    train_self = train_self.sample(n=len(train_nonself),random_state=42)

    unknown = pd.read_csv(unknown_path)
    
    # 加载测试数据
    test_self = pd.read_csv(test_self_path)
    test_self = test_self.sample(n=5000,random_state=42)
    test_nonself = pd.read_csv(test_nonself_path)
    test_nonself = test_nonself.sample(n=5000,random_state=42)

    # 添加标签：自体为0，非自体为1
    train_self['label'] = 0
    train_nonself['label'] = 1
    test_self['label'] = 0
    test_nonself['label'] = 1
    
    # 合并训练集和测试集
    train_data = pd.concat([train_self, train_nonself], axis=0).reset_index(drop=True)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    test_data = pd.concat([test_self, test_nonself], axis=0).reset_index(drop=True)
    test_data = test_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_data, test_data, unknown

# 数据预处理函数
def preprocess_data(train_data, test_data):
    # 处理缺失值
    train_data = train_data.dropna()
    test_data = test_data.dropna()
    # 分离特征和标签
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values
    
    X_train_seq = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test_seq = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    return X_train_seq, y_train, X_test_seq, y_test

# 训练模型函数
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50, patience=10):
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            outputs = model(features)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
        
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        
        model.eval()
        val_loss = 0.0
        val_outputs = []
        val_labels = []
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels.unsqueeze(1).float())
                val_loss += loss.item()
                val_outputs.extend(outputs.sigmoid().numpy())
                val_labels.extend(labels.numpy())
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), 'best_gru_lstm_model.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    model.load_state_dict(torch.load('best_gru_lstm_model.pth'))
    
    return model, train_losses, val_losses

# 评估未知覆盖率和误报率
def evaluate_unknown_coverage(model, unknown_data, threshold=0):
    # 预处理未知数据
    X_unknown = unknown_data.values
    X_unknown_seq = X_unknown.reshape(X_unknown.shape[0], 1, X_unknown.shape[1])
    unknown_dataset = torch.FloatTensor(X_unknown_seq)
    
    # 创建数据加载器
    unknown_loader = DataLoader(unknown_dataset, batch_size=64)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for features in unknown_loader:
            features = features.to(device)
            outputs = model(features)
            predicted = (outputs >= threshold).float()
            all_preds.extend(predicted.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    
    # 计算未知覆盖率 - 被检测为异常的未知样本比例
    unknown_coverage = np.mean(all_preds)
    
    return unknown_coverage

# 评估误报率 - 在正常数据上
def evaluate_false_positive_rate(model, normal_data, threshold=0):
    # 预处理正常数据
    X_normal = normal_data.drop('label', axis=1).values
    X_normal_seq = X_normal.reshape(X_normal.shape[0], 1, X_normal.shape[1])
    
    normal_dataset = IntrusionDataset(X_normal_seq, np.zeros(len(X_normal)))
    normal_loader = DataLoader(normal_dataset, batch_size=64)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for features, _ in normal_loader:
            features = features.to(device)
            outputs = model(features)
            predicted = (outputs >= threshold).float()
            all_preds.extend(predicted.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    
    # 计算误报率 - 正常样本被错误分类为异常的比例
    false_positive_rate = np.mean(all_preds)
    
    return false_positive_rate

# 评估模型性能
def evaluate_model(model, features, labels, threshold=0):
    model.eval()
    dataset = IntrusionDataset(features, labels)
    test_loader = DataLoader(dataset, batch_size=64)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            
            outputs = model(batch_features)
            predicted = (outputs >= threshold).float()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
    
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels)
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return accuracy, precision, recall, f1, conf_matrix

# 计算最佳阈值
def find_optimal_threshold(model, X_val_seq, y_val):
    val_dataset = IntrusionDataset(X_val_seq, y_val)
    val_loader = DataLoader(val_dataset, batch_size=64)
    
    # 收集所有预测分数
    model.eval()
    all_scores = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            all_scores.extend(outputs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_scores = np.array(all_scores).flatten()
    all_labels = np.array(all_labels)
    
    # 尝试不同阈值
    thresholds = np.linspace(0, 1, 1000)
    results = []
    
    for threshold in thresholds:
        # 计算验证集上的F1分数
        predicted = (all_scores >= threshold).astype(float)
        f1 = f1_score(all_labels, predicted)

        # 计算综合得分 (可以根据需要调整权重)
        score = f1
        results.append((threshold,f1,score))
    
    # 找到最佳阈值
    best_result = max(results, key=lambda x: x[1])
    return best_result

# 主函数
def main():
    unknown_types = ["bot", "bruteforce", "dos", "ddos", "infiltration", "sql_injection"]
    for unknown_type in unknown_types:
    
        # 设置数据路径
        train_self_path = '../../check/self/train_self.csv'
        train_nonself_path = f'../../check/train/seed_{unknown_type}.csv'
        test_self_path = '../../check/self/test_self.csv'
        test_nonself_path = '../../check/nonself/test_nonself.csv'
        unknown_path = f'../../check/unknown/{unknown_type}.csv'
        
        # 加载数据
        print("Loading data...")
        train_data, test_data, unknown = load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path,unknown_path)
        
        # 预处理数据
        print("Preprocessing data...")
        X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
        
        # 将训练集分为训练集和验证集（80%训练，20%验证）
        train_size = int(0.8 * len(X_train))
        val_size = len(X_train) - train_size
        
        X_train_split, X_val = X_train[:train_size], X_train[train_size:]
        y_train_split, y_val = y_train[:train_size], y_train[train_size:]
        
        train_dataset = IntrusionDataset(X_train_split, y_train_split)
        val_dataset = IntrusionDataset(X_val, y_val)
        test_dataset = IntrusionDataset(X_test, y_test)
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        input_size = X_train.shape[2]
        model = GRU_LSTM(input_size)
        
        # 权重
        pos_weight = torch.tensor([(y_train == 0).sum() / [(y_train == 1).sum()] ])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
        
        # 使用带权重衰减的优化器
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        
        # 添加学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
        
        # 训练模型
        print("Training model...")
        start_time = time.time()
        model, train_losses, val_losses= train_model(
            model, 
            train_loader, 
            val_loader, 
            criterion, 
            optimizer,
            scheduler,
            num_epochs=50, 
            patience=10
        )
        training_time = time.time() - start_time
        print(f"Done training, time: {training_time:.2f} 秒")

        # 使用验证集找到最佳阈值
        print("在验证集上寻找最佳阈值...")
        best_result = find_optimal_threshold(model, X_val, y_val)
        best_threshold = best_result[0]
        
        # 使用测试集进行最终评估
        print("在测试集上进行最终评估...")
        test_accuracy, test_precision, test_recall, test_f1, test_conf_matrix = evaluate_model(
            model, X_test, y_test, threshold=best_threshold
        )

        # 在测试集上评估未知覆盖率和误报率
        test_unknown_coverage = evaluate_unknown_coverage(model, unknown, threshold=best_threshold)
        test_self_data = test_data[test_data['label'] == 0]
        test_fpr = evaluate_false_positive_rate(model, test_self_data, threshold=best_threshold)
        
        with open(f'{unknown_type}_results.txt', 'w') as f:
            f.write(f"Best Threshold: {best_threshold:.6f}\n")
            f.write("\nTest Set Results:\n")
            f.write(f"Accuracy: {test_accuracy:.4f}\n")
            f.write(f"Precision: {test_precision:.4f}\n")
            f.write(f"Recall: {test_recall:.4f}\n")
            f.write(f"F1 Score: {test_f1:.4f}\n")
            f.write(f"Unknown Coverage: {test_unknown_coverage:.4f}\n")
            f.write(f"False Positive Rate: {test_fpr:.4f}\n")
            f.write(f"Confusion Matrix: {test_conf_matrix}\n")
            


if __name__ == "__main__":
    main()