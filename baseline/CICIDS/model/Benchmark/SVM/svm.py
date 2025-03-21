import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time

# 设置随机种子以确保结果可复现
np.random.seed(42)

# 数据加载
def load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path,unknown_path):
    train_nonself = pd.read_csv(train_nonself_path)
    train_self = pd.read_csv(train_self_path)
    train_self = train_self.sample(n=len(train_nonself),random_state=42)
    train_nonself = train_nonself.fillna(train_nonself.mean())
    train_self = train_self.fillna(train_self.mean())
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
    test_data = pd.concat([test_self, test_nonself], axis=0).reset_index(drop=True)
    print("训练集分布：")
    print(train_data['label'].value_counts())
    print("\n测试集分布：")
    print(test_data['label'].value_counts())
    return train_data, test_data, unknown

# 数据预处理函数
def preprocess_data(train_data, test_data):
    # 处理缺失值
    train_data = train_data.fillna(train_data.mean())
    test_data = test_data.fillna(test_data.mean())
    # 分离特征和标签
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values
    
    return X_train, y_train, X_test, y_test

# 训练支持向量机模型
def train_model(X_train, y_train, C=1.0, max_iter=100):
    # 创建SVM分类器
    model = SVC(
        C=C,
        kernel='rbf',  # 径向基函数核
        max_iter=max_iter,
        probability=True,  # 启用概率估计
        random_state=42
    )
    
    # 训练模型
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return model, training_time

# 评估未知覆盖率
def evaluate_unknown_coverage(model, unknown_data, threshold):
    # 预处理未知数据 - 填充缺失值
    unknown_data = unknown_data.dropna()
    X_unknown = unknown_data.values
    
    # 预测概率
    y_proba = model.predict_proba(X_unknown)[:, 1]
    
    # 根据阈值确定预测结果
    y_pred = (y_proba >= threshold).astype(int)
    
    # 计算未知覆盖率 - 被检测为异常的未知样本比例
    unknown_coverage = np.mean(y_pred)
    
    return unknown_coverage

# 评估误报率 - 在正常数据上
def evaluate_false_positive_rate(model, normal_data, threshold=0.5):
    # 预处理正常数据 - 填充缺失值
    normal_data = normal_data.fillna(normal_data.mean())
    X_normal = normal_data.drop('label', axis=1).values
    
    # 预测概率
    y_proba = model.predict_proba(X_normal)[:, 1]
    
    # 根据阈值确定预测结果
    y_pred = (y_proba >= threshold).astype(int)
    
    # 计算误报率 - 正常样本被错误分类为异常的比例
    false_positive_rate = np.mean(y_pred)
    
    return false_positive_rate

# 评估模型性能
def evaluate_model(model, X_test, y_test, threshold=0.5):
    # 预测概率
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 根据阈值确定预测结果
    y_pred = (y_proba >= threshold).astype(int)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, precision, recall, f1, conf_matrix

# 计算最佳阈值
def find_optimal_threshold(model, X_val, y_val, unknown_data, normal_data):
    # 收集所有预测分数
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # 尝试不同阈值
    thresholds = np.linspace(0.01, 0.99, 100)
    results = []
    
    for threshold in thresholds:
        # 计算验证集上的F1分数
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        pre = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        # 计算未知覆盖率
        unknown_cov = evaluate_unknown_coverage(model, unknown_data, threshold)
        
        # 计算误报率
        fpr = evaluate_false_positive_rate(model, normal_data, threshold)
        
        # 计算综合得分 (可以根据需要调整权重)
        score = f1
        
        results.append((threshold, acc, pre, recall, f1, unknown_cov, fpr, score))
    
    # 找到最佳阈值
    best_result = max(results, key=lambda x: x[4])
    return best_result

# 主函数
def main():
    unknown_types = ["bot", "bruteforce", "dos", "ddos", "infilteration", "sql_injection"]
    for unknown_type in unknown_types:
        # 设置数据路径
        train_self_path = '../../../check/self/train_self.csv'
        train_nonself_path = f'../../../check/train/unknown_{unknown_type}.csv'
        test_self_path = '../../../check/self/test_self.csv'
        test_nonself_path = '../../../check/nonself/test_nonself.csv'
        unknown_path = f'../../../check/unknown/{unknown_type}.csv'
        
        # 加载数据
        print(f"处理未知类型: {unknown_type}")
        print("加载数据...")
        train_data, test_data, unknown = load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path,unknown_path)
        
        # 预处理数据
        print("预处理数据...")
        X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
        
        # 将训练集分为训练集和验证集（80%训练，20%验证）
        train_size = int(0.8 * len(X_train))
        X_train_split, X_val = X_train[:train_size], X_train[train_size:]
        y_train_split, y_val = y_train[:train_size], y_train[train_size:]
        
        # 训练模型
        print("训练支持向量机模型...")
        model, training_time = train_model(X_train_split, y_train_split, C=1.0, max_iter=1000)
        print(f"训练完成，耗时: {training_time:.2f} 秒")
        
        # 使用验证集找到最佳阈值
        print("在验证集上寻找最佳阈值...")
        validation_self = train_data[train_data['label'] == 0].iloc[train_size:] 
        best_result = find_optimal_threshold(model, X_val, y_val, unknown, validation_self)
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

if __name__ == "__main__":
    main()