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
def load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path,test_unknown_path,train_unknown_path):
    
    train_self = pd.read_csv(train_self_path)
    train_self = train_self.sample(n=1000,random_state=42)
    train_nonself = pd.read_csv(train_nonself_path)
    
    test_unknown = pd.read_csv(test_unknown_path)
    train_unknown = pd.read_csv(train_unknown_path)
    unknown = pd.concat([test_unknown, train_unknown], axis=0).reset_index(drop=True)
    
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
def evaluate_unknown_coverage(model, unknown_data, threshold=0.5):
    # 预处理未知数据
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
    # 预处理正常数据
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
    thresholds = np.linspace(0.1, 0.9, 9)
    results = []
    
    for threshold in thresholds:
        # 计算验证集上的F1分数
        y_pred = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        
        # 计算未知覆盖率
        unknown_cov = evaluate_unknown_coverage(model, unknown_data, threshold)
        
        # 计算误报率
        fpr = evaluate_false_positive_rate(model, normal_data, threshold)
        
        # 计算综合得分 (可以根据需要调整权重)
        score = f1 * 0.4 + unknown_cov * 0.4 - fpr * 0.2
        
        results.append((threshold, f1, unknown_cov, fpr, score))
    
    # 找到最佳阈值
    best_result = max(results, key=lambda x: x[4])
    return best_result

# 主函数
def main():
    unknown_types = ["A", "B", "D", "E", "F", "G", "R", "S", "W"]
    for unknown_type in unknown_types:
        # 设置数据路径
        train_self_path = '../../../check/self/train_self_new.csv'
        train_nonself_path = f'../../../check/train/trainset_{unknown_type}_nonself.csv'
        test_self_path = '../../../check/self/test_self_new.csv'
        test_nonself_path = '../../../check/nonself/test_nonself.csv'
        test_unknown_path = f'../../../check/unknown/test/test{unknown_type}.csv'
        train_unknown_path = f'../../../check/unknown/train/train{unknown_type}.csv'
        
        # 加载数据
        print(f"处理未知类型: {unknown_type}")
        print("加载数据...")
        train_data, test_data, unknown = load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path,test_unknown_path,train_unknown_path)
        
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
        
        # 评估模型
        print("评估模型...")
        accuracy, precision, recall, f1, conf_matrix = evaluate_model(model, X_test, y_test)
        unknown_coverage = evaluate_unknown_coverage(model, unknown)
        test_self_data = test_data[test_data['label'] == 0]
        false_positive_rate = evaluate_false_positive_rate(model, test_self_data)
        
        # 保存结果
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'unknown_coverage': unknown_coverage,
            'false_positive_rate': false_positive_rate,
            'training_time': training_time,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        # 创建目录（如果不存在）
        import os
        if not os.path.exists(unknown_type):
            os.makedirs(unknown_type)
        
        # 将结果保存为文本文件
        with open(f'{unknown_type}/svm_results.txt', 'w') as f:
            for key, value in results.items():
                if key != 'confusion_matrix':
                    f.write(f"{key}: {value}\n")
            f.write(f"confusion_matrix:\n{conf_matrix}\n")
        
        print(f"结果已保存到 {unknown_type}/svm_results.txt")

        # 计算最佳阈值
        validation_data = pd.concat([train_data, test_data], axis=0).sample(frac=0.2, random_state=42)
        X_val_data = validation_data.drop('label', axis=1).values
        y_val_data = validation_data['label'].values
        best_result = find_optimal_threshold(model, X_val_data, y_val_data, unknown, test_self_data)
        best_threshold, best_f1, best_unknown_cov, best_fpr, best_score = best_result
        
        with open(f'{unknown_type}/best_threshold_results.txt', 'w') as f:
            f.write(f"Best Threshold: {best_threshold:.6f}\n")
            f.write(f"F1 Score at Best Threshold: {best_f1:.4f}\n")
            f.write(f"Unknown Coverage at Best Threshold: {best_unknown_cov:.4f}\n")
            f.write(f"False Positive Rate at Best Threshold: {best_fpr:.4f}\n")
            f.write(f"Combined Score: {best_score:.4f}\n")

if __name__ == "__main__":
    main()