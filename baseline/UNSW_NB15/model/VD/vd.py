import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import random
from scipy.spatial.distance import euclidean
import os

unknown_type = 'A'


class VDetector:
    """
    基于空间覆盖率的V-Detectors负选择算法
    
    参数:
    - self_radius: 自体半径，用于定义自体样本周围的保护区域
    - max_detectors: 最大检测器数量
    - coverage_threshold: 空间覆盖率阈值，达到此阈值时停止生成检测器
    - max_iterations: 生成检测器的最大尝试次数
    """
    
    def __init__(self, self_radius=0.05, max_detectors=20000, coverage_threshold=0.999, max_iterations=1000000):
        self.self_radius = self_radius
        self.max_detectors = max_detectors
        self.coverage_threshold = coverage_threshold
        self.max_iterations = max_iterations
        self.detectors = []
        self.self_samples = None
        self.dim = 42
        self.estimated_coverage = 0
        
    def _is_covered_by_self(self, point):
        """检查点是否被任何自体样本覆盖"""
        for self_point in self.self_samples:
            distance = euclidean(point, self_point)
            if distance <= self.self_radius:
                return True
        return False
    
    def _is_covered_by_detectors(self, point):
        """检查点是否被任何现有检测器覆盖"""
        for detector_center, detector_radius in self.detectors:
            distance = euclidean(point, detector_center)
            if distance <= detector_radius:
                return True
        return False
    
    def _calculate_detector_radius(self, point):
        """计算检测器的最大可能半径"""
        min_distance = float('inf')
        for self_point in self.self_samples:
            distance = euclidean(point, self_point)
            if distance - self.self_radius < min_distance:
                min_distance = distance - self.self_radius
        return max(0, min_distance)
    
    def _generate_random_point(self):
        """在[0,1]^dim空间内生成随机点"""
        return np.random.random(self.dim)
    
    def _estimate_coverage(self, num_samples=100000):
        """估计当前检测器对非自体空间的覆盖率"""
        covered_count = 0
        for _ in range(num_samples):
            point = self._generate_random_point()
            if self._is_covered_by_self(point) or self._is_covered_by_detectors(point):
                covered_count += 1
        return covered_count / num_samples
    
    def fit(self, X):
        """
        训练V-Detectors模型
        
        参数:
        - X: 自体样本集，形状为(n_samples, n_features)
        """
        self.self_samples = X
        self.dim = X.shape[1]
        
        iterations = 0
        while len(self.detectors) < self.max_detectors and iterations < self.max_iterations:
            iterations += 1
            
            # 生成随机点
            candidate = self._generate_random_point()
            
            # 如果点被自体样本覆盖，跳过
            if self._is_covered_by_self(candidate):
                continue
                
            # 如果点被现有检测器覆盖，跳过
            if self._is_covered_by_detectors(candidate):
                continue
                
            # 计算检测器半径
            radius = self._calculate_detector_radius(candidate)
            
            # 添加新检测器
            if radius > 0:
                self.detectors.append((candidate, radius))
            
            # 每生成10个检测器，估计一次覆盖率
            if len(self.detectors) % 10 == 0:
                self.estimated_coverage = self._estimate_coverage()
                if self.estimated_coverage >= self.coverage_threshold:
                    break
                
        print(f"生成了{len(self.detectors)}个检测器，估计覆盖率: {self.estimated_coverage:.2%}")
        return self
    
    def predict(self, X):
        """
        预测样本是否为异常(非自体)
        
        参数:
        - X: 测试样本，形状为(n_samples, n_features)
        
        返回:
        - y_pred: 预测结果，0表示正常(自体)，1表示异常(非自体)
        """
        y_pred = np.zeros(X.shape[0])
        
        for i, sample in enumerate(X):
            for detector_center, detector_radius in self.detectors:
                distance = euclidean(sample, detector_center)
                if distance <= detector_radius:
                    y_pred[i] = 1  # 标记为异常
                    break
                    
        return y_pred
    
    def score(self, X, y):
        """计算模型在测试集上的准确率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def evaluate_performance(self, self_predictions, nonself_predictions, unknown_predictions, test_self, test_nonself, unknown):
        """
        评价模型性能
        
        参数:
        - self_predictions: 对自体样本的预测结果
        - nonself_predictions: 对非自体样本的预测结果
        - unknown_predictions: 对未知样本的预测结果
        - test_self: 测试集自体样本
        - test_nonself: 测试集非自体样本
        - unknown: 未知样本
        
        返回:
        - metrics: 包含各项性能指标的字典
        """
        TP = np.sum(nonself_predictions)  
        FP = np.sum(self_predictions)     
        FN = len(test_nonself) - TP       
        TN = len(test_self) - FP  
        
        metrics = {
            "Accuracy": (TP + TN) / (len(test_self) + len(test_nonself)),
            "Precision": TP / (TP + FP) if (TP + FP) > 0 else 0,
            "Recall": TP / (TP + FN) if (TP + FN) > 0 else 0,
            "False Positive Rate": FP / (FP + TN) if (FP + TN) > 0 else 0,
            "Unknown Coverage Rate": np.sum(unknown_predictions) / len(unknown),
            "Confusion Matrix": np.array([[TP, FP], [FN, TN]])
        }
        
        if metrics["Precision"] + metrics["Recall"] > 0:
            metrics["F1 Score"] = 2 * (metrics["Precision"] * metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"])
        else:
            metrics["F1 Score"] = 0
        
        with open(f"{unknown_type}/{unknown_type}_results.txt", "w") as f:
            f.write("Performance Metrics:\n")
            f.write(f"Number of test self samples: {len(test_self)}\n")
            f.write(f"Number of test non-self samples: {len(test_nonself)}\n")
            f.write(f"Number of unknown samples: {len(unknown)}\n\n")
            
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    f.write(f"{metric}:\n{value}\n")
                else:
                    f.write(f"{metric}: {value:.2%}\n")
        
        plot_metrics = {k: v for k, v in metrics.items() if k != "Confusion Matrix"}
        plt.figure(figsize=(12, 6))
        bars = plt.bar(plot_metrics.keys(), plot_metrics.values())
        plt.title('V-detector Performance Metrics')
        plt.ylabel('Value')
        plt.xticks(rotation=45, ha='right')

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2%}',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{unknown_type}/{unknown_type}_metrics_bar_chart.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return metrics


if __name__ == "__main__":
    # 确保目标目录存在
    if not os.path.exists(unknown_type):
        os.makedirs(unknown_type)

    train_self = pd.read_csv('baseline/UNSW_NB15/check/self/train_self_new.csv')
    train_self = train_self.sample(n=10000, random_state=42)
    train_nonself = pd.read_csv(f'baseline/UNSW_NB15/check/train/trainset_{unknown_type}_nonself.csv')
    
    test_self = pd.read_csv(f'baseline/UNSW_NB15/check/self/test_self_new.csv')
    test_self = test_self.sample(n=5000, random_state=42)
    test_nonself = pd.read_csv(f'baseline/UNSW_NB15/check/nonself/test_nonself.csv')
    test_nonself = test_nonself.sample(n=5000, random_state=42)
    
    # 加载未知样本
    train_set_unknown = pd.read_csv(f'baseline/UNSW_NB15/check/unknown/train/train{unknown_type}.csv')
    test_set_unknown = pd.read_csv(f'baseline/UNSW_NB15/check/unknown/test/test{unknown_type}.csv')
    unknown = pd.concat([train_set_unknown, test_set_unknown])
    
    # 初始化和训练模型
    model = VDetector(self_radius=0.1, max_detectors=10000, coverage_threshold=0.999)
    model.fit(train_self.values)
    
    # 预测
    self_predictions = model.predict(test_self.values)
    nonself_predictions = model.predict(test_nonself.values)
    unknown_predictions = model.predict(unknown.values)
    
    # 评估模型性能
    metrics = model.evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, 
                                          test_self, test_nonself, unknown)
    
    print("性能评估结果:")
    for metric, value in metrics.items():
        if metric == "Confusion Matrix":
            print(f"{metric}:\n{value}")
        else:
            print(f"{metric}: {value:.2%}")
    