unknown_type = "B"
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import torch
import matplotlib.pyplot as plt
# 定义V-detector
class VDetector:
    def __init__(self, self_radius, max_detectors, threshold=0.999):
        self.self_radius = self_radius
        self.max_detectors = max_detectors
        self.threshold = threshold
        self.detectors = [] 
        
    def _calculate_detector_radius(self, candidate, self_samples):
        distances = cdist([candidate], self_samples)[0]
        return np.min(distances) - self.self_radius
    
    def _is_covered_by_detectors(self, points, detectors):
        if not detectors:
            return np.zeros(len(points), dtype=bool)
        
        # 批量计算距离
        centers = np.array([d[0] for d in detectors])
        radii = np.array([d[1] for d in detectors])
        
        # 使用向量化操作
        distances = cdist(points, centers)
        return np.any(distances <= radii[:, np.newaxis].T, axis=1)
    
    def generate_detectors(self, self_samples):
        self_samples = np.array(self_samples)
        batch_size = 5000  # 增大批量处理数
        attempts = 0
        max_attempts = self.max_detectors * 100
        
        while len(self.detectors) < self.max_detectors and attempts < max_attempts:
            # 批量生成候选点
            candidates = np.random.uniform(
                self_samples.min(axis=0),
                self_samples.max(axis=0),
                size=(batch_size, self_samples.shape[1])
            )
            
            # 批量计算与自体样本的距离
            distances = cdist(candidates, self_samples)
            radii = np.min(distances, axis=1) - self.self_radius
            
            # 筛选有效的检测器
            valid_mask = radii > 0
            valid_candidates = candidates[valid_mask]
            valid_radii = radii[valid_mask]
            
            # 批量检查覆盖情况
            not_covered_mask = ~self._is_covered_by_detectors(valid_candidates, self.detectors)
            
            # 一次性添加所有有效的检测器
            new_detectors = list(zip(valid_candidates[not_covered_mask], 
                                   valid_radii[not_covered_mask]))
            
            # 添加新检测器，但不超过最大数量
            remaining = self.max_detectors - len(self.detectors)
            self.detectors.extend(new_detectors[:remaining])
            
            # 保存当前的检测器到CSV
            if len(self.detectors) > 0:
                current_centers = np.array([d[0] for d in self.detectors])
                current_radii = np.array([d[1] for d in self.detectors])
                
                current_df = pd.DataFrame(current_centers)
                current_df.columns = [f'center_{i}' for i in range(current_centers.shape[1])]
                current_df['radius'] = current_radii
                
                current_df.to_csv(f'detectors_{unknown_type}.csv', index=False)
            attempts += batch_size
            
        print(f"Generated {len(self.detectors)} detectors after {attempts} attempts")
        
        
    
    def _estimate_coverage(self, sample_size=42000):
        if not self.detectors:
            return 0.0

        samples = np.random.uniform(
            np.min([d[0] for d in self.detectors], axis=0),
            np.max([d[0] for d in self.detectors], axis=0),
            size=(sample_size, len(self.detectors[0][0]))
        )
        
        covered = sum(self._is_covered_by_detectors(s, self.detectors) for s in samples)
        return covered / sample_size
    
    def predict(self, samples):
        # 批量预测
        samples = np.array(samples)
        return self._is_covered_by_detectors(samples, self.detectors)
# 评价指标
def evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, test_self, test_nonself, unknown):

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
        
    with open(f"{unknown_type}_results.txt", "w") as f:
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
    plt.savefig(f'{unknown_type}_metrics_bar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


self_radius = 0.066
self_count = 800
train_nonself = pd.read_csv(f"unknown/{unknown_type}_nonself.csv")  # 自体样本
train_self = pd.read_csv("../check/self/train_self_new.csv").sample(frac=1, random_state=42)
train_self = train_self.sample(n = self_count, random_state=42)

# 测试集自体数量
test_self = pd.read_csv("../check/self/test_self_new.csv")
test_self = test_self.sample(n = 5000, random_state=42)

# 测试集非自体数量
test_nonself = pd.read_csv("../check/nonself/test_nonself.csv")
test_nonself = test_nonself.sample(n = 5000, random_state=42)

train_set_unknown = pd.read_csv(f'../check/unknown/train/train{unknown_type}.csv')
test_set_unknown = pd.read_csv(f'../check/unknown/test/test{unknown_type}.csv')
unknown = pd.concat([train_set_unknown, test_set_unknown])
with open(f"{unknown_type}_results.txt", "w") as f:
    f.write(f"Unknown type: {unknown_type}\n")
    
# 初始化和训练检测器
detector = VDetector(self_radius=self_radius, max_detectors=20000, threshold=0.95)
detector.generate_detectors(train_self)

# 预测
self_predictions = detector.predict(test_self)
nonself_predictions = detector.predict(test_nonself)
unknown_predictions = detector.predict(unknown)
metrics = evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, test_self, test_nonself, unknown)    