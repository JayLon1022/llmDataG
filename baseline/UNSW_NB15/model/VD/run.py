# unknown_types = ["A"]
unknown_types = ["A", "B", "D", "E", "F", "G", "R", "S", "W"]
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt

# 定义V-detector
class VDetector:
    def __init__(self, self_radius, coverage_threshold=0.999, max_detectors=10000):
        self.self_radius = self_radius
        self.coverage_threshold = coverage_threshold
        self.max_detectors = max_detectors
        self.detectors = []
        
    def _calculate_detector_radius(self, candidate, self_samples):
        distances = cdist([candidate], self_samples)[0]
        return np.min(distances) - self.self_radius
    
    def _is_covered_by_detectors(self, point, detectors):
        if not detectors:
            return False
        
        centers = np.array([d[0] for d in detectors])
        radii = np.array([d[1] for d in detectors])
        
        distances = cdist([point], centers)[0]
        return np.any(distances <= radii)
    
    def generate_detectors(self, self_samples):
        # 确保self_samples是numpy数组并且是正确的格式
        self_samples = np.array(self_samples)
        attempts = 0
        max_attempts = self.max_detectors * 100000
        
        while len(self.detectors) < self.max_detectors and attempts < max_attempts:
            # 生成候选检测器时确保是2D数组
            candidate = np.random.uniform(0, 1, size=(1, self_samples.shape[1]))
            
            # 计算候选检测器到所有自体样本的距离
            distances_to_self = cdist(candidate, self_samples)[0]
            
            if np.all(distances_to_self > self.self_radius):
                radius = self._calculate_detector_radius(candidate, self_samples)
                
                if radius > 0 and not self._is_covered_by_detectors(candidate[0], self.detectors):
                    self.detectors.append((candidate[0], radius))
            
            attempts += 1
            
            coverage = self._estimate_coverage()
            
            if coverage >= self.coverage_threshold:
                break
                
        print(f"Generated {len(self.detectors)} detectors after {attempts} attempts")
        print(f"Final coverage: {self._estimate_coverage():.2%}")
        
        # 将检测器保存为CSV文件
        centers = np.array([d[0] for d in self.detectors])
        radii = np.array([d[1] for d in self.detectors])
        
        # 创建包含中心点和半径的DataFrame
        detector_df = pd.DataFrame(centers)
        detector_df.columns = [f'center_{i}' for i in range(centers.shape[1])]
        detector_df['radius'] = radii
        
        # 保存为CSV文件
        detector_df.to_csv(f'{unknown_type}/detectors_{unknown_type}.csv', index=False)
        print(f"Detectors saved to detectors_{unknown_type}.csv")
        
    def _estimate_coverage(self, sample_size=420000):
        if not self.detectors:
            return 0.0

        samples = np.random.uniform(0, 1, size=(sample_size, 42))
        
        
        covered = sum(self._is_covered_by_detectors(s, self.detectors) for s in samples)
        return covered / sample_size
    
    def predict(self, samples):
        samples = np.array(samples)
        results = np.zeros(len(samples), dtype=bool)
        
        for i, sample in enumerate(samples):
            results[i] = self._is_covered_by_detectors(sample, self.detectors)
            
        return results

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

self_radius = 0.1
self_count = 10000

for unknown_type in unknown_types:
    train_nonself = pd.read_csv(f"../../check/train/trainset_{unknown_type}_nonself.csv")
    train_self = pd.read_csv("../../check/self/train_self_new.csv").sample(frac=1, random_state=42)
    train_self = train_self.sample(n = self_count, random_state=42)

    # 测试集自体数量
    test_self = pd.read_csv("../../check/self/test_self_new.csv")
    test_self = test_self.sample(n = 5000, random_state=42)

    # 测试集非自体数量
    test_nonself = pd.read_csv("../../check/nonself/test_nonself.csv")
    test_nonself = test_nonself.sample(n = 5000, random_state=42)

    train_set_unknown = pd.read_csv(f'../../check/unknown/train/train{unknown_type}.csv')
    test_set_unknown = pd.read_csv(f'../../check/unknown/test/test{unknown_type}.csv')
    unknown = pd.concat([train_set_unknown, test_set_unknown])
    with open(f"{unknown_type}/{unknown_type}_results.txt", "w") as f:
        f.write(f"Unknown type: {unknown_type}\n")
        
    # 初始化和训练检测器
    detector = VDetector(self_radius=self_radius)
    detector.generate_detectors(train_self)

    # 预测
    self_predictions = detector.predict(test_self)
    nonself_predictions = detector.predict(test_nonself)
    unknown_predictions = detector.predict(unknown)
    metrics = evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, test_self, test_nonself, unknown)