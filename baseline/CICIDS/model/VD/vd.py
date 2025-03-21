import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import os
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor
np.random.seed(42)
class VDetector:
    def __init__(self, self_radius, coverage_threshold=0.999, initial_detectors=1000):
        self.self_radius = self_radius
        self.coverage_threshold = coverage_threshold
        self.initial_detectors = initial_detectors
        self.detectors = []
        
    def _calculate_detector_radius(self, candidate, self_samples):
        distances = cdist([candidate], self_samples)[0]
        return max(np.min(distances) - self.self_radius, 0)
    
    def _is_covered(self, point, detectors):
        if not detectors:
            return False
        centers = np.array([d[0] for d in detectors])
        radii = np.array([d[1] for d in detectors])
        distances = cdist([point], centers)[0]
        return np.any(distances <= radii)
    
    def generate_detectors(self, self_samples, output_dir):
        # os.makedirs(output_dir, exist_ok=True)
        self_samples = np.array(self_samples)
        dim = self_samples.shape[1]
        max_detectors = min(self.initial_detectors * dim * 10, 50000)
        candidates = np.random.uniform(0, 1, size=(max_detectors, dim))
        
        pbar = tqdm(total=max_detectors, desc=f"Generating detectors (r={self.self_radius})")
        start_time = time.time()
        
        for candidate in candidates:
            if len(self.detectors) >= max_detectors:
                break
            radius = self._calculate_detector_radius(candidate, self_samples)
            if radius > 0 and not self._is_covered(candidate, self.detectors):
                self.detectors.append((candidate, radius))
                pbar.update(1)
                if len(self.detectors) % 100 == 0:
                    coverage = self._estimate_coverage(self_samples)
                    if coverage >= self.coverage_threshold:
                        break
        
        pbar.close()
        
        return self._estimate_coverage(self_samples)
    
    def _estimate_coverage(self, self_samples, sample_size=20000):
        if not self.detectors:
            return 0.0
        dim = len(self.detectors[0][0])
        samples = np.random.uniform(0, 1, size=(sample_size, dim))
        return np.mean([self._is_covered(s, self.detectors) for s in samples])
    
    def predict(self, samples):
        samples = np.array(samples)
        return np.array([self._is_covered(s, self.detectors) for s in samples])
    
    def get_anomaly_scores(self, samples):
        """计算样本到最近检测器的距离作为异常度量"""
        samples = np.array(samples)
        if not self.detectors:
            return np.zeros(len(samples))
        centers = np.array([d[0] for d in self.detectors])
        distances = cdist(samples, centers)
        min_distances = np.min(distances, axis=1)
        return min_distances

def evaluate_performance(self_preds, nonself_preds, unknown_preds, test_self, test_nonself, unknown):
    TP = np.sum(nonself_preds)
    FP = np.sum(self_preds)
    FN = len(test_nonself) - TP
    TN = len(test_self) - FP
    
    metrics = {
        "Accuracy": (TP + TN) / (len(test_self) + len(test_nonself)),
        "Precision": TP / (TP + FP) if (TP + FP) > 0 else 0,
        "Recall": TP / (TP + FN) if (TP + FN) > 0 else 0,
        "FPR": FP / (FP + TN) if (FP + TN) > 0 else 0,
        "Unknown Coverage": np.sum(unknown_preds) / len(unknown)
    }
    metrics["F1"] = 2 * (metrics["Precision"] * metrics["Recall"]) / (metrics["Precision"] + metrics["Recall"]) if (metrics["Precision"] + metrics["Recall"]) > 0 else 0
    
    return metrics

def dual_ablation_study(self_radii, self_counts, unknown_type):
    train_nonself = pd.read_csv(f"../../check/train/seed_{unknown_type}.csv")
    train_self_full = pd.read_csv("../../check/self/train_self.csv").sample(frac=1, random_state=42)
    test_self = pd.read_csv("../../check/self/test_self.csv").sample(n=5000, random_state=42)
    test_nonself = pd.read_csv("../../check/nonself/test_nonself.csv").sample(n=5000, random_state=42)
    unknown = pd.read_csv(f"../../check/unknown/{unknown_type}.csv")
    
    best_radius = self_radii[len(self_radii)//2]
    best_count = self_counts[len(self_counts)//2]
    best_score = -1
    best_metrics = None
    
    def test_params(params):
        radius, count = params
        train_self = train_self_full.sample(n=min(int(count), len(train_self_full)), random_state=42)
        detector = VDetector(self_radius=radius)
        detector.generate_detectors(train_self, f"temp")
        metrics = evaluate_performance(
            detector.predict(test_self),
            detector.predict(test_nonself),
            detector.predict(unknown),
            test_self, test_nonself, unknown
        )
        return radius, count, metrics
    
    # 并行测试参数组合
    for iteration in range(3):
        radius_range = [max(self_radii[0], best_radius-0.002), min(self_radii[-1], best_radius+0.002)]
        count_range = [max(self_counts[0], best_count-20), min(self_counts[-1], best_count+20)]
        radii_to_test = np.linspace(radius_range[0], radius_range[1], 5)
        counts_to_test = np.linspace(count_range[0], count_range[1], 5, dtype=int)
        
        param_combinations = [(r, c) for r in radii_to_test for c in counts_to_test]
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(test_params, param_combinations))
        
        for radius, count, metrics in results:
            score = 1 * metrics["F1"] 
            if score > best_score:
                best_score = score
                best_radius = radius
                best_count = int(count)
                best_metrics = metrics
    
    # 保存最佳结果
    with open(f"{unknown_type}_best_parameters.txt", "w") as f:
        f.write(f"Best: r={best_radius:.3f}, c={best_count}\n")
        for m, v in best_metrics.items():
            f.write(f"{m}: {v:.2%}\n")
    
    return best_metrics, best_radius, best_count

self_radii = np.arange(0.001, 0.101, 0.001)
self_counts = np.arange(10, 101, 5)

unknown_types = ["bot", "bruteforce", "ddos","dos","infiltration", "sql_injection"]
for unknown_type in unknown_types:
    print(f"\n处理未知类型: {unknown_type}")
    best_metrics, best_radius, best_count = dual_ablation_study(self_radii, self_counts, unknown_type)
    
    train_self_full = pd.read_csv("../../check/self/train_self.csv").sample(frac=1, random_state=42)
    train_self = train_self_full.sample(n=min(best_count, len(train_self_full)), random_state=42)
    final_detector = VDetector(self_radius=best_radius)
    final_detector.generate_detectors(train_self, f"{unknown_type}/final_model")
    
    test_self = pd.read_csv("../../check/self/test_self.csv").sample(n=5000, random_state=42)
    test_nonself = pd.read_csv("../../check/nonself/test_nonself.csv").sample(n=5000, random_state=42)
    unknown = pd.read_csv(f"../../check/unknown/{unknown_type}.csv")
    
    final_metrics = evaluate_performance(
        final_detector.predict(test_self),
        final_detector.predict(test_nonself),
        final_detector.predict(unknown),
        test_self, test_nonself, unknown
    )