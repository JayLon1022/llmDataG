unknown_types = ["dos", "probe", "r2l", "u2r"]
import pandas as pd
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

# 定义V-detector
class VDetector:
    def __init__(self, self_radius, coverage_threshold=0.999, max_detectors=50000):
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
    
    def generate_detectors(self, self_samples, output_dir):
    # 确保目录存在
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        self_samples = np.array(self_samples)
        batch_size = 5000  
        attempts = 0
        max_attempts = self.max_detectors * 100
        
        start_time = time.time()
        pbar = tqdm(total=self.max_detectors, desc=f"generating detectors (r={self.self_radius})")
        
        while len(self.detectors) < self.max_detectors and attempts < max_attempts:
            # 批量生成候选点
            candidates = np.random.uniform(0, 1, size=(batch_size, self_samples.shape[1]))
            
            for candidate in candidates:
                radius = self._calculate_detector_radius(candidate, self_samples)
                
                if radius > 0 and not self._is_covered_by_detectors(candidate, self.detectors):
                    self.detectors.append((candidate, radius))
                    pbar.update(1)

                    # 每生成100个检测器检查一次覆盖率
                    if len(self.detectors) % 100 == 0:
                        coverage = self._estimate_coverage()
                        pbar.set_postfix({"coverage": f"{coverage:.2%}"})
                        
                        if coverage >= self.coverage_threshold:
                            break
                        
            attempts += batch_size
        
            
            if len(self.detectors) >= self.max_detectors or self._estimate_coverage() >= self.coverage_threshold:
                break
        
        pbar.close()
        elapsed_time = time.time() - start_time
        
        final_coverage = self._estimate_coverage()
        
        print(f"Generated {len(self.detectors)} detectors after {attempts} attempts")
        print(f"Final coverage: {final_coverage:.2%}")
        
        # 将检测器保存为CSV文件
        centers = np.array([d[0] for d in self.detectors])
        radii = np.array([d[1] for d in self.detectors])
        
        # 创建包含中心点和半径的DataFrame
        detector_df = pd.DataFrame(centers)
        detector_df.columns = [f'center_{i}' for i in range(centers.shape[1])]
        detector_df['radius'] = radii
        
        # 保存为CSV文件 - 使用os.path.join确保路径正确
        csv_path = os.path.join(output_dir, f'detectors_r{self.self_radius}.csv')
        detector_df.to_csv(csv_path, index=False)
        print(f"Detectors saved to {csv_path}")
        
        return final_coverage
        
    def _estimate_coverage(self, sample_size=420000):
        if not self.detectors:
            return 0.0

        samples = np.random.uniform(0, 1, size=(sample_size, len(self.detectors[0][0])))
      
        covered = sum(self._is_covered_by_detectors(s, self.detectors) for s in samples)
        return covered / sample_size
    
    def predict(self, samples):
        samples = np.array(samples)
        results = np.zeros(len(samples), dtype=bool)
        
        for i, sample in enumerate(samples):
            results[i] = self._is_covered_by_detectors(sample, self.detectors)
            
        return results

# 评价指标
def evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, test_self, test_nonself, unknown, output_dir, self_radius):
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
    
    # 使用os.path.join确保路径正确    
    results_path = os.path.join(output_dir, "results.txt")
    with open(results_path, "w") as f:
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
    plt.title(f'V-detector Performance Metrics (r={self_radius})')
    plt.ylabel('Value')
    plt.xticks(rotation=45, ha='right')

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "metrics_bar_chart.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return metrics


def ablation_study(self_radii, unknown_type, self_count):
    """
    对自体半径进行消融实验
    
    参数:
    self_radii: 要测试的自体半径列表
    unknown_type: 未知类型
    self_count: 使用的自体样本数量
    """
    results = {}
    
    print(f"加载数据集: {unknown_type}")
    train_nonself = pd.read_csv(f"../../check/train/seed_{unknown_type}.csv")
    train_self_full = pd.read_csv("../../check/self/train_self.csv").sample(frac=1, random_state=42)
    train_self = train_self_full.sample(n=min(self_count, len(train_self_full)), random_state=42)

    test_self = pd.read_csv("../../check/self/test_self.csv")
    test_self = test_self.sample(n=5000, random_state=42)

    test_nonself = pd.read_csv("../../check/nonself/test_nonself.csv")
    test_nonself = test_nonself.sample(n=5000, random_state=42)

    unknown = pd.read_csv(f'../../check/unknown/4type/{unknown_type}.csv')
    
    all_metrics = {}

    # 创建保存目录
    base_dir = f'{unknown_type}/radius_ablation'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(f"{base_dir}/ablation_results.txt", "w", encoding="utf-8") as f:
        f.write(f"自体半径消融实验 (未知类型: {unknown_type}, 自体样本数量: {self_count})\n")
        f.write("=" * 50 + "\n")
    
    for radius in self_radii:
        print(f"\n测试自体半径: {radius}")
        
        # 为每个半径创建单独的目录
        radius_dir = f"{base_dir}/radius_{radius}"
        if not os.path.exists(radius_dir):
            os.makedirs(radius_dir)
        
        # 初始化和训练检测器
        detector = VDetector(self_radius=radius)
        detector.generate_detectors(train_self, radius_dir)

        # 预测
        self_predictions = detector.predict(test_self)
        nonself_predictions = detector.predict(test_nonself)
        unknown_predictions = detector.predict(unknown)
        
        # 评估性能
        metrics = evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, 
                                      test_self, test_nonself, unknown, 
                                      radius_dir, radius)
        
        # 保存结果
        all_metrics[radius] = metrics
        
        # 将结果追加到消融实验文件
        with open(f"{base_dir}/ablation_results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n自体半径: {radius}\n")
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    f.write(f"{metric}:\n{value}\n")
                else:
                    f.write(f"{metric}: {value:.2%}\n")
            f.write("-" * 50 + "\n")
    
    # 绘制消融实验结果
    plot_radius_ablation(all_metrics, unknown_type, self_count, base_dir)
    
    # 找出最佳自体半径
    best_radius = find_best_radius(all_metrics)
    
    with open(f"{base_dir}/best_radius.txt", "w", encoding="utf-8") as f:
        f.write(f"最佳自体半径: {best_radius}\n")
        f.write(f"自体样本数量: {self_count}\n")
        f.write("最佳指标:\n")
        for metric, value in all_metrics[best_radius].items():
            if metric == "Confusion Matrix":
                f.write(f"{metric}:\n{value}\n")
            else:
                f.write(f"{metric}: {value:.2%}\n")
    
    return all_metrics, best_radius

def plot_radius_ablation(all_metrics, unknown_type, self_count, base_dir):
    """绘制自体半径消融实验结果"""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "Unknown Coverage Rate"]
    
    plt.figure(figsize=(15, 10))
    
    for metric in metrics_to_plot:
        radii = sorted(list(all_metrics.keys()))
        values = [all_metrics[r][metric] for r in radii]
        
        plt.plot(radii, values, marker='o', label=metric)
    
    plt.xlabel('自体半径')
    plt.ylabel('指标值')
    plt.title(f'V-Detector 自体半径消融实验 (类型: {unknown_type}, 样本数量: {self_count})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/ablation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def find_best_radius(all_metrics):
    """找出最佳自体半径"""
    best_score = -1
    best_radius = None
    
    for radius, metrics in all_metrics.items():
        # 使用F1分数和未知覆盖率的加权和作为评分标准
        score = 0.8 * metrics["F1 Score"] + 0.5 * metrics["Unknown Coverage Rate"] - 0.3 * metrics["False Positive Rate"]
        
        if score > best_score:
            best_score = score
            best_radius = radius
    
    return best_radius



def ablation_study_self_count(self_radius, self_counts, unknown_type):
    """
    对自体样本数量进行消融实验
    
    参数:
    self_radius: 固定的自体半径
    self_counts: 要测试的自体样本数量列表
    unknown_type: 未知类型
    """
    results = {}
    
    print(f"加载数据集: {unknown_type}")
    train_nonself = pd.read_csv(f"../../check/train/seed_{unknown_type}.csv")
    train_self_full = pd.read_csv("../../check/self/train_self.csv").sample(frac=1, random_state=42)

    test_self = pd.read_csv("../../check/self/test_self.csv")
    test_self = test_self.sample(n=5000, random_state=42)

    test_nonself = pd.read_csv("../../check/nonself/test_nonself.csv")
    test_nonself = test_nonself.sample(n=5000, random_state=42)

    unknown = pd.read_csv(f'../../check/unknown/4type/{unknown_type}.csv')
    
    all_metrics = {}

    # 创建保存目录
    base_dir = f'{unknown_type}/self_count_ablation'
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    with open(f"{base_dir}/ablation_results.txt", "w", encoding="utf-8") as f:
        f.write(f"自体样本数量消融实验 (未知类型: {unknown_type}, 自体半径: {self_radius})\n")
        f.write("=" * 50 + "\n")
    
    for count in self_counts:
        print(f"\n测试自体样本数量: {count}")
        # 为每个数量创建单独的目录
        count_dir = f"{base_dir}/count_{count}"
        if not os.path.exists(count_dir):
            os.makedirs(count_dir)
        
        # 从完整训练集中采样指定数量的自体样本
        train_self = train_self_full.sample(n=min(count, len(train_self_full)), random_state=42)
        
        # 初始化和训练检测器
        detector = VDetector(self_radius=self_radius)
        detector.generate_detectors(train_self, count_dir)

        # 预测
        self_predictions = detector.predict(test_self)
        nonself_predictions = detector.predict(test_nonself)
        unknown_predictions = detector.predict(unknown)
        
        # 评估性能
        metrics = evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, 
                                      test_self, test_nonself, unknown, 
                                      count_dir, self_radius)
        
        # 保存结果
        all_metrics[count] = metrics
        
        # 将结果追加到消融实验文件
        with open(f"{base_dir}/ablation_results.txt", "a", encoding="utf-8") as f:
            f.write(f"\n自体样本数量: {count}\n")
            for metric, value in metrics.items():
                if metric == "Confusion Matrix":
                    f.write(f"{metric}:\n{value}\n")
                else:
                    f.write(f"{metric}: {value:.2%}\n")
            f.write("-" * 50 + "\n")
    
    # 绘制消融实验结果 - 修复这里，传入base_dir参数
    plot_self_count_ablation(all_metrics, unknown_type, self_radius, base_dir)
    
    # 找出最佳自体样本数量
    best_count = find_best_self_count(all_metrics)
    
    with open(f"{base_dir}/best_count.txt", "w", encoding="utf-8") as f:
        f.write(f"最佳自体样本数量: {best_count}\n")
        f.write(f"自体半径: {self_radius}\n")
        f.write("最佳指标:\n")
        for metric, value in all_metrics[best_count].items():
            if metric == "Confusion Matrix":
                f.write(f"{metric}:\n{value}\n")
            else:
                f.write(f"{metric}: {value:.2%}\n")
    
    return all_metrics, best_count

def plot_self_count_ablation(all_metrics, unknown_type, self_radius, base_dir):

    """绘制自体样本数量消融实验结果"""
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "Unknown Coverage Rate"]
    
    plt.figure(figsize=(15, 10))
    
    for metric in metrics_to_plot:
        counts = sorted(list(all_metrics.keys()))
        values = [all_metrics[c][metric] for c in counts]
        
        plt.plot(counts, values, marker='o', label=metric)
    
    plt.xlabel('自体样本数量')
    plt.ylabel('指标值')
    plt.title(f'V-Detector 自体样本数量消融实验 (类型: {unknown_type}, 自体半径: {self_radius})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{base_dir}/ablation_results.png', dpi=300, bbox_inches='tight')
    plt.close()

def find_best_self_count(all_metrics):
    """找出最佳自体样本数量"""
    best_score = -1
    best_count = None
    
    for count, metrics in all_metrics.items():
        # 使用F1分数和未知覆盖率的加权和作为评分标准
        score = 0.8 * metrics["F1 Score"] + 0.5 * metrics["Unknown Coverage Rate"] - 0.3 * metrics["False Positive Rate"]
        
        if score > best_score:
            best_score = score
            best_count = count
    
    return best_count

# 双参数消融实验
def dual_ablation_study(self_radii, self_counts, unknown_type):
    """
    同时对自体半径和自体样本数量进行消融实验
    
    参数:
    self_radii: 要测试的自体半径列表
    self_counts: 要测试的自体样本数量列表
    unknown_type: 未知类型
    """
    results = {}
    
    print(f"加载数据集: {unknown_type}")
    train_nonself = pd.read_csv(f"../../check/train/seed_{unknown_type}.csv")
    train_self_full = pd.read_csv("../../check/self/train_self.csv").sample(frac=1, random_state=42)

    test_self = pd.read_csv("../../check/self/test_self.csv")
    test_self = test_self.sample(n=5000, random_state=42)

    test_nonself = pd.read_csv("../../check/nonself/test_nonself.csv")
    test_nonself = test_nonself.sample(n=5000, random_state=42)

    unknown = pd.read_csv(f'../../check/unknown/4type/{unknown_type}.csv')
    
    # 创建保存目录
    if not os.path.exists(f'{unknown_type}/dual_ablation'):
        os.makedirs(f'{unknown_type}/dual_ablation')
    
    # 记录所有参数组合的结果
    all_results = {}
    
    with open(f"{unknown_type}/dual_ablation/results.txt", "w", encoding="utf-8") as f:
        f.write(f"自体半径和样本数量双参数消融实验 (未知类型: {unknown_type})\n")
        f.write("=" * 60 + "\n")
    
    # 创建结果矩阵用于热力图
    f1_scores = np.zeros((len(self_radii), len(self_counts)))
    unknown_coverage = np.zeros((len(self_radii), len(self_counts)))
    combined_scores = np.zeros((len(self_radii), len(self_counts)))
    
    for i, radius in enumerate(self_radii):
        all_results[radius] = {}
        
        for j, count in enumerate(self_counts):
            print(f"\n测试参数组合: 自体半径={radius}, 自体样本数量={count}")
            
            # 从完整训练集中采样指定数量的自体样本
            train_self = train_self_full.sample(n=min(count, len(train_self_full)), random_state=42)
            
            # 初始化和训练检测器
            detector = VDetector(self_radius=radius)
            detector.generate_detectors(train_self, f"{unknown_type}/dual_ablation/r{radius}_c{count}")

            # 预测
            self_predictions = detector.predict(test_self)
            nonself_predictions = detector.predict(test_nonself)
            unknown_predictions = detector.predict(unknown)
            
            # 评估性能
            metrics = evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, 
                                          test_self, test_nonself, unknown, 
                                          f"{unknown_type}/dual_ablation/r{radius}_c{count}", radius)
            
            # 保存结果
            all_results[radius][count] = metrics
            
            # 更新结果矩阵
            f1_scores[i, j] = metrics["F1 Score"]
            unknown_coverage[i, j] = metrics["Unknown Coverage Rate"]
            combined_scores[i, j] = 0.7 * metrics["F1 Score"] + 0.3 * metrics["Unknown Coverage Rate"]
            
            # 将结果追加到文件
            with open(f"{unknown_type}/dual_ablation/results.txt", "a", encoding="utf-8") as f:
                f.write(f"\n自体半径: {radius}, 自体样本数量: {count}\n")
                for metric, value in metrics.items():
                    if metric == "Confusion Matrix":
                        f.write(f"{metric}:\n{value}\n")
                    else:
                        f.write(f"{metric}: {value:.2%}\n")
                f.write("-" * 60 + "\n")
    
    # 绘制热力图
    plot_heatmaps(self_radii, self_counts, f1_scores, unknown_coverage, combined_scores, unknown_type)
    
    # 找出最佳参数组合
    best_radius, best_count = find_best_parameters(combined_scores, self_radii, self_counts)
    
    with open(f"{unknown_type}/dual_ablation/best_parameters.txt", "w", encoding="utf-8") as f:
        f.write(f"最佳参数组合:\n")
        f.write(f"自体半径: {best_radius}\n")
        f.write(f"自体样本数量: {best_count}\n")
        f.write("\n最佳指标:\n")
        for metric, value in all_results[best_radius][best_count].items():
            if metric == "Confusion Matrix":
                f.write(f"{metric}:\n{value}\n")
            else:
                f.write(f"{metric}: {value:.2%}\n")
    
    return all_results, best_radius, best_count

def plot_heatmaps(self_radii, self_counts, f1_scores, unknown_coverage, combined_scores, unknown_type):
    """绘制热力图展示双参数消融实验结果"""
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # F1分数热力图
    im0 = axes[0].imshow(f1_scores, cmap='viridis', interpolation='nearest')
    axes[0].set_title('F1 Score')
    axes[0].set_xlabel('自体样本数量')
    axes[0].set_ylabel('自体半径')
    axes[0].set_xticks(np.arange(len(self_counts)))
    axes[0].set_yticks(np.arange(len(self_radii)))
    axes[0].set_xticklabels(self_counts)
    axes[0].set_yticklabels(self_radii)
    plt.colorbar(im0, ax=axes[0])
    
    # 未知覆盖率热力图
    im1 = axes[1].imshow(unknown_coverage, cmap='viridis', interpolation='nearest')
    axes[1].set_title('Unknown Coverage Rate')
    axes[1].set_xlabel('自体样本数量')
    axes[1].set_ylabel('自体半径')
    axes[1].set_xticks(np.arange(len(self_counts)))
    axes[1].set_yticks(np.arange(len(self_radii)))
    axes[1].set_xticklabels(self_counts)
    axes[1].set_yticklabels(self_radii)
    plt.colorbar(im1, ax=axes[1])
    
    # 综合得分热力图
    im2 = axes[2].imshow(combined_scores, cmap='viridis', interpolation='nearest')
    axes[2].set_title('Combined Score (0.7*F1 + 0.3*Coverage)')
    axes[2].set_xlabel('自体样本数量')
    axes[2].set_ylabel('自体半径')
    axes[2].set_xticks(np.arange(len(self_counts)))
    axes[2].set_yticks(np.arange(len(self_radii)))
    axes[2].set_xticklabels(self_counts)
    axes[2].set_yticklabels(self_radii)
    plt.colorbar(im2, ax=axes[2])
    
    # 添加数值标签
    for i in range(len(self_radii)):
        for j in range(len(self_counts)):
            axes[0].text(j, i, f"{f1_scores[i, j]:.2f}", ha="center", va="center", color="w" if f1_scores[i, j] < 0.7 else "black")
            axes[1].text(j, i, f"{unknown_coverage[i, j]:.2f}", ha="center", va="center", color="w" if unknown_coverage[i, j] < 0.7 else "black")
            axes[2].text(j, i, f"{combined_scores[i, j]:.2f}", ha="center", va="center", color="w" if combined_scores[i, j] < 0.7 else "black")
    
    plt.tight_layout()
    plt.savefig(f'{unknown_type}/dual_ablation/heatmaps.png', dpi=300, bbox_inches='tight')
    plt.close()

def find_best_parameters(combined_scores, self_radii, self_counts):
    """找出最佳参数组合"""
    max_idx = np.unravel_index(np.argmax(combined_scores), combined_scores.shape)
    best_radius = self_radii[max_idx[0]]
    best_count = self_counts[max_idx[1]]
    return best_radius, best_count




self_radii = np.arange(0.001, 5.001, 0.001)
self_counts = np.arange(1, 5000, 1)


for unknown_type in unknown_types:
    print(f"\n开始处理未知类型: {unknown_type}")
        
    # 1. 首先进行自体数量消融实验（使用中等自体半径）
    initial_radius = 0.01  # 使用中等自体半径作为初始值
    print(f"\n使用初始自体半径 {initial_radius} 执行自体数量消融实验...")
    count_metrics, best_count = ablation_study_self_count(initial_radius, self_counts, unknown_type)
    print(f"\n找到最佳自体数量: {best_count}")
    
    # 2. 使用最佳自体数量进行自体半径消融实验
    print(f"\n使用最佳自体数量 {best_count} 执行自体半径消融实验...")
    radius_metrics, best_radius = ablation_study(self_radii, unknown_type, self_count=best_count)
    print(f"\n找到最佳自体半径: {best_radius}")
    
    # 3. 可选：使用找到的最佳参数再次验证
    print(f"\n使用最佳参数组合进行最终验证 (半径={best_radius}, 数量={best_count})...")
    
    # 加载数据
    train_nonself = pd.read_csv(f"../../check/train/seed_{unknown_type}.csv")
    train_self_full = pd.read_csv("../../check/self/train_self.csv").sample(frac=1, random_state=42)
    train_self = train_self_full.sample(n=min(best_count, len(train_self_full)), random_state=42)

    test_self = pd.read_csv("../../check/self/test_self.csv")
    test_self = test_self.sample(n=5000, random_state=42)

    test_nonself = pd.read_csv("../../check/nonself/test_nonself.csv")
    test_nonself = test_nonself.sample(n=5000, random_state=42)

    unknown = pd.read_csv(f'../../check/unknown/4type/{unknown_type}.csv')
   
    
    # 使用最佳参数训练最终模型
    final_detector = VDetector(self_radius=best_radius)
    final_detector.generate_detectors(train_self, f"{unknown_type}/final_model")

    # 预测
    self_predictions = final_detector.predict(test_self)
    nonself_predictions = final_detector.predict(test_nonself)
    unknown_predictions = final_detector.predict(unknown)
    
    # 评估最终性能
    final_metrics = evaluate_performance(self_predictions, nonself_predictions, unknown_predictions, 
                                        test_self, test_nonself, unknown, 
                                        f"{unknown_type}/final_model", best_radius)
    
    
    # 将最终结果写入文件
    final_results_path = os.path.join(f"{unknown_type}", "final_results.txt")
    with open(final_results_path, "w", encoding="utf-8") as f:
        f.write(f"未知类型 {unknown_type} 的最终最佳参数:\n")
        f.write(f"自体半径: {best_radius}\n")
        f.write(f"自体样本数量: {best_count}\n")
        f.write(f"最佳F1分数: {final_metrics['F1 Score']:.2%}\n")
        f.write(f"最佳未知覆盖率: {final_metrics['Unknown Coverage Rate']:.2%}\n")
        f.write(f"误报率: {final_metrics['False Positive Rate']:.2%}\n")
        f.write(f"准确率: {final_metrics['Accuracy']:.2%}\n")
        f.write(f"精确率: {final_metrics['Precision']:.2%}\n")


