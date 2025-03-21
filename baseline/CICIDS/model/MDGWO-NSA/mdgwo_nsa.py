import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mutual_info_score, confusion_matrix
from scipy.spatial.distance import euclidean, cdist
from sklearn.cluster import KMeans
import os
import json
from datetime import datetime
import warnings
from sklearn.decomposition import PCA
np.random.seed(42)
SELF_RADIUS = 0.05
SELF_COUNT = 100
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics.cluster._supervised')

# 数据加载
def load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path, unknown_path):
    try:
        train_self = pd.read_csv(train_self_path)
        train_self = train_self.sample(n=SELF_COUNT, random_state=42)
        train_nonself = pd.read_csv(train_nonself_path)
        
        unknown = pd.read_csv(unknown_path)
            
        test_self = pd.read_csv(test_self_path).sample(n=5000, random_state=42)
        test_nonself = pd.read_csv(test_nonself_path).sample(n=5000, random_state=42)

        train_self['label'] = 0
        train_nonself['label'] = 1
        test_self['label'] = 0
        test_nonself['label'] = 1
        unknown['label'] = 1

        train_data = pd.concat([train_self, train_nonself], axis=0).reset_index(drop=True)
        test_data = pd.concat([test_self, test_nonself], axis=0).reset_index(drop=True)
        return train_data, test_data, unknown
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None, None, None

# 数据预处理
def preprocess_data(train_data, test_data, unknown):
    train_data = train_data.fillna(train_data.mean(numeric_only=True))
    test_data = test_data.fillna(test_data.mean(numeric_only=True))
    unknown = unknown.fillna(unknown.mean(numeric_only=True))

    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values
    X_unknown = unknown.drop('label', axis=1).values
    y_unknown = unknown['label'].values
    return X_train, y_train, X_test, y_test, X_unknown, y_unknown

# 特征选择：DP-SUMIC
def dp_sumic_feature_selection(X, y, k=5, threshold_mic=0.8):
    n_features = X.shape[1]
    # 1. 特征聚类
    distances = cdist(X.T, X.T)
    rho = local_density(distances, k)
    delta = np.zeros(n_features)
    for i in range(n_features):
        higher_density = rho > rho[i]
        delta[i] = np.min(distances[i, higher_density]) if higher_density.any() else np.max(distances[i])

    D = (rho / np.max(rho)) * (delta / np.max(delta))
    cluster_centers = np.argsort(-D)[:int(n_features * 0.3)]

    # 2. 类内 SU 和 类间 MIC
    su_scores = symmetric_uncertainty(X, y)
    mic_scores = np.corrcoef(X.T)
    redundant_features = set()
    for i in range(n_features):
        for j in range(i + 1, n_features):
            if mic_scores[i, j] > threshold_mic and su_scores[i] < su_scores[j]:
                redundant_features.add(i)
    
    selected_features = [i for i in range(n_features) if i not in redundant_features]
    return selected_features

def symmetric_uncertainty(X, y):
    su_scores = []
    for i in range(X.shape[1]):
        mi = mutual_info_score(X[:, i], y)
        h_x = entropy(X[:, i])
        h_y = entropy(y)
        su = 2 * mi / (h_x + h_y) if (h_x + h_y) > 0 else 0
        su_scores.append(su)
    return np.array(su_scores)

def entropy(data):
    hist, _ = np.histogram(data, bins=10, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist)) if hist.size > 0 else 0

def local_density(distances, k):
    n_features = distances.shape[0]
    rho = np.zeros(n_features)
    for i in range(n_features):
        sorted_dist = np.sort(distances[i])
        rho[i] = np.sum(np.exp(-sorted_dist[1:k+1]))
    return rho

# 边界检测器生成
def hdbd_boundary_detectors(X_self, grid_size=0.05, max_features=5):
    X_self = np.array(X_self)
    # 降维到 max_features（不超过 5）
    if X_self.shape[1] > max_features:
        pca = PCA(n_components=max_features)
        X_self_reduced = pca.fit_transform(X_self)
    else:
        X_self_reduced = X_self
    n_features = X_self_reduced.shape[1]

    grid_bounds = np.linspace(0, 1, int(1 / grid_size) + 1)
    grid_shape = [int(1 / grid_size)] * n_features
    grid_space = np.zeros(grid_shape, dtype=int)

    for sample in X_self_reduced:
        grid_idx = tuple(np.minimum(np.maximum(np.floor(sample / grid_size).astype(int), 0), np.array(grid_shape) - 1))
        grid_space[grid_idx] += 1

    boundary_detectors = []
    it = np.nditer(grid_space, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        if grid_space[idx] == 0:
            neighbors = get_neighbors(idx, grid_shape)
            if any(grid_space[n] > 0 for n in neighbors if all(0 <= ni < grid_shape[i] for i, ni in enumerate(n))):
                center_reduced = np.array([grid_bounds[i] + grid_size / 2 for i in idx])
                # 还原到原始维度
                center_full = pca.inverse_transform(center_reduced.reshape(1, -1)).flatten()
                distance = min([euclidean(center_full, X_self[i]) for i in range(len(X_self))])
                boundary_detectors.append((center_full, distance))
        it.iternext()
    return boundary_detectors

def get_neighbors(idx, shape):
    neighbors = []
    for dim in range(len(idx)):
        for offset in [-1, 1]:
            new_idx = list(idx)
            new_idx[dim] += offset
            if all(0 <= new_idx[i] < shape[i] for i in range(len(shape))):
                neighbors.append(tuple(new_idx))
    return neighbors

# 非边界探测器生成（MDGWO-NSA）
class MDGWO_NSA:
    def __init__(self, X_self, n_detectors=20, max_iter=50, min_radius=0.01, max_radius=0.3):
        self.X_self = np.array(X_self)
        self.n_detectors = n_detectors
        self.max_iter = max_iter
        self.dim = X_self.shape[1]
        self.min_radius = min_radius
        self.max_radius = max_radius

    def cluster_self(self):
        n_clusters = min(3, len(self.X_self))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(self.X_self)
        centers = kmeans.cluster_centers_
        labels = kmeans.labels_
        radii = []
        for i in range(n_clusters):
            cluster_points = self.X_self[labels == i]
            if len(cluster_points) > 0:
                distances = cdist(centers[i:i+1], cluster_points)
                radii.append(np.max(distances))
            else:
                radii.append(self.min_radius)
        return centers, np.array(radii)

    def fitness(self, detectors, centers, radii):
        overlap = 0
        coverage = 0
        for i, (center, radius) in enumerate(detectors):
            # 重叠惩罚
            for j in range(i + 1, len(detectors)):
                dist = euclidean(detectors[i][0], detectors[j][0])
                if dist < detectors[i][1] + detectors[j][1]:
                    overlap += (detectors[i][1] + detectors[j][1] - dist)

            # 自样本覆盖惩罚
            for x in self.X_self:
                if euclidean(center, x) < radius:
                    overlap += 2.0

            # 半径惩罚
            if radius > self.max_radius:
                overlap += (radius - self.max_radius) * 2

            coverage += radius ** 2

        return coverage - overlap * 2

    def optimize(self):
        centers, radii = self.cluster_self()
        wolves = np.random.rand(self.n_detectors, self.dim)
        wolf_radii = np.random.uniform(self.min_radius, self.max_radius, self.n_detectors)

        alpha, beta, delta = wolves[0], wolves[0], wolves[0]
        alpha_score, beta_score, delta_score = -np.inf, -np.inf, -np.inf

        for t in range(self.max_iter):
            a = 2 * (1 - t / self.max_iter)
            for i in range(self.n_detectors):
                detectors = [(wolves[j], wolf_radii[j]) for j in range(self.n_detectors)]
                score = self.fitness(detectors, centers, radii)

                if score > alpha_score:
                    alpha_score, alpha = score, wolves[i].copy()
                elif score > beta_score:
                    beta_score, beta = score, wolves[i].copy()
                elif score > delta_score:
                    delta_score, delta = score, wolves[i].copy()

                r1, r2 = np.random.rand(2)
                A = 2 * a * r1 - a
                C = 2 * r2

                D_alpha = abs(C * alpha - wolves[i])
                D_beta = abs(C * beta - wolves[i])
                D_delta = abs(C * delta - wolves[i])

                X1 = alpha - A * D_alpha
                X2 = beta - A * D_beta
                X3 = delta - A * D_delta

                wolves[i] = np.clip((X1 + X2 + X3) / 3, 0, 1)
                min_dist_to_self = np.min(cdist(wolves[i].reshape(1, -1), self.X_self))
                wolf_radii[i] = np.clip(min_dist_to_self / 2, self.min_radius, self.max_radius)

        return [(wolves[i], wolf_radii[i]) for i in range(self.n_detectors)]

# 孔洞修复
def hole_repair(detectors, X_nonself, threshold_density=0.2, r_min=0.01, r_max=0.1):
    repaired_detectors = detectors.copy()
    detector_centers = np.array([d[0] for d in detectors])
    detector_radii = np.array([d[1] for d in detectors])

    # 计算密度
    densities = []
    for i, (center, radius) in enumerate(detectors):
        dist_to_nonself = cdist(center.reshape(1, -1), X_nonself)
        density = np.sum(dist_to_nonself < radius) / len(X_nonself)
        densities.append(density)

    # Rp1: 在低密度区域添加新探测器
    for i, density in enumerate(densities):
        if density < threshold_density:
            new_center = detector_centers[i] + np.random.uniform(-0.1, 0.1, size=detector_centers[i].shape)
            new_center = np.clip(new_center, 0, 1)
            min_dist_to_self = np.min(cdist(new_center.reshape(1, -1), X_nonself))
            new_radius = np.clip(min_dist_to_self / 2, r_min, r_max)
            repaired_detectors.append((new_center, new_radius))

    return repaired_detectors

# 检测函数
def detect(test_data, detectors):
    predictions = []
    for sample in test_data:
        is_anomaly = any(euclidean(sample, d[0]) < d[1] for d in detectors)
        predictions.append(1 if is_anomaly else 0)
    return np.array(predictions)

# 评估函数
def evaluate(y_true, y_pred, y_unknown, u_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    uc = np.mean(u_pred == 1)
    return acc, precision, recall, f1, far, uc

# 保存结果
def save_results(unknown_type, metrics, unknown_detection_rate, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    result = {
        "unknown_type": unknown_type,
        "accuracy": metrics[0],
        "precision": metrics[1],
        "recall": metrics[2],
        "f1_score": metrics[3],
        "false_alarm_rate": metrics[4],
        "unknown_detection_rate": unknown_detection_rate,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(f"{output_dir}/results_{unknown_type}.json", "w") as f:
        json.dump(result, f, indent=4)
    print(f"结果已保存到 {output_dir}/results_{unknown_type}.json")

# 参数实验
def parameter_experiment(X_train, y_train, X_test, y_test, X_unknown, y_unknown,unknown_type):
    # 实验参数范围
    self_count_range = np.arange(10, 100, 10)
    self_radius_range = np.arange(0.01, 0.2, 0.01)
    # 存储结果
    results = {
        'self_radius': [],
        'self_count': [],
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'far': [],
        'uc': []
    }
    
    for radius in self_radius_range:
        for count in self_count_range:
            # 更新全局参数
            global SELF_COUNT,SELF_RADIUS
            SELF_COUNT = count
            SELF_RADIUS = radius
            print(f"当前自样本数量: {count}")
            # 在所有操作前应用特征选择
            selected_features = dp_sumic_feature_selection(X_train, y_train, k=5)
            X_train_selected = X_train[:, selected_features]
            X_test_selected = X_test[:, selected_features]
            X_unknown_selected = X_unknown[:, selected_features]
            
            X_self = X_train_selected[y_train == 0]
            X_self = pd.DataFrame(X_self).sample(n=count, random_state=42).values
            X_nonself = X_train_selected[y_train == 1]

            # print("生成边界探测器...")
            boundary_detectors = hdbd_boundary_detectors(X_self, grid_size=0.05, max_features=5)

            # print("使用MDGWO-NSA生成非边界探测器...")
            mdgwo = MDGWO_NSA(X_self, n_detectors=20, max_iter=50)
            nonboundary_detectors = mdgwo.optimize()

            # print("执行孔洞修复...")
            all_detectors = boundary_detectors + nonboundary_detectors
            repaired_detectors = hole_repair(all_detectors, X_nonself)
            # print(f"修复后的探测器数量: {len(repaired_detectors)}")

            # print("评估模型性能...")
            y_pred_test = detect(X_test_selected, repaired_detectors)
            u_pred = detect(X_unknown_selected, repaired_detectors)

            accuracy, precision, recall, f1, fpr, uc = evaluate(y_test, y_pred_test, y_unknown, u_pred)
            # print(f"准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1分数: {f1:.4f}, 误报率: {fpr:.4f}")

            # 存储结果
            results['self_radius'].append(radius)
            results['self_count'].append(count)
            results['accuracy'].append(accuracy)
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['far'].append(fpr)
            results['uc'].append(uc)
            
    return pd.DataFrame(results)

# 主函数
def main():
    unknown_types = ["bot", "bruteforce", "ddos", "dos", "infilteration", "sql_injection"]
    for unknown_type in unknown_types:
        # 加载数据
        train_self_path = '../../check/self/train_self.csv'
        train_nonself_path = f'../../check/train/seed_{unknown_type}.csv'
        test_self_path = '../../check/self/test_self.csv'
        test_nonself_path = '../../check/nonself/test_nonself.csv'
        unknown_path = f'../../check/unknown/{unknown_type}.csv'
       
        train_data, test_data, unknown = load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path, unknown_path)
        if train_data is None:
            continue

        X_train, y_train, X_test, y_test, X_unknown, y_unknown = preprocess_data(train_data, test_data, unknown)

        results_df = parameter_experiment(X_train, y_train, X_test, y_test, X_unknown, y_unknown,unknown_type)
        
        # 找到最佳参数组合
        best_result = results_df.loc[results_df['uc'].idxmax()]
        # 保存结果
        with open(f"MDGWO-NSA_{unknown_type}_best_params.txt", 'w') as f:
            f.write(f"Best Parameters:\n")
            f.write(f"Self Radius: {best_result['self_radius']:.2%}\n")
            f.write(f"Self Count: {int(best_result['self_count'])}\n")
            f.write(f"Accuracy: {best_result['accuracy']:.2%}\n")
            f.write(f"Precision: {best_result['precision']:.2%}\n")
            f.write(f"Recall: {best_result['recall']:.2%}\n")
            f.write(f"F1 Score: {best_result['f1']:.2%}\n")
            f.write(f"False Alarm Rate: {best_result['far']:.2%}\n")
            f.write(f"Unknown Coverage: {best_result['uc']:.2%}\n")
            
if __name__ == '__main__':
    main()