import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(42)
# 参数设置
SELF_RADIUS = 0.05
SELF_COUNT = 100
EXPECTED_COVERAGE = 0.999
LOCAL_LEARNING_FACTOR = 2.0  # C1
GLOBAL_LEARNING_FACTOR = 2.0  # C2
PERTURBATION_FACTOR = 0.4    # δ
NUM_PARTICLES = 100
MAX_ITERATIONS = 100
VELOCITY_RANGE = [-0.2, 0.2]
INERTIA_WEIGHT_START = 0.9
INERTIA_WEIGHT_END = 0.4

# 数据加载
def load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path, unknown_path):
    
    train_self = pd.read_csv(train_self_path)
    # train_self = train_self.sample(random_state=42)
    train_nonself = pd.read_csv(train_nonself_path)
    unknown = pd.read_csv(unknown_path)
    
    # 加载测试数据
    test_self = pd.read_csv(test_self_path)
    test_self = test_self.sample(n=5000, random_state=42)
    test_nonself = pd.read_csv(test_nonself_path)
    test_nonself = test_nonself.sample(n=5000, random_state=42)
    
    # 添加标签：自体为0，非自体为1
    train_self['label'] = 0
    train_nonself['label'] = 1
    test_self['label'] = 0
    test_nonself['label'] = 1
    unknown['label'] = 1
    
    # 合并训练集和测试集
    train_data = pd.concat([train_self, train_nonself], axis=0).reset_index(drop=True)
    test_data = pd.concat([test_self, test_nonself], axis=0).reset_index(drop=True)

    return train_data, test_data, unknown

# 数据预处理函数
def preprocess_data(train_data, test_data,unknown):
    # 处理缺失值
    train_data = train_data.fillna(train_data.mean())
    test_data = test_data.fillna(test_data.mean())
    unknown = unknown.fillna(unknown.mean())
    # 分离特征和标签
    X_train = train_data.drop('label', axis=1).values
    y_train = train_data['label'].values
    X_test = test_data.drop('label', axis=1).values
    y_test = test_data['label'].values
    X_unknown = unknown.drop('label', axis=1).values
    y_unknown = unknown['label'].values
    
    return X_train, y_train, X_test, y_test, X_unknown, y_unknown

# 计算欧几里得距离
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

# 初始化粒子群
def initialize_particles(num_particles, dim):
    positions = np.random.uniform(0, 1, (num_particles, dim))  # 位置
    velocities = np.random.uniform(VELOCITY_RANGE[0], VELOCITY_RANGE[1], (num_particles, dim))  # 速度
    pbest = positions.copy()  # 个体最优位置
    gbest = np.random.uniform(0, 1, dim)  # 全局最优位置（初始随机）
    return positions, velocities, pbest, gbest

# 适应度函数：检测器与自身样本的最小距离
def fitness(particle, self_data):
    min_dist = min([euclidean_distance(particle, s) for s in self_data])
    return min_dist

# 更新速度和位置
def update_velocity_position(positions, velocities, pbest, gbest, iteration, max_iterations):
    w = INERTIA_WEIGHT_START - (INERTIA_WEIGHT_START - INERTIA_WEIGHT_END) * (iteration / max_iterations)
    r1, r2 = np.random.rand(), np.random.rand()
    velocities = (w * velocities +
                 LOCAL_LEARNING_FACTOR * r1 * (pbest - positions) +
                 GLOBAL_LEARNING_FACTOR * r2 * (gbest - positions))
    velocities = np.clip(velocities, VELOCITY_RANGE[0], VELOCITY_RANGE[1])
    positions = positions + velocities
    positions = np.clip(positions, 0, 1)  # 保持在 [0, 1] 范围内
    return positions, velocities

# 第一阶段：候选检测器生成
def generate_candidate_detectors(self_data, num_particles, dim):
    positions, velocities, pbest, gbest = initialize_particles(num_particles, dim)
    detectors = []
    radii = []  # 存储每个检测器的半径
    coverage = 0
    iteration = 0

    while coverage < EXPECTED_COVERAGE and iteration < MAX_ITERATIONS:
        for i in range(num_particles):
            fit_value = fitness(positions[i], self_data)
            if fit_value > SELF_RADIUS:  # 自容忍过程
                detector_radius = fit_value - SELF_RADIUS  # 检测半径 = 到最近自身的距离 - 自半径
                detectors.append(positions[i].copy())
                radii.append(detector_radius)
            if fit_value > fitness(pbest[i], self_data):
                pbest[i] = positions[i].copy()
            if fit_value > fitness(gbest, self_data):
                gbest = positions[i].copy()

        positions, velocities = update_velocity_position(positions, velocities, pbest, gbest, iteration, MAX_ITERATIONS)
        coverage = len(detectors) / num_particles
        iteration += 1

    return np.array(detectors), np.array(radii), gbest

# 第二阶段：变异填充漏洞
def mutate_detectors(detectors, radii, self_data, perturbation_factor):
    mutated_detectors = []
    mutated_radii = []
    for detector, radius in zip(detectors, radii):
        mutation = detector + np.random.uniform(-perturbation_factor, perturbation_factor, detector.shape)
        mutation = np.clip(mutation, 0, 1)
        fit_value = fitness(mutation, self_data)
        if fit_value > SELF_RADIUS:
            mutated_detectors.append(mutation)
            mutated_radii.append(fit_value - SELF_RADIUS)  # 更新变异后的半径
    return np.array(mutated_detectors), np.array(mutated_radii)

# DGA-PSO 主函数
def dga_pso(self_data, num_particles, dim):
    start_time = time.time()
    candidate_detectors, candidate_radii, gbest = generate_candidate_detectors(self_data, num_particles, dim)
    mature_detectors, mature_radii = mutate_detectors(candidate_detectors, candidate_radii, self_data, PERTURBATION_FACTOR)
    tolerance_time = time.time() - start_time
    return mature_detectors, mature_radii, tolerance_time

# 检测函数
def detect(test_data, detectors, radii):
    predictions = []
    for sample in test_data:
        is_anomaly = any(euclidean_distance(sample, d) < r for d, r in zip(detectors, radii))
        predictions.append(1 if is_anomaly else 0)
    return np.array(predictions)

# 评估指标
def evaluate(y_true, y_pred, y_unknown, u_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)  # 准确率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # 精确率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0  # 召回率
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0  # F1分数
    far = fp / (fp + tn) if (fp + tn) > 0 else 0  # 误报率
    uc = np.mean(u_pred == 1)  # 未知覆盖率
    return acc, precision, recall, f1, far, uc

# 参数实验
def parameter_experiment(X_train, y_train, X_test, y_test, X_unknown, y_unknown):
    # 实验参数范围
    self_radius_range = np.arange(0.01, 0.1, 0.01)
    self_count_range = np.arange(10, 200, 10)
    
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
            global SELF_RADIUS, SELF_COUNT
            SELF_RADIUS = radius
            SELF_COUNT = count
            
            # 获取自身数据
            self_data = X_train[y_train == 0][:count]
            # 运行DGA-PSO
            detectors, radii, _ = dga_pso(self_data, NUM_PARTICLES, X_train.shape[1])
            
            # 检测
            y_pred = detect(X_test, detectors, radii)
            u_pred = detect(X_unknown, detectors, radii)
            
            # 评估
            acc, prec, rec, f1, far, uc = evaluate(y_test, y_pred, y_unknown, u_pred)
            
            # 存储结果
            results['self_radius'].append(radius)
            results['self_count'].append(count)
            results['accuracy'].append(acc)
            results['precision'].append(prec)
            results['recall'].append(rec)
            results['f1'].append(f1)
            results['far'].append(far)
            results['uc'].append(uc)
            
    return pd.DataFrame(results)

# 主程序
def main():
    unknown_types = ["A", "B", "D", "E", "F", "G", "R", "S", "W"]
    for unknown_type in unknown_types:
        # 加载数据
        train_self_path = '../../check/self/train_self.csv'
        train_nonself_path = f'../../check/train/seed_{unknown_type}.csv'
        test_self_path = '../../check/self/test_self.csv'
        test_nonself_path = '../../check/nonself/test_nonself.csv'
        unknown_path = f'../../check/unknown/{unknown_type}.csv'
       
        train_data, test_data, unknown = load_data(train_self_path, train_nonself_path, test_self_path, test_nonself_path, unknown_path)
        
        X_train, y_train, X_test, y_test, X_unknown, y_unknown = preprocess_data(train_data, test_data, unknown)
        
        
        # 分离自身（正常）和非自身（攻击）
        self_data = X_train[y_train == 0]  

        # 运行 DGA-PSO
        detectors, radii, tolerance_time = dga_pso(self_data, NUM_PARTICLES, X_train.shape[1])
        print(f"Number of detectors: {len(detectors)}")
        print(f"Tolerance time: {tolerance_time:.2f} seconds")

        # 检测测试集
        y_pred = detect(X_test, detectors, radii)
        u_pred = detect(X_unknown, detectors,radii)
        
        # 评估
        acc, precision, recall, f1, far, uc = evaluate(y_test, y_pred, y_unknown,u_pred)
        with open(f"DGA-PSO_{unknown_type}.txt", 'w') as f:
            f.write(f"Accuracy: {acc:.2%}\n")
            f.write(f"Precision: {precision:.2%}\n") 
            f.write(f"Recall: {recall:.2%}\n")
            f.write(f"F1 Score: {f1:.2%}\n")
            f.write(f"False Alarm Rate: {far:.2%}\n")
            f.write(f"Unknown Coverage: {uc:.2%}\n")
        # 运行参数实验
        results_df = parameter_experiment(X_train, y_train, X_test, y_test, X_unknown, y_unknown)
        
        
        # 找到最佳参数组合
        best_result = results_df.loc[results_df['f1'].idxmax()]
        
        # 保存结果
        with open(f"DGA-PSO_{unknown_type}_best_params.txt", 'w') as f:
            f.write(f"Best Parameters:\n")
            f.write(f"Self Radius: {best_result['self_radius']:.3f}\n")
            f.write(f"Self Count: {int(best_result['self_count'])}\n")
            f.write(f"Accuracy: {best_result['accuracy']:.2%}\n")
            f.write(f"Precision: {best_result['precision']:.2%}\n")
            f.write(f"Recall: {best_result['recall']:.2%}\n")
            f.write(f"F1 Score: {best_result['f1']:.2%}\n")
            f.write(f"False Alarm Rate: {best_result['far']:.2%}\n")
            f.write(f"Unknown Coverage: {best_result['uc']:.2%}\n")
            
if __name__ == "__main__":
    main()