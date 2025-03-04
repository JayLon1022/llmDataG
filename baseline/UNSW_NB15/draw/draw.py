import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_model_comparison(models, metrics, values, title="Model Performance Comparison", figsize=(12, 8), save_path=None):
    """
    绘制不同模型各项指标的对比图
    
    参数:
    models: list, 模型名称列表
    metrics: list, 评估指标名称列表
    values: list of lists, 每个模型对应的各项指标值
    title: str, 图表标题
    figsize: tuple, 图表大小
    save_path: str, 保存路径，如果为None则不保存

    """
    # 创建DataFrame以便使用seaborn
    data = []
    for i, model in enumerate(models):
        for j, metric in enumerate(metrics):
            data.append({
                'model': model,
                'metric': metric,
                'value': values[i][j]
            })
    df = pd.DataFrame(data)
    
    # 设置样式
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)
    
    # 绘制条形图
    ax = sns.barplot(x='model', y='value', hue='metric', data=df)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('model', fontsize=14)
    plt.ylabel('value', fontsize=14)
    plt.ylim(0, 1.0)  # 假设指标值在0-1之间
    
    # 在条形上方添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # 调整图例位置
    plt.legend(title='metric', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()

def plot_model_comparison_radar(models, metrics, values, title="model performance radar chart", figsize=(10, 8), save_path=None):
    """
    绘制不同模型各项指标的雷达图对比
    
    参数:
    models: list, 模型名称列表
    metrics: list, 评估指标名称列表
    values: list of lists, 每个模型对应的各项指标值
    title: str, 图表标题
    figsize: tuple, 图表大小
    save_path: str, 保存路径，如果为None则不保存
    """
    # 设置图表
    plt.figure(figsize=figsize)
    
    # 计算雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合雷达图
    
    # 设置极坐标图
    ax = plt.subplot(111, polar=True)
    
    # 添加每个模型的数据
    for i, model in enumerate(models):
        values_model = values[i]
        values_model += values_model[:1]  # 闭合雷达图
        ax.plot(angles, values_model, 'o-', linewidth=2, label=model)
        ax.fill(angles, values_model, alpha=0.1)
    
    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    
    # 设置y轴范围
    ax.set_ylim(0, 1)
    
    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title(title, fontsize=16)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()

def plot_model_comparison_heatmap(models, metrics, values, title="model performance heatmap", figsize=(10, 8), save_path=None):
    """
    绘制不同模型各项指标的热力图对比
    
    参数:
    models: list, 模型名称列表
    metrics: list, 评估指标名称列表
    values: list of lists, 每个模型对应的各项指标值
    title: str, 图表标题
    figsize: tuple, 图表大小
    save_path: str, 保存路径，如果为None则不保存
    """
    # 创建数据矩阵
    data_matrix = np.array(values)
    
    # 设置图表
    plt.figure(figsize=figsize)
    
    # 绘制热力图
    ax = sns.heatmap(data_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                     xticklabels=metrics, yticklabels=models, vmin=0, vmax=1)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # plt.show()


# DL模型数据
dl_models = ['LLM', 'CNN+LSTM', 'RNN+LSTM', 'GRU+LSTM']
dl_values_A = [
    [1, 0.9048, 0.9051, 0.9044, 0.9048, 0.0948],  # LLM的各项指标
    [0.5976, 0.7912, 0.2652, 0.3972, 0.6220, 0.0700],  # CNN+LSTM的各项指标 
    [0.9333, 0.9026, 0.9714, 0.9357, 0.9981, 0.1048],  # RNN+LSTM的各项指标
    [0.9325, 0.9044, 0.9672, 0.9348, 0.9978, 0.1022]   # GRU+LSTM的各项指标
]
dl_values_B = [
    [0.9983, 0.8987, 0.9093, 0.8858, 0.8974, 0.0884],  # LLM的各项指标
    [0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000],  # CNN+LSTM的各项指标 
    [0.9785, 0.8767, 0.8941, 0.8546, 0.8739, 0.1012],  # RNN+LSTM的各项指标
    [0.9717, 0.9101, 0.9034, 0.9184, 0.9108, 0.0982]   # GRU+LSTM的各项指标
]
dl_values_D = [
    [0.9819, 0.9024, 0.9118, 0.8910, 0.9013, 0.0862],  # LLM的各项指标
    [0.8617, 0.8199, 0.7583, 0.9392, 0.8391, 0.2994],  # CNN+LSTM的各项指标 
    [0.8620, 0.8932, 0.9294, 0.8510, 0.8885, 0.0646],  # RNN+LSTM的各项指标
    [0.9705, 0.9195, 0.9048, 0.9376, 0.9209, 0.0986]   # GRU+LSTM的各项指标
]
dl_values_E = [
    [0.8968, 0.9000, 0.9073, 0.8910, 0.8991, 0.0910],  # LLM的各项指标
    [0.8266, 0.8456, 0.8068, 0.9088, 0.8548, 0.2176],  # CNN+LSTM的各项指标 
    [0.3837, 0.7829, 0.8558, 0.6804, 0.7581, 0.1146],  # RNN+LSTM的各项指标
    [0.4200, 0.8242, 0.8692, 0.7632, 0.8128, 0.1148]   # GRU+LSTM的各项指标
]
dl_values_F = [
    [0.7494, 0.8987, 0.9151, 0.8790, 0.8967, 0.0816],  # LLM的各项指标
    [0.3636, 0.7070, 0.7269, 0.6632, 0.6936, 0.2492],  # CNN+LSTM的各项指标 
    [0.8031, 0.8441, 0.8984, 0.7760, 0.8327, 0.0878],  # RNN+LSTM的各项指标
    [0.8761, 0.8488, 0.8965, 0.7886, 0.8391, 0.0910]   # GRU+LSTM的各项指标
]
dl_values_G = [
    [0.9969, 0.9127, 0.9144, 0.9106, 0.9125, 0.0852],  # LLM的各项指标
    [0.9937, 0.8240, 0.9028, 0.7262, 0.8049, 0.0782],  # CNN+LSTM的各项指标 
    [0.9984, 0.9121, 0.9091, 0.9158, 0.9124, 0.0916],  # RNN+LSTM的各项指标
    [0.9973, 0.8903, 0.8932, 0.8866, 0.8899, 0.1060]   # GRU+LSTM的各项指标
]
dl_values_R = [
    [0.9233, 0.9125, 0.9141, 0.9106, 0.9123, 0.0856],  # LLM的各项指标
    [0.7735, 0.8906, 0.8846, 0.8984, 0.8914, 0.1172],  # CNN+LSTM的各项指标 
    [0.9476, 0.8887, 0.9259, 0.8450, 0.8836, 0.0676],  # RNN+LSTM的各项指标
    [0.7699, 0.7966, 0.8653, 0.7026, 0.7755, 0.1094]   # GRU+LSTM的各项指标
]
dl_values_S = [
    [1.0000, 0.9201, 0.9144, 0.9270, 0.9206, 0.0868],  # LLM的各项指标
    [0.0000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000],  # CNN+LSTM的各项指标 
    [0.4964, 0.9016, 0.9226, 0.8768, 0.8991, 0.0736],  # RNN+LSTM的各项指标
    [0.5083, 0.8892, 0.9068, 0.8676, 0.8868, 0.0892]   # GRU+LSTM的各项指标
]
dl_values_W = [
    [1.0000, 0.9066, 0.9129, 0.8990, 0.9059, 0.0858],  # LLM的各项指标
    [0.8391, 0.9315, 0.8817, 0.9968, 0.9357, 0.1338],  # CNN+LSTM的各项指标 
    [0.9828, 0.8656, 0.9099, 0.8116, 0.8579, 0.0804],  # RNN+LSTM的各项指标
    [0.9713, 0.7900, 0.8620, 0.6906, 0.7668, 0.1106]   # GRU+LSTM的各项指标
]

# ML模型数据
ml_models = ['LLM','SVM', 'Random Forest', 'Decision Tree', 'Logistic Regression', 'XGBoost']
ml_values_A = [
    [1, 0.9048, 0.9051, 0.9044, 0.9048, 0.0948],  # LLM的各项指标
    [0.9907, 0.9346, 0.9249, 0.9460, 0.9353, 0.0768],  # SVM的各项指标
    [0.9940, 0.9989, 0.9996, 0.9982, 0.9989, 0.0004],  # Random Forest的各项指标
    [0.7755, 0.6781, 0.9944, 0.3582, 0.5267, 0.0020],  # Decision Tree的各项指标
    [0.9955, 0.9267, 0.9129, 0.9434, 0.9279, 0.0900],  # Logistic Regression的各项指标
    [0.9892, 0.9961, 0.9984, 0.9938, 0.9961, 0.0016]   # XGBoost的各项指标
]
ml_values_B = [
    [0.9983, 0.8987, 0.9093, 0.8858, 0.8974, 0.0884],  # LLM的各项指标
    [0.9897, 0.9339, 0.9245, 0.9450, 0.9346, 0.0772],  # SVM的各项指标
    [0.9751, 0.9974, 0.9996, 0.9952, 0.9974, 0.0004],  # Random Forest的各项指标
    [0.7673, 0.6787, 0.9961, 0.3588, 0.5276, 0.0014],  # Decision Tree的各项指标
    [0.9888, 0.9297, 0.9192, 0.9422, 0.9306, 0.0828],  # Logistic Regression的各项指标
    [0.9760, 0.9976, 0.9996, 0.9956, 0.9976, 0.0004]   # XGBoost的各项指标
]
ml_values_D = [
    [0.9819, 0.9024, 0.9118, 0.8910, 0.9013, 0.0862],  # LLM的各项指标
    [0.9782, 0.9312, 0.9242, 0.9394, 0.9318, 0.0770],  # SVM的各项指标
    [0.9985, 0.9986, 0.9996, 0.9976, 0.9986, 0.0004],  # Random Forest的各项指标
    [0.8361, 0.6766, 0.9884, 0.3574, 0.5250, 0.0042],  # Decision Tree的各项指标
    [0.9688, 0.9339, 0.9248, 0.9446, 0.9346, 0.0768],  # Logistic Regression的各项指标
    [0.9926, 0.9967, 0.9996, 0.9938, 0.9967, 0.0004]   # XGBoost的各项指标
]
ml_values_E = [
    [0.8968, 0.9000, 0.9073, 0.8910, 0.8991, 0.0910],  # LLM的各项指标
    [0.9907, 0.9346, 0.9249, 0.9460, 0.9353, 0.0768],  # SVM的各项指标 (使用A类型数据，因为没有E类型数据)
    [0.9952, 0.9969, 0.9998, 0.9940, 0.9969, 0.0002],  # Random Forest的各项指标
    [0.9274, 0.6755, 0.9861, 0.3560, 0.5231, 0.0050],  # Decision Tree的各项指标
    [0.4950, 0.8722, 0.9134, 0.8224, 0.8655, 0.0780],  # Logistic Regression的各项指标
    [0.9910, 0.9961, 0.9984, 0.9938, 0.9961, 0.0016]   # XGBoost的各项指标
]
ml_values_F = [
    [0.7494, 0.8987, 0.9151, 0.8790, 0.8967, 0.0816],  # LLM的各项指标
    [0.5610, 0.8985, 0.9449, 0.8464, 0.8929, 0.0494],  # SVM的各项指标
    [0.9993, 0.9993, 0.9998, 0.9988, 0.9993, 0.0002],  # Random Forest的各项指标
    [0.8981, 0.6756, 0.9944, 0.3532, 0.5213, 0.0020],  # Decision Tree的各项指标
    [0.8844, 0.9110, 0.9225, 0.8974, 0.9098, 0.0754],  # Logistic Regression的各项指标
    [0.9975, 0.9961, 0.9984, 0.9938, 0.9961, 0.0016]   # XGBoost的各项指标
]
ml_values_G = [
    [0.9969, 0.9127, 0.9144, 0.9106, 0.9125, 0.0852],  # LLM的各项指标
    [0.9986, 0.9348, 0.9254, 0.9458, 0.9355, 0.0762],  # SVM的各项指标
    [1.0000, 0.9994, 0.9994, 0.9994, 0.9994, 0.0006],  # Random Forest的各项指标
    [0.6890, 0.6766, 0.9884, 0.3574, 0.5250, 0.0042],  # Decision Tree的各项指标
    [0.9982, 0.9321, 0.9167, 0.9506, 0.9333, 0.0864],  # Logistic Regression的各项指标
    [0.9999, 0.9967, 0.9996, 0.9938, 0.9967, 0.0004]   # XGBoost的各项指标
]
ml_values_R = [
    [0.9233, 0.9125, 0.9141, 0.9106, 0.9123, 0.0856],  # LLM的各项指标
    [0.7906, 0.9349, 0.9250, 0.9466, 0.9357, 0.0768],  # SVM的各项指标
    [0.9989, 0.9992, 0.9996, 0.9988, 0.9992, 0.0004],  # Random Forest的各项指标
    [0.9357, 0.8030, 0.9925, 0.6106, 0.7561, 0.0046],  # Decision Tree的各项指标
    [0.9215, 0.9314, 0.9171, 0.9486, 0.9326, 0.0858],  # Logistic Regression的各项指标
    [0.9990, 0.9961, 0.9984, 0.9938, 0.9961, 0.0016]   # XGBoost的各项指标
]
ml_values_S = [
    [1.0000, 0.9201, 0.9144, 0.9270, 0.9206, 0.0868],  # LLM的各项指标
    [0.4950, 0.9349, 0.9250, 0.9466, 0.9357, 0.0768],  # SVM的各项指标
    [0.4964, 0.9992, 0.9996, 0.9988, 0.9992, 0.0004],  # Random Forest的各项指标
    [0.1317, 0.8030, 0.9925, 0.6106, 0.7561, 0.0046],  # Decision Tree的各项指标
    [0.4964, 0.9314, 0.9171, 0.9486, 0.9326, 0.0858],  # Logistic Regression的各项指标
    [0.0126, 0.9961, 0.9984, 0.9938, 0.9961, 0.0016]   # XGBoost的各项指标
]
ml_values_W = [
    [1.0000, 0.9066, 0.9129, 0.8990, 0.9059, 0.0858],  # LLM的各项指标
    [0.9598, 0.9349, 0.9250, 0.9466, 0.9357, 0.0768],  # SVM的各项指标
    [0.8793, 0.9992, 0.9996, 0.9988, 0.9992, 0.0004],  # Random Forest的各项指标
    [0.7126, 0.8030, 0.9925, 0.6106, 0.7561, 0.0046],  # Decision Tree的各项指标
    [0.9713, 0.9314, 0.9171, 0.9486, 0.9326, 0.0858],  # Logistic Regression的各项指标
    [0.5172, 0.9961, 0.9984, 0.9938, 0.9961, 0.0016]   # XGBoost的各项指标
]

# AIS模型数据
als_model = ['LLM','VD', 'VD+KNN', 'VD+SVM', 'VD+RF', 'VD+XGB']
als_values_A = [
    [1, 0.9048, 0.9051, 0.9044, 0.9048, 0.0948],  # LLM的各项指标
    [0.2069, 0.5369, 0.8773, 0.0858, 0.1563, 0.0012],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]
als_values_B = [
    [0.9983, 0.8987, 0.9093, 0.8858, 0.8974, 0.0884],  # LLM的各项指标
    [0.8536, 0.6660, 0.6613, 0.6806, 0.6708, 0.3486],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)

]
als_values_D = [
    [0.9819, 0.9024, 0.9118, 0.8910, 0.9013, 0.0862],  # LLM的各项指标
    [0.9063, 0.5797, 0.5605, 0.7388, 0.6374, 0.5794],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]
als_values_E = [
    [0.8968, 0.9000, 0.9073, 0.8910, 0.8991, 0.0910],  # LLM的各项指标
    [0.1549, 0.6020, 0.7157, 0.3384, 0.4595, 0.1344],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]
als_values_F = [
    [0.7494, 0.8987, 0.9151, 0.8790, 0.8967, 0.0816],  # LLM的各项指标
    [0.8753, 0.6119, 0.5704, 0.9070, 0.7003, 0.6832],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]
als_values_G = [
    [0.9969, 0.9127, 0.9144, 0.9106, 0.9125, 0.0852],  # LLM的各项指标
    [0.9971, 0.5772, 0.5469, 0.9002, 0.6804, 0.7458],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]
als_values_R = [
    [0.9233, 0.9125, 0.9141, 0.9106, 0.9123, 0.0856],  # LLM的各项指标
    [0.0749, 0.6101, 0.8469, 0.2688, 0.4081, 0.0486],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]
als_values_S = [
    [1.0000, 0.9201, 0.9144, 0.9270, 0.9206, 0.0868],  # LLM的各项指标
    [0.5255, 0.5809, 0.5515, 0.8668, 0.6741, 0.7050],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]
als_values_W = [
    [1.0000, 0.9066, 0.9129, 0.8990, 0.9059, 0.0858],  # LLM的各项指标
    [0.7529, 0.5024, 0.5036, 0.3320, 0.4002, 0.3272],  # VD的各项指标
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+KNN的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+SVM的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],  # VD+RF的各项指标 (数据缺失)
    [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]   # VD+XGB的各项指标 (数据缺失)
]

metrics = ['Unknown Coverage', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'False Positive Rate']

def plot_all_models(unknown_type):

    if unknown_type == 'A':
        dl_values = dl_values_A
        ml_values = ml_values_A
        als_values = als_values_A
    elif unknown_type == 'B':
        dl_values = dl_values_B
        ml_values = ml_values_B
        als_values = als_values_B
    elif unknown_type == 'D':
        dl_values = dl_values_D
        ml_values = ml_values_D
        als_values = als_values_D
    elif unknown_type == 'E':
        dl_values = dl_values_E
        ml_values = ml_values_E
        als_values = als_values_E
    elif unknown_type =='F':
        dl_values = dl_values_F
        ml_values = ml_values_F
        als_values = als_values_F
    elif unknown_type =='G':
        dl_values = dl_values_G
        ml_values = ml_values_G
        als_values = als_values_G
    elif unknown_type =='R':
        dl_values = dl_values_R
        ml_values = ml_values_R
        als_values = als_values_R
    elif unknown_type =='S':
        dl_values = dl_values_S
        ml_values = ml_values_S
        als_values = als_values_S
    elif unknown_type =='W':
        dl_values = dl_values_W
        ml_values = ml_values_W
        als_values = als_values_W

        
    # DL模型    
    plot_model_comparison(dl_models, metrics, dl_values, 
        title=f"UNSW_NB15({unknown_type}) DL Model Comparison", 
        save_path=f"UNSW_NB15/draw/bar/{unknown_type}_dl_model_comparison_bar.png")
    plot_model_comparison_radar(dl_models, metrics, dl_values,
        title=f"UNSW_NB15({unknown_type}) DL Model Comparison Radar",
        save_path=f"UNSW_NB15/draw/radar/{unknown_type}_dl_model_comparison_radar.png")
    plot_model_comparison_heatmap(dl_models, metrics, dl_values,
        title=f"UNSW_NB15({unknown_type}) DL Model Comparison Heatmap",
        save_path=f"UNSW_NB15/draw/heatmap/{unknown_type}_dl_model_comparison_heatmap.png")
    # ML模型
    plot_model_comparison(ml_models, metrics, ml_values,
        title=f"UNSW_NB15({unknown_type}) ML Model Comparison",
        save_path=f"UNSW_NB15/draw/bar/{unknown_type}_ml_model_comparison_bar.png")
    plot_model_comparison_radar(ml_models, metrics, ml_values,
        title=f"UNSW_NB15({unknown_type}) ML Model Comparison Radar",
        save_path=f"UNSW_NB15/draw/radar/{unknown_type}_ml_model_comparison_radar.png")
    plot_model_comparison_heatmap(ml_models, metrics, ml_values,
        title=f"UNSW_NB15({unknown_type}) ML Model Comparison Heatmap",
        save_path=f"UNSW_NB15/draw/heatmap/{unknown_type}_ml_model_comparison_heatmap.png")
    # AIS模型    
    plot_model_comparison(als_model, metrics, als_values,
        title=f"UNSW_NB15({unknown_type}) AIS Model Comparison",
        save_path=f"UNSW_NB15/draw/bar/{unknown_type}_ais_model_comparison_bar.png")
    plot_model_comparison_radar(als_model, metrics, als_values,
        title=f"UNSW_NB15({unknown_type}) AIS Model Comparison Radar",
        save_path=f"UNSW_NB15/draw/radar/{unknown_type}_ais_model_comparison_radar.png")
    plot_model_comparison_heatmap(als_model, metrics, als_values,
        title=f"UNSW_NB15({unknown_type}) AIS Model Comparison Heatmap",
        save_path=f"UNSW_NB15/draw/heatmap/{unknown_type}_ais_model_comparison_heatmap.png")


plot_all_models('A')
# plot_all_models('B')
# plot_all_models('D')
# plot_all_models('E')
# plot_all_models('F')
# plot_all_models('G')
# plot_all_models('R')
# plot_all_models('S')
# plot_all_models('W')
