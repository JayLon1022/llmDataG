import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def plot_model_comparison(models, metrics, values, title="模型性能对比", figsize=(12, 8), save_path=None):
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
                '模型': model,
                '指标': metric,
                '值': values[i][j]
            })
    df = pd.DataFrame(data)
    
    # 设置样式
    sns.set(style="whitegrid")
    plt.figure(figsize=figsize)
    
    # 绘制条形图
    ax = sns.barplot(x='模型', y='值', hue='指标', data=df)
    
    # 添加标题和标签
    plt.title(title, fontsize=16)
    plt.xlabel('模型', fontsize=14)
    plt.ylabel('性能值', fontsize=14)
    plt.ylim(0, 1.0)  # 假设指标值在0-1之间
    
    # 在条形上方添加数值标签
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=10)
    
    # 调整图例位置
    plt.legend(title='评估指标', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_model_comparison_radar(models, metrics, values, title="模型性能雷达图", figsize=(10, 8), save_path=None):
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
    
    plt.show()

def plot_model_comparison_heatmap(models, metrics, values, title="模型性能热力图", figsize=(10, 8), save_path=None):
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
    
    plt.show()
    
    
    # 示例数据
models = ['LLM', 'CNN+LSTM', 'RNN+LSTM', 'GRU+LSTM','SVM','Random Forest','Decision Tree','Logistic Regression','XGBoost']
metrics = ['Unknown Coverage', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'False Positive Rate']
values = [
    [0.92, 0.90, 0.88, 0.89, 0.05],  # SVM的各项指标
    [0.95, 0.94, 0.93, 0.93],  # Random Forest的各项指标
    [0.97, 0.96, 0.95, 0.95],  # XGBoost的各项指标
    [0.94, 0.92, 0.91, 0.91]   # Neural Network的各项指标
]

# 绘制条形图对比
plot_model_comparison(models, metrics, values, title="UNSW_NB15 Model Comparison", save_path="model_comparison_bar.png")

# 绘制雷达图对比
plot_model_comparison_radar(models, metrics, values, title="UNSW_NB15 Model Comparison Radar", save_path="model_comparison_radar.png")

# 绘制热力图对比
plot_model_comparison_heatmap(models, metrics, values, title="UNSW_NB15 Model Comparison Heatmap", save_path="model_comparison_heatmap.png")