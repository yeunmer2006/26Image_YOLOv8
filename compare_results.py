"""
结果对比分析
对比微调前后的性能指标并生成报告
"""
import json
import os
from pathlib import Path
import matplotlib.pyplot as plt
import config


def load_metrics(results_dir, model_name):
    """
    从训练结果中加载指标

    Args:
        results_dir: 结果目录
        model_name: 模型名称

    Returns:
        metrics: 指标字典
    """
    # 尝试从 results.json 读取
    results_file = Path(results_dir) / f"{model_name}/results.json"

    if results_file.exists():
        with open(results_file, 'r') as f:
            data = json.load(f)
            # 返回最后一行的指标
            if len(data) > 0:
                return data[-1]

    return None


def extract_final_metrics(results_dir):
    """
    从 results/after 目录提取最终训练指标

    Args:
        results_dir: 训练输出目录

    Returns:
        metrics: 指标字典
    """
    results_file = Path(results_dir) / 'results.csv'

    if not results_file.exists():
        return None

    # 读取最后一行（最终结果）
    with open(results_file, 'r') as f:
        lines = f.readlines()
        if len(lines) > 1:
            # 解析表头
            headers = lines[0].strip().split(',')
            # 解析最后一行数据
            values = lines[-1].strip().split(',')

            metrics = {}
            for h, v in zip(headers, values):
                try:
                    metrics[h.strip()] = float(v)
                except ValueError:
                    metrics[h.strip()] = v

            return metrics

    return None


def compare_metrics(before_metrics, after_metrics):
    """
    对比前后指标并打印

    Args:
        before_metrics: 微调前指标
        after_metrics: 微调后指标
    """
    print("\n" + "="*70)
    print("性能对比报告")
    print("="*70)
    print(f"{'指标':<20} {'微调前':<15} {'微调后':<15} {'提升':<15}")
    print("-"*70)

    # 需要对比的指标
    metrics_to_compare = [
        ('mAP50', 'metrics/mAP50(B)'),
        ('mAP50-95', 'metrics/mAP50-95(B)'),
        ('precision', 'metrics/precision(B)'),
        ('recall', 'metrics/recall(B)'),
    ]

    for name, key in metrics_to_compare:
        before_val = before_metrics.get(key, 0) if before_metrics else 0
        after_val = after_metrics.get(key, 0) if after_metrics else 0

        if before_val > 0:
            improvement = ((after_val - before_val) / before_val) * 100
            improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
        else:
            improvement_str = "N/A"

        print(f"{name:<20} {before_val:<15.4f} {after_val:<15.4f} {improvement_str:<15}")

    print("="*70)


def plot_comparison(before_metrics, after_metrics, save_path='results/comparison/comparison.png'):
    """
    绘制对比柱状图

    Args:
        before_metrics: 微调前指标
        after_metrics: 微调后指标
        save_path: 保存路径
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 指标配置
    metrics_config = [
        ('mAP50', 'mAP@0.5'),
        ('mAP50-95', 'mAP@0.5:0.95'),
        ('precision', 'Precision'),
        ('recall', 'Recall'),
    ]

    keys = ['metrics/mAP50(B)', 'metrics/mAP50-95(B)', 'metrics/precision(B)', 'metrics/recall(B)']

    before_vals = [before_metrics.get(k, 0) if before_metrics else 0 for k in keys]
    after_vals = [after_metrics.get(k, 0) if after_metrics else 0 for k in keys]

    # 绘制
    fig, ax = plt.subplots(figsize=(10, 6))

    x = range(len(metrics_config))
    width = 0.35

    bars1 = ax.bar([i - width/2 for i in x], before_vals, width, label='Pretrained', color='lightblue')
    bars2 = ax.bar([i + width/2 for i in x], after_vals, width, label='Finetuned', color='salmon')

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('YOLOv8 Pretrained vs Finetuned Performance')
    ax.set_xticks(x)
    ax.set_xticklabels([m[1] for m in metrics_config])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"对比图已保存: {save_path}")
    plt.close()


def generate_report(before_metrics, after_metrics, output_file='results/comparison/report.txt'):
    """
    生成文本报告

    Args:
        before_metrics: 微调前指标
        after_metrics: 微调后指标
        output_file: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("YOLOv8 微调性能对比报告\n")
        f.write("="*60 + "\n\n")

        f.write(f"模型: {config.MODEL_NAME}\n")
        f.write(f"数据集: {config.DATA_YAML}\n")
        f.write(f"训练轮数: {config.EPOCHS}\n\n")

        f.write("-"*60 + "\n")
        f.write(f"{'指标':<20} {'微调前':<15} {'微调后':<15} {'提升':<15}\n")
        f.write("-"*60 + "\n")

        metrics_keys = [
            ('mAP50', 'metrics/mAP50(B)'),
            ('mAP50-95', 'metrics/mAP50-95(B)'),
            ('Precision', 'metrics/precision(B)'),
            ('Recall', 'metrics/recall(B)'),
        ]

        for name, key in metrics_keys:
            before_val = before_metrics.get(key, 0) if before_metrics else 0
            after_val = after_metrics.get(key, 0) if after_metrics else 0

            if before_val > 0:
                improvement = ((after_val - before_val) / before_val) * 100
                improvement_str = f"+{improvement:.2f}%" if improvement > 0 else f"{improvement:.2f}%"
            else:
                improvement_str = "N/A"

            f.write(f"{name:<20} {before_val:<15.4f} {after_val:<15.4f} {improvement_str:<15}\n")

        f.write("-"*60 + "\n")

    print(f"报告已保存: {output_file}")


if __name__ == '__main__':
    # 从训练结果中提取指标
    train_dir = f"{config.PROJECT_NAME}/{config.RUN_NAME}"

    # 微调前使用预训练模型在数据集上的默认评估结果
    # 这里假设有 results/before 目录
    before_dir = 'results/before'
    after_dir = f'{train_dir}'

    # 提取指标
    after_metrics = extract_final_metrics(after_dir)

    # 由于预训练模型没有训练历史，使用验证结果
    # 如果有 results/before 的数据可以加载
    before_metrics = None

    if after_metrics:
        # 对比并生成报告
        compare_metrics(before_metrics, after_metrics)
        plot_comparison(before_metrics, after_metrics)
        generate_report(before_metrics, after_metrics)
    else:
        print("未找到微调后的训练结果，请先运行训练")