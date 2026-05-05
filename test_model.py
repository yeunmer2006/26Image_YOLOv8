"""
模型测试脚本 - 预训练/微调后通用
"""
from ultralytics import YOLO
import config
import os
from pathlib import Path


def test_model(model_path, save_dir, model_name='pretrained'):
    """
    测试模型性能并保存结果

    Args:
        model_path: 模型权重路径
        save_dir: 结果保存目录
        model_name: 模型名称标识

    Returns:
        metrics: 评估指标字典
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 加载模型
    model = YOLO(model_path)
    print(f"已加载模型: {model_path}")

    # 在验证集上评估
    print("正在评估模型...")
    metrics = model.val(
        data=config.DATA_YAML,
        batch=config.BATCH_SIZE,
        imgsz=config.IMG_SIZE,
        device=config.DEVICE,
        save=True,
        save_json=True,
        project=save_dir,
        name=model_name,
    )

    # 打印关键指标
    print(f"\n{'='*50}")
    print(f"模型: {model_name}")
    print(f"{'='*50}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"{'='*50}\n")

    # 返回指标字典便于对比
    return {
        'mAP50': metrics.box.map50,
        'mAP50-95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr,
    }


def get_model_path(is_pretrained=True):
    """
    获取模型路径

    Args:
        is_pretrained: True=预训练模型, False=微调后模型

    Returns:
        model_path: 模型路径
    """
    if is_pretrained:
        return config.MODEL_NAME
    else:
        # 微调后模型路径
        best_path = f"{config.PROJECT_NAME}/{config.RUN_NAME}/weights/best.pt"
        last_path = f"{config.PROJECT_NAME}/{config.RUN_NAME}/weights/last.pt"

        if os.path.exists(best_path):
            return best_path
        elif os.path.exists(last_path):
            return last_path
        else:
            raise FileNotFoundError(f"微调模型未找到: {best_path} 或 {last_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='测试 YOLOv8 模型')
    parser.add_argument('--pretrained', action='store_true',
                        help='测试预训练模型')
    parser.add_argument('--finetuned', action='store_true',
                        help='测试微调后模型')
    args = parser.parse_args()

    if args.pretrained or not args.finetuned:
        # 测试预训练模型
        model_path = get_model_path(is_pretrained=True)
        test_model(model_path, config.SAVE_DIR, 'pretrained')

    if args.finetuned:
        # 测试微调后模型
        model_path = get_model_path(is_pretrained=False)
        test_model(model_path, 'results/after', 'finetuned')