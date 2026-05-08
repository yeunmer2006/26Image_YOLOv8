"""
YOLOv8 微调主运行脚本
按顺序执行: 预训练测试 -> 训练 -> 微调后测试 -> 结果对比

使用说明:
1. 修改 config.py 中的参数（如需要）
2. 准备额外5张测试图片到 test_images/ 目录
3. 运行: python run.py
"""
import os
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

import config
from train import train_model
from test_model import test_model, get_model_path


def main():
    print("="*70)
    print("YOLOv8 模型微调流程")
    print("="*70)
    print(f"模型: {config.MODEL_NAME}")
    print(f"数据集: {config.DATA_YAML}")
    print(f"训练轮数: {config.EPOCHS}")
    print(f"图像大小: {config.IMG_SIZE}")
    print(f"批量大小: {config.BATCH_SIZE}")
    print("="*70)

    # 步骤1: 测试预训练模型（在数据集验证集上）
    print("\n[步骤 1/4] 测试预训练模型...")
    print("-"*50)
    try:
        pretrained_path = get_model_path(is_pretrained=True)
        print(f"预训练模型路径: {pretrained_path}")

        # 由于预训练模型是原始 COCO 权重
        # 直接进入训练阶段

        print("预训练模型测试跳过（将在额外图片上测试）")
    except Exception as e:
        print(f"预训练模型加载失败: {e}")
        print("继续进行训练...")

    # 步骤2: 训练模型
    print("\n[步骤 2/4] 开始训练模型...")
    print("-"*50)
    try:
        train_results = train_model()
        print("训练完成!")
    except Exception as e:
        print(f"训练失败: {e}")
        sys.exit(1)

    # 步骤3: 测试微调后模型
    print("\n[步骤 3/4] 测试微调后模型...")
    print("-"*50)
    try:
        finetuned_path = get_model_path(is_pretrained=False)
        print(f"微调模型路径: {finetuned_path}")

        # 在验证集上评估
        from ultralytics import YOLO
        model = YOLO(finetuned_path)
        metrics = model.val(
            data=config.DATA_YAML,
            batch=config.BATCH_SIZE,
            imgsz=config.IMG_SIZE,
            device=config.DEVICE,
            save=True,
            project='results/after',
            name='finetuned',
        )

        print(f"\nmAP50: {metrics.box.map50:.4f}")
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"Precision: {metrics.box.mp:.4f}")
        print(f"Recall: {metrics.box.mr:.4f}")

    except Exception as e:
        print(f"微调后测试失败: {e}")
        print("请手动运行 test_model.py 进行测试")

    # 步骤4: 结果对比分析
    print("\n[步骤 4/4] 生成对比报告...")
    print("-"*50)
    try:
        from compare_results import extract_final_metrics, compare_metrics, plot_comparison, generate_report

        train_dir = f"{config.PROJECT_NAME}/{config.RUN_NAME}"
        after_metrics = extract_final_metrics(train_dir)

        if after_metrics:
            # 预训练模型指标（基于默认 COCO 训练的预期值，这里设为None表示无对比）
            before_metrics = None

            compare_metrics(before_metrics, after_metrics)
            plot_comparison(before_metrics, after_metrics)
            generate_report(before_metrics, after_metrics)
        else:
            print("未找到训练结果指标")

    except Exception as e:
        print(f"对比分析失败: {e}")

    print("\n" + "="*70)
    print("流程完成!")
    print("="*70)
    print("\n后续步骤:")
    print("1. 运行 python test_extra_images.py 测试额外5张图片")
    print("2. 查看 results/ 目录下的训练和测试结果")
    print("3. 如需调整参数，修改 config.py 后重新运行")


if __name__ == '__main__':
    main()