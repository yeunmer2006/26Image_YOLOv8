"""
训练 YOLOv8 模型
"""
from ultralytics import YOLO
import config


def train_model(resume=False):
    """
    使用 SKU-110K 数据集训练 YOLOv8 模型

    Args:
        resume: 是否从上次中断处继续训练

    Returns:
        results: 训练结果对象
    """
    # 加载预训练模型（如果 resume=True，加载 last.pt）
    if resume:
        model_path = f"{config.PROJECT_ROOT}/yolo_finetune/{config.RUN_NAME}/weights/last.pt"
        model = YOLO(model_path)
        print(f"从断点恢复: {model_path}")
    else:
        model = YOLO(config.MODEL_NAME)
        print(f"已加载模型: {config.MODEL_NAME}")

    # 项目保存路径（Colab 下自动保存到 Google Drive）
    project_path = f"{config.PROJECT_ROOT}/yolo_finetune"

    # 开始训练（关闭 Mosaic 增强用于数据增强消融实验）
    results = model.train(
        data=config.DATA_YAML,
        epochs=config.EPOCHS,
        imgsz=config.IMG_SIZE,
        batch=config.BATCH_SIZE,
        device=config.DEVICE,
        lr0=config.LR0,
        lrf=config.LRF,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY,
        warmup_epochs=config.WARMUP_EPOCHS,
        project=project_path,
        name=config.RUN_NAME,
        save=True,
        save_period=config.SAVE_PERIOD,
        patience=config.PATIENCE,
        verbose=True,
        mosaic=config.MOSAIC,  # 关闭 Mosaic 增强（数据增强消融实验）
        resume=resume,         # 断点续训
    )

    print("训练完成!")
    print(f"最佳模型路径: {model.trainer.best}")
    return results


if __name__ == '__main__':
    import sys
    resume = '--resume' in sys.argv
    train_model(resume=resume)