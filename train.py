"""
训练 YOLOv8 模型
"""
from ultralytics import YOLO
import config


def train_model():
    """
    使用 SKU-110K 数据集训练 YOLOv8 模型

    Returns:
        results: 训练结果对象
    """
    # 加载预训练模型
    model = YOLO(config.MODEL_NAME)
    print(f"已加载模型: {config.MODEL_NAME}")

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
        project=config.PROJECT_NAME,
        name=config.RUN_NAME,
        save=True,
        save_period=config.SAVE_PERIOD,
        patience=config.PATIENCE,
        verbose=True,
        mosaic=config.MOSAIC,  # 关闭 Mosaic 增强（数据增强消融实验）
    )

    print("训练完成!")
    print(f"最佳模型路径: {model.trainer.best}")
    return results


if __name__ == '__main__':
    train_model()