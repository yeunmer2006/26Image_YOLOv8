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
    # 根据训练模式加载模型
    if config.TRAIN_MODE == 'resume':
        model = YOLO(config.RESUME_WEIGHTS)
        print(f"继续训练，权重: {config.RESUME_WEIGHTS}")
    else:
        model = YOLO(config.MODEL_NAME)
        print(f"从头训练，加载预训练模型: {config.MODEL_NAME}")

    # 开始训练
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
        mosaic=config.MOSAIC,
        resume=(config.TRAIN_MODE == 'resume'),
    )

    print("训练完成!")
    print(f"最佳模型路径: {model.trainer.best}")
    return results


if __name__ == '__main__':
    import sys
    resume = '--resume' in sys.argv
    train_model(resume=resume)