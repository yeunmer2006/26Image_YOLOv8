"""
额外5张图片测试 - 个人完成部分
使用微调前后的模型对5张外部图片进行测试对比
"""
from ultralytics import YOLO
import config
import os
from pathlib import Path
import cv2


def test_extra_images(model_path, save_dir, model_name='test'):
    """
    对额外5张图片进行测试

    Args:
        model_path: 模型路径
        save_dir: 结果保存目录
        model_name: 模型标识

    Returns:
        results_list: 测试结果列表
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/images", exist_ok=True)

    # 加载模型
    model = YOLO(model_path)
    print(f"已加载模型: {model_path}")

    results_list = []

    for i, img_path in enumerate(config.TEST_IMAGES, 1):
        if not os.path.exists(img_path):
            print(f"图片不存在: {img_path}")
            continue

        # 执行推理
        results = model.predict(
            source=img_path,
            imgsz=config.IMG_SIZE,
            device=config.DEVICE,
            save=True,
            project=save_dir,
            name=f'img{i}',
            verbose=False,
        )

        results_list.append(results[0])

        # 获取检测结果
        boxes = results[0].boxes
        print(f"\n图片 {i}: {img_path}")
        print(f"  检测到 {len(boxes)} 个对象")

        if len(boxes) > 0:
            # 打印类别和置信度
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                cls_name = model.names[cls_id]
                print(f"    - {cls_name}: {conf:.2f}")

    return results_list


def compare_detections(pretrained_results, finetuned_results):
    """
    对比预训练和微调后的检测结果

    Args:
        pretrained_results: 预训练模型检测结果
        finetuned_results: 微调模型检测结果
    """
    print("\n" + "="*60)
    print("额外5张图片检测结果对比")
    print("="*60)

    total_pretrained = 0
    total_finetuned = 0

    for i, (pre, post) in enumerate(zip(pretrained_results, finetuned_results), 1):
        pre_count = len(pre.boxes)
        post_count = len(post.boxes)
        total_pretrained += pre_count
        total_finetuned += post_count

        print(f"图片 {i}: 预训练={pre_count}个, 微调后={post_count}个")

    print(f"\n总计: 预训练={total_pretrained}个, 微调后={total_finetuned}个")
    print("="*60)


if __name__ == '__main__':
    from test_model import get_model_path

    # 测试预训练模型
    print("="*60)
    print("测试预训练模型 (在额外5张图片上)")
    print("="*60)
    pretrained_path = get_model_path(is_pretrained=True)
    pretrained_results = test_extra_images(
        pretrained_path,
        'results/extra_test',
        'pretrained'
    )

    # 测试微调后模型
    print("\n" + "="*60)
    print("测试微调后模型 (在额外5张图片上)")
    print("="*60)
    finetuned_path = get_model_path(is_pretrained=False)
    finetuned_results = test_extra_images(
        finetuned_path,
        'results/extra_test',
        'finetuned'
    )

    # 对比结果
    if pretrained_results and finetuned_results:
        compare_detections(pretrained_results, finetuned_results)