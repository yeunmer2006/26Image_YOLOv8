"""
额外5张图片测试 - 个人完成部分
使用微调前后的模型对5张外部图片进行测试对比
"""
import json
import os
from pathlib import Path

from ultralytics import YOLO

import config


def test_extra_images(
	model_path,
	save_dir,
	model_name='test',
	conf=None,
	iou=None,
	max_det=None,
):
	"""
	对额外5张图片进行测试

	Args:
		model_path: 模型路径
		save_dir: 结果保存目录
		model_name: 模型标识

	Returns:
		results_list: 测试结果列表
	"""
	conf = config.PREDICT_CONF if conf is None else conf
	iou = config.PREDICT_IOU if iou is None else iou
	max_det = config.PREDICT_MAX_DET if max_det is None else max_det
	output_dir = Path(save_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 加载模型
	model = YOLO(model_path)
	print(f"已加载模型: {model_path}")

	results_list = []
	summary = {
		"model_name": model_name,
		"model_path": str(model_path),
		"imgsz": config.IMG_SIZE,
		"device": str(config.DEVICE),
		"conf": conf,
		"iou": iou,
		"max_det": max_det,
		"images": [],
	}

	for i, img_path in enumerate(config.TEST_IMAGES, 1):
		if not os.path.exists(img_path):
			print(f"图片不存在: {img_path}")
			continue

		# 显式记录阈值和最大检测数，确保结果可以复现
		results = model.predict(
			source=img_path,
			imgsz=config.IMG_SIZE,
			device=config.DEVICE,
			conf=conf,
			iou=iou,
			max_det=max_det,
			save=True,
			project=str(output_dir),
			name=f'{model_name}_img{i}',
			exist_ok=True,
			verbose=False,
		)

		result = results[0]
		results_list.append(result)

		# 获取检测结果
		boxes = result.boxes
		confidences = [float(box.conf[0]) for box in boxes]
		summary["images"].append(
			{
				"image": str(img_path),
				"detection_count": len(boxes),
				"reached_max_det": len(boxes) >= max_det,
				"min_confidence": min(confidences) if confidences else None,
				"max_confidence": max(confidences) if confidences else None,
			}
		)
		print(f"\n图片 {i}: {img_path}")
		print(f"  检测到 {len(boxes)} 个对象")
		if len(boxes) >= max_det:
			print(f"  警告: 检测数已达到 max_det={max_det}，实际候选框可能更多")

		if len(boxes) > 0:
			# 打印类别和置信度
			for box in boxes:
				cls_id = int(box.cls[0])
				box_conf = float(box.conf[0])
				cls_name = model.names[cls_id]
				print(f"    - {cls_name}: {box_conf:.2f}")

	summary_path = output_dir / f"{model_name}_summary.json"
	summary_path.write_text(
		json.dumps(summary, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)
	print(f"汇总文件: {summary_path}")
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
