"""
模型测试脚本 - 预训练/微调后通用
"""
import argparse
import json
import os
from pathlib import Path

from ultralytics import YOLO

import config


def test_model(
	model_path,
	save_dir,
	model_name='pretrained',
	split='val',
	imgsz=None,
	batch=None,
	device=None,
):
	"""
	在指定数据划分上评估模型并保存结构化指标

	Args:
		model_path: 模型权重路径
		save_dir: 结果保存目录
		model_name: 模型名称标识
		split: 数据划分，可选 val 或 test
		imgsz: 评估输入尺寸，默认读取 config.IMG_SIZE
		batch: 评估批大小，默认读取 config.BATCH_SIZE
		device: 评估设备，默认读取 config.DEVICE

	Returns:
		metrics: 评估指标字典
	"""
	# 显式固定评估参数，避免验证集和测试集结果混淆
	imgsz = config.IMG_SIZE if imgsz is None else imgsz
	batch = config.BATCH_SIZE if batch is None else batch
	device = config.DEVICE if device is None else device
	output_dir = Path(save_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# 加载模型
	model = YOLO(model_path)
	print(f"已加载模型: {model_path}")

	# 在指定数据划分上评估
	print(f"正在评估模型，数据划分: {split}...")
	metrics = model.val(
		data=config.DATA_YAML,
		split=split,
		batch=batch,
		imgsz=imgsz,
		device=device,
		save=True,
		save_json=True,
		project=str(output_dir),
		name=model_name,
		exist_ok=True,
	)

	# 保存机器可读结果，便于报告统一引用和计算变化量
	result = {
		"model_name": model_name,
		"model_path": str(model_path),
		"data": str(config.DATA_YAML),
		"split": split,
		"imgsz": imgsz,
		"batch": batch,
		"device": str(device),
		"precision": float(metrics.box.mp),
		"recall": float(metrics.box.mr),
		"mAP50": float(metrics.box.map50),
		"mAP50-95": float(metrics.box.map),
	}
	metrics_path = output_dir / f"{model_name}_{split}_metrics.json"
	metrics_path.write_text(
		json.dumps(result, ensure_ascii=False, indent=2),
		encoding="utf-8",
	)

	# 打印关键指标
	print(f"\n{'='*50}")
	print(f"模型: {model_name}")
	print(f"数据划分: {split}")
	print(f"{'='*50}")
	print(f"mAP50: {result['mAP50']:.4f}")
	print(f"mAP50-95: {result['mAP50-95']:.4f}")
	print(f"Precision: {result['precision']:.4f}")
	print(f"Recall: {result['recall']:.4f}")
	print(f"指标文件: {metrics_path}")
	print(f"{'='*50}\n")

	return result


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
		# 优先：weights/best.pt（集中管理）
		central_path = "weights/best.pt"
		if os.path.exists(central_path):
			return central_path

		# 回退：yolo_finetune/{RUN_NAME}/weights/best.pt（ultralytics 默认输出）
		best_path = f"{config.PROJECT_NAME}/{config.RUN_NAME}/weights/best.pt"
		last_path = f"{config.PROJECT_NAME}/{config.RUN_NAME}/weights/last.pt"

		if os.path.exists(best_path):
			return best_path
		elif os.path.exists(last_path):
			return last_path
		else:
			raise FileNotFoundError(
				f"微调模型未找到: {central_path} 或 {best_path} 或 {last_path}"
			)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='测试 YOLOv8 模型')
	parser.add_argument('--pretrained', action='store_true', help='测试预训练模型')
	parser.add_argument('--finetuned', action='store_true', help='测试微调后模型')
	parser.add_argument(
		'--split',
		choices=('val', 'test'),
		default='val',
		help='评估数据划分，课程最终比较应使用 test',
	)
	parser.add_argument('--imgsz', type=int, default=config.IMG_SIZE, help='评估输入尺寸')
	parser.add_argument('--batch', type=int, default=config.BATCH_SIZE, help='评估批大小')
	parser.add_argument('--device', default=config.DEVICE, help='评估设备，例如 0 或 cpu')
	parser.add_argument(
		'--output',
		default='results/evaluation',
		help='评估结果根目录',
	)
	args = parser.parse_args()
	pretrained_metrics = None
	finetuned_metrics = None

	if args.pretrained or not args.finetuned:
		model_path = get_model_path(is_pretrained=True)
		pretrained_metrics = test_model(
			model_path,
			args.output,
			'pretrained',
			split=args.split,
			imgsz=args.imgsz,
			batch=args.batch,
			device=args.device,
		)

	if args.finetuned:
		model_path = get_model_path(is_pretrained=False)
		finetuned_metrics = test_model(
			model_path,
			args.output,
			'finetuned',
			split=args.split,
			imgsz=args.imgsz,
			batch=args.batch,
			device=args.device,
		)

	if pretrained_metrics and finetuned_metrics:
		metric_names = ("precision", "recall", "mAP50", "mAP50-95")
		comparison = {
			"split": args.split,
			"pretrained": pretrained_metrics,
			"finetuned": finetuned_metrics,
			"delta": {
				name: finetuned_metrics[name] - pretrained_metrics[name]
				for name in metric_names
			},
		}
		comparison_path = Path(args.output) / f"comparison_{args.split}.json"
		comparison_path.write_text(
			json.dumps(comparison, ensure_ascii=False, indent=2),
			encoding="utf-8",
		)
		print(f"对比文件: {comparison_path}")
