# YOLOv8 模型微调与性能验证

本项目用于完成图像处理与识别大作业：基于 Ultralytics YOLOv8 在 SKU-110K 数据集上进行目标检测模型微调，并对微调前后性能进行量化和定性验证。

我的个人实验方向是数据增强消融组：使用 `yolov8n.pt` 作为预训练权重，关闭 Mosaic 增强，观察密集货架商品检测场景下训练指标和检测效果的变化。

## 项目信息

| 项目 | 内容 |
| --- | --- |
| 任务 | YOLOv8 目标检测模型微调与性能验证 |
| 数据集 | SKU-110K，非 COCO 数据集 |
| 模型 | YOLOv8n |
| 框架 | PyTorch，Ultralytics YOLOv8 |
| 环境 | WSL2，Python 3.8+，RTX 4060 8GB |
| 团队成员 | 刘易函、卓识、陈奕莱、皇甫泊宁 |

SKU-110K 面向超市货架密集排列商品检测，适合验证模型在拥挤、遮挡、小目标密集分布场景中的迁移学习效果。

## 目录说明

| 路径 | 用途 |
| --- | --- |
| `train.py` | 读取 `config.py` 并启动 YOLOv8 微调 |
| `test_model.py` | 在验证集上测试预训练模型或微调模型 |
| `test_extra_images.py` | 对个人额外 5 张图片做微调前后对比测试 |
| `config.py` | 训练、验证、额外测试的统一配置 |
| `requirements.txt` | Python 依赖列表 |
| `docs/finetune_results/` | 报告优先引用的三次微调核心结果 |
| `Yolo_finetune/` | 三次微调的完整原始训练输出 |
| `test_images/` | 个人额外测试图片目录 |

`docs/finetune_results/` 只保留报告需要直接引用的数据，包括训练参数、逐 epoch 指标和训练曲线图。`Yolo_finetune/` 保留完整原始输出，用于复核、补充图表和审查，不在本次整理中删除。

当前 `config.py` 默认会将新训练输出写入 `yolo_finetune/`。已归档的三次历史训练输出保存在 `Yolo_finetune/`，复现实验时需要注意目录大小写差异。

## 实验设计

团队整体采用控制变量法进行平行实验，比较模型容量、数据增强策略和优化器参数对 YOLOv8 微调效果的影响。我的个人实验聚焦关闭 Mosaic 增强后的三次微调对比。

| 实验 | 模型 | 数据集 | 关键变量 | 目的 |
| --- | --- | --- | --- | --- |
| `finetune1` | `yolov8n.pt` | SKU-110K | `imgsz=640`，`batch=32`，`mosaic=0.0` | 作为个人微调的基础对照 |
| `finetune2` | `yolov8n.pt` | SKU-110K | `imgsz=800`，`batch=32`，`mosaic=0.0` | 观察更高输入分辨率对密集目标检测的影响 |
| `finetune3` | `yolov8n.pt` | SKU-110K | `imgsz=640`，`batch=16`，`mosaic=0.0` | 观察较小 batch size 对训练趋势和最终指标的影响 |

三次实验均训练 100 个 epoch，使用相同预训练权重和相同数据集，主要对比输入尺寸和 batch size 对训练过程的影响。

## 微调结果归档

报告直接引用 `docs/finetune_results/` 中的结果文件：

| 目录 | 来源实验 | 归档文件 |
| --- | --- | --- |
| `docs/finetune_results/finetune1/` | `Yolo_finetune/finetune1/` | `args.yaml`、`results.csv`、`results.png` |
| `docs/finetune_results/finetune2/` | `Yolo_finetune/finetune2/` | `args.yaml`、`results.csv`、`results.png` |
| `docs/finetune_results/finetune3/` | `Yolo_finetune/finetune3/` | `args.yaml`、`results.csv`、`results.png` |

文件含义如下：

| 文件 | 报告用途 |
| --- | --- |
| `args.yaml` | 记录模型、数据集、epoch、batch、imgsz、优化器、学习率、Mosaic、保存间隔等训练参数 |
| `results.csv` | 记录每个 epoch 的训练损失、验证损失、Precision、Recall、mAP 和学习率 |
| `results.png` | 展示 Ultralytics 自动生成的训练曲线，可用于报告中的趋势图 |

## 个人三次微调对比

以下最终指标来自各 `results.csv` 的第 100 个 epoch：

| 实验 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | 结论摘要 |
| --- | ---: | ---: | ---: | ---: | --- |
| `finetune1` | 0.90961 | 0.83550 | 0.88337 | 0.53988 | 基础输入尺寸下关闭 Mosaic 后的结果 |
| `finetune2` | 0.91051 | 0.85209 | 0.89545 | 0.55846 | 输入尺寸提升到 800 后，Recall 和 mAP 更高 |
| `finetune3` | 0.90861 | 0.83649 | 0.88425 | 0.54186 | batch size 降到 16 后，最终表现接近 `finetune1` |

从最终指标看，`finetune2` 在 Recall、mAP@0.5 和 mAP@0.5:0.95 上均为三次实验中最高。对于 SKU-110K 这种密集货架商品检测任务，提高输入分辨率更有利于小目标和拥挤目标的识别。

`finetune1` 与 `finetune3` 的最终指标接近，说明在当前设置下，将 batch size 从 32 降到 16 对最终性能影响较小。报告中仍需要结合逐 epoch 曲线分析其训练过程是否存在收敛速度、震荡幅度或验证指标稳定性的差异。

完整课程报告位于 `docs/project_report.md`，实验数据提交要求位于 `docs/experiment_data_spec.md`。

生成三次微调的统一对比图和统计表：

```bash
python analyze_experiments.py \
	--input finetune1=docs/finetune_results/finetune1 \
	--input finetune2=docs/finetune_results/finetune2 \
	--input finetune3=docs/finetune_results/finetune3 \
	--output docs/analysis/personal
```

四位成员的数据到位后，复制 `group_experiments.example.json` 的结构建立 `group_experiments.json`，再使用 `--manifest` 生成同口径比较结果。

## 训练趋势分析方式

报告中的微调过程分析以 `results.csv` 为主，`results.png` 为辅助图。建议按以下维度梳理：

| 分析维度 | 关注字段 | 写作重点 |
| --- | --- | --- |
| 训练收敛 | `train/box_loss`、`train/cls_loss`、`train/dfl_loss` | 比较三次实验的损失下降速度和后期稳定性 |
| 验证泛化 | `val/box_loss`、`val/cls_loss`、`val/dfl_loss` | 判断验证损失是否同步下降，是否出现过拟合迹象 |
| 检测质量 | `metrics/precision(B)`、`metrics/recall(B)` | 分析误检和漏检倾向的变化 |
| 综合性能 | `metrics/mAP50(B)`、`metrics/mAP50-95(B)` | 比较整体检测能力和严格 IoU 条件下的表现 |
| 学习率变化 | `lr/pg*` | 解释训练前期 warmup 和后期学习率衰减对曲线的影响 |

个人报告可先比较三次关闭 Mosaic 微调的训练趋势，再把最终指标作为量化结论。这样可以避免只看最后一个 epoch，而忽略训练过程中的收敛速度和稳定性。

## 组内对比计划

后续组内比较建议固定 `batch size = 16`，再比较不同参数设置对训练趋势的影响。对比重点包括：

| 对比方向 | 关注问题 |
| --- | --- |
| 模型容量 | `yolov8n.pt` 与更大模型是否提升 mAP，是否带来更慢收敛或更高显存压力 |
| 数据增强 | 开启与关闭 Mosaic 后，前期收敛速度、Recall 和误检情况是否变化 |
| 优化器策略 | 不同优化器或学习率设置是否影响震荡幅度和最终泛化能力 |
| 输入尺寸 | 更高 `imgsz` 是否稳定提升密集小目标检测效果 |

组内报告应优先比较相同 epoch 区间内的趋势变化，而不只比较最终指标。固定 batch size 后，可以减少显存和梯度更新规模差异带来的干扰。

## 运行与复现

安装依赖：

```bash
pip install -r requirements.txt
```

开始训练：

```bash
python train.py
```

测试预训练模型：

```bash
python test_model.py --pretrained
```

测试微调后模型：

```bash
python test_model.py --finetuned
```

对额外 5 张图片进行微调前后对比：

```bash
python test_extra_images.py
```

运行前需要确认 `config.py` 中的 `DATA_YAML`、`IMG_SIZE`、`BATCH_SIZE`、`PROJECT_NAME`、`RUN_NAME` 和 `TEST_IMAGES` 与当前实验一致。若要复现 `finetune1`、`finetune2` 或 `finetune3`，应参考对应 `docs/finetune_results/*/args.yaml` 中的参数。

## 报告引用建议

| 报告内容 | 推荐引用 |
| --- | --- |
| 微调参数和设置 | `docs/finetune_results/*/args.yaml` |
| 性能比较表 | `docs/finetune_results/*/results.csv` 第 100 个 epoch |
| 训练过程曲线 | `docs/finetune_results/*/results.png` |
| 额外趋势分析 | `results.csv` 中每个 epoch 的 loss、Precision、Recall、mAP |
| 原始输出复核 | `Yolo_finetune/*/` 中的完整训练图和验证可视化结果 |

个人报告建议先写三次微调内部对比，再写组内固定 `batch size = 16` 条件下的横向比较，最后结合额外 5 张真实图片的检测结果分析模型泛化能力。
