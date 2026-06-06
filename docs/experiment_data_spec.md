# 实验数据提交规范

## 目的

本规范用于统一个人三次微调和四位成员对照试验的数据格式，确保所有实验采用相同统计口径生成逐 epoch 曲线、最佳指标表和结论。

## 每组实验必须提供的文件

每组实验建立一个独立目录，至少包含：

| 文件 | 是否必需 | 用途 |
| --- | --- | --- |
| `results.csv` | 是 | Ultralytics 逐 epoch 训练与验证指标 |
| `args.yaml` | 建议 | 模型、输入尺寸、batch、学习率、增强策略等训练参数 |
| `weights/best.pt` | 最终复核时建议 | 在相同验证集或测试集上统一评估 |
| `results.png` | 否 | 与分析程序生成的图进行人工交叉检查 |

不要手工修改 `results.csv` 的列名或数值。若实验中断后续训，应优先提交 Ultralytics 合并后的完整结果；如果存在多个分段文件，需要同时说明续训起始 epoch。

## `results.csv` 必需字段

| 字段 | 含义 |
| --- | --- |
| `epoch` | 训练轮次 |
| `train/box_loss` | 训练集边界框损失 |
| `train/cls_loss` | 训练集分类损失 |
| `train/dfl_loss` | 训练集分布焦点损失 |
| `metrics/precision(B)` | 验证集 Precision |
| `metrics/recall(B)` | 验证集 Recall |
| `metrics/mAP50(B)` | 验证集 mAP@0.5 |
| `metrics/mAP50-95(B)` | 验证集 mAP@0.5:0.95 |
| `val/box_loss` | 验证集边界框损失 |
| `val/cls_loss` | 验证集分类损失 |
| `val/dfl_loss` | 验证集分布焦点损失 |

学习率列允许因 Ultralytics 版本不同而变化，分析程序会自动识别所有以 `lr/` 开头的字段。

## 四组对照试验清单

收到组员数据后，在项目根目录建立 `group_experiments.json`：

```json
{
	"experiments": [
		{
			"name": "成员1-实验名称",
			"member": "成员姓名",
			"results": "group_results/member1/results.csv",
			"args": "group_results/member1/args.yaml",
			"description": "仅描述相对公共基线发生变化的变量"
		},
		{
			"name": "成员2-实验名称",
			"member": "成员姓名",
			"results": "group_results/member2/results.csv",
			"args": "group_results/member2/args.yaml",
			"description": "例如：模型由 YOLOv8n 改为 YOLOv8s"
		}
	]
}
```

实际文件需要包含四条实验记录。`name` 必须唯一，路径相对于清单文件所在目录解析。

## 公平比较要求

四组实验只有在以下条件一致或被明确记录时，指标才适合直接横向比较：

- 使用相同的数据集版本和训练集、验证集划分。
- 使用相同的验证脚本、IoU 设置和最大检测数。
- 报告最终 epoch 指标的同时，也报告最佳 `mAP@0.5:0.95` 所在 epoch。
- 模型规模、输入尺寸或 batch 不同时，同时报告训练耗时和资源开销。
- 随机种子不一致时，不把小幅单次差异直接解释为稳定改进。
- 所有最终模型应在同一测试集上重新评估，训练过程中的验证指标只用于分析收敛趋势。

## 运行方式

分析当前三次微调：

```bash
python analyze_experiments.py \
	--input finetune1=docs/finetune_results/finetune1 \
	--input finetune2=docs/finetune_results/finetune2 \
	--input finetune3=docs/finetune_results/finetune3 \
	--output docs/analysis/personal
```

分析四组对照试验：

```bash
python analyze_experiments.py \
	--manifest group_experiments.json \
	--output docs/analysis/group
```

输出包括逐 epoch 检测指标图、损失图、学习率图、最终指标柱状图、最终指标表、最佳指标表、数据质量告警和自动摘要。
