# YOLO模型微调与性能验证

## 项目概述

基于 Ultralytics YOLOv8 + HomeObjects-3K 数据集的模型微调和性能验证实验。

## 团队成员

- 成员1
- 成员2
- 成员3
- 成员4

## 任务分工

| 任务 | 负责人 | 状态 |
|------|--------|------|
| 模型选择与下载 | 待定 | - |
| 数据集准备 | 待定 | - |
| 模型微调训练 | 待定 | - |
| 性能测试与分析 | 待定 | - |
| 个人5张图片验证 | 各成员 | - |

## 快速开始

### 1. 安装依赖

```bash
pip install ultralytics torch torchvision opencv-python matplotlib pandas pillow
```

### 2. 修改配置（可选）

编辑 `config.py` 中的参数：
- `MODEL_NAME`: 选择模型大小 (yolov8n/s/m/l/x.pt)
- `EPOCHS`: 训练轮数
- `BATCH_SIZE`: 批量大小（RTX 4060 建议 16-32）
- `IMG_SIZE`: 输入图像大小

### 3. 准备额外测试图片

将5张测试图片放入 `test_images/` 目录，命名为 `img1.jpg` ~ `img5.jpg`

### 4. 运行完整流程

```bash
python run.py
```

## 项目结构

```
Image_Processing/
├── config.py              # 参数配置（主要修改此文件）
├── train.py               # 训练脚本
├── test_model.py          # 模型测试脚本
├── test_extra_images.py   # 额外5张图片测试
├── compare_results.py     # 结果对比分析
├── run.py                 # 主运行脚本
├── README.md
├── CLAUDE.md
├── test_images/           # 额外5张测试图片
├── results/
│   ├── before/            # 微调前测试结果
│   ├── after/             # 微调后测试结果
│   ├── extra_test/        # 额外图片测试结果
│   └── comparison/        # 对比报告
└── yolo_finetune/         # 训练输出目录
```

## 各脚本功能

| 脚本 | 功能 |
|------|------|
| `config.py` | 所有可调参数 |
| `train.py` | 模型训练 |
| `test_model.py` | 在验证集上测试模型 |
| `test_extra_images.py` | 对额外5张图片测试 |
| `compare_results.py` | 生成对比报告 |
| `run.py` | 运行完整流程 |

## 常用命令

```bash
# 单独训练
python train.py

# 测试预训练模型
python test_model.py --pretrained

# 测试微调后模型
python test_model.py --finetuned

# 测试额外图片
python test_extra_images.py

# 完整流程
python run.py
```

## 数据集说明

**HomeObjects-3K**: Ultralytics 内置的家庭物品检测数据集，包含多种日常物品。

- 自动下载，无需手动准备
- 图像大小: 640
- 类别: 家庭物品（杯子、椅子、桌子等）

## 参考资源

- [Ultralytics YOLO 文档](https://docs.ultralytics.com/)
- [YOLOv8 GitHub](https://github.com/ultralytics/ultralytics)
- [训练指南](https://docs.ultralytics.com/zh/modes/train/)