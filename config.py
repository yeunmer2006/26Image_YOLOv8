"""
YOLOv8 微调配置 - 修改此文件调整所有参数
"""

import os

def is_colab():
    """检测是否在 Google Colab 环境中"""
    return os.path.exists('/content/drive')

# ============================================================
# Colab 环境配置（用户上传项目到 Google Drive 后使用）
# ============================================================
if is_colab():
    # Google Drive 挂载路径（根据实际创建的文件夹调整）
    DRIVE_BASE = '/content/drive/MyDrive/YOLOv8_project'
    DATA_YAML = f'{DRIVE_BASE}/datasets/SKU-110K.yaml'
    TEST_IMAGES_BASE = f'{DRIVE_BASE}/test_images'
    PROJECT_ROOT = DRIVE_BASE
else:
    # 本地环境配置
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_YAML = 'datasets/SKU-110K.yaml'
    TEST_IMAGES_BASE = 'test_images'

# ============ 模型配置 ============
MODEL_NAME = 'yolov8n.pt'      # 预训练模型: yolov8n/s/m/l/x.pt
MODEL_VARIANT = 'n'            # 模型大小标识: n/s/m/l/x

# ============ 数据集配置 ============
# DATA_YAML 在上方根据环境自动设置
# SKU-110K 是 Ultralytics 内置数据集，直接使用 YAML 名称（数据集约 13.6GB）

# ============ 训练参数 ============
EPOCHS = 100               # 训练轮数
IMG_SIZE = 640             # 输入图像大小
BATCH_SIZE = 16            # RTX 4060 建议: 16-32
DEVICE = 0                 # GPU设备，0表示第一块GPU，''表示CPU

# ============ 优化器参数 ============
LR0 = 0.01                 # 初始学习率
LRF = 0.01                 # 最终学习率 = LR0 * LRF
MOMENTUM = 0.937           # SGD动量
WEIGHT_DECAY = 0.0005      # 权重衰减
WARMUP_EPOCHS = 3          # 预热轮数

# ============ 数据增强配置 ============
MOSAIC = 0.0                   # Mosaic增强比例，0.0表示关闭（数据增强消融实验）

# ============ 输出配置 ============
PROJECT_NAME = 'yolo_finetune'   # 项目目录名
RUN_NAME = 'homeobjects_n_no_mosaic'  # 本次运行名称
SAVE_DIR = 'results/before'      # 测试结果保存目录

# ============ 测试配置 ============
# 额外5张测试图片路径（根据环境自动适配）
TEST_IMAGES = [
    f'{TEST_IMAGES_BASE}/img1.jpg',
    f'{TEST_IMAGES_BASE}/img2.jpg',
    f'{TEST_IMAGES_BASE}/img3.jpg',
    f'{TEST_IMAGES_BASE}/img4.jpg',
    f'{TEST_IMAGES_BASE}/img5.jpg',
]

# ============ 训练输出 ============
SAVE_PERIOD = 5            # 每隔多少轮保存一次模型（每5个Epoch记录一次）
PATIENCE = 50              # 早停耐心值（无改进轮数）