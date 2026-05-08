"""
YOLOv8 微调配置 - 修改此文件调整所有参数
"""

# ============ 模型配置 ============
MODEL_NAME = 'yolov8n.pt'      # 预训练模型: yolov8n/s/m/l/x.pt
MODEL_VARIANT = 'n'            # 模型大小标识: n/s/m/l/x

# ============ 数据集配置 ============
# SKU-110K 是 Ultralytics 内置数据集，直接使用 YAML 名称（数据集约 13.6GB）
DATA_YAML = 'SKU-110K.yaml'

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
# 额外5张测试图片路径
TEST_IMAGES = [
    'test_images/img1.jpg',
    'test_images/img2.jpg',
    'test_images/img3.jpg',
    'test_images/img4.jpg',
    'test_images/img5.jpg',
]

# ============ 训练输出 ============
SAVE_PERIOD = 5            # 每隔多少轮保存一次模型（每5个Epoch记录一次）
PATIENCE = 50              # 早停耐心值（无改进轮数）