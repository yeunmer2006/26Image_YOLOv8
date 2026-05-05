#!/bin/bash
# YOLOv8 环境创建脚本

echo "创建 conda 环境: yolo"
conda env create -f environment.yml

echo ""
echo "激活环境: conda activate yolo"
echo "运行训练: python run.py"
echo ""
echo "如需在 PyCharm/VS Code 中使用，选择对应编译器的这个环境路径"