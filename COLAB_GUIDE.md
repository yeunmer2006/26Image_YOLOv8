# Google Colab 训练指南

## 前提条件

- Google 账号
- 能访问 Google Colab (需要稳定网络)

---

## Step 1: 准备 Git 仓库（本地操作）

1. 确保 `colab` 分支已推送到远程仓库（GitHub / Gitee 等）
2. 仓库地址例如：`https://github.com/yeunmer2006/YOLOv8_project.git`

---

## Step 2: 在 Colab 中克隆仓库

1. 打开 [Google Colab](https://colab.research.google.com/)
2. 新建笔记本
3. 在代码格中粘贴以下命令（将 `YOUR_REPO` 替换为你的仓库地址）：

```python
# 克隆仓库（使用 colab 分支）
!git clone -b colab https://github.com/yeunmer2006/26Image_YOLOv8.git
%cd /26Image_YOLOv8
```

---

## Step 3: 环境初始化

```python
# 安装依赖
!pip install ultralytics

# 验证 GPU 可用
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
```

---

## Step 4: 可选 - 运行数据集转换（如需转换 CSV 到 YOLO 格式）

```python
# 如果你使用本地 CSV 标注数据，先转换格式
!python datasets/convert_to_yolo.py
```

---

## Step 5: 开始训练

```python
# 方式一：使用 run.py（完整流程）
!python run.py

# 方式二：仅运行训练
!python train.py
```

训练过程中：
- Ultralytics 会自动下载 SKU-110K 数据集（如需）
- 模型保存在 `yolo_finetune/homeobjects_n_no_mosaic/weights/`
- 每 5 个 epoch 自动保存 `last.pt`

---

## Step 6: 断点续训（如中断后继续）

```python
# 加载上次保存的 checkpoint（last.pt 在项目目录的 weights 文件夹下）
model = YOLO('yolo_finetune/homeobjects_n_no_mosaic/weights/last.pt')

# 从断点继续训练
results = model.train(resume=True)
```

或直接修改 `config.py` 中的 `resume` 设置后运行 `train.py`。

---

## Step 7: 防止 Colab 断连（可选但推荐）

在浏览器控制台中运行以下 JS 代码，防止长时间 idle 断连：

1. 按 `F12` 打开开发者工具
2. 切换到 `Console` 标签
3. 粘贴以下代码并回车：

```javascript
function ClickConnect() {
    console.log("保持连接中...");
    document.querySelector("#top-toolbar > colab-connect-button")
        .shadowRoot.querySelector("#connect").click()
}
setInterval(ClickConnect, 5 * 60000); // 每5分钟点击一次
```

停止方法：在控制台运行 `clearInterval(id)`

---

## 重要注意事项

| 项目 | 说明 |
|------|------|
| **免费时长** | 每次最长 12 小时，超时会被重置 |
| **GPU 配额** | 长期使用可能遇到配额不足，需要等待或升级 Pro |
| **数据保存** | 通过 git push 同步到远程仓库，断连不会丢失 |
| **下载模型** | 训练完成后可从仓库下载 `best.pt` 或 `last.pt` |
| **数据集** | SKU-110K 约 13.6GB，每次新会话需要重新下载 |
| **代码同步** | 修改代码后记得 git push 保存到远程仓库 |

---

## 常用命令速查

```python
# 查看 GPU
!nvidia-smi

# 查看当前目录
!pwd

# 查看文件
!ls -la

# 安装特定版本
!pip install ultralytics==8.2.0

# 提交并推送代码到远程仓库
!git add .
!git commit -m "更新训练结果"
!git push
```

---

## 故障排除

**Q: 提示找不到文件**
→ 检查 `config.py` 中的 `DRIVE_BASE` 路径是否与实际文件夹名称一致

**Q: 显存不足 (OOM)**
→ 在 `config.py` 中减小 `BATCH_SIZE`（如改为 8 或 4）

**Q: GPU 配额用尽**
→ 等待几个小时后再试，或升级 Colab Pro

**Q: 训练中断后无法恢复**
→ 检查 `yolo_finetune/homeobjects_n_no_mosaic/weights/` 是否有 `last.pt`