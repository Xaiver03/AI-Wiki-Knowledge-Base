---
tags: [PyTorch, 入门, 深度学习, 学习路径]
updated: 2025-09-09
---

# PyTorch 入门教程（新手友好）

本文面向第一次接触 PyTorch 的同学，目标是在 1–2 天内跑通基础训练循环，并在 1–2 周内具备独立完成一个小项目的能力。内容覆盖安装、张量与自动求导、模型构建与训练、数据管道、设备管理、常见坑与调试，以及可执行的学习路径与高质量参考资料。

提示：文末附有权威参考链接（官方教程、课程与中文资料）。

## 1. 适用人群与前置

- 适用人群：具备基础 Python（函数、类、列表/字典）与线性代数/概率初步概念的同学。
- 前置环境：Python 3.9+，建议使用 Conda 或 venv 管理环境；有可选 GPU 更佳（CUDA/CuDNN 随 PyTorch 安装指引选择）。

## 2. 安装与环境验证

推荐按官方“Start Locally”页面安装（自动根据 OS / CUDA 给出命令）。常见命令示例：

```bash
# Conda（推荐）
conda create -n torch101 python=3.10 -y
conda activate torch101

# CPU 版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 或 GPU 版本（以 CUDA 12.x 为例，详见官网“Start Locally”）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

验证安装：

```python
import torch
print(torch.__version__)
print('CUDA available:', torch.cuda.is_available())
```

## 3. 张量（Tensor）与基础操作

张量是 PyTorch 的核心数据结构，类似更“会自动求导”的 NumPy 数组。

```python
import torch

# 创建
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.ones((2, 2))
c = torch.randn(2, 3)       # 高斯分布

# 设备（CPU/GPU）
device = 'cuda' if torch.cuda.is_available() else 'cpu'
a = a.to(device)

# 基本运算与广播
res = a + b.to(device)
res = res @ torch.randn(2, 2, device=device)  # 矩阵乘法

# 形状与切片
print(res.shape)
print(res[:, 0])

# 与 NumPy 互转
res_cpu = res.to('cpu')
np_arr = res_cpu.numpy()
back = torch.from_numpy(np_arr)
```

要点：
- 广播规则类似 NumPy，注意维度对齐。
- `.to(device)` 或 `tensor.cuda()`/`tensor.cpu()` 控制设备。
- `torch.*` 函数与 `Tensor.*` 方法多数等价；优先使用 `torch.*`（易于组合）。

## 4. 自动求导（autograd）

PyTorch 通过构建计算图跟踪运算，实现反向传播。

```python
import torch

x = torch.randn(3, requires_grad=True)
W = torch.randn(3, 3, requires_grad=True)
y = W @ x
loss = y.pow(2).sum()  # 举例：平方和损失

loss.backward()        # 反向传播
print(x.grad)
print(W.grad)

# 推理阶段禁止跟踪（节省显存/加速）
with torch.no_grad():
    y2 = W @ x

# 或从图中分离
y_detached = y.detach()
```

要点：
- 仅对 `requires_grad=True` 的张量跟踪梯度。
- 反向传播后，梯度会累积在 `.grad` 上；下一步前要清零（见训练循环）。

## 5. 神经网络模块（nn.Module）与前向传播

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, in_dim=784, hidden=256, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
print(sum(p.numel() for p in model.parameters()))
```

要点：
- 将层定义在 `__init__`，计算在 `forward`。
- `nn.Sequential` 适合顺序结构；复杂结构直接在 `forward` 编写张量逻辑。

## 6. 训练循环（最小可用版）

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 假数据（以二分类为例）
X = torch.randn(1024, 20)
y = (X.sum(dim=1) > 0).long()
ds = TensorDataset(X, y)
dl = DataLoader(ds, batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(20, 64), nn.ReLU(),
    nn.Linear(64, 2)
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss, total, correct = 0.0, 0, 0
    for xb, yb in dl:
        xb, yb = xb.to(device), yb.to(device)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimizer.zero_grad(set_to_none=True)  # 推荐：更高效、更干净
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += xb.size(0)

    print(f"epoch {epoch}: loss={total_loss/total:.4f}, acc={correct/total:.3f}")

# 推理 / 评估
model.eval()
with torch.no_grad():
    # ...在验证/测试集上评估
    pass
```

要点：
- `model.train()` / `model.eval()` 切换训练/推理模式（影响 BN/Dropout）。
- `optimizer.zero_grad(set_to_none=True)` 优于将 `.grad` 置零为 0；性能更好。
- 分类任务用 `CrossEntropyLoss`（标签是类别索引，不是 one-hot）。

## 7. 数据集与 DataLoader

常用方式：TorchVision/TorchText/TorchAudio 的内置数据集或自定义 `Dataset`。

```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImageFolderSimple(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        # 假设子目录名为类别名
        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fn in os.listdir(cls_dir):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fn), cls))

        self.class_to_idx = {c: i for i, c in enumerate(sorted({c for _, c in self.samples}))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.class_to_idx[cls]
        return img, label

dl = DataLoader(ImageFolderSimple('/data/train'), batch_size=32, shuffle=True,
                num_workers=4, pin_memory=True)
```

要点：
- CPU 数据加载通过 `num_workers` 并行；GPU 训练时 `pin_memory=True` 往往更快。
- 图像任务建议使用 TorchVision 的 `transforms`（标准化、数据增强）。

## 8. 设备、随机性与性能

- 设备：`device = 'cuda' if torch.cuda.is_available() else 'cpu'`，模型与数据需在同一设备。
- 随机性：
  ```python
  torch.manual_seed(42)
  if torch.cuda.is_available():
      torch.cuda.manual_seed_all(42)
  ```
  如需严格可复现，参考官方“Reproducibility”文档（涉及 CUDNN）。
- 性能：
  - Mixed Precision（AMP）：`torch.cuda.amp.autocast()` + `GradScaler` 可在 GPU 上提速与省显存。
  - PyTorch 2.0+：`torch.compile(model)` 可图优化（需 PyTorch 2+）。

## 9. 保存与加载

```python
# 仅保存权重（推荐）
torch.save(model.state_dict(), 'model.pt')

# 加载
model = MLP()
model.load_state_dict(torch.load('model.pt', map_location='cpu'))
model.eval()
```

## 10. 常见坑与调试建议

- 形状不匹配：在关键节点 `print(x.shape)` 或用断言校验；善用 `einops`（可选）。
- `.view` vs `.reshape`：前者需内存连续；不确定时用 `.reshape` 更稳。
- 梯度累积：忘记 `zero_grad` 会导致损失不降；推荐 `zero_grad(set_to_none=True)`。
- 精度与溢出：损失为 `nan` 时检查学习率、初始化、是否未归一化输入。
- 模式切换：推理前 `model.eval()`，训练前 `model.train()`；Dropout/BN 依赖模式。
- 性能瓶颈：数据加载常是瓶颈；用 `num_workers`、`pin_memory`，或预处理缓存。

## 11. 7–14 天学习路径（可执行）

- 第 1–2 天：
  - 安装与环境验证；完成官方“Learn the Basics/60-min Blitz”。
  - 理解 Tensor、autograd、Module、Optimizer、Loss、DataLoader。
- 第 3–5 天：
  - 复现一个小任务（如 MNIST/CIFAR10 分类），写出干净训练循环与验证代码。
  - 探索学习率、批大小、数据增强的影响；记录曲线（Matplotlib 或 W&B）。
- 第 6–7 天：
  - 加入学习率调度器（StepLR/Cosine），尝试 AMP 混合精度。
  - 梳理项目结构（`src/`, `configs/`, `data/`, `scripts/`）。
- 第 8–14 天：
  - 完成一个对你有意义的小项目（如猫狗分类、文本情感分析）。
  - 可选：尝试 PyTorch Lightning/Hydra/W&B 提升工程化能力。

## 12. 参考资料（权威/精选）

以下链接为常用的高质量入门与进阶资料（若访问失败，请稍后重试或使用镜像）：

- 官方总览：
  - PyTorch Tutorials（官方教程）：https://docs.pytorch.org/tutorials/
  - PyTorch Docs（稳定版文档）：https://pytorch.org/docs/stable/index.html
  - Start Locally（安装向导）：https://pytorch.org/get-started/locally/
  - Reproducibility（可复现性说明）：https://pytorch.org/docs/stable/notes/randomness.html
  - TorchVision 文档（官方）：https://pytorch.org/vision/stable/index.html
- 快速上手：
  - Learn the Basics / 60-Minute Blitz（新版/经典）：https://docs.pytorch.org/tutorials/beginner/basics/intro.html
  - What is PyTorch：https://docs.pytorch.org/tutorials/beginner/introyt/introyt1_tutorial.html
  - Neural Networks（nn 模块教程）：https://docs.pytorch.org/tutorials/beginner/nn_tutorial.html
  - Recipes（常用配方速查）：https://docs.pytorch.org/tutorials/recipes/recipes_index.html
- 课程与书籍：
  - Dive into Deep Learning（动手学深度学习，PyTorch 版，中文）：https://zh.d2l.ai/
  - fast.ai Practical Deep Learning for Coders：https://course.fast.ai/
  - Hugging Face Computer Vision Course（社区课程，含 PyTorch）：https://huggingface.co/learn/computer-vision-course
- 中文社区译本（注意版本可能滞后，仅作辅助）：
  - ApacheCN PyTorch 译文集合：https://pytorch.apachecn.org/
  - PyTorch 中文文档（社区维护，可能不完全同步）：https://pytorch-cn.readthedocs.io/

—— 如果你希望，我可以把上面链接在 Obsidian 中进一步细分为卡片，并为每节补充练习题与可运行代码模板。
