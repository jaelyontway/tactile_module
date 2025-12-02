# 模型推理工作原理详解

## 一、整体流程

```
输入数据          →  模型处理  →  输出预测
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Image (224×224)      多模态       Gripper
  +                  融合网络      Delta
Tactile (50×6)                   (标量)
```

## 二、详细数据流

### 阶段1: 数据准备

```python
# 1. 图像数据
image = torch.randn(1, 3, 224, 224)
# 形状: (批次, RGB通道, 高度, 宽度)
# 值域: [0, 1] (归一化后的像素值)

# 2. 触觉数据
tactile = torch.randn(1, 50, 6)
# 形状: (批次, 时间步, 传感器通道)
# 50个时间步 = 过去50帧的触觉历史
# 6个通道 = 6个触觉传感器

# 移到GPU
image = image.to("cuda")
tactile = tactile.to("cuda")
```

### 阶段2: 图像编码

```
输入:  (1, 3, 224, 224)
   ↓
┌────────────────────────┐
│  DINOv3 Vision Transformer │
│  - 预训练在ImageNet上    │
│  - 冻结权重（不训练）    │
│  - 提取视觉特征         │
└────────────────────────┘
   ↓
Patch Tokens: (1, 196, 384)
Register Tokens: (1, 4, 384)
# 196 = 14×14 patches
# 4 = DINOv3的register tokens
# 384 = DINOv3-vits16的特征维度

   ↓
┌────────────────────────┐
│  Perceiver Resampler    │
│  - 压缩patch tokens     │
│  - 196 → 10 tokens     │
│  - 保留重要信息         │
└────────────────────────┘
   ↓
压缩后: (1, 10, 256)  # 投影到d_model=256

   ↓
拼接register tokens
最终图像特征: (1, 14, 256)
# 14 = 10 compressed patches + 4 register tokens
```

### 阶段3: 触觉编码

```
输入:  (1, 50, 6)
   ↓
转置为Conv1D格式: (1, 6, 50)
   ↓
┌────────────────────────┐
│  Conv1D Layer 1        │
│  6 → 64 channels       │
│  stride=2              │
│  50 → 25 time steps    │
└────────────────────────┘
   ↓  (1, 64, 25)
┌────────────────────────┐
│  Conv1D Layer 2        │
│  64 → 128 channels     │
│  stride=2              │
│  25 → 13 time steps    │
└────────────────────────┘
   ↓  (1, 128, 13)
┌────────────────────────┐
│  Conv1D Layer 3        │
│  128 → 256 channels    │
│  stride=2              │
│  13 → 7 time steps     │
└────────────────────────┘
   ↓  (1, 256, 7)
┌────────────────────────┐
│  Conv1D Layer 4        │
│  256 → 256 channels    │
│  stride=2              │
│  7 → 4 time steps      │
└────────────────────────┘
   ↓  (1, 256, 4)
┌────────────────────────┐
│  Adaptive Pool         │
│  4 → 3 tokens          │
└────────────────────────┘
   ↓
转置回: (1, 3, 256)
   ↓
投影: (1, 3, 256)
最终触觉特征: (1, 3, 256)
```

### 阶段4: 多模态融合

```
CLS token:      (1, 1, 256)   ← 可学习的全局token
Image tokens:   (1, 14, 256)  ← 视觉信息
Tactile tokens: (1, 3, 256)   ← 触觉信息
   ↓
拼接在一起
   ↓
Combined: (1, 18, 256)
# 18 = 1 CLS + 14 image + 3 tactile

Token序列结构:
[CLS] [Reg1] [Reg2] [Reg3] [Reg4] [P1] [P2] ... [P10] [T1] [T2] [T3]
  ↑      ↑─────────────────↑        ↑──────────────↑    ↑──────────↑
 全局    Register tokens     压缩的patch tokens    触觉tokens

   ↓
┌────────────────────────────────────────┐
│  Transformer Encoder Layer 1           │
│  - Multi-head Self-Attention (8 heads) │
│  - FFN (256 → 512 → 256)               │
│  - LayerNorm + Residual                │
└────────────────────────────────────────┘
   ↓  (1, 18, 256)
┌────────────────────────────────────────┐
│  Transformer Encoder Layer 2           │
└────────────────────────────────────────┘
   ↓  (1, 18, 256)
┌────────────────────────────────────────┐
│  Transformer Encoder Layer 3           │
└────────────────────────────────────────┘
   ↓  (1, 18, 256)
┌────────────────────────────────────────┐
│  Transformer Encoder Layer 4           │
└────────────────────────────────────────┘
   ↓
融合后的特征: (1, 18, 256)
```

### 阶段5: 回归预测

```
取出CLS token: (1, 256)
# CLS token在transformer中收集了所有token的信息

   ↓
┌────────────────────────┐
│  Regression Head       │
│  LayerNorm             │
│  Linear: 256 → 128     │
│  GELU激活              │
│  Dropout               │
│  Linear: 128 → 1       │
└────────────────────────┘
   ↓
输出: (1, 1)
最终预测值: tensor([[0.123456]])

例如: 0.123456 表示gripper需要移动0.123456个单位
```

## 三、关键技术细节

### 1. 为什么用torch.no_grad()?

```python
# ❌ 错误做法（推理时）
prediction = model(image, tactile)
# 会计算梯度，浪费内存和时间

# ✅ 正确做法
with torch.no_grad():
    prediction = model(image, tactile)
# 不计算梯度，更快更省内存
```

**原因:**
- 训练时需要梯度来更新参数
- 推理时只需要前向传播，不需要梯度
- `torch.no_grad()` 可以节省约50%内存，加速约2x

### 2. 为什么用model.eval()?

```python
model.eval()  # 设置为评估模式
```

**效果:**
- Dropout层: 训练时随机丢弃神经元 → 评估时保留所有神经元
- BatchNorm层: 训练时用当前批次统计 → 评估时用全局统计
- 确保推理结果稳定、可重复

### 3. Perceiver Resampler如何工作?

```
输入: 196个patch tokens (太多!)
目标: 压缩到10个tokens (高效!)

方法: 交叉注意力 (Cross-Attention)
┌─────────────────────────────────┐
│  10个可学习的查询向量 (learnable)  │
│        ↓ Query                   │
│  Cross-Attention                 │
│        ↑ Key, Value              │
│  196个输入tokens                 │
└─────────────────────────────────┘

结果: 10个输出tokens包含了196个输入的核心信息
就像把一篇长文章总结成10句话
```

### 4. 为什么需要位置编码?

```python
image_tokens = image_tokens + self.image_positional
tactile_tokens = tactile_tokens + self.tactile_positional
```

**原因:**
- Transformer本身对token顺序不敏感
- 位置编码告诉模型每个token的位置信息
- 图像: 哪个token来自图像的哪个区域
- 触觉: 哪个token对应哪个时间步

## 四、实际运行示例

### 单样本推理

```python
import torch
from model import MultimodalForceTransformer, MultimodalTransformerConfig

# 1. 加载模型
device = torch.device("cuda")
config = MultimodalTransformerConfig(
    dinov3_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    # ... 其他配置
)
model = MultimodalForceTransformer(config)
model.load_state_dict(torch.load("checkpoints/delta_gripper.pt"))
model.to(device)
model.eval()

# 2. 准备数据
image = torch.randn(1, 3, 224, 224).to(device)
tactile = torch.randn(1, 50, 6).to(device)

# 3. 推理
with torch.no_grad():
    prediction = model(image, tactile)

# 4. 获取结果
delta = prediction.item()  # 转为Python标量
print(f"Predicted gripper delta: {delta:.6f}")
```

### 批量推理

```python
# 批量处理更高效
batch_size = 16
images = torch.randn(batch_size, 3, 224, 224).to(device)
tactile_data = torch.randn(batch_size, 50, 6).to(device)

with torch.no_grad():
    predictions = model(images, tactile_data)

# predictions shape: (16, 1)
for i, pred in enumerate(predictions):
    print(f"Sample {i}: {pred.item():.6f}")
```

## 五、常见问题

### Q1: checkpoint文件很大吗？

```
delta_gripper.pt 大小分析:
- DINOv3 backbone (冻结): ~21M 参数, 但权重来自HuggingFace
- 可训练部分: ~5M 参数
- 文件大小: 约 20-100 MB (取决于保存的内容)
```

### Q2: 推理速度如何？

```
在GPU上 (单样本):
- DINOv3处理图像: ~20ms
- 触觉编码: <1ms
- Transformer融合: ~5ms
- 回归头: <1ms
总计: ~30ms/样本

批量处理 (batch=16):
- 总时间: ~100ms
- 每样本: ~6ms (提速5x!)
```

### Q3: 可以不用GPU吗？

```python
# CPU推理
device = torch.device("cpu")
model.to(device)

# 速度会慢约10-50倍，但能运行
# 单样本: ~300ms - 1s
```

### Q4: 如何处理不同大小的输入？

```python
# 图像会自动resize到224×224
image = torch.randn(1, 3, 480, 640)  # 任意尺寸
# 内部会调整为 (1, 3, 224, 224)

# 触觉会自动pad/truncate到50
tactile = torch.randn(1, 30, 6)  # 只有30个时间步
# 内部会pad为 (1, 50, 6)
```

## 六、模型的"智能"来自哪里？

```
训练时学到的模式:

1. 视觉线索:
   - 物体距离 → gripper移动距离
   - 目标位置 → 移动方向
   - 障碍物 → 谨慎移动

2. 触觉线索:
   - 接触压力 → 抓取力度
   - 滑动检测 → 调整gripper
   - 震动模式 → 物体特性

3. 多模态融合:
   - 看到物体 + 触觉确认 → 高置信度
   - 视觉遮挡 + 触觉补充 → 鲁棒性
   - 时序关联 → 预测趋势
```

## 七、总结

**推理 = 前向传播 = 一系列矩阵运算**

```
输入数据 → 编码 → 融合 → 回归 → 输出预测

关键要点:
✅ 使用 torch.no_grad() 节省内存
✅ 使用 model.eval() 确保一致性
✅ 批处理提高效率
✅ checkpoint包含所有学到的知识
✅ 推理很快（毫秒级）
```
