# 计划：文本嵌入方向 + 层级深度范数初始化

## 设计哲学

### 核心洞察：向量的解耦

在向量空间中，任何向量都可以分解为两个独立的信息维度：

```
向量 = 方向 × 范数
     = (语义是什么) × (层级有多深)
```

- **方向（Direction）** → 编码"是什么"的语义关系
- **范数（Norm）** → 编码"在哪一层"的结构位置

### 信息与几何的精确对应

| 信息来源 | 编码内容 | 对应向量分量 |
|---------|---------|-------------|
| 文本嵌入（大规模语料预训练） | 概念之间的语义相似性 | **方向** |
| 层级树（领域知识） | 类别的深度/粗细粒度 | **范数** |

这不是随意的组合：
- 文本嵌入天然编码语义相似性 → 相似概念方向接近
- 双曲空间天然编码层级结构 → 越深节点范数越大

### 设计目标

> "我相信文本嵌入已经捕获了语义关系，我相信层级深度定义了结构位置，让我把它们放对地方，剩下的交给数据量，让模型在巨人的肩膀上学到最适合当前任务的特征分布。"

**初始化 ≠ 答案，初始化 = 起点**

- **先验知识** → 缩小搜索空间（不用从随机出发）
- **数据驱动** → 找到真正的最优解（不被先验锁死）

可学习的 `tangent_vectors` 正是这个哲学的体现：注入先验，但不固定先验。

---

## 三种模式对比

| 模式 | 标志 | 方向来源 | 范数来源 | 可学习参数 | 本质 |
|------|------|----------|----------|------------|------|
| 1. 随机+深度 | (默认) | 随机 | 层级深度 | tangent_vectors | 结构对，语义随机 |
| 2. 文本+投影器 | `--use_text_embeddings` | MLP输出 | MLP输出 | projector | 语义结构混合，让MLP学分离 |
| **3. 文本方向+深度** | `--use_text_direction_init` | 文本嵌入(PCA) | 层级深度 | tangent_vectors | **显式解耦：语义归语义，结构归结构** |

---

## 技术路径总览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          初始化阶段 (一次性)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  文本嵌入文件                        层级深度                                │
│  [n_classes, 768/512]                CLASS_DEPTHS                           │
│         │                                  │                                │
│         ▼                                  │                                │
│  ┌──────────────────┐                      │                                │
│  │ 重排为 label_id  │                      │                                │
│  │ 顺序 (0-69)      │                      │                                │
│  └────────┬─────────┘                      │                                │
│           ▼                                │                                │
│  ┌──────────────────┐                      │                                │
│  │ PCA 降维         │                      │                                │
│  │ 768→embed_dim    │                      │                                │
│  └────────┬─────────┘                      │                                │
│           ▼                                │                                │
│  ┌──────────────────┐                      │                                │
│  │ 归一化为单位向量  │                      │                                │
│  │ direction = v/‖v‖│                      │                                │
│  └────────┬─────────┘                      │                                │
│           │                                │                                │
│           │         方向                深度范数                              │
│           │           │                    │                                │
│           │           ▼                    ▼                                │
│           │    ┌─────────────────────────────────────┐                      │
│           └───►│ tangent_vector = direction × norm   │                      │
│                │ (类别 0 特殊处理：设为零向量)         │                      │
│                └──────────────────┬──────────────────┘                      │
│                                   │                                         │
│                                   ▼                                         │
│                    ┌──────────────────────────────┐                         │
│                    │ self.tangent_vectors         │                         │
│                    │ nn.Parameter, requires_grad  │                         │
│                    └──────────────────────────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          前向传播阶段 (每次调用)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│            切空间 (原点处)                          双曲面                    │
│                                                                             │
│    ┌──────────────────────┐              ┌──────────────────────────┐       │
│    │   tangent_vectors    │   exp_map0   │      embeddings          │       │
│    │   [n_classes, dim]   │ ──────────►  │   [n_classes, dim]       │       │
│    │                      │              │   (空间分量)              │       │
│    └──────────────────────┘              └──────────────────────────┘       │
│                                                                             │
│    exp_map0 公式：                                                          │
│    output = sinh(√c · ‖v‖) × v / (√c · ‖v‖)                                │
│                                                                             │
│    注：返回空间分量，时间分量在需要时计算：                                    │
│    x_time = √(1/c + ‖x_space‖²)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**深度范数计算公式**（与 Mode 1 一致）：
```
tangent_norm = min_radius + (max_radius - min_radius) × (depth / MAX_DEPTH)
```
- 默认参数：`min_radius=0.1`, `max_radius=2.0`
- 浅层类（如 skeletal_system, depth=1）→ 范数 ≈ 0.34
- 深层类（如 rib_left_1, depth=5）→ 范数 ≈ 1.29

---

## 实现方案

### 1. 核心模块：`pasco/models/hyperbolic/label_embedding.py`

**目标**：在 `LorentzLabelEmbedding` 中添加第三种初始化模式

**改动点**：
- 添加 `use_text_direction_init` 参数
- 添加 `text_embedding_type` 参数（sat/clip/biomedclip）
- 在 `__init__` 早期添加互斥检查：`use_text_direction_init` 和 `use_text_embeddings` 不能同时为 True
- 添加 `_init_from_text_direction_with_depth()` 方法

**新方法 `_init_from_text_direction_with_depth()` 的详细逻辑**：

```python
def _init_from_text_direction_with_depth(self, embedding_type: str):
    """
    Initialize tangent vectors using text embedding directions + hierarchy depth norms.

    核心思想：方向来自语义，范数来自结构
    - 方向：文本嵌入 PCA 降维后归一化
    - 范数：与 Mode 1 一致，基于层级深度计算
    """
    from pasco.data.body.organ_hierarchy import CLASS_DEPTHS, MAX_DEPTH

    # Step 1: 加载文本嵌入
    path = self._get_embedding_path(embedding_type)
    data = torch.load(path, map_location="cpu", weights_only=False)
    label_ids = data["label_ids"]  # [N_file]
    embeddings = data["embeddings"]  # [N_file, text_dim] (768 or 512)

    # Step 2: 重新排列为 label_id 顺序（只处理实类 0-69）
    text_dim = embeddings.shape[1]
    reordered = torch.zeros(self.n_classes, text_dim)
    for idx in range(min(self.n_classes, len(label_ids))):
        lid = label_ids[idx].item()
        if 0 <= lid < self.n_classes:
            reordered[lid] = embeddings[idx]

    # Step 3: PCA 降维 (text_dim → embed_dim)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=self.embed_dim)
    reduced = pca.fit_transform(reordered.numpy())  # [n_classes, embed_dim]
    reduced = torch.from_numpy(reduced).float()

    # Step 4: 归一化为单位向量（提取纯方向）
    directions = reduced / (reduced.norm(dim=-1, keepdim=True) + 1e-8)

    # Step 5: 计算深度范数并组合
    with torch.no_grad():
        for class_id in range(self.n_classes):
            if class_id == self.ignore_class:
                # 类别 0 在原点（零切向量）
                self.tangent_vectors.data[class_id] = 0
            else:
                # 获取深度
                depth = CLASS_DEPTHS.get(class_id, MAX_DEPTH)

                # 深度范数公式（与 Mode 1 一致）
                # norm = min_radius + (max_radius - min_radius) × (depth / MAX_DEPTH)
                if MAX_DEPTH > 0:
                    tangent_norm = self.min_radius + (self.max_radius - self.min_radius) * (depth / MAX_DEPTH)
                else:
                    tangent_norm = (self.min_radius + self.max_radius) / 2

                # 方向 × 范数
                self.tangent_vectors.data[class_id] = tangent_norm * directions[class_id]
```

**关键技术要点**：

1. **深度范数公式**（与 Mode 1 完全一致）：
   ```
   tangent_norm = min_radius + (max_radius - min_radius) × (depth / MAX_DEPTH)
   ```
   - 默认 `min_radius=0.1`, `max_radius=2.0`
   - `depth` 来自 `CLASS_DEPTHS` 字典
   - `MAX_DEPTH` 来自 `organ_hierarchy.py`

2. **类别 0 特殊处理**（与 Mode 1 一致）：
   ```python
   if class_id == self.ignore_class:
       self.tangent_vectors.data[class_id] = 0
   ```

3. **投影到双曲空间**：
   - `tangent_vectors` 存储的是**切空间中的向量**
   - 在 `forward()` 中通过 `exp_map0()` 映射到双曲面：
     ```python
     embeddings = exp_map0(tangent_vectors, curv=self.curv)
     ```
   - `exp_map0` 公式：`sinh(√c · ||v||) · v / (√c · ||v||)`
   - 返回的是双曲面上点的**空间分量**（不含时间分量）

4. **不创建 projector**：
   - 与 Mode 2 的关键区别
   - `tangent_vectors` 直接作为可学习参数
   - `self.use_text_embeddings` 保持 `False`

### 2. 模型层：`pasco/models/body_net_hyperbolic.py`

**目标**：透传参数到 `LorentzLabelEmbedding`

**改动点**：
- `__init__` 添加 `use_text_direction_init` 参数
- 传递给 `LorentzLabelEmbedding` 构造函数

### 3. 训练入口：`scripts/body/train_body.py`

**目标**：暴露 CLI 参数

**改动点**：
- 添加 `--use_text_direction_init` 命令行参数
- 传递给模型构造函数

---

## 测试要点

测试文件：`tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py`

### 功能正确性
- 参数 `use_text_direction_init` 存在且默认为 `False`
- 启用时 `tangent_vectors` 是 `nn.Parameter` 且 `requires_grad=True`
- 不创建 `projector`（与 Mode 2 的关键区别）
- `self.use_text_embeddings` 保持 `False`

### 几何约束
- 类别 0 在原点（tangent_vector 为零向量）
- 更深的类别有更大的切向量范数（层级结构保持）
- 范数范围符合 `[min_radius, max_radius]`
- 经过 `exp_map0` 后的嵌入在双曲面上

### 语义保持
- 相似类别有相似方向（余弦相似度高）
- 方向信息来自文本嵌入的 PCA 降维

### 兼容性
- 所有嵌入类型可用（sat, clip, biomedclip）
- 与 `use_text_embeddings=True` 互斥（不能同时启用）
- 反向传播正常工作（梯度可以流经 `tangent_vectors`）

### 边界情况
- PCA 降维时 `embed_dim < text_dim` 正常工作
- 类别深度未定义时使用 `MAX_DEPTH`

---

## 使用方式

```bash
# 新模式：文本方向 + 深度范数
python scripts/body/train_body.py \
    --use_hyperbolic \
    --use_text_direction_init \
    --text_embedding_type sat

# 可选择不同的文本嵌入类型
python scripts/body/train_body.py \
    --use_hyperbolic \
    --use_text_direction_init \
    --text_embedding_type clip  # 或 biomedclip
```

**注意**：当前实现暂不支持虚拟节点，后续可扩展。

---

## 验证清单

1. **单元测试**
   ```bash
   pytest tests/hyperbolic/Lorentz/text_embedding/test_text_direction_init.py -v
   ```

2. **快速验证**：确认可学习、无projector、范数符合预期

3. **训练运行**：5个epoch的短训练，确认流程通畅
