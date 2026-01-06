# Voxel Dataset Structure

## 数据集概览

人体3D医学数据集，包含**皮肤表面点云**和**体素化的内部器官标注**。

### 数据集规模

- 总计: 4,028 样本（已去重）
- 原始数据: 5,048 样本（去除重复 1,020 个）

## 数据结构

每个样本是一个 `.npz` 文件，包含以下字段:

### 1. sensor_pc (皮肤点云)

- **类型**: `float32 (N, 3)`
- **描述**: 人体表面3D点云（世界坐标系，单位: 毫米）

### 2. voxel_labels (器官体素标注)

- **类型**: `uint8 (D, H, W)` - 动态尺寸
- **描述**: 3D体素网格，每个体素标注一个类别
- **体素分辨率**: 4mm × 4mm × 4mm (固定)

### 3. grid_world_min / grid_world_max

- **类型**: `float32 (3,)`
- **描述**: 体素网格在世界坐标系中的边界框

### 4. grid_voxel_size

- **类型**: `float32 (3,)`
- **值**: `[4.0, 4.0, 4.0]` - 每个体素的物理尺寸(毫米)

### 5. grid_occ_size

- **类型**: `int32 (3,)`
- **描述**: 体素网格的实际尺寸 `(D, H, W)`

## 类别标注 (72类)

### 特殊类别

- **class 0**: `outside_body` - 体外空点 (~35-65%)
- **class 1**: `inside_body_empty` - 体内空点 (~15-25%)

### 器官类别 (class 2-71, 共70类)

- **软组织器官**: liver, spleen, kidney, stomach, pancreas, gallbladder, etc.
- **骨骼**: spine, ribs, skull, sternum, etc.
- **肌肉**: gluteus_maximus, gluteus_medius, iliopsoas, etc.
- **其他**: lung, heart, brain, spinal_cord, etc.

完整类别列表见 `dataset_info.json`

## 坐标转换

- **体素索引 → 世界坐标**: `world_coord = grid_world_min + voxel_index * grid_voxel_size`
- **世界坐标 → 体素索引**: `voxel_index = (world_coord - grid_world_min) / grid_voxel_size`

## 数据特点

1. **动态网格尺寸**: 每个样本体素网格大小不同，根据扫描范围自适应
2. **固定体素分辨率**: 统一使用 4mm³ 体素
3. **完整人体**: 包含皮肤表面 + 内部器官 + 空间标注
4. **高类别数**: 72个类别，包含详细的器官和骨骼分割
5. **稀疏性**: 器官体素仅占 10-15%，其余为空点
