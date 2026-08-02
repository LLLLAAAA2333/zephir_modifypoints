# AGENTS.md

## 项目概述

`zephir_modifypoints` 是一个基于 [ZephIR](https://github.com/venkatachalamlab/ZephIR)（v1.0.5）的神经元点云坐标可视化修改工具。核心流程：将 `.npy` 图像和神经元坐标转换为 ZephIR 格式 → 通过 Web GUI 手动修正坐标 → 导出回 `.npy`。

## 环境与执行

- **包管理器**: [Pixi](https://pixi.sh)（conda-forge + PyPI），Python 3.9
- **安装**: `pixi install`
- **进入环境**: `pixi shell`
- **启动 GUI**: `annotator --dataset=.\data\<dataset> --port=5000`，浏览器打开 `localhost:5000`
- **平台**: win-64, linux-64
- **可编辑安装**: `zephir/` 以 editable mode 安装为本地包（参见 `pixi.toml:31`）

## 数据文件约定

ZephIR GUI 需要一个目录包含三个文件：

| 文件 | 内容 | 生成方式 |
|------|------|----------|
| `data.h5` | 图像 volume 数据，shape `(T, C, Z, Y, X)`, uint8 | `create_zephir_data_from_npy()` |
| `annotations.h5` | 点云坐标，含 `/x`, `/y`, `/z`, `/id`, `/parent_id`, `/worldline_id`, `/provenance`, `/t_idx`, `/abs_t_idx` | `create_zephir_annotations_from_npy()` |
| `metadata.json` | 图像维度说明：`shape_t`, `shape_c`, `shape_z`, `shape_y`, `shape_x`, `dtype` | `create_metadata_json_from_npy()` |

坐标采用归一化存储：`x/width`, `y/height`, `z/(z_ratio * depth)`。反向转换时乘以对应系数还原。

## 目录结构

```
zephir_modifypoints/
├── utils/                    # 自定义工具脚本（核心业务逻辑）
│   ├── convert_npy2zephir_format.py   # 主要转换：.npy ↔ ZephIR HDF5 格式
│   ├── convert_cbmi_data_to_ZephIR_format.py  # CBMI/MATLAB 数据转换
│   ├── check_annotation.py            # 校验 annotations.h5 的 worldline 完整性
│   ├── generate_zephir_video.py       # 从 annotations 生成标注视频
│   ├── neuron_pt_tuple_npz_pair.py    # 神经元坐标配对导出
│   ├── HDF5Toolkit.py                 # HDF5 读写工具
│   ├── MatToolkit.py                  # MATLAB v7.3 .mat 文件读取
│   ├── logged_operation.py            # 计时装饰器/上下文管理器
│   └── cyk_logging.py                 # 日志配置
├── zephir/                   # ZephIR 库本体（editable install）
│   └── zephir/
│       ├── annotator/        # Flask Web GUI（server + React/WebGL 前端）
│       │   ├── main.py       # Flask 路由、RPC dispatch
│       │   ├── data/         # 数据模型、I/O、坐标变换
│       │   └── rpc/          # CRUD 操作实现（增删改查 annotation/worldline）
│       ├── methods/          # 追踪管线（build_*, track_*, save_*, extract_*）
│       ├── models/           # PyTorch 模型（ZephIR 注册、ZephOD 检测、损失函数）
│       ├── zephod/           # 特征检测模型训练/推理
│       └── utils/            # I/O、getter 辅助
├── data/                     # 运行时数据（含 context.py、process.ipynb）
├── pixi.toml                 # Pixi 项目配置
└── README.md                 # 中文使用说明
```

## 关键数据结构

- **`neuron_pt_tuple`**: numpy 数组，shape `(T, N, 8)`。每行 8 个特征：`[x, y, z, feature1, feature2, ...]`。前三维是空间坐标。
- **`annotations.h5` datasets**: 所有数据集为一维数组，按行对应。`/id` 全局唯一且递增（从 1 开始），`/parent_id` 默认为 0（无父节点），`/provenance` 值为 `ANTT`（annotation）或 `DETD`（detected），`/worldline_id` 按帧内序号排列（0-based）。
- **`worldlines.h5`**: 自动生成文件，存储 worldline 元信息。删除后会在下次加载时自动重新生成。

## 核心转换流程

```python
# utils/convert_npy2zephir_format.py 中的入口函数（尚未导出为独立函数，见 __main__ 示例）
# 1. 图像: .npy (y,x,z) → transpose → rescale → data.h5
# 2. 坐标: neuron_pt_tuple (T,N,8) → 归一化 → annotations.h5
# 3. 元数据: 参数 → metadata.json
# 4. 反向: annotations.h5 → load → 去归一化 → neuron_pt_tuple.npy
#    同时清理已删除的 track，删除 worldlines.h5 触发重新生成
```

## GUI 快捷键（常用）

- `f` / `d`: 前进/后退一个 volume
- `v` / `c`: z 坐标 ±0.05
- `shift + f/d/v/c`: 加速移动（×10 或 ×4）
- `r` / `e`: 切换/回退视图（切片 / 3D / MIP）
- `w` / `q`: 切换上/下一个神经元
- `o`: 切换神经元显示样式（空心圆/实心圆）
- `a`: 切换所有切片 / 当前切片显示
- `1`: 删除当前选中的神经元（track）
- `2`: 在当前位置插入神经元

## 开发注意事项

- `utils/` 中的脚本使用 `if __name__ == "__main__"` 时动态添加 `sys.path`，因此 `from utils.xxx import` 的导入方式仅在顶层脚本（如 notebook）中有效。直接运行 `utils/` 下的脚本作为入口点可以正常工作。
- `annotations.h5` 修改后需删除 `worldlines.h5` 以强制重新生成，否则可能出现不一致。
- `zephir/annotator/rpc/insert_annotation.py` 中插入时需要校验 `next_id` 与现有最大 ID 对齐（最近修复）。
- 坐标归一化时 z 轴使用 `z_ratio * depth` 而非 `depth`，因为 z 轴物理单位可能与 x/y 不同。
- Pixi 环境中 `zephir` 包通过 `[pypi-dependencies]` 以 path editable 方式安装，修改 `zephir/` 下源码后直接生效，无需重新安装。
- 上游 ZephIR 使用 MIT 许可证，论文发表在 PLOS Computational Biology (2024)。
