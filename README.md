# 介绍
该工具主要使用zephir的annotator GUI进行已有点云坐标的修改。
使用GUI需要以下文件：
- 图像文件，保存为`data.h5`。
- 点云坐标，保存为`annotations.h5`。
- 数据格式的说明，即对图像文件各通道大小的说明，保存为`metadata.json`。

# 环境安装
1. 安装[pixi](https://github.com/prefix-dev/pixi?tab=readme-ov-file#installation),可通过powershell运行以下命令安装：
   ```powershell
    powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"
   ```
2. git clone该仓库。
3. 进入该仓库，运行命令`pixi install`安装依赖。

# 使用方法
1. 将上述文件放在同一目录下，假设路径为`.\data`。
2. 运行命令`annotator --dataset=.\data --port=5000`，再浏览器使用`localhost:5000`打开GUI。
3. 在GUI中进行点云坐标的修改，保存后关闭GUI。

## GUI的快捷键
- `f`: 前进一个volume
- `d`: 后退一个volume
- `shift + f`: 前进十个volume
- `shift + d`: 后退十个volume
- `v`: 当前z+0.05(z坐标经过了归一化处理)
- `c`: 当前z-0.05
- `shift + v`: 当前z+0.2
- `shift + c`: 当前z-0.2
- `ctrl + v`: 当前z+0.1
- `ctrl + c`: 当前z-0.1
- `r`: 切换当前视图为(切片，3D，MIP)
- `e`: 切换为上一个视图
- `w`: 切换显示下一个神经元
- `q`: 切换显示上一个神经元
- `o`: 切换神经元显示为⭕或者●
- `a`: 显示所有切片的神经元/当前切片的神经元
- `0`: 为神经元重置颜色
- `1`: 删除当前选中的神经元
- `2`: 插入当前选中的神经元

使用以上快捷键已经足够进行点云坐标的修改，更多具体介绍可见[zephir文档](https://github.com/venkatachalamlab/ZephIR/blob/main/docs/Guide-annotatorGUI.md)。具体使用示例可见[视频](https://www.youtube.com/watch?v=4O9aIftvoqM)。