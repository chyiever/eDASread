# eDASread 文档总览（合并版）

本文档为原 `architecture.md`、`README.md`、`run_guide.md` 的合并版本，面向使用者与维护者，覆盖软件结构、运行方式、核心能力与操作流程。

## 1. 项目简介

`eDASread` 是一个基于 PyQt5 的桌面程序，用于 `.bin` 格式 eDAS 数据的浏览、可视化、滤波与当前视图导出。

主要能力：
- 自动扫描目录内 `.bin` 文件并分页浏览
- 从文件名解析采样率、点数、时间戳
- Time-space 图 + 指定点时域波形联动显示
- 低通/高通/带通快速滤波
- 当前视图窗口导出（原始或滤波后）

## 2. 运行环境

推荐依赖：

```bash
pip install pyqt5 pyqtgraph numpy scipy
```

启动方式（项目根目录）：

```bash
python -m scripts.app
```

打包单文件 exe：

```bash
cd scripts
python build_exe.py
```

## 3. 当前界面结构

- 顶部标题栏：左侧 logo，中部标题
- 状态栏：右侧固定文字“中国科学院半导体所开发”
- 左侧：文件浏览区（每页 100）
- 右侧上部：参数与导出区（两列）
  - 左列：Display Range（含 Actions 行）
  - 右列：Filter + Export
- 右侧下部：双图
  - 上：指定位置点时域波形
  - 下：time-space 图（右侧色标）

## 4. 文件与数据模型

文件名要求：
- 包含 `-xxxHz-`（采样率）
- 包含 `-xxxpt-`（点数）
- 可选结尾时间戳 `-YYYYMMDDTHHMMSS(.fff).bin`

二进制格式：
- `int32`，4 字节/采样点
- 逻辑矩阵：`frame_count x points`
- 存储顺序：frame-major

单位换算：
- 显示/滤波：`raw -> rad`
- 导出滤波后：`rad -> raw int32`

## 5. 交互与操作流程

1. 选择或输入数据目录
2. 选择目标文件
3. 设置 Display Range / Filter 参数
4. 输入指定位置点（可滚轮 ±1 调整）
5. 点击 `Apply`
6. 使用缩放、Reset View、Back View、Zoom Out 2x 浏览
7. 设置导出目录，点击 `Export Current View`

关键交互规则：
- 指定点输入必须为整数
- 指定点限制在 `[Point Start, Point End]`
- 时域图与 time-space 图共用时间范围，缩放联动

## 6. 滤波与导出规则

滤波类型：
- `none`
- `lowpass`（需要 High Cut）
- `highpass`（需要 Low Cut）
- `bandpass`（需要 Low Cut + High Cut）

频率约束：
- 需满足 Nyquist 条件（`0 < cutoff < fs/2`）

导出规则：
- 仅导出当前可视窗口
- 按视图横轴计算帧切片，按纵轴计算点切片
- 命名：`eDAS-<sample-rate>-<point-count>-<start-time>.bin`
- 结果写回 frame-major `int32`

## 7. 性能与稳定性策略

- 使用 `numpy.memmap` 按需映射大文件
- 滤波/导出通过 `QThreadPool + QRunnable` 后台执行
- 处理结果使用 LRU 缓存复用
- 绘图矩阵使用 `float32` 与连续内存减少开销

## 8. 常见问题

- 无法解析元数据：文件名缺少 `Hz`/`pt`
- 点范围非法：`Point Start > Point End` 或超界
- 指定点非法：非整数或超出范围
- 导出为空：当前可视区域无有效帧或点

## 9. 相关文档

- 开发文档：`docs/dev.md`
- 用户手册源：`docs/eDAS_user_manual_source.txt`
- 用户手册（可用 Word 打开）：`docs/eDAS_user_manual.doc`
