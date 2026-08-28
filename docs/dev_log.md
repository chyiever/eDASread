# eDASread 开发文档与开发日志

## 一、开发文档（dev）

本文档面向开发者，聚焦：绘图实现、卡顿控制、bin 读取方式、数据处理链路、多通道快速滤波、导出规则。

### 1. 主要技术栈

- UI 框架：`PyQt5`
- 绘图库：`pyqtgraph`
- 数值计算：`numpy`
- 数字滤波：`scipy.signal`
- 异步执行：`QThreadPool + QRunnable`

### 2. 绘图实现方式

#### 2.1 双图结构与时间轴联动

核心类：`scripts/render/space_time_view.py` 的 `SpaceTimeView`。

关键点：
- time-space 图：`pg.PlotWidget + pg.ImageItem`
- 时域图：`pg.PlotWidget + plot curve`
- 共用时间范围：

```python
self.waveform_plot.setXLink(self.plot_widget)
```

- time-space 横轴格式化：`TimeAxisItem`

```python
self.time_axis = TimeAxisItem(orientation="bottom")
self.plot_widget = pg.PlotWidget(axisItems={"bottom": self.time_axis})
```

#### 2.2 ImageItem 坐标变换

将“帧索引”映射到“秒”，并将 y 轴映射到点号：

```python
transform = QTransform()
transform.scale(1.0 / metadata.sample_rate_hz, 1.0)
transform.translate(0.0, start_point)
self.image_item.setTransform(transform)
```

#### 2.3 指定位置点时域波形

时域数据直接从 time-space 使用的同一矩阵抽取：

```python
waveform_index = self._waveform_point - self._y_start_point
waveform = np.asarray(self._image_data[waveform_index], dtype=np.float32)
x_values = np.arange(waveform.shape[0], dtype=np.float32) / self._metadata.sample_rate_hz
self.waveform_curve.setData(x=x_values, y=waveform)
```

### 3. 防止卡顿的实现

#### 3.1 大文件读取：memmap

- 不一次性把原始 bin 全部复制到内存
- 通过 `numpy.memmap` 进行只读映射

```python
return np.memmap(
    metadata.file_path,
    dtype=RAW_DTYPE,
    mode="r",
    shape=(metadata.frame_count, metadata.points),
)
```

#### 3.2 耗时任务异步化

- 读取/滤波/导出放入线程池
- UI 线程只负责交互与绘图

```python
worker = FunctionWorker(self._load_file_data, metadata, filter_config)
self.thread_pool.start(worker)
```

`FunctionWorker`（`scripts/workers/tasks.py`）使用 `QRunnable` + 信号回传结果/错误。

#### 3.3 结果缓存

- `main_window.py` 使用 `LruByteCache`
- key = `(file_path + filter_config)`
- 避免频繁重复滤波

```python
cache_key = (str(metadata.file_path),) + filter_config.cache_key()
processed = self.processed_cache.get(cache_key)
```

#### 3.4 渲染数据内存优化

- 统一使用 `float32`
- 绘图前转换连续内存，减少拷贝成本

```python
return np.ascontiguousarray(frame_data[:, point_slice].T)
```

### 4. bin 数据读取方式

#### 4.1 当前读取流程

1. 从文件名解析采样率与点数
2. 校验文件长度与 `int32` 对齐
3. 计算 `frame_count`
4. memmap 为 `frames x points`

#### 4.2 关键代码（单文件完整读取函数）

下面给出一个可直接复用的“读取单个 bin 为二维矩阵”的完整函数示例（与当前项目规则一致）：

```python
from pathlib import Path
import re
import numpy as np

POINTS_PATTERN = re.compile(r"-(\d+)pt-")
SCAN_RATE_PATTERN = re.compile(r"-(\d+)Hz-")
RAW_DTYPE = np.int32
RAW_BYTES_PER_POINT = np.dtype(RAW_DTYPE).itemsize


def read_single_bin_file(file_path: str | Path):
    """
    返回:
      data_memmap: np.memmap, shape=(frame_count, points), dtype=int32
      sample_rate_hz: float
      points: int
      frame_count: int
      duration_s: float
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    name = path.name
    m_pt = POINTS_PATTERN.search(name)
    m_hz = SCAN_RATE_PATTERN.search(name)
    if m_pt is None or m_hz is None:
        raise ValueError(f"Unable to infer points/sample rate from file name: {name}")

    points = int(m_pt.group(1))
    sample_rate_hz = float(m_hz.group(1))

    file_size_bytes = path.stat().st_size
    total_values, remainder = divmod(file_size_bytes, RAW_BYTES_PER_POINT)
    if remainder != 0:
        raise ValueError(f"Invalid file size {file_size_bytes} bytes for int32 data")

    frame_count, frame_remainder = divmod(total_values, points)
    if frame_remainder != 0:
        raise ValueError(f"File size does not match inferred point count {points}")

    data_memmap = np.memmap(
        path,
        dtype=RAW_DTYPE,
        mode="r",
        shape=(frame_count, points),
    )

    duration_s = frame_count / sample_rate_hz
    return data_memmap, sample_rate_hz, points, frame_count, duration_s
```

### 5. 数据处理流程

主流程（`MainWindow`）：

1. `inspect_bin_file()` 解析与校验元数据
2. `read_bin_memmap()` 映射原始 int32
3. `convert_raw_to_radians()` 转为 rad
4. `apply_filter()`（可选）
5. `prepare_image_matrix()` 抽取点段并转置为 `points x frames`
6. `space_time_view.set_data()` 渲染图像与时域曲线

关键调用：

```python
raw = read_bin_memmap(metadata)
phase_data = convert_raw_to_radians(raw)
processed = apply_filter(phase_data, metadata.sample_rate_hz, filter_config)
image_data = prepare_image_matrix(processed, start_point, end_point)
```

### 6. 多通道信号快速滤波实现

当前实现位于 `scripts/processing/filters.py`：

- 使用 `scipy.signal.butter(..., output="sos")`
- 使用 `scipy.signal.sosfiltfilt(..., axis=0)`
- `axis=0` 表示沿时间轴处理，每列对应一个空间点（多通道并行向量化）

关键代码：

```python
sos = signal.butter(config.order, wn, btype=btype, output="sos")
filtered = signal.sosfiltfilt(sos, np.asarray(data, dtype=np.float32), axis=0)
```

设计要点：
- `filtfilt` 零相位，避免相位延迟
- `float32` 降低内存开销
- 配合 LRU 缓存减少重复计算

### 7. 数据导出规则

导出逻辑在 `MainWindow._export_current_view_data()`：

- 导出范围来自当前可视窗口：
  - x 轴秒 -> 帧索引
  - y 轴点号 -> 点索引
- 规则：
  - 未勾选 `Save Filtered Data`：直接从原始 memmap 切片导出
  - 勾选后：从滤波结果切片，再 `rad -> int32`
- 勾选 `Save as txt`：导出为 `txt`，每列对应一个位置点，数据单位为 `rad`
- `txt` 文件名首段保留源文件原有序号前缀，例如 `0000377-`
- 文件命名：`build_export_filename()`，格式 `eDAS-<Hz>-<pt>-<start-time>.bin`
- 二进制导出使用 `tofile()`，保持 frame-major `int32`
- 文本导出使用 `numpy.savetxt()`，输出 frames x points 的 `rad` 浮点矩阵

关键代码：

```python
frame_start = max(0, int(np.floor(x_range[0] * metadata.sample_rate_hz)))
frame_end = min(metadata.frame_count, int(np.ceil(x_range[1] * metadata.sample_rate_hz)))
point_start = max(1, int(np.floor(y_range[0])))
point_end = min(metadata.points, int(np.ceil(y_range[1]) - 1))
...
export_slice.tofile(output_path)
```

### 8. 2026-05-11 更新记录

本次源码更新目标：

- 调整参数区布局，使 `Point Start / Point End`、`VMin / VMax`、`Low Cut / High Cut` 各自位于同一行
- 为上述输入框增加明确的数值范围约束，并保持原有字体字号
- 在导出区域新增 `Save as txt` 选项，和 `Save Filtered Data` 并列显示
- 新增脚本侧正式 `txt` 导出实现，不直接引用 `singledataread/` 下的临时脚本

涉及源码：

- `scripts/ui/control_panel.py`
- `scripts/ui/main_window.py`
- `scripts/io/bin_reader.py`
- `scripts/io/text_export.py`

实现说明：

- 参数区改为“同一行双参数”布局，并统一增大同一行控件间距，避免标签、输入框和复选框过于拥挤
- `Point Start`、`Point End` 输入范围限制为 `0~100000`
- `VMin`、`VMax` 输入范围限制为 `-1000~1000`，保留最多 3 位小数
- `Low Cut`、`High Cut` 输入范围限制为 `0~100000`
- 当勾选 `Save as txt` 时：
  - 导出后缀改为 `.txt`
  - 文件名前缀保留源文件首端序号
  - 文件名中的时间仍按当前视图起始采样时间生成
  - 导出矩阵按 `frames x points` 写出，每列表示一个位置点
  - 导出数值单位为 `rad`
- 当未勾选 `Save Filtered Data` 且导出 `txt` 时，数据来源为原始 `bin` 切片，只做 `raw -> rad` 转换，不使用滤波结果
- 未勾选 `Save Filtered Data` 且导出 `bin` 时，仍直接写原始 `int32` 切片，避免无意义的往返转换

验证记录：

- 已执行 `python -m py_compile scripts\\ui\\control_panel.py scripts\\ui\\main_window.py scripts\\io\\bin_reader.py scripts\\io\\text_export.py`
- 语法检查通过

GitHub 分支记录：

- 本次变更提交目标分支为 `dev`
- 推送时仅同步源码与文档，不包含 `data`、`dist` 等目录

### 9. 修改建议（后续可选）

- 为绘图和导出增加性能计时日志
- 为极大文件增加分块显示/分辨率降采样
- 为文档中的关键函数补充单元测试样例

## 二、开发日志

### 2026-07-28 bz 压缩格式支持

#### 背景

软件原先只支持直接读取 `.bin` 原始相位数据文件。为减小数据文件体积，本次按 `E:\codes\compress\docs` 中的 Bitshuffle + Zstd 上位机 Python 开发指南，补充 `.bz/.bzs` 压缩文件读取和 `.bin -> .bz` 批量转换能力。

#### 数据格式

- `.bin` 仍为原始 `int32`，逻辑形状为 `frames x points`，frame-major 存储。
- `.bz/.bzs` 真实格式为 `BZF1` 文件头 + 多个 `BZS1` packet。
- `BZF1` 文件头结构为 `<4sHHQIIIIIIIiI>`，记录采样率、每帧点数、通道数、数据源、保存抽点因子、packet 帧数、文件时长、Zstd level 和 Bitshuffle block。
- `BZS1` packet header 结构为 `<4sHHQQIIIHHIiQQQQII>`，记录 packet 序号、时间戳、帧数、原始/压缩字节数和 CRC。
- 读取端同时兼容早期无文件头的 `BZS1` packet stream。
- packet header 结构为 `<4sHHQQIIIHHIiQQQQII`。
- 解压流程为 `Zstd decompress -> Bitshuffle unshuffle -> int32 reshape(frames, points)`。
- 每个 packet 校验 `payload_crc32` 和 `raw_crc32`，避免损坏数据静默进入绘图链路。

#### 代码变更

- 新增 `scripts/io/bz_codec.py`：
  - 实现 `BZF1` file header 与 `BZS1` packet header 打包、解析和校验。
  - 实现 NumPy 版 bitshuffle 正逆变换。
  - Zstd 后端优先使用 `python-zstandard`，缺失时尝试加载 `zstd.dll`。
  - 提供 `.bz` 扫描、逐包读取、整文件解压。
- 更新 `scripts/io/bin_reader.py`：
  - `list_bin_files()` 扫描 `.bin/.bz/.bzs`。
  - `inspect_bin_file()` 兼容 `.bz/.bzs` 元数据读取。
  - `read_bin_memmap()` 对 `.bin` 继续返回 memmap，对 `.bz/.bzs` 返回解压后的二维 `ndarray`。
  - 新增 `compress_bin_to_bz_file()` 和 `batch_compress_bin_folder()`。
  - 批量压缩支持按已处理原始字节数汇报进度，并返回耗时、压缩比、压缩效率等统计。
- 更新 `scripts/workers/tasks.py`：
  - `FunctionWorker` 增加 `progress` signal。
  - 支持把 worker 的 progress signal 作为 `progress_callback` 注入到后台函数。
- 更新 `scripts/ui/file_browser.py` 和 `scripts/ui/main_window.py`：
  - 左侧文件区新增 `Batch Bin -> bz` 按钮。
  - 批量压缩前确认源目录、输出目录和文件数。
  - 压缩期间显示非模态进度条，不阻塞主窗口读取其他文件和绘图。
  - 压缩完成后弹窗显示压缩文件数、原始数据体积、压缩后体积、压缩比、用时和压缩效率。
- 更新 `scripts/build_exe.py`：
  - 打包文件名改为 `eDASread-YYYYMMDD.exe`。
  - 如果同名 exe 已存在，自动使用 `eDASread-YYYYMMDD-2.exe`、`eDASread-YYYYMMDD-3.exe` 等名称。
  - 不再删除 `dist/` 目录，避免覆盖或清理旧 exe。
  - 打包时尝试把当前环境的 `zstd.dll` 加入 exe。

#### 使用方式

1. 在左侧选择数据目录。
2. 点击 `Batch Bin -> bz`。
3. 如果右侧 `Export folder` 已填写，`.bz` 输出到该目录；否则输出到原数据目录。
4. 已存在的同名 `.bz` 默认跳过，不覆盖。
5. 压缩完成后文件列表会刷新，`.bz` 文件可直接选择、滤波和绘图。

#### 验证记录

- 语法检查：

```powershell
python -m py_compile scripts\io\bz_codec.py scripts\io\bin_reader.py scripts\workers\tasks.py scripts\ui\file_browser.py scripts\ui\main_window.py scripts\build_exe.py
```

- 小样本往返：
  - 生成临时 `.bin`。
  - 压缩为 `.bz`。
  - 使用 `inspect_bin_file()` 和 `read_bin_memmap()` 读取 `.bz`。
  - `np.array_equal(original, restored) == True`。

- 滤波链路：
  - `.bz -> raw int32 -> rad float32 -> apply_filter(FilterConfig(filter_type="none"))`。
  - 输出形状与原始数据一致。

#### 注意事项

- `.bz` 随机访问暂未建立索引，当前读取会解压整个文件到内存；`.bin` 仍保持 memmap。
- NumPy 版 bitshuffle 以兼容和可验证为主，高速大文件实时压缩性能仍建议使用原生 C/SIMD 后端复测。
- 若运行环境既没有 `python-zstandard` 又找不到 `zstd.dll`，`.bz` 读取和转换会失败并提示安装或放置 Zstd 后端。

### 2026-07-29 文件列表排序

- 左侧文件浏览区新增 `Sort` 下拉框。
- 支持三种排序方式：
  - `Name`：按文件名升序排列，保持原有默认行为。
  - `Modified: Newest`：按文件修改时间倒序排列，最新修改的文件显示在前。
  - `Modified: Oldest`：按文件修改时间升序排列，最早修改的文件显示在前。
- 切换排序方式时会重新分页，并尽量保持当前选中文件不变。
- 刷新目录和批量压缩完成后的文件列表刷新会沿用当前排序方式。

### 2026-08-12 数据预处理、通道分析与 f-k 谱

#### 背景

参照 `E:\codes\DFSpy\surfacewave-C3-inv\notebooks\01_das_preprocess_ncf.ipynb` 中的数据降噪流程，并结合 DASPy 中文教程中的 f-k 波场分解说明，为 eDASread 增加面向 DAS 数据浏览的交互式预处理和分析功能。

#### 功能变更

- 右侧绘图区改为 3 个 Tab：
  - Tab1：数据预处理、指定位置点时域图、time-space 图。
  - Tab2：指定多个通道的时域波形和 PSD。
  - Tab3：指定通道范围和通道间距的 f-k 谱。
- Tab1 新增 4 个独立预处理按钮：
  - `去尖峰噪声`
  - `去共模噪声`
  - `去随机噪声`
  - `FK滤波`
- Tab1 新增 `播放波形` 按钮，可播放当前指定通道波形，并按 99 分位幅值归一化放大音量。
- Tab2 支持输入如 `5,10,15` 的通道列表，时域图自动增加垂直偏移，PSD 使用 dB 单位。
- Tab3 支持输入如 `5-100` 的通道范围和 `5` m 通道间距，使用 Tab1 当前处理后的数据计算 f-k 谱。
- `Save Filtered Data` 导出逻辑调整为导出当前处理矩阵的可视窗口切片，使去噪结果可随当前视图导出。

#### 代码变更

- 新增 `scripts/processing/fdct.py`，引入参考项目中的 curvelet 变换实现。
- 新增 `scripts/processing/denoise.py`，引入 spike、common-mode、curvelet random denoising，并增加 common mode 除零保护。
- 新增 `scripts/processing/denoise_pipeline.py`，封装 UI 调用的单步降噪、FK 滤波和 FK 谱计算。
- 新增 `scripts/render/analysis_views.py`，实现 Tab2 和 Tab3 绘图组件。
- 更新 `scripts/ui/control_panel.py`，新增预处理按钮、FK 参数和播放按钮。
- 更新 `scripts/ui/main_window.py`，改为 3 Tab 布局，接入预处理 worker、Tab2/Tab3 绘图和 WAV 播放。
- 新增 `docs/preprocessing_development.md`，记录算法来源、处理链路、UI 行为、验证记录和后续维护建议。

#### 验证记录

- 语法检查通过：

```powershell
python -m py_compile scripts\processing\fdct.py scripts\processing\denoise.py scripts\processing\denoise_pipeline.py scripts\render\analysis_views.py scripts\ui\control_panel.py scripts\ui\main_window.py
```

- 小样本数值验证通过：`spike`、`common`、`fk`、`random`、`compute_fk_spectrum_db`。
- Qt 构造检查通过：`MainWindow()` 可实例化，右侧 Tab 数量为 3。

### 2026-08-12 Tab 参数区压缩与 PSD/FK 参数化

#### 功能变更

- Tab1 顶部参数区由多个大分组压缩为一个紧凑的 `Tab1 Parameters` 分组，减少空白区域，为 time-space 图保留更大显示空间。
- 调整 Tab 标题样式：
  - 标题字号加大。
  - 当前选中 Tab 使用蓝色背景和白色文字。
  - 当前 Tab 字重更高，和未选中 Tab 区分更明显。
- Tab2/Tab3 绘图区启用鼠标缩放、拖动平移和框选放大。
- Tab2 新增 PSD 参数输入：
  - `PSD窗长`
  - `重叠(%)`
  - `FMin / FMax`
  - `PSD Min / PSD Max`
- Tab3 新增 f-k 谱计算和显示参数输入：
  - `空间taper`
  - `时间taper`
  - `NFFT-K`
  - `NFFT-T`
  - `FMin / FMax`
  - `KMin / KMax`
  - `扣除背景`
- README 增加 Tab1、Tab2、Tab3 主要参数含义说明。

#### 代码变更

- 更新 `scripts/ui/control_panel.py`：
  - 将 Tab1 控制区改为紧凑网格布局。
  - 控件高度收紧到 24-28 px。
- 更新 `scripts/ui/main_window.py`：
  - 增加 Tab 样式表。
  - 增加 PSD/FK 参数读取、`auto` 解析和范围校验。
- 更新 `scripts/render/analysis_views.py`：
  - 增加 Tab2 PSD 参数控件并传入 Welch 计算。
  - 增加 Tab3 FK 参数控件并控制 f-k 谱显示范围。
  - 对 Tab2/Tab3 PlotWidget 启用交互式缩放和平移。
- 新增 `scripts/render/plot_style.py`：
  - 统一 pyqtgraph 前景色、坐标轴线、刻度文字和坐标标签为黑色。
  - Tab1、Tab2、Tab3 的主图、波形图、PSD 图、FK 图和色标轴共用同一套轴样式。
- 更新 `scripts/processing/denoise_pipeline.py`：
  - `compute_fk_spectrum_db()` 支持手动指定空间和时间方向 NFFT。
- 更新 `README.md`：
  - 补充 PSD、FK 和 Tab1 预处理参数说明。

#### 验证记录

- 语法检查通过：

```powershell
python -m py_compile scripts\processing\denoise_pipeline.py scripts\render\plot_style.py scripts\render\analysis_views.py scripts\render\space_time_view.py scripts\ui\control_panel.py scripts\ui\main_window.py
```

- Qt 构造检查通过：右侧 Tab 数量为 3。

### 2026-08-15 大文件读取/滤波/绘图性能优化

#### 背景

面对数百 MB 的 `.bz/.bin` 数据文件，原先的链路存在三处瓶颈：

1. `.bz` 读取必须先解压出完整的 `int32` 数组再转 `rad`，峰值内存翻倍。
2. Bitshuffle 逆变换按块 Python 循环，单文件耗时明显。
3. 绘图总是把可视窗口的完整矩阵上传到 GPU，缩放时大范围视图（几百万列）卡顿明显。

本次优化原则：**只降低显示层分辨率，绝不降低数据采样率、不丢失数据完整性**。所有降采样均为"显示信封"（min/max 或 peak 包络），底层 `current_data` 全分辨率数组不变。

#### 代码变更

- `scripts/io/bz_codec.py`：
  - `shuffle_int32` / `unshuffle_int32` 重写为分块向量化 NumPy 实现，块大小默认 `65536`，每次向量化处理 `1 << 20` 个采样点（`_VECTOR_CHUNK_ITEMS`，为块大小的整数倍，保持块内位平面主序不变）；文件末尾不完整块沿用标量路径。输出与原逐块实现逐字节一致。
  - 修正向量化 `unshuffle_int32` 中 `np.packbits` 缺失 `bitorder="little"` 的问题（默认 `"big"` 会导致位序错误，已用逐块参考实现做往返 + 逐字节比对验证）。
  - `scan_bz_file` 增加 LRU 扫描缓存 `_SCAN_CACHE`（上限 `_SCAN_CACHE_MAX = 32`，key 为 `(路径, st_size, st_mtime_ns)`），避免每次读取重复解析文件头与 packet 索引。
  - `read_bz_file(path, converter=None)`：传入 converter 时逐 packet 将 `int32` 直接转换为 `float32 rad` 写入目标数组，完整的 `int32` 大数组永不物化，峰值内存减半。
- `scripts/io/bin_reader.py`：
  - 新增 `read_phase_radians(metadata)`：`.bz` 走 converter 直接产出 `float32 rad`；`.bin` 走 memmap + 一次向量化 `raw -> rad`。
- `scripts/ui/main_window.py`：
  - `_load_file_data` 改用 `read_phase_radians(metadata)`；`_render_current_data` 把 `frame_data`（全分辨率 `frames x points`）直接传给 `space_time_view.set_data`，移除中间 `prepare_image_matrix` 拷贝。
- `scripts/processing/denoise.py`：
  - `common_mode_noise_removal` 向量化：`xc = np.sum(data * common[None, :], axis=1)` 一次算完全部通道的投影系数，去掉逐通道 Python 循环。
- `scripts/processing/fdct.py`、`denoise_pipeline.py`：
  - 引入 `_FFT_WORKERS = os.cpu_count() or 1`，`denoise_pipeline.py` 的 `numpy.fft` 全部换成 `scipy.fft` 并给 `fft2 / ifft2 / rfft2` 传 `workers`，多核并行加速 f-k 滤波与谱计算。
- `scripts/processing/view_data.py`：
  - 新增 `MAX_IMAGE_DISPLAY_COLUMNS = 16384`、`downsample_image_for_display(image, max_columns)`（按块取 min/max，每块输出两列，返回 `(display, samples_per_col)`，保证全局极值不丢失）。
  - 新增 `robust_median` / `robust_abs_percentile`：对超长序列用步进子集估计，仅用于显示缩放。
- `scripts/render/space_time_view.py`：
  - LOD 渲染：`set_data` 保存 `frame_data`，`sigRangeChanged` 触发 60ms 单发 QTimer，延迟重切可见帧切片并按 `MAX_IMAGE_DISPLAY_COLUMNS` 降采样后上传；图像 transform 用 `samples_per_col / sample_rate` 缩放，保证坐标精确。
  - 时域波形曲线启用 `setDownsampling(auto=True, method="peak")` + `setClipToView(True)`（只影响显示点数，保留每 bin 的 min/max 包络）。
- `scripts/render/analysis_views.py`：
  - Tab2 通道曲线同样启用 peak 降采样；基线中位数与幅值分布改用 `robust_median` / `robust_abs_percentile`。

#### 验证记录

- 语法检查全部通过（见下），共 9 个文件：

```powershell
python -m py_compile scripts\io\bz_codec.py scripts\io\bin_reader.py scripts\ui\main_window.py scripts\processing\denoise.py scripts\processing\fdct.py scripts\processing\denoise_pipeline.py scripts\processing\view_data.py scripts\render\space_time_view.py scripts\render\analysis_views.py
```

- 数值验证（临时脚本）：
  - Bitshuffle：`shuffle -> unshuffle` 往返（含整块、跨 chunk 边界、末尾不完整块），并与逐块参考实现逐字节一致。
  - `bz 压缩 -> read_phase_radians`：输出 `float32 rad`，与 `raw * (pi/32767)` 逐点一致。
  - `common_mode_noise_removal` 向量化版本与逐通道循环版本数值一致。
  - `downsample_image_for_display` 的全局 min/max 与原始矩阵一致（极值不丢失）。
  - `robust_median` 与全量中位数接近。

#### 注意事项

- 位序约定：Bitshuffle 输出为"块内位平面主序"，`unpackbits`/`packbits` 必须使用 `bitorder="little"`，否则解出的 `int32` 字节序错误。
- 显示降采样不影响导出：`Export Current View` 仍按可视窗口直接切全分辨率数据。
- 更详细的可读/滤波/绘图策略见 `docs/eDAS快速读取-处理-绘图策略总结.md`。

### 2026-08-15 修复 time-space 图刷新/色块问题并新增 Tab2 时间范围选区

本次针对用户反馈的 5 个问题逐一定位并修复，并新增 Tab2 的矩形时间选区功能。

#### 问题 1：调整通道范围后 101~200 区间空白

- **现象**：Point End 从 100 改为 200 并 Apply 后，y 轴刻度变为 1~200，但 101~200 区间为空白，需反复切换文件才显示。
- **原因**：`SpaceTimeView._refresh_image()` 用 `(frame0, frame_stop, display_cols)` 作为缓存键判断是否需要重新上传图像，该键只包含时间维，**不包含通道范围**；同时 `set_data()` 没有重置 `_applied_image_key`。通道范围只改变图像行数（点数），不影响键，因此新点段被 `if key == self._applied_image_key: return` 提前跳过，图像仍保持旧的 100 行。
- **解决**：
  - `set_data()` 开头将 `self._applied_image_key = None`，确保每次完整重载都强制重绘。
  - 键改为 `(y_start, y_end, frame_start, frame_stop, display_cols)`，把通道范围纳入缓存键，即使不重置也能识别点段变化。

#### 问题 2：左右滚动时 time-space 图变空白

- **现象**：放大后左右滚动，time-space 图有时整片空白，右侧明明还有数据却不显示。
- **原因**（两个叠加）：
  1. **图像坐标变换顺序错误**（根本原因）：`_refresh_image()` 用 `QTransform` 先 `scale()` 后 `translate()`。Qt 的 `QTransform` 采用后乘（`T = Scale * Translate`），点映射为 `p' = Scale * (p + d)`，即平移量也被缩放。当 `frame0 != 0` 且存在降采样（`scale != 1`）时，图像被错误定位到 `frame0 / sample_rate^2`，远离可视窗口，表现为空白。只有整轴视图（`frame0 = 0`）不受影响，因此问题只在放大/滚动后显现。
  2. LOD 刷新只在 `sigRangeChanged` 后经 60ms **单发**定时器延迟重切，且每次只切"当前可见窗口"这一窄片；滚动过程中定时器被不断重置，图像仍停留在上一次的窄切片。
- **解决**（`space_time_view.py`）：
  - 修正变换顺序为**先 `translate()` 后 `scale()`**，得到 `p' = Scale * p + d`，图像原点正确落在 `frame0 / sample_rate`，放大/滚动后坐标精确。
  - `_frame_slice_for_view()` 改为：当可见帧跨度 `<= MAX_IMAGE_DISPLAY_COLUMNS` 时，切"可见窗口 + 两侧 20% 余量"的全分辨率切片；当缩小到跨度超过上限时，直接上传"整条时间轴 + 降采样"的一张粗图，覆盖全部时长。
  - `_schedule_lod_refresh()` 由"防抖（每次重启计时）"改为"节流（仅当定时器未激活才启动）"，并把间隔降到 30ms，连续滚动时每约 30ms 重切一次，滚动结束即可恢复。
  - 因屏幕宽度远小于 16384 列，缩小态用整轴粗图在视觉上无损。

#### 问题 3：调整滤波范围后很久不更新 / 切换文件后才更新

- **现象**：修改 Low Cut/High Cut 并 Apply 后长时间不更新，或只有切换文件后才更新。
- **原因**：滤波参数变化会使 LRU 缓存 miss，后台会重新读文件并滤波（耗时正常），但结果回填后 `set_data()` 没有重置 `_applied_image_key`，而新数据形状不变，缓存键与旧图相同，`_refresh_image()` 提前返回，新滤波结果没有被上传显示。切换文件后因帧数不同、键发生变化才被迫重绘。
- **解决**：同问题 1，`set_data()` 重置 `_applied_image_key` 并把点段纳入缓存键，滤波结果回填后必然重绘。

#### 问题 4：time-space 图被切成红蓝相间竖条色块

- **现象**：整图呈大量红蓝相间的竖条，而非连续波场。
- **原因**：`downsample_image_for_display()` 把每个时间块降采样为**相邻两列**（块内 min 与块内 max）。在 `seismic` 等发散色标下，负向 min 呈蓝色、正向 max 呈红色，两列并排即形成蓝红交替竖条。
- **解决**（`view_data.py`）：改为每块输出**单列**，取该块内绝对值最大的采样（保留正负主导特征），`samples_per_col` 由 `stride/2` 改为 `stride`；色阶自动范围改用新增的 `robust_min_max()` 从原始切片稳健估计，避免丢失全局极值。

#### 问题 5：新增 Tab2 矩形时间选区

- **需求**：在 Tab2 时域图上用鼠标框选时间范围；左键框选=放大，Ctrl+左键框选=不放大、仅把 PSD 计算数据限定到所选时间范围，并为选区加淡黄色阴影。
- **实现**（`analysis_views.py`）：
  - 新增 `TimeSelectionViewBox(pg.ViewBox)`：左键 + Ctrl 按下时进入选区模式（不调用父类，禁用 RectMode 缩放），拖动过程连续发出 `region_dragged`，释放时发出 `region_selected`；无 Ctrl 的左键拖动仍走默认 RectMode 放大。
  - Tab2 时域图改用该 ViewBox，并添加 `pg.LinearRegionItem`（淡黄阴影、不可拖动）标记选区。
  - `ChannelAnalysisView` 保存最近一次数据/通道/PSD 参数，`set_data` 拆分为 `_render_time_plot()`（全波形）与 `_render_psd()`（PSD 可限定时间范围）；Ctrl 框选后仅重算 PSD，波形视图不缩放。
  - 选区内采样数 < 8 时不做 PSD，避免 `scipy.signal.welch` 报错。

#### 验证记录

- 语法检查通过：

```powershell
python -m py_compile scripts\processing\view_data.py scripts\render\space_time_view.py scripts\render\analysis_views.py
```

- Qt 构造检查通过：`MainWindow()` 可实例化，Tab2 时域图 ViewBox 类型为 `TimeSelectionViewBox`，`psd_region` 为 `LinearRegionItem`。
- 数值/行为验证：
  - `downsample_image_for_display` 输出单列/块，列数不超过上限，无 NaN。
  - 同帧数下 Point End 100→200 重载后图像行数由 100 变为 200；替换等形数据后图像内容同步刷新（问题 1/3 修复）。
  - `_frame_slice_for_view`：缩小态返回整轴 `slice(0, frame_count)`，放大态返回可见窗口 + 余量（问题 2 修复）。
  - Tab2 Ctrl 框选后 `_psd_time_range` 正确更新、阴影显示、PSD 标题显示所选时间范围。

### 2026-08-16 修复自动色阶不稳定 / PSD 刻度重叠并优化 Tab2 交互与标注

#### 问题 6：time-space 图放大变浅、缩小变深

- **现象**：矩形放大后颜色变浅，慢慢缩小（显示范围变大）后颜色突然变深。
- **原因**：当 VMin/VMax 留空（自动色阶）时，`_resolve_levels()` 每次都用**当前可见切片** `robust_min_max()` 计算色阶。放大时切片小、局部 min/max 范围窄 → 同幅值映射到更宽色带、整体偏浅；缩小时切片回到整轴、min/max 范围宽 → 对比度突然增强、整体变深。
- **解决**（`space_time_view.py`）：新增 `_compute_auto_levels()`，在 `set_data()` 时从整条时间轴（选定点段）一次性稳健估计 `_auto_levels` 并缓存；`_resolve_levels()` 直接复用该缓存，缩放/平移过程中色阶恒定。

#### 问题 7：Tab2 PSD x 轴刻度重叠 / 科学计数法难读

- **现象**：PSD 频率轴（对数坐标）刻度文字重叠，且显示为 `2·10¹`、`10²` 等科学计数/上标形式，不易读。
- **原因**（两处）：
  1. pyqtgraph 对数轴在跨度 < ~3 个十倍频时，`AxisItem.logTickValues()` 过滤掉间距 < 1 的层级后只剩一个 `spacing=None` 的"minor"层，含 1~9 每十倍频共 20+ 个刻度，`generateDrawSpecs()` 对顶层无条件绘制标签 → 重叠。
  2. `AxisItem.logTickStrings()` 用 `%0.1g`（1 位有效数字），20/100 被格式化为 `2e+01`/`1e+02` 再转成上标 `2·10¹`/`10²`。
- **解决**（`analysis_views.py`）：
  - 新增 `FrequencyLogAxis(pg.AxisItem)`，重写 `logTickStrings()` 输出普通十进制 `10`、`20`、`100`。
  - 该轴用于 PSD x 轴；`setTickSpacing(major=1.0, minor=0.25)` 强制每十倍频一个主刻度，`setStyle(maxTextLevel=0)` 只给主刻度画标签，1~9 次刻度不再产生标签，消除重叠。
  - 修正对数轴范围单位：`viewRange()`/`setXRange()` 对对数轴以 `log10(Hz)` 为量纲，故 `_apply_psd_ranges()` 在设 FMin/FMax 与默认范围时统一做 `log10` 换算。

#### 任务：Tab2 交互与标注优化

- **键盘左右滚动**：时域图/PSD 图被点击后设为 `_active_plot`（通过 `scene().sigMouseClicked`），新增 `pan_active_plot()` 按可视跨度的 30% 平移；主窗口 `eventFilter()` 在 Tab2 激活时把左/右键转交 `channel_analysis_view.pan_active_plot()`，不再劫持给 Tab1。时域图按 `[0, 时长]` 限位，PSD 按 `[10, Nyquist]`（log10 量纲）限位。
- **两图分割线**：Tab2 时域图与 PSD 图由固定拉伸改为 `QSplitter(Qt.Vertical)`，可拖动分割线调整两图高度。
- **标注调整**：时域图 y 标签由 `Phase + offset (rad)` 改为 `Amplitude (rad)`；两图图例 `Ch N` 改为 `CH N`；PSD 横轴单位 Hz，默认频率范围改为 `[10 Hz, Fs/2]`（FMin/FMax 留空时）。

#### 验证记录

- 语法检查通过：

```powershell
python -m py_compile scripts\processing\view_data.py scripts\render\space_time_view.py scripts\render\analysis_views.py scripts\ui\main_window.py
```

- Qt 构造检查通过：`MainWindow()` 可实例化，Tab2 内为 `QSplitter`（2 个 child）。
- 行为验证：
  - `_auto_levels` 计算一次后 `_resolve_levels()` 放大/缩小返回值一致（色阶稳定）。
  - PSD 轴默认范围 `[10, 500] Hz`（`log10` 域 `[1.0, 2.699]`），主刻度标签为 `10`、`100`，`maxTextLevel=0` 抑制 1~9 次刻度标签。
  - `pan_active_plot()`：时域图右移后 x 范围被限位到 `[0, ...]`，PSD 图按 `log10` 量纲平移。
  - 时域图 y 标签为 `Amplitude`(units=`rad`)，图例显示 `CH 5 / CH 10 / CH 15`。

### 2026-08-22 参数跨文件/跨会话持久化（Tab1/Tab2/Tab3）

#### 需求

- Tab1/Tab2/Tab3 的参数不随更换/打开数据文件而变化；
- 每次打开软件，默认显示最近一次使用的参数。

#### 实现

- 新增 `scripts/core/settings.py`：`ParameterStore` 基于 `QSettings`，显式使用 `IniFormat + UserScope`（配置文件位于 `%APPDATA%\eDASread\eDASread.ini`）。显式指定格式可避免两参构造落到 Windows 注册表，也便于测试用 `QSettings.setPath()` 整体重定向。
- `main_window.py`：
  - 删除 `_sync_analysis_defaults()`：原先每次加载数据都会把 Tab2 通道重置为 `5,10,15`、Tab3 通道范围重置为 `5-100`，现完全不再随文件切换改动。
  - `_apply_auto_point_limit()` 改造：Point Start/End 一旦来自用户编辑或持久化恢复（`_point_end_manual_override=True`），不再被自动重置为 `min(点数,100)`；新增 `_clamp_point_range_to_points()` 仅当数值超出新文件通道数 `[1, points]` 时夹取，保证渲染始终有效。
  - 新增 `_parameter_widgets()` 注册表统一管理持久化字段：Tab1 全部输入/下拉/复选项与导出目录、Tab2 通道分析全部参数、Tab3 f-k 谱全部参数，另含左侧数据目录。
  - 自动保存：所有注册控件的用户编辑信号（`textEdited`/`currentTextChanged`/`toggled`）与波形点滚轮 `wheelStepped` 触发 600ms 单发防抖保存；`Reset` 恢复出厂值后同样触发保存；`closeEvent()` 停止计时器并强制落盘。
  - 启动流程：构建 UI 后先 `_restore_persisted_parameters()`，再处理 `default_directory`（后者降级为"无历史记录时的回退值"），最后同步加载数据目录。
- 兼容性：首次运行（无历史配置）仍使用原出厂默认值；恢复的 Point End 视为用户值，切换文件时只做越界夹取。

#### 细节与坑

- `QSettings(org, app)` 两参构造使用 NativeFormat（Windows 注册表），不受 `setDefaultFormat()`/`setPath()` 影响——测试隔离必须让生产代码与测试使用同一显式构造方式。
- 非可编辑 `QComboBox` 恢复时先 `findText()` 校验，非法历史值直接跳过，防止脏数据破坏下拉框。
- `WheelIntLineEdit.set_bounds()` 会把显示值静默夹取进新边界，天然满足"换文件后指定位置点仍在 Point Start~End 内"。

#### 验证记录

- 语法检查通过：

```powershell
python -m py_compile scripts\core\settings.py scripts\ui\main_window.py
```

- 新增离屏冒烟测试 `scripts/tests/smoke_test_parameters.py`（`QT_QPA_PLATFORM=offscreen`，`QSettings.setPath` 重定向至临时 INI），27 项断言全部通过：
  - 启动恢复最近参数（含复选框布尔、下拉合法值校验）；
  - 切换文件后 Tab2/Tab3 参数不变；大文件下 Point Start/End 保持原值，小文件（40 点）仅把 Point End 夹取到 40；
  - `closeEvent` 落盘后第二次启动完整恢复修改值；
  - 恢复的 Point End 正确置位 `_point_end_manual_override`。
- 测试期间误写入注册表 `HKCU:\Software\eDASread` 的临时数据已清理。

### 2026-08-28 新增 Tab4 单通道时域/时频图

#### 需求

- 新增 Tab4，用于显示指定通道的时域波形（上）和短时时频图（下）。
- Tab4 使用 Tab1 当前处理后的 `current_data`，但可在 Tab4 内对指定通道进一步低通/高通/带通滤波。
- 支持通道号、滤波参数、时频窗长、重叠比例、PSD/Amplitude 模式、Log/Linear 数值显示、频率轴范围、颜色范围与色标设置。
- 时域图和时频图均支持矩形放大；键盘左右键按最近点击的 Tab4 图进行水平平移。
- 时频图纵轴按对数频率显示，并参考 FIPread 的实现，避免直接把线性频率谱贴到对数轴造成显示失真。

#### 实现

- 新增 `scripts/processing/time_frequency.py`：
  - `compute_time_frequency_map()` 使用 `scipy.signal.spectrogram()` 计算短时谱；
  - PSD 模式使用 `scaling="density", mode="psd"`；
  - Amplitude 模式使用 `scaling="spectrum", mode="magnitude"`；
  - `build_log_frequency_display_grid()` 将线性频率输出重采样到等间距 `log10(f)` 网格，再交给 `ImageItem` 显示。
- 更新 `scripts/render/analysis_views.py`：
  - 新增 `LogFrequencyAxis`，内部坐标为 `log10(Hz)`，刻度标签显示真实 Hz；
  - 新增 `TimeFrequencyAnalysisView`；
  - 上方 `PlotWidget` 绘制单通道时域波形，启用 peak downsampling 和 clip-to-view；
  - 下方 `PlotWidget + ImageItem + HistogramLUTWidget` 绘制时频图和颜色条；
  - 两个图均使用 `ViewBox.RectMode` 支持鼠标矩形放大；
  - 记录最近点击的时域图/时频图，`pan_active_plot()` 用于键盘左右平移。
- 更新 `scripts/ui/main_window.py`：
  - 创建并添加 `"Tab4 时域/时频"`；
  - 新增 `plot_time_frequency()`，通过 `FunctionWorker` 后台执行 Tab4 单通道滤波与时频计算；
  - 新增 `_build_tab4_filter_config()`、`_read_tab4_time_frequency_options()`、`_read_channel_number()`；
  - Tab4 控件加入 `_parameter_widgets()`，支持跨文件、跨会话持久化；
  - `eventFilter()` 在 Tab4 激活时把左右键交给 `time_frequency_view.pan_active_plot()`。

#### 参数约定

- `通道` 使用 1-based 通道号，与 Tab1/Tab2/Tab3 一致。
- `滤波` 支持 `none / lowpass / highpass / bandpass`：
  - highpass 使用 `Low Cut`；
  - lowpass 使用 `High Cut`；
  - bandpass 同时使用 `Low Cut` 和 `High Cut`。
- `窗长(s)` 为 spectrogram 的 `nperseg = round(window_seconds * sample_rate)`，并限制在 `[8, trace_length]`。
- `重叠(%)` 对应 `noverlap = nperseg * overlap_percent / 100`，有效范围 `[0, 100)`。
- `FMin/FMax` 是时频图纵轴显示范围，单位 Hz；对数频率显示下必须大于 0。
- `VMin/VMax` 是时频图颜色范围；勾选 `颜色自动` 时使用数据分位数自动设置。

#### 验证记录

- 语法检查通过：
```powershell
python -m py_compile scripts\processing\time_frequency.py scripts\render\analysis_views.py scripts\ui\main_window.py
```

- Qt 构造检查通过：
```powershell
$env:QT_QPA_PLATFORM='offscreen'; python -c "import sys; from PyQt5.QtWidgets import QApplication; from scripts.ui.main_window import MainWindow; app=QApplication(sys.argv); w=MainWindow(); print(w.plot_tabs.count(), [w.plot_tabs.tabText(i) for i in range(w.plot_tabs.count())]); w.close(); app.quit()"
```
输出 Tab 数量为 `4`，标题为 `['Tab1 数据预处理', 'Tab2 通道时域/PSD', 'Tab3 f-k谱', 'Tab4 时域/时频']`。

- 合成信号时频计算与 Tab4 渲染入口验证通过：
  - 1000 Hz 采样率、4096 点、30 Hz 正弦信号；
  - `window_seconds=0.256`、`overlap_percent=50`；
  - spectrogram 输出矩阵 `(129, 31)`；
  - 去掉 0 Hz 后的对数显示矩阵 `(128, 31)`；
  - Tab4 `set_result()` 可完成时域曲线、时频图、色标和频率范围设置。

#### 2026-08-28 补充：修正 Tab4 时频图 y 轴刻度标签

- 问题：Tab4 时频图 y 轴使用 `log10(Hz)` 坐标，但原先直接格式化 pyqtgraph 自动生成的普通线性刻度，导致显示 `158`、`251` 等不符合常规对数频率图习惯的标签。
- 修正：
  - `LogFrequencyAxis.tickStrings()` 对非整数 decade 位置返回空标签；
  - `TimeFrequencyAnalysisView` 新增 `_update_frequency_axis_ticks()`，参考 FIPread 的实际做法，手动设置左轴 ticks；
  - 主刻度只显示 `10`、`100`、`1000` 等 10 的整数次幂；
  - 2..9 倍频位置作为次刻度保留但不显示文本；
  - 时频图 y 轴范围变化后自动刷新刻度。
- 验证：
```powershell
python -m py_compile scripts\render\analysis_views.py scripts\ui\main_window.py scripts\processing\time_frequency.py
$env:QT_QPA_PLATFORM='offscreen'; python -c "import sys, numpy as np; from PyQt5.QtWidgets import QApplication; from scripts.render.analysis_views import LogFrequencyAxis; app=QApplication(sys.argv); axis=LogFrequencyAxis('left'); print(axis.tickStrings([1.0, 2.0, 2.2, 2.4, 3.0], 1, 1)); app.quit()"
```
输出为 `['10', '100', '', '', '1000']`。

#### 2026-08-28 补充：Tab4 时域图与时频图横轴联动

- 新增 Tab4 上下两个子图的 x 轴双向同步：
  - 时域图矩形放大或水平平移时，时频图自动切换到相同时间范围；
  - 时频图矩形放大或水平平移时，时域图自动切换到相同时间范围；
  - 键盘左右键平移继续按最近点击的子图触发，但最终两个子图保持一致。
- 两个子图统一使用原始通道 trace 的时间范围 `[0, trace_length / sample_rate]` 作为横轴边界，避免时频图因 spectrogram 窗中心时间而产生独立边界。
- 实现细节：
  - `TimeFrequencyAnalysisView` 连接两个 ViewBox 的 `sigXRangeChanged`；
  - 新增 `_syncing_x_ranges` 防止双向同步递归；
  - 新增 `_set_linked_x_range()`、`_sync_x_range()`、`_clamp_x_range()`，保证同步后的范围有效且不越界；
  - 内部状态初始化提前到信号连接之前，避免 Qt 初始化阶段的 range 信号访问未初始化属性。
- 验证：
```powershell
python -m py_compile scripts\render\analysis_views.py scripts\ui\main_window.py scripts\processing\time_frequency.py
```
合成信号离屏验证中，`time_plot.setXRange(1.0, 2.0)` 后两个子图均为 `[1.0, 2.0]`；`tf_plot.setXRange(2.0, 3.0)` 后两个子图均为 `[2.0, 3.0]`。

#### 2026-08-28 补充：Tab2/Tab4 坐标轴刻度方向

- 调整 `scripts/render/analysis_views.py` 中 Tab2 和 Tab4 的绘图轴样式：
  - 左侧纵轴刻度线朝外；
  - 底部横轴刻度线朝下。
- 实现方式：
  - 在 `ChannelAnalysisView._style_plot()` 和 `TimeFrequencyAnalysisView._style_plot()` 中，按 pyqtgraph 的轴方向规则分别设置 `left: tickLength=-8`、`bottom: tickLength=8`；
  - 调整范围仅限 Tab2/Tab4，不影响 Tab1 time-space 图和 Tab3 f-k 图。
- 验证：
```powershell
python -m py_compile scripts\render\analysis_views.py
```
离屏构造检查确认 Tab2/Tab4 的 `left` 轴 `tickLength=-8`、`bottom` 轴 `tickLength=8`；Tab4 时频图 y 轴 decade 标签仍输出 `['10', '100', '', '', '1000']`。

#### 2026-08-28 补充：空 bin 文件读取保护

- 问题：目录中存在 0 字节 `.bin` 文件时，`inspect_bin_file()` 会生成 `frame_count=0` 的 metadata，随后 `read_bin_memmap()` 调用 `np.memmap()` 时报错 `ValueError: cannot mmap an empty file`，并通过后台 worker 弹出完整 traceback。
- 修正：
  - `_inspect_raw_bin_file()` 在解析阶段检查 `file_size_bytes <= 0`，空文件直接报 `File <name> is empty and cannot be loaded.`；
  - `_inspect_raw_bin_file()` 检查 `frame_count <= 0`，避免无完整帧文件继续进入读取流程；
  - `read_bin_memmap()` 增加二次保护，拒绝 `file_size_bytes <= 0` 或 `frame_count <= 0` 的 metadata；
  - `FunctionWorker` 的错误信号改为发送 `str(exc)`，用户界面显示简洁错误，不再显示完整 traceback。
- 验证：
```powershell
python -m py_compile scripts\io\bin_reader.py scripts\workers\tasks.py scripts\ui\main_window.py
```
临时创建 0 字节测试文件后，`inspect_bin_file()` 输出 `ValueError File eDAS-2000Hz-0100pt-20260828T000000.bin is empty and cannot be loaded.`。

#### 2026-08-28 补充：Tab4 初始显示窗口与时间轴像素对齐

- 新增 Tab4 `显示窗口(s)` 参数：
  - 默认值 `0.1`；
  - 绘制完成后，上方时域图和下方时频图默认显示 `[0, 显示窗口]` 时间段；
  - 填写 `auto` 或留空时显示完整 trace 时长；
  - 参数加入持久化键 `tab4/visible_window_seconds`。
- 修正 Tab4 上下时间轴刻度像素对齐：
  - 下方时频图右侧有 150 px 色标，上方时域图右侧新增等宽空白占位，使上下两个绘图区宽度一致；
  - 上下两个 PlotWidget 的左轴固定为 `96` px，底部时间轴固定为 `62` px；
  - 同步 x 轴范围后，两个子图的时间刻度在屏幕像素位置上保持上下对齐。
- 验证：
```powershell
python -m py_compile scripts\render\analysis_views.py scripts\ui\main_window.py scripts\processing\time_frequency.py
```
合成信号离屏验证中，`visible_window_seconds=0.1` 时上下两个子图初始 x 范围均为 `[0.0, 0.1]`；Tab4 两个左轴 `fixedWidth=96`，两个底轴 `fixedHeight=62`。

#### 2026-08-28 补充：增大 Tab4 时频图显示高度

- 问题：Tab4 下方 `tf_plot + HistogramLUTWidget` 位于同一个水平行内，右侧色标固定宽度但高度随父容器扩展；左侧 `PlotWidget` 内部还有标题、坐标轴和固定底轴高度，若垂直 splitter 初始尺寸由默认 `sizeHint` 分配，容易出现色标很高、时频图实际绘图区偏矮且下方留白较大的观感。
- 修正：在 `scripts/render/analysis_views.py` 中为 Tab4 时域图、时频图、下方时频行和色标设置明确的 `QSizePolicy` 与最小高度，并给垂直 `QSplitter` 设置初始尺寸 `[220, 520]`，让主要高度分配给短时时频图。
- 范围：仅调整 Tab4 布局尺寸策略，不改变时频计算、坐标轴联动、颜色范围或数据渲染逻辑。

验证：

```powershell
python -m py_compile scripts\render\analysis_views.py scripts\ui\main_window.py scripts\processing\time_frequency.py
```

离屏构造 `TimeFrequencyAnalysisView` 并填入合成时频数据，在 `1200x900` 窗口中读到：时域图高度 `220 px`、时频图高度 `521 px`、色标高度 `521 px`。

### 2026-08-28 文档同步与版本推送

- 更新 `README.md`：
  - 主要功能列表新增 Tab4 条目（单通道时域波形与短时时频图、对数频率轴）；
  - 新增完整的 "Tab4 时域/时频" 参数说明小节，覆盖通道、滤波、模式（PSD/Amplitude）、数值标度（Log/Linear）、窗长、重叠、FMin/FMax、VMin/VMax、显示窗口、Colormap、颜色自动、x 轴双向联动与键盘平移；
  - 功能列表补充空 bin 文件读取保护说明。
- 更新 `docs/readme.md`：文档总览的主要能力列表补充 Tab1 预处理、Tab2 PSD、Tab3 f-k 谱、Tab4 时频图与空文件保护条目。
- 验证：`py_compile` 全部通过；离屏 Qt 构造检查输出 4 个页签标题 `['Tab1 数据预处理', 'Tab2 通道时域/PSD', 'Tab3 f-k谱', 'Tab4 时域/时频']`。
- 提交 `0219310`（Add Tab4 waveform/time-frequency view, harden empty bin file handling, update README and docs）并推送到 `origin/dev`；按用户决定 main 分支保持不变。

### 2026-08-29 Tab4 改进：时间刻度对齐、刻度朝向、紧凑布局、色标 dB 单位、logo 更换

#### 需求

- 放大 Tab4 时频图时，y 轴刻度数字位数会变化（如 `100` → `1000`），导致时频图左侧轴自动宽度发生变化，进而使时域图与时频图下方的时间刻度无法在像素上对齐。
- 时频图 y 轴刻度需要朝外。
- Tab4 参数区原来占用 4 行、每个输入框过宽，浪费空间，需优化布局。
- `数值` 选择 `Log` 模式时，时频图右侧色标需要显示相对单位 `dB`。
- 将软件所有 logo 更换为新图（源图为 `assets/eDASread-logo2.png`）。

#### 实现（scripts/render/analysis_views.py）

- **时间刻度对齐（根因修复）**
  - 根因：pyqtgraph `AxisItem` 在 `fixedWidth=None` 时按刻度文本宽度自动调整轴宽（`_updateWidth()` 中 `w = self.textWidth`）。时频图左轴 `LogFrequencyAxis` 的主刻度标签是 `10`、`100`、`1000` 等 10 的幂，缩放时可见 decade 变化使标签位数变化，轴宽随之变化，导致绘图区 x 方向位移。
  - 修复：对 `time_plot` 与 `tf_plot` 的左轴均调用 `setWidth(96)` 固定宽度，缩放时轴宽不再变化，两图绘图区 x 起点与宽度始终一致。
  - 追加修正右侧对齐：原 `time_right_spacer` 宽度为 `156`，但 `tf_row_layout` 中 histogram（`setFixedWidth(150)`）与 plot 之间还有 `6` px 布局间距，两者右侧占位分别为 `156+6=162` 与 `150+6=156`，导致时域图 ViewBox 恒比时频图窄 `6` px。将 `time_right_spacer` 改为 `150`，使两侧占位一致。
- **y 轴刻度朝外**：pyqtgraph 中 `tickLength` 为负值时刻度线伸向绘图区内侧（`p2[axis] += tickLength*tickDir`，左轴 `tickDir=-1`），原实现 `left: -8` 为朝内。统一改为正值 `8`，左轴刻度朝外、底轴刻度朝下（仅调整 Tab4 的 `_style_plot`，与 Tab2 独立）。
- **紧凑布局**：将参数区从 `4 行 × 4 列` 改为 `2 行 × 8 列`；数值输入框 `setMaximumWidth(80)`、下拉框 `setMaximumWidth(100)`，并压缩 `QGridLayout` 的 margins/spacing，减少占用高度。
- **色标 dB 单位**：`set_result()` 中根据 `value_scale` 动态设置色标轴标签——`Log` 时 `setLabel(text="Amplitude", units="dB")`，`Linear` 时 `units=""`；初始化时保持默认无单位，每次绘图按当前 `数值` 选项刷新。

#### logo 更换（assets/）

- 源图：`assets/eDASread-logo2.png`（1132×1132 RGBA）。
- 生成运行时 logo `assets/eDASread_logo.png`（256×256，程序实际引用此文件）。
- 用 `QImage` 生成打包用 `assets/eDASread_logo.ico`（与 `build_exe.py::_build_icon` 逻辑一致）。
- 删除冗余旧 logo：`eDASread-logo1.png`、`eDASread-logo2.ico`、`eDASread_logo.ico1`、`eDASread_logo1.png`。

#### 验证记录

- 语法检查通过：
```powershell
python -m py_compile scripts\render\analysis_views.py scripts\ui\main_window.py scripts\processing\time_frequency.py scripts\io\bin_reader.py scripts\workers\tasks.py
```
- 离屏 `TimeFrequencyAnalysisView` 合成信号（2000 Hz、4096 点、30 Hz 正弦、0.256 s 窗、50% 重叠）验证：
  - `time_plot` 与 `tf_plot` 左轴 `fixedWidth=96`、`tickLength=8`；
  - 缩放前与 `setYRange(1.0, 2.4)` 缩放后，两图 ViewBox 场景矩形均为 `x=97.0, width=946.5`，x 起点与宽度精确一致；
  - `值 scale=log` 时色标 `labelText=Amplitude, labelUnits=dB`；`value_scale=linear` 时 `labelUnits` 为空。
- 离屏 `MainWindow` 构造检查：4 个页签、标题 logo `setPixmap` 非空。

### 2026-08-29 移除标题栏 logo + 修复 Tab1 上下时间刻度对齐

#### 需求

- 软件左上角标题栏（软件名字左侧）不再显示 logo；但窗口最小化、任务栏以及打包生成的 exe 仍需保留 logo 图标。
- Tab1（数据预处理）上方的指定点时域波形图与下方的 time-space 图，在 ViewBox 左侧未对齐：点号范围较大（如数千点）时，Point Index 左轴标签位数增多导致轴宽变大，下方图整体右移。

#### 实现

- **移除标题栏 logo（`scripts/ui/main_window.py`）**
  - 删除 `self.title_logo_label` 的创建、`_build_layout()` 中的 `setFixedSize/setAlignment/setPixmap` 设置，以及标题行布局中 `title_logo_label` 与其对称占位 `right_spacer` 的加入；标题栏改由两个 `addStretch(1)` 夹住标题居中。
  - 清理不再使用的 `QPixmap` 导入。
  - 保留 `_apply_window_icons()`（`__init__` 中调用）与 `app.py` 的 `app.setWindowIcon()`，用于窗口最小化/任务栏图标；打包图标由 `build_exe.py --icon`（使用 `assets/eDASread_logo.ico`）提供。
- **修复 Tab1 上下时间刻度对齐（`scripts/render/space_time_view.py`）**
  - 根因：两个子图左轴 `fixedWidth=None`，pyqtgraph 按刻度文本宽度自动调整轴宽。当可见点号范围增大时，下方 `Point Index` 轴标签（如 `1000`/`8000`）比上方 `Phase` 轴（小数，如 `-0.05`）更宽，实测差距可达 `87.4 px` vs `57.4 px`，使下方 ViewBox 起点右移约 `30 px`。
  - 修复：与 Tab4 一致，对 `waveform_plot` 与 `plot_widget` 的左轴均 `setWidth(96)`，固定为相同宽度；两子图本就通过 `waveform_plot.setXLink(self.plot_widget)` 同步 x 轴且右侧占位对称（`waveform_right_spacer` 150 vs `histogram_widget` 150），固定左轴后左右完全对齐，缩放/大数据量下不再偏移。

#### 验证记录

- 语法检查通过：
```powershell
python -m py_compile scripts\ui\main_window.py scripts\render\space_time_view.py scripts\render\analysis_views.py
```
- 离屏 `SpaceTimeView` 合成数据验证（2000 帧 × 9000 点、采样率 500 Hz、显示点号 1-9000、波形点 500）：
  - 上方波形图与下方 time-space 图 ViewBox 场景矩形均为 `x=97.0, width=946.5`，x 起点与宽度精确一致；
  - 两者左轴 `fixedWidth=96`。
- 离屏 `MainWindow` 检查：`hasattr(window, 'title_logo_label')` 为 `False`（标题栏 logo 已移除）；`window.windowIcon()` 与 `app.windowIcon()` 均非空（最小化/任务栏图标保留）。
