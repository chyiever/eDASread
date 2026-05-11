# eDASread 开发文档（dev）

本文档面向开发者，聚焦：绘图实现、卡顿控制、bin 读取方式、数据处理链路、多通道快速滤波、导出规则。

## 1. 主要技术栈

- UI 框架：`PyQt5`
- 绘图库：`pyqtgraph`
- 数值计算：`numpy`
- 数字滤波：`scipy.signal`
- 异步执行：`QThreadPool + QRunnable`

## 2. 绘图实现方式

### 2.1 双图结构与时间轴联动

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

### 2.2 ImageItem 坐标变换

将“帧索引”映射到“秒”，并将 y 轴映射到点号：

```python
transform = QTransform()
transform.scale(1.0 / metadata.sample_rate_hz, 1.0)
transform.translate(0.0, start_point)
self.image_item.setTransform(transform)
```

### 2.3 指定位置点时域波形

时域数据直接从 time-space 使用的同一矩阵抽取：

```python
waveform_index = self._waveform_point - self._y_start_point
waveform = np.asarray(self._image_data[waveform_index], dtype=np.float32)
x_values = np.arange(waveform.shape[0], dtype=np.float32) / self._metadata.sample_rate_hz
self.waveform_curve.setData(x=x_values, y=waveform)
```

## 3. 防止卡顿的实现

### 3.1 大文件读取：memmap

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

### 3.2 耗时任务异步化

- 读取/滤波/导出放入线程池
- UI 线程只负责交互与绘图

```python
worker = FunctionWorker(self._load_file_data, metadata, filter_config)
self.thread_pool.start(worker)
```

`FunctionWorker`（`scripts/workers/tasks.py`）使用 `QRunnable` + 信号回传结果/错误。

### 3.3 结果缓存

- `main_window.py` 使用 `LruByteCache`
- key = `(file_path + filter_config)`
- 避免频繁重复滤波

```python
cache_key = (str(metadata.file_path),) + filter_config.cache_key()
processed = self.processed_cache.get(cache_key)
```

### 3.4 渲染数据内存优化

- 统一使用 `float32`
- 绘图前转换连续内存，减少拷贝成本

```python
return np.ascontiguousarray(frame_data[:, point_slice].T)
```

## 4. bin 数据读取方式

### 4.1 当前读取流程

1. 从文件名解析采样率与点数
2. 校验文件长度与 `int32` 对齐
3. 计算 `frame_count`
4. memmap 为 `frames x points`

### 4.2 关键代码（单文件完整读取函数）

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

## 5. 数据处理流程

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

## 6. 多通道信号快速滤波实现

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

## 7. 数据导出规则

导出逻辑在 `MainWindow._export_current_view_data()`：

- 导出范围来自当前可视窗口：
  - x 轴秒 -> 帧索引
  - y 轴点号 -> 点索引
- 规则：
  - 未勾选 `Save Filtered Data`：直接从原始 memmap 切片导出
  - 勾选后：从滤波结果切片，再 `rad -> int32`
- 勾选 `Save as txt`：导出为 `txt`，每列对应一个位置点，数据单位为 `rad`
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

## 8. 2026-05-11 更新记录

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
  - 文件名中的时间仍按当前视图起始采样时间生成
  - 导出矩阵按 `frames x points` 写出，每列表示一个位置点
  - 导出数值单位为 `rad`
- 未勾选 `Save Filtered Data` 且导出 `bin` 时，仍直接写原始 `int32` 切片，避免无意义的往返转换

验证记录：

- 已执行 `python -m py_compile scripts\\ui\\control_panel.py scripts\\ui\\main_window.py scripts\\io\\bin_reader.py scripts\\io\\text_export.py`
- 语法检查通过

GitHub 分支记录：

- 本次变更提交目标分支为 `dev`
- 推送时仅同步源码与文档，不包含 `data`、`dist` 等目录

## 9. 修改建议（后续可选）

- 为绘图和导出增加性能计时日志
- 为极大文件增加分块显示/分辨率降采样
- 为文档中的关键函数补充单元测试样例
