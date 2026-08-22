# eDAS 快速读取-处理-绘图策略总结

面向开发者，总结 eDASread 针对数百 MB 级 `.bz/.bin` 数据文件采用的读取、滤波、绘图加速策略。

## 0. 核心原则

- **不降低数据采样率、不丢失数据完整性**：所有"降采样"都只发生在显示层（信封/包络），底层 `current_data` 全分辨率数组不变，导出仍按可视窗口切全分辨率数据。
- **延迟求值**：能切片时切片，能缓存时缓存，能异步时异步，能向量化时向量化。
- **峰值内存控制**：避免同一份数据的 `int32` 与 `float32` 两个全量副本同时存在。

## 1. 读取策略

### 1.1 `.bin`：memmap 懒映射

```python
raw = np.memmap(metadata.file_path, dtype=np.int32, mode="r",
                shape=(metadata.frame_count, metadata.points))
```

- 不把整个文件读进内存，按页按需访问。
- 转换为 `float32 rad` 时做一次向量化乘法 `raw * (pi/32767)`。

### 1.2 `.bz/.bzs`：converter 逐包转换

格式：`BZF1` 文件头 + 多个 `BZS1` packet（`Zstd 解压 -> Bitshuffle 逆变换 -> int32`）。

```python
read_bz_file(path, converter=lambda p: np.asarray(p, dtype=np.float32) * PHASE_RAD_SCALE)
```

- converter 在**每个 packet 解压后立即**把 `int32` 转成 `float32 rad` 写入目标数组。
- 全量 `int32` 数组永不物化，峰值内存减半，且省去一次整文件往返拷贝。
- `scan_bz_file` 结果做 LRU 缓存（key 含路径、大小、mtime_ns），重复读取不重复解析文件头与 packet 索引。

### 1.3 Bitshuffle 向量化

- 分块向量化，一次处理 `1 << 20` 个采样点（块大小 `65536` 的整数倍）。
- 保持"块内位平面主序"不变：`unpackbits(..., bitorder="little")` -> `transpose(0,2,1)` -> `packbits(..., bitorder="little")`。
- 末尾不完整块走标量路径。
- 位序约定：**必须用 `bitorder="little"`**，否则解出的 `int32` 字节序错误（曾因 `packbits` 缺省 `"big"` 出现 bug，已用参考实现逐字节比对修复）。

### 1.4 统一入口

```python
def read_phase_radians(metadata):
    if is_bz_path(metadata.file_path):
        return read_bz_file(metadata.file_path, converter=...)
    raw = read_bin_memmap(metadata)
    return convert_raw_to_radians(raw)
```

主流程 `main_window._load_file_data` 直接调用 `read_phase_radians(metadata)`，随后可选滤波并写入 LRU 缓存。

## 2. 滤波/处理策略

### 2.1 多通道滤波

- `scipy.signal.sosfiltfilt(..., axis=0)`：沿时间轴处理，每列一个空间点，多通道向量化。
- 零相位，`float32`，配合 LRU 缓存复用结果。

### 2.2 共模噪声去除

向量化投影，去掉逐通道 Python 循环：

```python
common = np.median(data.T, axis=0)          # 共模信号
xc = np.sum(data.T * common[None, :], axis=1)   # 所有通道投影系数一次算完
data_dn = data.T - (xc / np.sum(common ** 2))[:, None] * common[None, :]
```

除零保护：共模信号能量趋零时直接返回原数据。

### 2.3 f-k 滤波与谱计算并行

- `numpy.fft` 全部换成 `scipy.fft`，给 `fft2 / ifft2 / rfft2` 传 `workers=_FFT_WORKERS`（`os.cpu_count() or 1`）。
- `fdct.py`（curvelet）内部 FFT 同样使用多 worker。

## 3. 绘图策略

### 3.1 显示层降采样（单列块，无红蓝竖条）

```python
def downsample_image_for_display(image, max_columns=16384):
    stride = ceil(ncols / max_columns)
    # 每个 stride 列块 -> 1 列：块内 |值| 最大的采样（保留正负主导特征）
    return display, samples_per_col  # samples_per_col = stride
```

- 每块输出**单列**（取绝对值最大的采样），避免发散色标下 min/max 双列造成的红蓝相间竖条，同时保留正负主导特征。
- `samples_per_col` 用于坐标变换；注意 Qt `QTransform` 后乘语义，须**先 `translate` 后 `scale`**，即 `transform.translate(frame0/sample_rate, y_start)` 再 `transform.scale(samples_per_col/sample_rate, 1)`，否则平移量会被缩放、放大/滚动后图像错位空白。
- 色阶自动范围用 `robust_min_max()` 从原始切片稳健估计，不因降采样丢失全局极值。
- 时间轴波形曲线用 pyqtgraph 的 `setDownsampling(auto=True, method="peak")` + `setClipToView(True)`，同样只保留显示包络。

### 3.2 LOD（Level of Detail）刷新

- 可见帧切片由当前 viewBox 范围计算：`slice(floor(x0*fs), ceil(x1*fs))`。
- 缩小态（可见跨度 > 16384 列）：直接上传整条时间轴 + 降采样的粗图，滚动不留白。
- 放大态（可见跨度 <= 16384 列）：上传"可见窗口 + 两侧 20% 余量"的全分辨率切片。
- `sigRangeChanged` 触发 30ms **节流**（仅当定时器未激活才启动）QTimer，连续滚动时每约 30ms 重切一次，滚动结束即恢复，避免拖动时连续全量重建。
- 完整刷新只在 `set_data` 时发生一次；之后所有重绘都是"只重算可视部分"。

### 3.3 上采样看细节

- 放大时重新从全分辨率 `frame_data` 切片，`max_columns` 未超限则原样上传，细节不丢。
- 图像行列数上界约 `16384 列 x 可见点数行`，GPU 纹理保持在百万像素量级。

### 3.4 统计量的稳健估计

```python
robust_median(values)           # 超长序列用步进子集估计，仅用于显示
robust_abs_percentile(values, 95.0)
```

- 只在超采样时抽样（`1_000_000` 上限），避免对全量数据做 `nanpercentile`。

## 4. 缓存与异步

- 处理结果：`LruByteCache`，key = `(file_path, filter_config.cache_key())`，重复滤波免计算。
- `.bz` 扫描：`bz_codec._SCAN_CACHE` LRU（`_SCAN_CACHE_MAX = 32`）。
- 读取/滤波/导出：`QThreadPool + QRunnable` 后台执行，UI 线程只做交互与绘图。

## 5. 导出规则（不受显示降采样影响）

- `Export Current View` 按可视窗口：
  - `frame_start = floor(x0 * fs)`，`frame_end = ceil(x1 * fs)`
  - `point_start = floor(y0)`，`point_end = ceil(y1) - 1`
- 未勾选 `Save Filtered Data`：直接切原始 memmap（`.bin`）或原始 `int32` 数组，不往返转换。
- 勾选后：从滤波结果切片，`rad -> int32`。
- 勾选 `Save as txt`：导出 `frames x points` 的 `rad` 浮点矩阵。

## 6. 快速定位要点

| 关注点 | 文件 |
| --- | --- |
| 向量化 bitshuffle / 扫描缓存 / converter 读取 | `scripts/io/bz_codec.py` |
| 统一读取入口（rad 直接产出） | `scripts/io/bin_reader.py:read_phase_radians` |
| 主流程接入 | `scripts/ui/main_window.py:_load_file_data` / `_render_current_data` |
| 显示层降采样与稳健统计 | `scripts/processing/view_data.py` |
| LOD 图像渲染 | `scripts/render/space_time_view.py:_refresh_image` / `_schedule_lod_refresh` |
| 峰值波形降采样 | `scripts/render/space_time_view.py:_update_waveform_curve` |
| 通道曲线降采样 | `scripts/render/analysis_views.py` |
| 共模向量化 | `scripts/processing/denoise.py:common_mode_noise_removal` |
| FFT 多核并行 | `scripts/processing/fdct.py` / `denoise_pipeline.py` |

## 7. 验证清单

- Bitshuffle 往返：`shuffle -> unshuffle` 与原数据逐元素相等，且 shuffle 输出与逐块参考实现逐字节一致（含整块、跨 chunk、末尾不完整块）。
- `read_phase_radians`：与 `raw * (pi/32767)` 逐点一致，dtype 为 `float32`。
- `common_mode_noise_removal`：向量化版本与逐通道循环数值一致。
- `downsample_image_for_display`：单列/块、列数不超过上限、无 NaN，色阶范围经 `robust_min_max` 与全量 min/max 一致（极值不丢）。
- 修改文件后执行：

```powershell
python -m py_compile scripts\io\bz_codec.py scripts\io\bin_reader.py scripts\ui\main_window.py scripts\processing\denoise.py scripts\processing\fdct.py scripts\processing\denoise_pipeline.py scripts\processing\view_data.py scripts\render\space_time_view.py scripts\render\analysis_views.py
```