# eDASread 数据预处理功能开发文档

日期：2026-08-12

本文档记录本次新增的数据预处理、通道分析、PSD、f-k 谱和波形播放功能，面向后续维护和二次开发。

## 1. 参考来源

- 本地 notebook：`E:\codes\DFSpy\surfacewave-C3-inv\notebooks\01_das_preprocess_ncf.ipynb`
- 参考处理模块：`E:\codes\DFSpy\surfacewave-C3-inv\scripts\preprocessing\denoise.py`
- 参考 curvelet 变换：`E:\codes\DFSpy\surfacewave-C3-inv\scripts\preprocessing\fdct.py`
- DASPy 中文教程 f-k 波场分解：`https://daspy-tutorial-cn.readthedocs.io/zh-cn/latest/波场分解.html`

notebook 中的 DAS 降噪流程为：

1. `spike_removal()`：去尖峰噪声。
2. `common_mode_noise_removal()`：去共模噪声。
3. `curvelet_denoising()`：去随机噪声。
4. f-k 对比谱使用二维 FFT、Tukey taper、dB 归一化和背景扣除。

DASPy 教程中的 FK 滤波说明为：将数据变换到频率-波数域，在 f-k 域乘速度窗，再反变换回时空域。教程示例使用 `vmin=(800, 1200)` 作为速度过渡窗，用于压制低视速度交通信号并保留较高视速度成分。

## 2. 新增源码文件

### 2.1 `scripts/processing/fdct.py`

从参考项目引入 Fast Discrete Curvelet Transform 实现。该文件来自 DASPy/CurveLab 风格的 Python 实现，作为随机噪声去除的底层变换。

### 2.2 `scripts/processing/denoise.py`

从参考项目引入 DAS 降噪函数：

- `spike_removal(data, nch=50, nsp=5, thresh=10)`
- `common_mode_noise_removal(data, method='median')`
- `curvelet_denoising(...)`

本项目数据约定保持为 `frames x channels`，函数内部按参考实现转置为 `channels x frames` 处理，输出再转回 `frames x channels`。

本次额外增加了一个共模全零保护：当估计出的 common mode 能量接近 0 时直接返回原数据副本，避免除零产生 NaN。

### 2.3 `scripts/processing/denoise_pipeline.py`

新增 UI 侧调用封装：

- `apply_denoise_step(data, step, sample_rate_hz, channel_spacing_m, fk_vmin_mps, fk_vmax_mps)`
- `fk_filter_high_velocity(...)`
- `compute_fk_spectrum_db(...)`

默认参数与 notebook 保持一致：

- spike：`nch=10`、`nsp=10`、`thresh=10`
- common：`method='median'`
- random：`curvelet pad=0.3`、`knee_fac=0.2`、`soft_thresh=True`、`nbangles=16`
- FK：默认通道间距 `5 m`、速度窗 `800-1200 m/s`

## 3. Tab1 数据预处理

Tab1 保留原有功能：

- 文件读取
- Point Start / Point End
- VMin / VMax
- colormap
- highpass / lowpass / bandpass
- 指定位置点时域图
- time-space 图
- 滚动条和视图回退
- 当前视图导出

本次新增 `Data Preprocess` 控制组：

- `去尖峰噪声`
- `去共模噪声`
- `去随机噪声`
- `FK滤波`
- `DX(m)`
- `VMin`
- `VMax`

四个按钮每次只执行一种处理，处理顺序由用户点击顺序决定。例如先点 `去尖峰噪声`，再点 `去共模噪声`，则第二次处理会作用在第一次处理后的矩阵上。

处理逻辑位于 `MainWindow.run_preprocess_step()`：

1. 检查当前文件和当前矩阵是否存在。
2. 读取 FK 参数。
3. 使用 `FunctionWorker` 放入线程池执行。
4. Worker 调用 `apply_denoise_step()`。
5. 返回后更新 `self.current_data`。
6. 调用 `_render_current_data()` 刷新 Tab1 的时域图和 time-space 图。
7. 状态栏记录已执行的预处理步骤。

注意：点击 `Apply` 或重新选择文件会从原始文件重新读取并应用当前频率滤波配置，因此会清空已点击的预处理步骤。

## 4. FK 滤波实现

FK 滤波函数为 `fk_filter_high_velocity()`，输入矩阵形状为 `frames x channels`。

处理流程：

1. 转置为 `channels x frames`。
2. 对时间方向应用 Tukey taper，默认比例为 `0.05`。
3. 对空间方向默认不 taper，避免在通道数少时过度压低边缘通道。
4. 使用 `numpy.fft.fft2()` 做二维 FFT。
5. 根据频率 `f` 和空间频率 `k` 计算视速度：

```text
apparent_velocity = abs(f / k)
```

6. 构造高视速度保留窗：

- `abs(v) <= VMin`：权重 0
- `abs(v) >= VMax`：权重 1
- `VMin < abs(v) < VMax`：余弦过渡
- `k = 0`：保留，权重 1

7. f-k 域乘窗。
8. 使用 `ifft2()` 回到时空域。
9. 裁剪回原始尺寸并返回 `float32`。

当前按钮语义为：压制低视速度能量，保留高视速度成分。默认 `800-1200 m/s` 与 DASPy 教程示例一致。

## 5. Tab2 通道时域图和 PSD

新增文件：`scripts/render/analysis_views.py`

新增类：`ChannelAnalysisView`

Tab2 控件：

- 通道输入框，默认 `5,10,15`
- `绘制` 按钮

通道号使用 1-based 编号，与 Tab1 的 Point 编号一致。

绘图行为：

- 子图 1：指定通道时域波形。
- 子图 2：PSD 曲线。
- 时域图中每条曲线先扣除自身中位数，再自动增加不同直流偏移，避免多通道波形重叠。
- PSD 使用 `scipy.signal.welch()` 计算。
- PSD 单位显示为 dB：

```text
PSD_dB = 10 * log10(power + eps)
```

字号：

- 坐标刻度字号：16
- 坐标标签字号：16

Tab2 使用 `self.current_data`，也就是 Tab1 当前处理后的数据矩阵。因此先在 Tab1 执行去噪，再切到 Tab2 绘制，会看到去噪后的通道曲线和 PSD。

## 6. Tab3 f-k 谱

新增类：`FkSpectrumView`

Tab3 控件：

- 通道范围，格式如 `5-100`
- 通道间距，单位 m，默认 `5`
- dB 色标范围，默认 `-20` 到 `0`
- `绘制 f-k 谱` 按钮

数据来源：

- 直接使用 Tab1 当前处理后的 `self.current_data`
- 按 Tab3 输入的通道范围切片
- 不重新读取原始文件
- 不重新应用频率滤波或预处理

f-k 谱计算函数为 `compute_fk_spectrum_db()`：

1. 输入矩阵 `frames x channels`。
2. 转置为 `channels x frames`。
3. 应用默认 Tukey taper。
4. 使用 `rfft2()` 计算正频率二维谱。
5. 空间频率轴使用 `fftshift(fftfreq(...))`。
6. 按 99 分位幅值归一化到 dB。
7. 默认按每个频率扣除空间频率方向的中位背景。
8. 返回：

- `db`
- `frequency_hz`
- `wavenumber_cpm`

`FkSpectrumView.set_spectrum()` 使用 `pyqtgraph.ImageItem` 绘制 dB 图，x 轴为空间频率 `cycles/m`，y 轴为频率 `Hz`。

## 7. 波形播放

Tab1 新增按钮：`播放波形`

入口函数：`MainWindow.play_current_waveform()`

播放数据：

- 当前 `waveform_point_edit` 指定的通道
- 当前 `self.current_data` 中的该通道完整波形
- 因此播放内容包含已执行的频率滤波和降噪结果

音频生成流程：

1. 提取通道波形。
2. 将 NaN/Inf 替换为有限值。
3. 扣除中位数。
4. 用 99 分位绝对幅值归一化。
5. 放大到接近满幅并限幅，提升电脑外放音量。
6. 若采样率低于 8000 Hz，则线性重采样到 8000 Hz。
7. 若采样率高于 44100 Hz，则线性重采样到 44100 Hz。
8. 写入临时 WAV 文件。
9. Windows 下使用 `winsound.PlaySound(..., SND_ASYNC)` 异步播放。

临时 WAV 文件最多保留 6 个旧路径，后续播放时会尝试清理更早文件。

## 8. 导出规则调整

原有导出逻辑中，`Save Filtered Data` 会重新按当前滤波参数处理原始数据。

本次改为：

- 勾选 `Save Filtered Data`：导出当前 `self.current_data` 的可视窗口切片。
- 未勾选 `Save Filtered Data`：仍直接从原始文件读取 int32 切片。

这样可以保证用户在 Tab1 执行去噪后导出当前视图时，不会丢失去噪结果。

## 9. 验证记录

已执行语法检查：

```powershell
python -m py_compile scripts\processing\fdct.py scripts\processing\denoise.py scripts\processing\denoise_pipeline.py scripts\render\analysis_views.py scripts\ui\control_panel.py scripts\ui\main_window.py
```

已执行小样本数值验证：

- `spike`
- `common`
- `fk`
- `random`
- `compute_fk_spectrum_db`

验证结果：

- 输出形状保持为输入 `frames x channels`
- 输出 dtype 为 `float32`
- 输出值全部有限
- f-k 谱输出包含 `db`、`frequency_hz`、`wavenumber_cpm`

已执行 Qt 构造检查：

- `MainWindow()` 可实例化
- 右侧 `QTabWidget` 包含 3 个 Tab：
  - `Tab1 数据预处理`
  - `Tab2 通道时域/PSD`
  - `Tab3 f-k谱`

## 10. 后续维护建议

- 如果现场数据通道数很大，`去随机噪声` 的 curvelet 处理可能耗时明显，建议后续增加进度提示或范围处理。
- FK 滤波当前固定为高视速度保留模式，后续如需交通信号提取，可增加“低视速度保留”或“分解输出”模式。
- Tab3 当前绘制完整时间长度的 f-k 谱，后续可增加时间窗输入，用于局部波场分析。
