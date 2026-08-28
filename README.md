# eDASread

`eDASread` 是一个面向 `.bin` / `.bz` / `.bzs` 格式 eDAS 数据的桌面程序，提供文件浏览、时空图显示、指定点时域波形联动、滤波预处理、f-k 分析与当前视图导出功能。

## 主要功能

- 扫描并分页浏览目录中的 `.bin` / `.bz` / `.bzs` 文件
- 从文件名解析采样率、点数和时间戳
- 支持 `.bin -> .bz` 批量压缩
- 显示 time-space 图与指定位置点时域波形
- 支持低通、高通、带通滤波
- 支持 Tab1 交互式数据预处理：去尖峰噪声、去共模噪声、去随机噪声、FK 滤波
- 支持 Tab2 指定多通道时域图与 PSD(dB) 绘制
- 支持 Tab3 指定通道范围 f-k 谱绘制
- 支持 Tab4 指定单通道时域波形与短时时频图绘制（对数频率轴）
- 空文件（0 字节）或无完整数据帧的文件会被直接拒绝并给出简洁错误提示
- 支持将指定位置点波形缩放后以声音播放
- 按当前可视窗口导出原始或滤波后的数据

## 主要界面参数说明

### Tab1 数据预处理

- `Point Start / Point End`：time-space 图显示的位置点范围，采用 1-based 点号。
- `VMin / VMax`：time-space 图颜色范围，单位为 rad；两者同时填写才生效。
- `Colormap`：time-space 图色标，可选 `jet`、`hsv`、`seismic`。
- `Type`：时间方向滤波类型，可选 `none`、`lowpass`、`highpass`、`bandpass`。
- `Low Cut / High Cut`：滤波截止频率，单位 Hz；高通使用 `Low Cut`，低通使用 `High Cut`，带通同时使用两者。
- `指定位置点时域绘图`：Tab1 上方时域图显示的单个位置点，采用 1-based 点号。
- `播放波形`：播放当前指定位置点的完整波形；播放前会扣除直流、按 99 分位幅值放大并限幅。
- `去尖峰噪声`：参考 DASPy spike removal，使用局部中值检测异常尖峰并按邻近通道插值替换。
- `去共模噪声`：参考 DASPy common-mode removal，估计各通道共有波形并从每个通道中投影扣除。
- `去随机噪声`：参考 DASPy curvelet denoising，使用 curvelet 系数阈值压制随机噪声，计算耗时相对更高。
- `FK滤波`：在 f-k 域压制低视速度能量、保留高视速度成分。
- `DX(m)`：FK 滤波使用的通道间距，单位 m，默认 `5`。
- `VMin / VMax`：FK 滤波速度过渡窗，单位 m/s，默认 `800-1200`；低于 `VMin` 的视速度成分被压制，高于 `VMax` 的成分被保留，中间为余弦过渡。
- `Save Filtered Data`：导出当前处理后矩阵的可视窗口切片；如果未勾选，则从原始文件导出当前窗口。
- `Save as txt`：将导出格式改为文本矩阵，单位为 rad。

### Tab2 通道时域/PSD

- `通道`：需要绘制的通道列表，采用 1-based 点号，可输入 `5,10,15`，也支持空格、中文逗号或分号分隔。
- `PSD窗长`：Welch PSD 的每段采样点数；填写整数时手动指定，填写 `auto` 或留空时自动取 `min(max(256, 采样率), 4096, 数据长度)`。
- `重叠(%)`：Welch PSD 相邻窗口重叠比例，范围 `[0, 100)`，默认 `50`。
- `FMin / FMax`：PSD 图 x 轴频率显示范围，单位 Hz；两者同时填写才生效，留空或 `auto` 表示自动范围。
- `PSD Min / PSD Max`：PSD 图 y 轴显示范围，单位 dB；两者同时填写才生效，留空或 `auto` 表示自动范围。
- 时域图会对每个通道扣除自身中位数，并自动施加不同垂直偏移，避免多条波形重叠。
- PSD 计算使用 `scipy.signal.welch()`，显示单位为 dB：`10 * log10(power + eps)`。

### Tab3 f-k谱

- `通道范围`：用于计算 f-k 谱的连续通道范围，采用 1-based 点号，格式如 `5-100`。
- `间距(m)`：通道间距，单位 m，用于计算空间频率轴。
- `空间taper`：空间方向 Tukey taper 比例，范围 `0-1`，默认 `0`。
- `时间taper`：时间方向 Tukey taper 比例，范围 `0-1`，默认 `0.05`。
- `NFFT-K`：空间方向 FFT 点数；填写整数时手动指定，必须不小于通道数；填写 `auto` 或留空时自动取不小于通道数的 2 的幂。
- `NFFT-T`：时间方向 FFT 点数；填写整数时手动指定，必须不小于采样点数；填写 `auto` 或留空时自动取不小于采样点数的 2 的幂。
- `FMin / FMax`：f-k 谱 y 轴频率显示范围，单位 Hz；两者同时填写才生效，留空或 `auto` 表示自动范围。
- `KMin / KMax`：f-k 谱 x 轴空间频率显示范围，单位 cycles/m；两者同时填写才生效，留空或 `auto` 表示自动范围。
- `VMin(dB) / VMax(dB)`：f-k 谱颜色显示范围，单位 dB，默认 `-20` 到 `0`。
- `扣除背景`：按每个频率扣除空间频率方向的中位背景，使相对能量更突出；默认开启。
- Tab3 使用 Tab1 当前处理后的数据，不重新读取原始文件，也不重新执行 Tab1 的滤波或去噪。

### Tab4 时域/时频

- `通道`：需要绘制的单个通道，采用 1-based 点号。
- `滤波`：在 Tab1 处理后数据的基础上，对所选通道进一步滤波，可选 `none`、`lowpass`、`highpass`、`bandpass`；高通使用 `Low Cut`，低通使用 `High Cut`，带通同时使用两者。
- `模式`：时频数值类型，可选 `PSD`（功率谱密度，`scaling="density"`）或 `Amplitude`（幅值谱，`scaling="spectrum"`）。
- `数值`：颜色数值标度，可选 `Log`（PSD 按 `10*log10`、Amplitude 按 `20*log10`）或 `Linear`。
- `窗长(s)`：短时谱的窗口长度，单位 s，对应 `nperseg = round(窗长 × 采样率)`，限制在 `[8, 采样点数]`，默认 `0.01`。
- `重叠(%)`：相邻窗口重叠比例，范围 `[0, 100)`，默认 `50`。
- `FMin / FMax`：时频图 y 轴频率显示范围，单位 Hz；对数频率轴下必须大于 0；两者同时填写才生效，留空或 `auto` 表示自动范围。
- `VMin / VMax`：时频图颜色范围；勾选 `颜色自动` 时按数据分位数自动设置，取消勾选时需同时填写两者。
- `显示窗口(s)`：绘制完成后默认显示的时间段长度，默认 `0.1`；填写 `auto` 或留空时显示完整时长。
- `Colormap`：时频图色标，可选 `jet`、`hsv`、`seismic`。
- Tab4 使用 Tab1 当前处理后的数据，不重新读取原始文件，也不重新执行 Tab1 的滤波或去噪。
- 时频计算使用 `scipy.signal.spectrogram()`（Hann 窗），线性频率谱会重采样到等间距 `log10(f)` 网格后再显示，y 轴标签直接显示真实频率（Hz）。
- 上方时域图与下方时频图的 x 轴双向联动：任一子图矩形放大或平移，另一子图自动切换到相同时间范围；键盘左右键按最近点击的子图水平平移。

Tab2 和 Tab3 的绘图区支持鼠标滚轮缩放、拖动平移和框选放大，方便查看局部时窗、频段或波数范围。

## 目录说明

- `scripts/`：主程序与界面、渲染、处理逻辑
- `assets/`：程序图标等静态资源
- `docs/`：用户手册与开发文档
- `singledataread/`：单文件读取与示例 Notebook
- `bin2segy/`：bin 数据转 segy 的相关脚本
- `run.py`：项目根目录启动入口
- `build_exe.py`：根目录打包脚本（输出到 `dist/`）
- `requirements.txt`：运行时依赖清单

## 运行环境

推荐依赖（完整清单见 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

核心运行时依赖：

- `PyQt5`：界面框架
- `pyqtgraph`：绘图
- `numpy`：数值计算
- `scipy`：信号滤波与频谱分析
- `python-zstandard`：可选，`.bz/.bzs` 压缩格式支持（缺失时尝试加载 `zstd.dll`）

启动方式（项目根目录）：

```bash
python run.py
```

打包单文件 EXE（项目根目录，exe 名自动包含打包日期，构建完成后自动清理 `build/` 等临时文件）：

```bash
python build_exe.py
```

## 数据规则

- 文件名应包含 `-xxxHz-` 和 `-xxxpt-`
- 二进制数据类型为 `int32`
- 数据组织方式为 `frame_count x points`
- 存储顺序为 frame-major

## 文档

- 使用说明见 `docs/eDAS_user_manual.doc`
- 文档总览见 `docs/readme.md`
- 开发文档见 `docs/dev_log.md`
