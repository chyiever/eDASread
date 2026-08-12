# eDASread 开发日志

## 2026-07-28 bz 压缩格式支持

### 背景

软件原先只支持直接读取 `.bin` 原始相位数据文件。为减小数据文件体积，本次按 `E:\codes\compress\docs` 中的 Bitshuffle + Zstd 上位机 Python 开发指南，补充 `.bz/.bzs` 压缩文件读取和 `.bin -> .bz` 批量转换能力。

### 数据格式

- `.bin` 仍为原始 `int32`，逻辑形状为 `frames x points`，frame-major 存储。
- `.bz/.bzs` 真实格式为 `BZF1` 文件头 + 多个 `BZS1` packet。
- `BZF1` 文件头结构为 `<4sHHQIIIIIIIiI>`，记录采样率、每帧点数、通道数、数据源、保存抽点因子、packet 帧数、文件时长、Zstd level 和 Bitshuffle block。
- `BZS1` packet header 结构为 `<4sHHQQIIIHHIiQQQQII>`，记录 packet 序号、时间戳、帧数、原始/压缩字节数和 CRC。
- 读取端同时兼容早期无文件头的 `BZS1` packet stream。
- packet header 结构为 `<4sHHQQIIIHHIiQQQQII`。
- 解压流程为 `Zstd decompress -> Bitshuffle unshuffle -> int32 reshape(frames, points)`。
- 每个 packet 校验 `payload_crc32` 和 `raw_crc32`，避免损坏数据静默进入绘图链路。

### 代码变更

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

### 使用方式

1. 在左侧选择数据目录。
2. 点击 `Batch Bin -> bz`。
3. 如果右侧 `Export folder` 已填写，`.bz` 输出到该目录；否则输出到原数据目录。
4. 已存在的同名 `.bz` 默认跳过，不覆盖。
5. 压缩完成后文件列表会刷新，`.bz` 文件可直接选择、滤波和绘图。

### 验证记录

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

### 注意事项

- `.bz` 随机访问暂未建立索引，当前读取会解压整个文件到内存；`.bin` 仍保持 memmap。
- NumPy 版 bitshuffle 以兼容和可验证为主，高速大文件实时压缩性能仍建议使用原生 C/SIMD 后端复测。
- 若运行环境既没有 `python-zstandard` 又找不到 `zstd.dll`，`.bz` 读取和转换会失败并提示安装或放置 Zstd 后端。

## 2026-07-29 文件列表排序

- 左侧文件浏览区新增 `Sort` 下拉框。
- 支持三种排序方式：
  - `Name`：按文件名升序排列，保持原有默认行为。
  - `Modified: Newest`：按文件修改时间倒序排列，最新修改的文件显示在前。
  - `Modified: Oldest`：按文件修改时间升序排列，最早修改的文件显示在前。
- 切换排序方式时会重新分页，并尽量保持当前选中文件不变。
- 刷新目录和批量压缩完成后的文件列表刷新会沿用当前排序方式。

## 2026-08-12 数据预处理、通道分析与 f-k 谱

### 背景

参照 `E:\codes\DFSpy\surfacewave-C3-inv\notebooks\01_das_preprocess_ncf.ipynb` 中的数据降噪流程，并结合 DASPy 中文教程中的 f-k 波场分解说明，为 eDASread 增加面向 DAS 数据浏览的交互式预处理和分析功能。

### 功能变更

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

### 代码变更

- 新增 `scripts/processing/fdct.py`，引入参考项目中的 curvelet 变换实现。
- 新增 `scripts/processing/denoise.py`，引入 spike、common-mode、curvelet random denoising，并增加 common mode 除零保护。
- 新增 `scripts/processing/denoise_pipeline.py`，封装 UI 调用的单步降噪、FK 滤波和 FK 谱计算。
- 新增 `scripts/render/analysis_views.py`，实现 Tab2 和 Tab3 绘图组件。
- 更新 `scripts/ui/control_panel.py`，新增预处理按钮、FK 参数和播放按钮。
- 更新 `scripts/ui/main_window.py`，改为 3 Tab 布局，接入预处理 worker、Tab2/Tab3 绘图和 WAV 播放。
- 新增 `docs/preprocessing_development.md`，记录算法来源、处理链路、UI 行为、验证记录和后续维护建议。

### 验证记录

- 语法检查通过：

```powershell
python -m py_compile scripts\processing\fdct.py scripts\processing\denoise.py scripts\processing\denoise_pipeline.py scripts\render\analysis_views.py scripts\ui\control_panel.py scripts\ui\main_window.py
```

- 小样本数值验证通过：`spike`、`common`、`fk`、`random`、`compute_fk_spectrum_db`。
- Qt 构造检查通过：`MainWindow()` 可实例化，右侧 Tab 数量为 3。

## 2026-08-12 Tab 参数区压缩与 PSD/FK 参数化

### 功能变更

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

### 代码变更

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

### 验证记录

- 语法检查通过：

```powershell
python -m py_compile scripts\processing\denoise_pipeline.py scripts\render\plot_style.py scripts\render\analysis_views.py scripts\render\space_time_view.py scripts\ui\control_panel.py scripts\ui\main_window.py
```

- Qt 构造检查通过：右侧 Tab 数量为 3。
