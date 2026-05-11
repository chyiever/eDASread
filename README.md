# eDASread

`eDASread` 是一个面向 `.bin` 格式 eDAS 数据的桌面程序，提供文件浏览、时空图显示、指定点时域波形联动、滤波处理和当前视图导出功能。

## 主要功能

- 扫描并分页浏览目录中的 `.bin` 文件
- 从文件名解析采样率、点数和时间戳
- 显示 time-space 图与指定位置点时域波形
- 支持低通、高通、带通滤波
- 按当前可视窗口导出原始或滤波后的数据

## 目录说明

- `scripts/`：主程序与界面、渲染、处理逻辑
- `docs/`：用户手册与开发文档
- `singledataread/`：单文件读取与示例 Notebook
- `bin2segy/`：bin 数据转 segy 的相关脚本

## 运行环境

推荐依赖：

```bash
pip install pyqt5 pyqtgraph numpy scipy
```

启动方式：

```bash
python -m scripts.app
```

打包单文件 EXE：

```bash
cd scripts
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
- 开发文档见 `docs/dev.md`
