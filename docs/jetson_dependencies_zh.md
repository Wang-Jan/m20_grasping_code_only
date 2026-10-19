# Jetson R36.4.3 依赖安装

本说明仅覆盖 JetPack R36.4.3、Python 3.10、aarch64 的依赖准备。安装脚本不枚举、不打开也不连接 DM-B601、D435、D455 或 M20；真实设备初始化仍只能由 `--run-real` 触发。

## 固定版本

| 组件 | 版本 | 来源 |
| --- | --- | --- |
| PyTorch | 2.8.0 | Jetson AI Lab `jp6/cu126` aarch64 wheel 索引 |
| torchvision | 0.23.0 | 同一 Jetson wheel 索引，和 PyTorch 2.8 对应 |
| NumPy | 2.2.6 | 清华 PyPI 镜像；Pinocchio 4.1 的 aarch64 wheel 要求 `>=2.0` |
| OpenCV | 4.12.0.88 | 清华 PyPI 镜像；与 NumPy 2 ABI 兼容，也是 Ultralytics 的运行依赖 |
| PyYAML | 6.0.2 | 清华 PyPI 镜像，满足 DM SDK 的 `>=6.0` |
| Pinocchio | `pin==4.1.0` | 清华 PyPI 镜像的 `cp310/aarch64` wheel；导入名为 `pinocchio` |
| Ultralytics | 8.4.120 | 清华 PyPI 镜像；支持本工程的 YOLOE 模型加载路径 |
| OpenAI CLIP | 1.0.1 | YOLOE 的开放词汇 `set_classes()` 文本编码依赖 |

本项目使用 wheelhouse 中已验证的 Jetson 专用 `torch-2.8.0-cp310-cp310-linux_aarch64.whl` 和
`torchvision-0.23.0-cp310-cp310-linux_aarch64.whl`。曾存在的 2.12.1/0.27.1 ARM64 wheel
在本机 Orin CC 8.7 上无法执行 CUDA kernel，因此不采用。安装脚本会校验文件名、wheel 标签
和 SHA-256；不会从普通 PyPI 下载 Torch，也不会用 x86_64 或通用 CUDA wheel 替代。PyTorch
是否可用以 Orin 上的 GPU 张量 smoke test 为最终验收标准。

该 Jetson Torch wheel 按 NumPy 1.x 构建，而 Pinocchio 4.1 和 OpenCV 4.12 要求 NumPy 2.x。
项目在 D455/D435 的 YOLO 入口安装了兼容转换层，避开 `torch.from_numpy()` 的失效路径，
不降低 NumPy 版本，也不替换 Orin 的 CUDA wheel。YOLOE 开放词汇还需要本地
`weights/mobileclip2_b.ts`；安装脚本会优先通过国内代理下载，失败时尝试官方 Ultralytics
asset URL，运行时不会临时联网下载。

SDK 声明 `pin>=3.9.0`，但 `pin==3.9.0` 在当前 aarch64 镜像中会解析到 URDFDOM ABI 6，而其二进制扩展需要 ABI 4，导致导入失败。已验证的 `pin==4.1.0` 明确要求 `cmeel-urdfdom>=6`，与该 ABI 一致，因此采用它；该版本仍满足 SDK 的最低版本约束。

系统自带 OpenCV 按 NumPy 1.x ABI 编译，不能与 Pinocchio 4.1 所需的 NumPy 2 同时运行。用户目录中的 OpenCV 4.12.0.88 已验证可同时导入，因此固定安装该版本。

## 安装

在项目根目录运行：

```bash
bash scripts/install_jetson_dependencies.sh
```

脚本会验证 `wheelhouse/jetson-jp62/` 中两个 Jetson wheel 的 `cp310/aarch64` 标签和 SHA-256，
然后优先使用 uv 安装到指定的 Python 环境。未找到 uv 时才回退到该 Python 的 pip。
通用依赖默认使用清华镜像，失败时自动改用中科大镜像；Torch/torchvision 始终只使用本地
Jetson wheel，不会回退到通用 PyPI 包。

如果需要重新准备 wheelhouse，应从 Jetson AI Lab `jp6/cu126` 获取这两个确切文件，完成后再运行脚本；
不要改用清华或中科大提供的通用 `torch` wheel，它们不是本方案验证的 Jetson CUDA 构建。

若 wheelhouse 需要放在其他磁盘，可设置：

```bash
WHEELHOUSE=/data/wheelhouse PYTHON_BIN="$PWD/.venv/bin/python" bash scripts/install_jetson_dependencies.sh
```

脚本不会自动下载 2.8.0/0.23.0；如使用其他磁盘或内部制品库，必须保持上述文件名和
SHA-256，并通过 `WHEELHOUSE`、`TORCH_WHEEL` 和 `TORCHVISION_WHEEL` 指定路径。

## 验证

安装脚本会验证 PyTorch CUDA 状态、Ultralytics 的 `YOLO` 类、Pinocchio、motorbridge 和 pyrealsense2 的导入。之后运行以下不连接硬件的检查：

```bash
source .venv/bin/activate
uv pip check
python -m unittest discover -s tests -v
python -m compileall -q src tests
PYTHONPATH=src python -m m20_rebot_moving_grasp.main --dry-run
```

不使用 `--run-real`，也不要在依赖准备阶段运行任何 SDK 的扫描、串口或 RealSense 打开命令。
