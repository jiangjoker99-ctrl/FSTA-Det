# 环境与安装

## 实验环境

已有实验和发布检查使用以下 Linux 环境：

| 组件 | 版本 |
|---|---|
| Python | 3.10.21 |
| PyTorch | 2.7.0+cu128 |
| MMCV | 1.7.2，含编译算子 |
| MMDetection 源码 | 2.23.0，包含本工程扩展 |

本工程使用 MMCV 1.x 的 runner、Config 和 registry，依赖版本范围为 1.3.17–1.7.2。MMCV 2.x 的接口不同。

## 安装步骤

1. 创建独立 Python 环境，安装适合 GPU 与驱动的 PyTorch。
2. 安装与该 PyTorch/CUDA 组合匹配的 MMCV 1.x 编译版本。
3. 安装本工程依赖和源码。

```bash
python -m pip install -r requirements.txt
python -m pip install -e .
python -c "import torch, mmcv; from mmcv.ops import nms; print(torch.__version__, mmcv.__version__)"
```

旧版 MMCV 的预编译包未必覆盖新的 PyTorch/CUDA 组合。需要源码构建时，应参考 [MMCV 安装说明](https://mmcv.readthedocs.io/en/v1.7.2/get_started/installation.html)，并确认 `mmcv.ops` 能在目标设备执行。以上环境记录来自现有可运行环境，未宣称从空环境的一次安装即可覆盖所有 GPU 型号。

若已有另一份 MMDetection，请使用独立环境，避免同名 `mmdet` 包遮蔽本工程。可核对实际导入位置：

```bash
python -c "import mmdet; print(mmdet.__file__)"
```

脚本支持 `PYTHON_BIN` 指定解释器；源码根目录会加入 `PYTHONPATH`。

## 检查

```bash
python tools/run_tests.py --ddp
python tools/smoke_test.py --device cpu
```

测试覆盖奇数尺寸、padding、梯度、关闭模块的精确恒等性、状态恢复、评估坐标和队列失败处理。smoke test 使用小尺寸合成输入，完成前向、反向、优化步骤、空目标及混合尺寸批次、推理和权重重载。

准备好 VisDrone 后，还可检查真实数据管线和 evaluator：

```bash
python tools/validate_setup.py
```
