# FSTA-Det

基于 FCOS 的无人机小目标检测工程，包含 VisDrone2019 与 AI-TOD 的训练、评估和消融配置。

FSTA-Det 在特征金字塔中接入 TAFE 和 CSTA：TAFE 利用空间尺度差分提取细节，并通过纹理与前景门控调节增强强度；CSTA 在有限搜索区域内按特征相似度重排 token，完成跨窗口注意力聚合后恢复空间顺序。VisDrone 配置作用于 P2/P3，AI-TOD 配置作用于 P3/P4。

## 工程结构

```text
configs/
  _base_/                 数据管线、训练计划与运行设置
  visdrone/               A0–A6 消融与 640×640 对照实验
  aitod/                  P3–P7 基线、单模块与组合实验
mmdet/
  models/plugins/fsta/    TAFE、CSTA、前景预测与残差组合
  models/necks/fsta_fpn.py FPN 接入
  models/detectors/       FSTADet 与训练损失接入
  datasets/visdrone.py    VisDrone 标注及评估适配
scripts/                  单卡训练、评估与顺序队列
tools/                    数据转换、检查与性能统计
tests/                    模块、评估和队列测试
docs/                     方法、数据、实验与环境说明
third_party/              第三方许可证
```

`mmdet/` 以 MMDetection 2.23.0 为基础，保留框架目录结构。项目新增代码集中在上述文件中。

## 环境与安装

需要 Linux、PyTorch 和带编译算子的 MMCV 1.x。原实验与本次发布检查使用 Python 3.10、PyTorch 2.7.0+cu128、MMCV 1.7.2。MMCV 的编译版本必须与 PyTorch/CUDA 对应，详见 [环境说明](docs/installation.md)。

```bash
git clone https://github.com/jiangjoker99-ctrl/FSTA-Det.git
cd FSTA-Det
python -m pip install -r requirements.txt
python -m pip install -e .
```

## 数据准备

数据根目录通过环境变量设置；默认分别为 `data/visdrone` 和 `data/aitod`。

```bash
export VISDRONE_ROOT=/path/to/visdrone
export AITOD_ROOT=/path/to/aitod

python tools/convert_visdrone_to_coco.py --root "$VISDRONE_ROOT"
```

标注文件、图像目录和 AI-TOD 划分见 [数据说明](docs/data.md)。

## 训练与评估

以 VisDrone A5 为例，第二个参数是物理 GPU 编号：

```bash
bash scripts/train.sh configs/visdrone/a5_tafe_csta_appearance.py 0 work_dirs/a5
bash scripts/eval.sh configs/visdrone/a5_tafe_csta_appearance.py \
  work_dirs/a5/best_bbox_mAP_epoch_12.pth 0
```

VisDrone 默认使用 FP32、batch size 2 和 12 轮训练，每轮验证并保留最佳 `bbox_mAP` 权重。输入按 `(1600, 1000)` 等比例缩放，并补齐到 32 的倍数；实际张量尺寸取决于原图比例。

AI-TOD 的配置使用 800×800、batch size 16、12 轮训练和 1,250 步预热，结束时验证。`data.test` 指向独立测试集，需调用评估入口进行最终测试。

```bash
bash scripts/train.sh configs/aitod/tafe_csta_p3_p4.py 0 work_dirs/aitod_fsta
bash scripts/eval.sh configs/aitod/tafe_csta_p3_p4.py \
  work_dirs/aitod_fsta/best_bbox_mAP_epoch_12.pth 0
```

使用 `--cfg-options` 可覆盖 batch size、学习率和训练轮数，续训说明见 [实验配置](docs/experiments.md)。

## 消融配置

| 实验 | 结构 | 配置 |
|---|---|---|
| A0 | FCOS P3–P7 | [a0_fcos_p3_p7.py](configs/visdrone/a0_fcos_p3_p7.py) |
| A1 | FCOS P2–P5 | [a1_fcos_p2_p5.py](configs/visdrone/a1_fcos_p2_p5.py) |
| A2 | A1 + 纹理门控 TAFE | [a2_tafe_texture_only.py](configs/visdrone/a2_tafe_texture_only.py) |
| A3 | A1 + TAFE | [a3_tafe.py](configs/visdrone/a3_tafe.py) |
| A4 | A1 + 外观相似度 CSTA | [a4_csta_appearance.py](configs/visdrone/a4_csta_appearance.py) |
| A5 | A1 + TAFE + 外观相似度 CSTA | [a5_tafe_csta_appearance.py](configs/visdrone/a5_tafe_csta_appearance.py) |
| A6 | TAFE + 前景引导 CSTA | [a6_fsta_det.py](configs/visdrone/a6_fsta_det.py) |

A5 的 TAFE 使用前景门控与辅助监督；A6 进一步使用前景概率加权 CSTA 窗口代表特征。

## VisDrone 实验记录

以下为已有训练日志中的最佳验证结果，AP 按百分数展示。输入缩放上限为 1600×1000。

| 实验 | 总轮数 | 最佳轮次 | AP | AP50 | AP75 | APs | APm | APl |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 | 12 | 12 | 23.6 | 42.9 | 22.8 | 15.4 | 34.2 | 38.7 |
| A1 | 12 | 12 | 25.5 | 46.3 | 24.2 | 18.2 | 33.8 | 36.7 |
| A3 | 20 | 18 | 25.5 | 46.6 | 24.2 | 18.2 | 34.1 | 35.9 |
| A4 | 20 | 19 | 25.3 | 46.2 | 24.1 | 18.0 | 33.6 | 36.3 |
| A5 | 12 | 12 | 27.1 | 48.3 | 26.4 | 19.6 | 35.8 | 38.4 |
| A6 | 12 | 12 | 27.1 | 48.3 | 26.1 | 19.6 | 35.7 | 38.0 |

A2 未完成有效训练。A3/A4 与其余行训练时长不同，表格用于记录已有实验；控制变量比较应使用相同训练设置。评估采用 COCO 风格 AP 与 VisDrone 忽略区域适配，详见 [实验记录](docs/experiments.md)。

## 顺序队列与检查

队列使用 GPU 锁，等待已有计算进程退出后启动下一项。训练失败时停止，完成的任务可跳过，未完成的非空目录需要手动指定续训。

```bash
bash scripts/queue.sh 0 work_dirs/ablation_run1 \
  configs/visdrone/a1_fcos_p2_p5.py \
  configs/visdrone/a3_tafe.py \
  configs/visdrone/a4_csta_appearance.py \
  configs/visdrone/a5_tafe_csta_appearance.py

python tools/run_tests.py --ddp
python tools/smoke_test.py --device cpu
python tools/validate_setup.py
```

最后一项检查需要已准备好的 VisDrone 数据；模块测试与默认 smoke test 使用合成输入。本次检查记录见 [验证记录](docs/validation.md)。权重、数据和运行输出通过 `.gitignore` 排除。

## 来源与许可证

基础框架来自 [MMDetection](https://github.com/open-mmlab/mmdetection/tree/v2.23.0)，实验设置参考 [SET](https://github.com/HuixinSun/SET)。TAFE 的尺度残差与 CSTA 的相似度分组思路参考 [CUST](https://github.com/jwgdmkj/CUST)，在检测特征上加入了有效区域掩码、前景门控和辅助监督。

本工程采用 Apache-2.0，CUST 的原始 MIT 许可证保留于 [third_party/CUST_LICENSE](third_party/CUST_LICENSE)。具体来源见 [NOTICE](NOTICE)。
