# 数据准备

## VisDrone2019-DET

从 [VisDrone](https://github.com/VisDrone/VisDrone-Dataset) 获取检测数据。目录如下：

```text
visdrone/
  VisDrone2019-DET-train/
    images/
    annotations/
  VisDrone2019-DET-val/
    images/
    annotations/
  annotations/
    visdrone_train_coco.json
    visdrone_val_coco.json
```

```bash
export VISDRONE_ROOT=/path/to/visdrone
python tools/convert_visdrone_to_coco.py --root "$VISDRONE_ROOT"
```

转换保留 10 个检测类别，并将忽略区域、score=0 和 others 类别作为忽略标注处理。类别顺序见数据配置，训练和权重必须保持一致。

本工程的验证结果来自 COCO 风格 AP：IoU 阈值为 0.50–0.95，并按 COCO 面积阈值统计 APs/APm/APl。VisDrone 适配器使所有 AP 项使用 maxDets=1000，并提供类别无关忽略区域处理。这与 VisDrone 官方竞赛评测并非完全相同的协议。

`data.test` 默认指向有公开标注的验证集；若使用其他测试数据，应覆盖标注与图像路径。

## AI-TOD

数据来源：[AI-TOD](https://github.com/jwwangchn/AI-TOD)。配置对应已有工程的数据布局：

```text
aitod/
  annotations/
    aitod_train.json       11,214 张训练图像
    aitod_val.json          2,804 张验证图像
    aitod_val_coco.json    14,018 张独立测试图像
  data/images/
    train/                训练及验证图像
    val/                  独立测试图像
```

第三个 JSON 的名称沿用原实验，配置中的用途是测试集。不同下载包的文件名和图像目录可能不同；使用对应划分的标注，并调整 `configs/_base_/datasets/aitod_detection.py` 的路径即可。

```bash
export AITOD_ROOT=/path/to/aitod
```

若使用 YOLO 标注镜像，可先转换格式：

```bash
python tools/convert_aitod_yolo_to_coco.py --root "$AITOD_ROOT"
```

此工具只转换镜像已有的 train/val 文件夹，不会自动生成官方训练/验证划分。复现实验还需要提供对应的 `aitod_train.json` 和 `aitod_val.json`，不能把镜像文件夹名直接当作官方划分。

AI-TOD 配置使用标准 COCO evaluator，APs/APm 的含义仍是 COCO 面积分组。原始 AI-TOD evaluator 的 very-tiny/tiny 分组指标不在当前输出中。

## 路径覆盖

环境变量在配置加载时生效，也可通过命令行逐项覆盖：

```bash
python tools/train.py configs/visdrone/a5_tafe_csta_appearance.py \
  --cfg-options data.train.ann_file=/path/to/train.json \
    data.train.img_prefix=/path/to/train/images \
    data.val.ann_file=/path/to/val.json \
    data.val.img_prefix=/path/to/val/images
```
