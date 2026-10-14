# 实验与复现

## VisDrone

A0 为 P3–P7 的 FCOS 结构参考；A1 为 P2–P5 的消融基线。A2–A6 继承 A1 的 backbone、检测头、数据管线、优化器及训练计划，仅调整模块开关。

| 设置 | 默认值 |
|---|---|
| Backbone | ResNet-50 |
| FPN 通道 | 256 |
| 模块作用层 | P2/P3 |
| 输入 | 等比例缩放至 1600×1000 范围，补齐到 32 的倍数 |
| 单卡 batch size | 2 |
| 精度 | FP32 |
| 优化器 | SGD，momentum 0.9，weight decay 0.0001 |
| 基础学习率 | 0.005 |
| 预热 | constant，10,000 steps，ratio 1/3 |
| 训练计划 | 12 epochs，step [8, 11] |
| 验证 | 每轮验证，按 bbox_mAP 保留一个最佳权重 |

README 中的 AP 表来自已完成训练的 JSON 日志；A3/A4 采用过 20 轮训练，其最佳轮次分别为 18、19。它们与 12 轮实验应作为实验记录阅读。

本次构建统计为：A0 32.132M、A1 31.608M、A3 31.816M、A4 31.715M、A5/A6 31.923M。相对 A1，A5 增加 315,780 个参数，约 0.316M（1.00%），已有 12 轮结果的 AP 为 25.5→27.1。

640×640 对照配置为 `configs/visdrone/a5_tafe_csta_640.py`，等比例缩放后补齐为方形。其已有 12 轮最佳结果为 AP 14.3、AP50 27.7、AP75 13.2；不能与高分辨率结果混为同一输入设置。

## AI-TOD

基础消融配置统一使用 P3–P7、800×800、batch size 16、12 轮训练、step [8, 11] 和 1,250 步预热。TAFE/CSTA 接入 P3/P4。结束时在验证集评估，再使用 `scripts/eval.sh` 在独立测试集评估。

原始 batch-16 组合实验使用过 10,000 步预热，12 轮内没有结束预热，随后修正计划并续训至 24 轮。当前基础配置使用已修正的预热长度。续训配置和从头训练配置分别列出，原有不同训练时长的结果不在此合并为公平消融结论。

```bash
bash scripts/train.sh configs/aitod/resume/tafe_csta_24ep.py 0 work_dirs/aitod_resume \
  --resume-from /path/to/epoch_12.pth

bash scripts/train.sh configs/aitod/resume/fcos_24ep_bs4_fp16.py 0 work_dirs/fcos_resume \
  --resume-from /path/to/epoch_12.pth
```

续训配置使用总轮数 24 和 step [20, 23]，关闭预热。FCOS batch-4 FP16 配置将学习率从 0.005 按 batch 比例调整为 0.00125。`--resume-from` 恢复模型、优化器、轮次和迭代状态；`load_from` 只用于加载模型参数。

## 自定义训练轮数

修改总轮数时需同时检查衰减节点、预热步数和验证间隔。例如 20 轮 VisDrone：

```bash
bash scripts/train.sh configs/visdrone/a3_tafe.py 0 work_dirs/a3_20ep \
  --cfg-options runner.max_epochs=20 lr_config.step='[14,18]'
```

这是指定新训练设置的示例；应以保存的配置和日志记录实际运行条件。

## 计算量与速度

```bash
python tools/complexity.py configs/visdrone/a5_tafe_csta_appearance.py --shape 1000 1600
python tools/benchmark.py --config configs/visdrone/a5_tafe_csta_appearance.py \
  --checkpoint /path/to/best.pth --image /path/to/image.jpg --device cuda:0
```

`complexity.py` 报告参数量与 MMCV 注册模块的 MAC 统计。CSTA 内部函数式注意力矩阵乘法、插值和检测后处理不被该统计器完整覆盖，因此输出不能作为完整推理 FLOPs。历史资料中的 GFLOPs 数字也不用于当前仓库的完整计算量声明。

`benchmark.py` 包含图像读取、预处理、网络前向和 NMS，使用预热后的重复测量。报告速度时应同时记录硬件、输入、精度、batch size 与计时范围。
