# 方法与代码对应

## 前向路径

VisDrone 使用 ResNet-50 的 C2–C5 输出，经 FPN 形成 P2–P5。TAFE、CSTA 依次接在 P2 和 P3 上，P4/P5 保留普通 FPN 路径。AI-TOD 配置形成 P3–P7，在 P3/P4 使用同一组模块接口。

```text
图像 → ResNet-50 → FPN → 选定层的 TAFE → CSTA → FCOSHead
                         ↓
                     前景辅助监督
```

检测头输出类别、边界距离和 centerness。模块分文件存放，配置中的注册名仍为 `FSTAFPN` 和 `FSTADet`；已有检查点的参数键不因文件拆分而改变。

## TAFE

代码：[tafe.py](../mmdet/models/plugins/fsta/tafe.py)。

给定特征 F，先池化至约一半尺寸，再将池化结果继续池化至约四分之一尺寸。两张低分辨率特征分别双线性上采样回原尺寸，形成 L2、L4：

```text
E2 = F - L2
E4 = L2 - L4
E  = dilated_depthwise_conv(E2 + E4) → GELU → pointwise_conv
Gtexture = sigmoid(pointwise_conv(GELU(pointwise_conv(concat(|E2|, |E4|)))))
Fout = F + alpha × Gtexture × Gforeground × E
```

`alpha` 为逐通道参数，初始化为 0.001。前景概率来自当前 FPN 特征，不需要真实框参与推理。批次补齐区域被有效掩码排除，增强分支不会读取这些区域的填充值。

此处提取的是空间尺度残差。`E2 + E4 = F - L4`，因此精炼分支的两项会相消；纹理门控仍分别读取 E2、E4。它不是严格的傅里叶频带分解，也不保证残差仅来自目标。

## CSTA

代码：[csta.py](../mmdet/models/plugins/fsta/csta.py)。

1. 通过 1×1 卷积将 256 个通道投影到 64 个内部通道。
2. 划分 8×8 窗口；4×4 个窗口组成一个 32×32 的搜索区域。
3. 在每个窗口内形成代表特征。A6 使用预测前景概率加权，A4/A5 使用外观特征。
4. 根据相似度，将 token 分配给区域内的代表特征并排序。
5. 对排序后的固定长度块做注意力，每块可读取前一块的后一半、当前块和后一块的前一半。
6. 按逆排序恢复 token，再恢复二维布局、移除填充、投影回 256 通道。
7. 使用 `X + beta × Delta_CSTA` 完成残差融合，`beta` 初始化为 0.001。

搜索范围受限于区域，避免整张特征图的全局注意力。`argmax` 和排序索引是离散操作，梯度不会通过分配索引；Q/K/V、门控与残差参数仍参与梯度计算。特征相似不等于实例或类别相同。

## 前景监督

代码：[foreground.py](../mmdet/models/plugins/fsta/foreground.py)、[fsta_det.py](../mmdet/models/detectors/fsta_det.py)。

前景头由深度卷积、GELU 与输出一个通道的 1×1 卷积组成。辅助目标复用 FCOS 点分配：类别点为正样本，背景点为负样本，补齐区域与忽略框区域不参与辅助损失。使用二元 focal loss，默认权重为 0.1。

```text
Ltotal = Lclassification + Lbox + Lcenterness + 0.1 × Lforeground
```

FCOS 主检测头仍沿用基础框架的分配语义；辅助损失中的忽略处理不代表主头额外实现了忽略区域分配。目标分配使用 FP32，避免 FP16 无法表示最终回归范围上限 1e8。

## 框架改动

`mmdet/datasets/coco.py` 增加可覆写的 evaluator 构造接口与空预测结果处理。`VisDroneDataset` 用类别无关忽略区域和统一 maxDets 进行 COCO 风格评估。

`mmdet/core/evaluation/eval_hooks.py` 使用 MMDetection 的测试函数恢复原图坐标；最佳权重先写临时文件，再替换正式文件，成功后清理旧最佳权重。有限的 0 AP 是合法结果，验证时也能保留检查点。
