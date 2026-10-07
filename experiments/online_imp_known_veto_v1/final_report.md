# IMP veto known目标：明显未知收益，同时明显已知代价

两arm同warm4续训6，实际12新轮。代码b2272ab8，T4/A→W/ResNet50/256teacher/C10/seed3/source-only3/frozenencoderBN。exec17/collector成功，223.2/238.2秒，无NaN/OOM。

| known资格 | fullbest epoch/HOS | post4best epoch/OS*/UNK/HOS | final OS* | final UNK | final HOS |
| --- | --- | --- | --- | --- | --- |
| 原RTA权重（相同IMP骨架，非原版RTA） | 4 / 79.6696 | 5 / 91.1287 / 64.1854 / 75.3200 | 98.0108 | 54.7566 | 70.2602 |
| IMP新候选屏蔽known entropy/target adv | 同共同warm | 10 / 92.6007 / 69.9051 / 79.6681 | 92.6007 | 69.9051 | 79.6681 |

final HOS+9.4078pp、UNK+15.1486pp，但OS*−5.4101pp触发guard。**不晋升整套veto**，不能因HOS更高忽略已知边界。fullbest同warm，第10轮几乎追平warm HOS，但非完整预算/论文成绩。

## 实际权重记录

结束后按保存样本索引注释，不重评模型、不给训练真值。

- 控制known加权曝光3046，其中784来自真实未知，说明该已知目标确实作用在部分未知上。
- veto分支原权重2830→有效2287：移除382真实未知权重，也移除161真实已知；43个独立已知/96个独立未知受屏蔽。
- 这些是加权样本曝光，不是loss幅度或完整因果分解。分支下游模型变化，不能把控制784与veto原565的变化都算作直接屏蔽效果。
- veto仍有183真实未知known权重；校准支持未知污染26.06%→19.44%，仍非纯净。当前原型也错把17.63%已知当候选（控制5.42%），有已知误伤风险。
- 未知实际结构标签：控制806真未知/7真已知，veto785/11；独立未知136→157。unknown选样规则没改，支持计数变化是学习下游。

## 结论与下一项

屏蔽两个known目标改变了已知/未知取舍，符合存在目标冲突的可能性，但不证明全部退化来自它，也不能分清entropy与alignment各自贡献。下一项分别仅veto known entropy、仅veto target adversarial，仍同warm4各续训6；复用本控制/both结果作为同设置参考，不重复训练参考。目标是保留未知收益且保护已知，不放宽1pp guard。

小结果已下载，模型留runtime、小记录Drive；关闭前备份。事后seed/oraclebest/额外前置BN/单seed短程以及目标标签只评价的边界均保持。
