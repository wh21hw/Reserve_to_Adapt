# 固定分类头BN：未解决已知/未知的取舍

同完整warm4，各续训6，T4/ResNet50/seed3/source3/frozenencoderBN，唯一因素是CLS BN是否固定running统计，affine仍训练。每轮IMP、原entropy/alignment权重、virtual、未知选择和预测保持相同。

| 模式 | final OS* | final UNK | final HOS | post4best epoch / HOS |
| --- | --- | --- | --- | --- |
| 原batch统计 | 98.3441 | 53.4242 | 69.2365 | 5 / 75.3200 |
| 固定running统计 | 92.3548 | 62.2183 | 74.3487 | 9 / 79.5166 |

final HOS+5.1122pp但known−5.9892pp，guard失败，不采用。两组fullbest均来自共同warm4：OS*86.4022/UNK73.9103/HOS79.6696，不将预热峰值当改造收益。两组各epoch5–10，实际训练454.30秒，ZIP与summary已下载，collector退出0。

未知预测ARI .4758→.3710、NMI .7073→.6494，拒识改善不等于语义簇更好。原未知selector实际选中已知的加权曝光106→147，IMP覆盖已知伪标签7→16，unique未知结构训练覆盖134→132；更多unknown槽（final K19→29）没有带来更多训练覆盖。上述是记录后的语义注释，不进入训练或阈值，不是完整因果结论。

下一步停止BN/筛查网格，回到每轮IMP重建与簇身份承接：需要区分“特征变化后重新估计”与“丢掉上一轮结构重新发现”。保留原BN，不叠加失败veto；是否应改原型承接由实际重建/成员匹配记录决定。

best使用目标标签选epoch，seed3事后选择、source-only3与冻结encoderBN均为额外设置；非三seed均值，非原论文完整复现。OfficeHome/VisDA仍未验证，整体目标未达成。
