# 第二次确认也通过，下一步验证完整训练

完全同设置：T4/ResNet50/C10/source3/frozenencoderBN/seed3，source_only每轮IMP、原BN/权重/损失/预测，共同warm4后每arm续训6。没有新seed或新参数。

| 重复 / 方法 | final OS* | final UNK | final HOS |
| --- | --- | --- | --- |
| 首次原入口 | 97.3441 | 49.8748 | 65.9564 |
| 首次可靠成员union | 97.1115 | 58.1918 | 72.7749 |
| 确认原入口 | 97.6774 | 54.3523 | 69.8415 |
| 确认可靠成员union | 96.7679 | 60.6717 | 74.5819 |

首次Δknown−0.2326/UNK+8.3170/HOS+6.8185pp；确认Δ−0.9096/+6.3194/+4.7404pp，两次通过。第二次known损失距1pp上限仅0.0904pp，不称完全无损，不做第三次重复直到挑更好。两次都是seed3，非三seed均值或统计显著性证明。

四arm fullbest均共享warm4（86.4022/73.9103/79.6696）；两批post4best均5，control HOS75.3200、union75.3096，收益来自后期final而非选best。确认final K/V=22/9→18/6。

两批post4best完整OS*/UNK/HOS均为：control 91.1287/64.1854/75.3200，union 91.1287/64.1702/75.3096。首次final K/V=20/8→17/7；确认K范围17–38（两组），V范围3–20（两组）。首次ARI .4858→.4777、NMI .7099→.7070，新增194曝光中177unknown/17known（8.76%污染），独立73unknown/5known；总独立unknown结构监督136→152。

确认teacher新增197实际曝光，181unknown/16known（8.12%污染）、独立76unknown/5known；总独立unknown结构监督138→152。误known结构标签7→31，类别21新增仍0，已知边界风险和覆盖缺口存在。ARI .4707→.4946、NMI .6947→.7140，本次上升，但首次下降，语义结构提升未稳定验证。目标真值仅结束注释，不能据此把语义标签加入训练。

确认训练464.39秒、collector退出0，ZIP/summary/posthoc/图保存。下一阶段保持方法固定，从同warm4各续训5–70，预声明更长预算并验证收益持续性；不叠加熵/阈值/BN/预测改变。完整70轮结果未出前不声称超过论文。

事后seed选择、target-oracle方法筛选与best、额外source预热和encoderBN均需披露。当前比较是IMP框架内模块，不将control伪称未改动原RTA。OfficeHome/VisDA与总体目标仍未完成，实例/cache/mount/models保留。
