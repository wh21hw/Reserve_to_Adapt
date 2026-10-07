# 补训练入口：首次守住已知并改善未知识别

T4/ResNet50/source3/frozenencoderBN/seed3，同完整warm4各续训6(epoch5–10)。只改变unknown CE人口：原r保持，再加入几何可靠IMP成员。source_only/正常BN/原entropy与alignment/损失系数/预测规则保持相同，不叠加失败承接或merge。

| 入口 | final OS* | final UNK | final HOS | post4best epoch / HOS |
| --- | --- | --- | --- | --- |
| 原RTA选择器 | 97.3441 | 49.8748 | 65.9564 | 5 / 75.3200 |
| 原选择器＋可靠IMP成员 | 97.1115 | 58.1918 | 72.7749 | 5 / 75.3096 |

同批known−0.2326/UNK+8.3170/HOS+6.8185pp，通过短程guard，保留为候选，默认入口尚未更改，需同设置确认。两组fullbest均共同warm4（86.4022/73.9103/79.6696），不是改造收益；收益体现在final而非挑最高best。相对已完成source_only背景97.6774/54.9944/70.3694，HOS+2.4055/known−0.5659pp；背景控制与本批有数值漂移，不冒称统计显著。

新增194次实际CE训练：177unknown、17known，独立73unknown/5known。73不是相对对照多覆盖73个样本；两个动态训练轨迹不同，总独立unknown结构监督覆盖136→152，普通unknown CE覆盖194→204。未知结构成员未获得CE的问题确实部分缓解，第10轮eligible且无CE的unknown11→3。

可靠性仍有限：新增污染8.76%，总误known结构标签曝光7→35；类别21新增0次、总结构监督2次。最终ARI .4858→.4777、NMI .7099→.7070，不能称语义聚类改善或真实类别数恢复。K最终20→17，容量变化不是成功的唯一原因。目标真值仅结束注释，不用于阈值/K/训练。

两组新训练446.91秒，collector退出0、结果ZIP/summary已保存。下一步同设置独立确认，暂不叠加新预测/熵/阈值，不直接用此10轮single-seed当论文成绩。若从头完整训练，扩展需仅在warm4后启用，避免零系数unknown CE前向改变warm BN统计；本pilot已共享warm4。

seed3事后选择、target-oracle best及探索性方法筛选、额外source预热/encoderBN需披露；非三seed均值、非完整论文复现。OfficeHome/VisDA与总体未知结构目标尚未完成，runtime/cache/mount/models保留。
