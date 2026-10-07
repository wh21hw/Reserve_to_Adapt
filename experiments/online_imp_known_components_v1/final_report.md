# 模块分解：未知收益主要出现在entropy veto，已知代价仍在

共同warm4各续训6，T4/ResNet50/256teacher/source-only3/frozenencoderBN/seed3；代码9c9c6103，exec20/collector完成。原none/both参考同warm，独立进程数值噪声存在，不作显著性或完整交互因果结论。

| known mask位置 | final OS* | final UNK | final HOS | post4best epoch/HOS |
| --- | --- | --- | --- | --- |
| none（已完成参考） | 98.0108 | 54.7566 | 70.2602 | 5 / 75.3200 |
| both（已完成参考） | 92.6007 | 69.9051 | 79.6681 | 10 / 79.6681 |
| 仅entropy | 93.0658 | 71.1560 | 80.6493 | 10 / 80.6493 |
| 仅target alignment | 98.0108 | 51.0482 | 67.1315 | 5 / 75.3200 |

entropy相对none UNK+16.3994/HOS+10.3891pp，但OS*−4.9450pp guard失败。alignment相对none known相同、UNK−3.7084/HOS−3.1288pp，不改善。fullbest：entropy为10/80.6493，其余共同warm4/79.6696；不把warm peak当改造收益。

实际记录：entropy删除167已知/412未知权重，alignment保持；alignment删除166已知/448未知权重，但entropy保持。两种mask都删掉了部分未知，只有entropy分支明显改变拒识。这支持known entropy的错误强化值得处理，而不是泛称所有对齐都应关掉；不证明任何域对齐都无风险。

entropy仍误伤44独立已知、原型错候选16.95%，不能直接采用硬veto。下一项只比较raw vs既有reliable候选的entropy资格，alignment保留、其他目标不改，检验保护已知的同时还能保留收益。不扫描强度/阈值，不放宽1pp guard。

小结果ZIP/summary已下载；图/posthoc正常记录处理，无重评模型。旧参考复用及事后seed/oraclebest/前置BN明确披露，三任务尚未完成。
