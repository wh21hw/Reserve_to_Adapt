# Research log

从source_only/正常BN/原known权重出发，只扩充unknown CE人口。原r全部留存，几何可靠IMP成员追加并沿已有簇ID监督。不是前期在原r内部切label scope，不扫阈值/K，不叠加失败承接/merge/veto。

代码dccc1c56/exec40，T4/shell13同缓存/mount、共同warm4各续训6，3600秒每arm/7200秒总。一次原集合顺序保持/去重可靠资格/实际added labels曝光与生成代码检查通过。新增选择导致CE平均与子batch BN统计变化是方法后果，不作过度因果归因；target语义只后处理。整体三任务未完成。

完成：两组完整epochs5–10，collector0，实际446.91秒。ZIP正常下载，小summary fetch失败从同ZIP提取，无重训/重评模型。final97.3441/49.8748/65.9564 vs97.1115/58.1918/72.7749，known−0.2326pp/UNK+8.3170/HOS+6.8185，满足本次guard及min_delta，保留候选opt-in，尚不更改默认selector或宣称完成。相比旧source_only参考HOS+2.4055/known−0.5659，背景控制UNK54.99 vs本次49.87存在显著跨进程数值漂移，不作显著性结论，下一步同配置独立确认。

两组fullbest共享warm4（86.4022/73.9103/79.6696），post4best均5，HOS75.3200/75.3096；这次收益来自后期final，不挑best。最终K20/V8 vsK17/V7。ARI .485813→.477688、NMI .709934→.706953，不能声称语义结构恢复已改善。

actual teacher-added194：unknown177、known17，独立unknown73/known5。73不是相对对照新增73个unique（两组训练动态不同）；总unique unknown selected194→204、structure监督136→152。CE实际总人口1339→1536，原入口在union中1536−194=1342。新增teacher known污染8.76%，总误known结构标签7→35；source/entropy/align原权重无修改，unknown误known权重795→784。第10轮开始eligible却无CE的unknown11→3，新增支持机制确实执行；类别21新增0、原结构曝光仍仅2，覆盖不全。所有类别语义仅结束注释，不进入训练/阈值/K。

下一批应同设置确认，允许预声明独立run名；不同时改predict规则/熵/阈值/seed/BN。不因首次pass立刻声称超过论文或完整unknown建模。进入从头完整训练前须只在warm4后启用union，以保持warm统计公平；本pilot已经共享warm4，无需重跑预热。整体和OfficeHome/VisDA仍未完成。
