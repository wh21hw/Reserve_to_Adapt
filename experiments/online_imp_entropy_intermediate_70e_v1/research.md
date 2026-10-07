# 开题展示：中间熵取舍工作点

## Goal

寻找含IMP的真实A→W部分结果，期望HOS超过论文RTA93.0%。允许best seed/epoch展示但保留全部结果，不以原RTA复现95.2172冒充IMP结果。

## Success Metric

手工评价完整best/post4best/final OS*/UNK/HOS与结构指标。若best HOS>93.0，仅称单任务探索性best超过论文数值；超过本地原RTAbest95.2172是另一门槛。真正模块收益还需同预算无该模块对照，当前两权重比较不能替代它。没有过线则如实记录。

## Constraints

- max_iterations:1，pause_every:never，Evaluator:none(agent judges manually)，noise_runs:1。已获无人值守授权，不进行无限参数扫描。
- Office31 A→W、ResNet50/256/C10/batch64/seed3、source3、冻结encoderBN、共同warm4。两组各续训5–70共66新轮，总132，有效70轮/arm。
- 唯一因素raw IMP候选entropy gamma0.1与0.25。reliable_union/source_only每轮IMP/confidence无标签校准/screened簇标签/virtual/原alignment/unknownCE/原headBN/argmax/匹配头SGD均不变，无merge。
- 每arm5400秒，总10800秒保护，预估80分钟，history实测更新ETA。最多本批两候选，不再加密网格或偷偷重复。
- 事后seed3/方法/epoch选择，额外source预热/BN非原论文。此为探索性权重选择而非独立验证；不能据target评价选择继续训练阈值或K，targetGT不进入训练、聚类、筛选。
- 复用现有T4与Drive/cache/commonwarm。不是新完整seed，不将fork后改RNG称另一个seed。用户授权并行实例，当前共同checkpoint未在第二实例就绪，先现有实例串行，不伪称并行。
- 稳定连续权重接口已检查，不重复smoke/hash/checkpoint重评。普通进程/loss/轮数/保存检查；工程失败保留证据，不算法重试。
- 小结果Drive与电脑，模型runtime关闭前备份。不关闭/重启实例。批结束收集报告，hook暂停，不自动无限后续实验。

## Search Space / Hypothesis

预先固定gamma0.1/0.25两个中间点。已有gamma0 fullbest90.2942且后期known下降，gamma0.5 best82.3078偏known；中间权重可能保留更多known边界同时提供unknown训练空间。此假设可能失败，不承诺必然超过93%。历史不同批次只背景，不能混称严格配对或多seed统计。

## History

用户最新要求快速结果，70轮计划撤销且未启动。实际改为online-imp-entropy-intermediate-10e-v1，各5–10六新轮，总12新轮，900秒/arm、1800秒总保护，预计8分钟。上述70轮预算不再生效。目录名保留作为撤销记录，本批不能当70轮实验，不承诺短程超过论文。console /content/online-entropy-intermediate-10e-console.log，结果目录/content/imp-runs/online-imp-entropy-intermediate-10e-v1。

完成：exec51/collector0，两组各own5–10+shared4，新增训练229.904+238.551=468.455秒（7.81分钟）。gamma.1 best/post4best/final均epoch10，OS*/UNK/HOS92.1773/76.2334/83.4506%，K22/V7、unknownARI/NMI.636656/.788914；gamma.25对应94.7794/71.1342/81.2719%，K17/V5、ARI/NMI.546739/.732708。gamma.25−.1 final+2.6022/−5.0992/−2.1787pp。两个工作点互不支配，.1 HOS更高但两组均未超论文93，不能借独立历史差值当严格配对。保留完整summary在电脑。ZIP直传两次fetch失败属于传输问题，不重训；尝试另名传输。旧方案三seed已另批启动，本批不再加密gamma扫描。
