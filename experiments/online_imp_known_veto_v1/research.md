# 协调IMP与RTA的known目标：新候选不被known目标拉回

## Goal

前几批扩大空间、标签覆盖或换层均未恢复未知；UNK从warm73.91%退到约52–57%。检验一个可证伪冲突：RTA GMM认为known的样本可能当前IMP已经属于新候选，但仍被known entropy与target adversarial拉向known。阻断这一known支持，不强制未知标签或新增loss。

## Success Metric

有效第10轮final OS*/UNK/HOS和epoch5–10轨迹，HOS至少+1pp、OS*/UNK各不下降超过1pp筛查，人工综合；实际known权重veto中的已知/未知分布、unknown曝光、ARI/NMI/槽使用/校准污染。target真值仅结束注释，不入known veto或K/阈值。

## Constraints

- max_iterations:1；pause_every:never；Evaluator: _(none — agent judges manually)_。
- 现存共同warm4恢复，两arm rta_known/imp_veto各续训6（epoch5–10），实际12新轮、有效各10，不再warm/source或下载。
- T4/ResNet50/256teacher/A→W/C10/seed3/source-only3/frozenencoderBN/batch64/原损失系数与最终argmax。每轮IMP动态K，confidence阈值、screened结构标签和screened virtual两组相同；不启用此前未达筛查的all_candidates。
- 唯一因素：target known权重原RTA vs 原权重乘当前IMP分配到已知原型的mask。只作用同一个weight本来控制的known entropy/target adversarial，source CE/source adversarial/virtual、r未知选择规则与unknownCE标签范围不变。**先计算r，再veto known weight**，不偷偷把mask样本全部送unknownCE。
- 一部分真正已知可能被IMP错标新候选；必须衡量丢失known支持和OS*，不假设安全。无known权重时原损失按batch N平均为0，不除weight.sum，不引入NaN。
- 两组同head CPU fork_rng隔离，已知/对应未知头与SGD/schedule/RNG恢复。新weight接口做一次针对性检查，300秒。
- 每arm3600秒/总7200秒，覆盖技能默认5分钟；预计约8分钟，观察近期elapsed估ETA，安静hook。K0/100cap失败不强制K或调阈值。
- 不加loss，不改BN/架构/预测，不70轮/换seed或其他任务；结束分析再下一因素。只小记录Drive，大模型留runtime，关闭前备份，不擅自删云盘。

## Search Space

只known支持资格。mask导致K/虚拟/后续候选改变是该因素下游，不把它们作为独立消融。

## History

预声明，尚未启动，复用挂载T4 gpu-t4-s-kkb-ass1c2-1j7bns9e7dhke和完整共同warm4；上一标签覆盖+0.9871pp不晋升。

已启动：代码b2272ab8，background exec17，独立shell13，控制台`/content/online-known-veto-console.log`，root`/content/imp-runs/online-imp-known-veto-v1`。新weight接口一次通过，不改原tensor/assignments；原Entropy/BCE均按batch N平均，权重全0有效。模型不复制Drive；hook已替换17，每5分钟安静检查，不重复旧结果。

rta_known已完成epoch5–7，elapsed113.2秒、近期36.66秒/轮；总批预计约7.5–8.5分钟。按imp_veto实际阶段再估时，ETA不是硬截止。

完成：exec17/collector成功，final HOS70.2602→79.6681%、UNK+15.1486pp但OS*−5.4101pp触发guard，不晋升。屏蔽382真实未知和161真实已知加权曝光，下一项仅entropy/仅alignment分解，同warm参考，见final_report。
