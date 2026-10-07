# PPT阶段：IMP与原RTA编码器BN兼容性

用户于2026-10-07要求讨论7分钟开题PPT，并取得含IMP、能超过RTA的真实任务结果；允许best seed。现含IMP最高A→W HOS90.2942%，未超过论文93.0%，也未超过本地原RTA单seed best95.2172%。不保证实验必然达标，不把原版复现成绩当IMP效果。

## 本批预声明

- Office31 A→W，ResNet50/256，C10，batch64，seed3；复用source3与共同warm4。各续训epoch5–10共6轮，总12新轮。
- 唯一因素：续训时ResNet编码器BN运行统计冻结与否。frozen_encoder使用RTA_FREEZE_ENCODER_BN=1，normal_encoder=0。编码器结构、BN affine可训练性、分类头BN不变。
- 两组均为每轮source_only IMP、confidence无标签target阈值校正、reliable_union、screened簇标签、raw候选entropy gamma0、原alignment、virtual、unknown CE、argmax、头状态匹配；无merge。
- 每arm3600秒/总7200秒硬保护，预计8–10分钟，实际history近期elapsed重估。使用已有T4/Drive/数据缓存，不新下载或重新预热。
- 已稳定BN接口不重复smoke/hash/checkpoint重评。普通进程/loss/轮数/保存检查。
- 原论文不含额外source3/冻结BN共同warm4，因此normal续训也不是完全原版复现。此批不是模块移除消融，不能凭组间收益声称IMP独立有效。
- 报告全部best/post4best/final OS*/UNK/HOS、K/V及ARI/NMI，不只取最高best。seed3、方法和epoch为事后target-oracle探索；target真值只评价，不进入聚类/训练。
- 若没有信号不重复直到通过；下一长程验证需另行声明预算。模型保留runtime，关闭前备份。结束收集后hook停止，不自动无限实验。

## 状态

已启动：T4 gpu-t4-s-kkb-ass1c2-1j7bns9e7dhke，exec49/launcher219942/shell13，console /content/online-encoder-bn-console.log。frozen_encoder已进入epoch5，现有Drive挂载与共同warm状态复用成功。5分钟临时hook已开启，结果后报告并停止此批等待；不关闭实例。
