# 可靠IMP训练入口：完整70轮配对

## Goal

两次10轮短程确认均改善final拒识，但已知边界损失接近上限、语义分组改善不稳定。验证同一方法的收益能否持续到70轮，不再做第三次短程重复。

## Success Metric

配对final HOS至少+1pp，OS*/UNK各下降不超过1pp。共同warm4、post4best与final完整报告，ARI/NMI、K/V、实际新增监督及污染同时报告。不只挑best，不把更多簇当恢复真实类别。若完整训练不通过，标为短程收益不延续；不重跑至通过。

## Constraints

- max_iterations: 1; pause_every: never; Evaluator: _(none — agent judges manually)_。
- T4、Office31 A→W、seed3、ResNet50/C10/source3/frozenencoderBN/batch64。共享已有完整warm4，各续训epoch5–70共66新轮，实际132新轮；有效70轮/arm。共享source3前置训练已存在，不重复训练或下载。
- 每组5400秒、总10800秒，覆盖技能默认5min。按短程约37秒/轮预计总80分钟；实际几轮及换组后重估ETA。超时保留证据，不自动延长或重跑。
- source_only每轮IMP、confidence target无标签阈值、screened簇标签、无merge/无veto/原headBN/virtual/损失系数/单槽argmax/头与SGD匹配不变。
- 唯一组间因素：原RTA入口r vs 保留全部原r并追加去重的可靠IMP成员。两组均已有IMP动态K/virtual及簇标签骨架，control不是发布版原RTA。
- 复用warm4公平分叉，union从epoch5开始；不重新跑warm并引入zeroCE前向BN差异。原RTA第10轮后的筛选分支按原代码自然执行，两组一致，不偷偷延长早期top16策略。
- target真值仅评价/曝光事后注释，不能用于K/阈值/训练；事后seed3、target-oracle方法筛选/best及额外预热BN均披露，非三seed统计。OfficeHome/VisDA待验证。
- 小结果复制mounted Drive并下载电脑；API项目quota无法独立验证云同步，模型runtime关闭前另行备份。不hash、checkpoint重评、重复smoke，不关闭/重启实例。

## Search Space

仅延长已固定方法至70轮配对。无参数网格或新损失。Runner只参数化轮数与预算/collector，不修改算法。

## History

预声明：首次final ΔOS*/UNK/HOS=−0.2326/+8.3170/+6.8185pp；确认−0.9096/+6.3194/+4.7404pp。两次single-seed均通过，确认known余量仅0.0904pp。完整70轮尚未启动。

已启动：代码9241b007、exec45/launcher109227/shell13，现有T4端点gpu-t4-s-kkb-ass1c2-1j7bns9e7dhke。首个续训epoch5完成、loss有限，实际41.15秒（含初始提取/恢复开销）；待近期轮间差估计稳定ETA。10分钟hook已替换旧确认hook，完成后自动收集，不关闭实例。

epoch6 elapsed77.8955秒，epoch5→6实际36.7456秒/轮；当前第一组余64轮约39分钟，第二组估41分钟，总剩约80分钟（非硬截止）。epoch10后筛选/K变化和换组时继续重估，无OOM或非有限loss。
