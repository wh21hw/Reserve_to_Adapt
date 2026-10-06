# 残差方向：保护已知但未激活新增未知槽

单次CPU缓存测试exit0，0训练。source958与target564个样本最终全头预测均与原模型一致，新增三个槽全头赢家均0；source未知预测仍0，保护guard通过但功能激活目标未达成，因此不进入长训练。

unknown-only条件argmax中target新槽各10/6/6个赢家。这不等于最终拒识或未知类别恢复，因为C+K全头竞争没有变化。候选行12最近source类3（残差范数12.1069），行13/14最近source类2（4.7191/3.7611），这些距离未用于额外调参。

与上一均值候选的区别仅为三个新增行的方向定义；K、norm、成员和未修改分类行相同。不能把两次失败组合成训练因果定论。仍然使用epoch10成员与epoch70表示，source是训练数据，没读目标语义标签或重评accuracy。后续应检查域偏移与未知证据如何进入训练，而不是继续无监督地调高新槽分数。

完整报告：pipeline-results/a2w-new-slot-known-residual-proxy-v1-summary.json。按autoresearch规则记录reverted，原checkpoint未改，不需回滚。
