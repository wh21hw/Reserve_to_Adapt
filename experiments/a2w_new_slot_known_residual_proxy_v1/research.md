# 未知方向的已知中心残差：单因素缓存诊断

## Goal / Hypothesis

均值方向初始化已激活三个新槽，却误拒识97/958个source已知样本。检验一次替换方向：候选均值减去head空间最近的source已知类均值，再单位化并取相同固定norm。假设去掉共享已知成分可降低已知竞争。残差也可能只是域偏移，不宣称必为未知语义。

## Metric / Guard

完整target全头argmax中至少一个新槽有赢家；source未知预测相对未改原模型增加不超过1pp。source是监督训练数据，仅机制保护，不是泛化证明。不读target标签，不选择norm、距离阈值或其他参数。

## Constraints

max_iterations: 1；pause_every: never；Evaluator: agent按JSON综合判断。0训练轮，CPU一次缓存计算，上限300秒。与均值候选仅差方向定义；known10/retained unknown2不动、K5不动、norm保留12行均值不动、固定epoch10成员不动。预声明guard未通过就不接长训练；不扫缩放、不加损失、不改预测规则。

复用final70 C10/K5缓存和实际BN→LeakyReLU head输入，使用允许的source监督标签算10类中心。跨时段限制同前：epoch10成员在epoch70表示计算，不冒充epoch10干净训练消融。结果单独目录，不覆盖均值候选或任何checkpoint。

## History

| Iteration | Change | Observation | Decision |
| --- | --- | --- | --- |
| 0 | 未改final70分类头 | target新槽使用0；source未知预测0/958 | 原始模型基线 |
| 1 | 新槽方向=簇均值−最近source已知均值 | source误拒识0；target新槽全头使用0，source/target最终预测均完全不变 | reverted：通过保护但未激活，不接长训练 |
