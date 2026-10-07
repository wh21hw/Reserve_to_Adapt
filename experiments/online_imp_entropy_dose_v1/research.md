# 候选未知的已知熵：三个取舍工作点

## Goal

补充可靠训练入口完整70轮提高HOS4.97pp、known0.67pp、UNK5.73pp，但final UNK仍57%、known近100%；骨架继续训练使unknown低于warm4。旧entropy硬veto明显改善未知但误伤known。检验适度减弱候选unknown的known entropy，寻找更好的工作点，而非仅扩大K。

## Success Metric

主看同批final HOS及OS*/UNK完整组合，HOS+1pp为候选收益线，不使用旧known下降1pp作为单独否决。各点报告final/best/post4best/ARI/NMI/污染/覆盖，标出Pareto取舍，不只挑最高best或忽视known塌缩。无统计显著性或最终无标签最优点承诺，具体保留由综合手工判断。

## Constraints

- max_iterations:1; pause_every:never; Evaluator: _(none — agent judges manually)_。
- T4/Office31 A→W/seed3/ResNet50/C10/batch64/source3/frozenencoderBN，共享已有commonwarm4，各续训epoch5–10共6新轮，三arm共18新轮；每arm3600秒/总10800秒，预计12分钟，按实测重估。覆盖默认5min，不加新seed或无上限网格。
- 三arm统一reliable_union训练入口、source_only每轮IMP/confidence阈值/screened簇标签/virtual/无merge/原headBN/原adv/未知CE/单槽argmax/头SGD匹配不变。
- 唯一因素候选unknown所受原known entropy权重的乘数gamma=1/0.5/0，已有known posterior样本权重照旧；不改变原r或target alignment。raw occupied IMP unknown作为资格，已知身份成员权重保留1。不是全局减半熵、不修改source监督，候选误known也可能被减弱，需要记录。
- 共用warm4恢复，gamma从epoch5启用；不重新预热。gamma1控制数学上等价原熵权重；默认旧硬veto保持gamma0，稳定原流程不更改默认。
- 新权重接口做一次针对性检查（gamma1/0.5/0精确权重、alignment不变），不重跑训练smoke/hash/checkpoint重评。
- target真值仅结束评价和曝光注释；seed3/方法工作点选择/best为target-oracle探索，非三seed统计。先短程探索，不伪称70轮或论文复现；跨数据集未验证。
- 独立目录不覆盖前结果，复用缓存/挂载，普通小结果Drive及电脑，模型runtime关闭前备份。API项目quota非容量，不擅自删除数据或重启销毁。

## Search Space

预声明三个点gamma1（原熵）、0.5（半强度）、0（硬veto）。预算用完先评价，再决定下一单因素，不追加gamma网格直到挑通过。

## History

预声明，尚未启动。上一完整70轮支持入口模块有收益，但不是本批γ控制的干净对照；本批三个点同预算重新配对，旧entropy-only无union参考只能作背景。

已启动：代码aa20128b，exec46/launcher146130/shell13，console /content/online-entropy-dose-console.log。新权重接口一次检查gamma1/0.5/0及alignment不变均通过，无模型/训练smoke。5分钟hook已更新为三点取舍，完整结果后不采用旧1pp单项否决。
