# 未知结构训练入口：原RTA选择器 vs 加可靠IMP成员

## Goal

以恢复的source_only骨架直接检验结构成员是否因原RTA选择器未覆盖而缺少未知CE支持。不继续扫K/惩罚/BN或稳定结构。

## Success Metric

同批rta_only对照final HOS至少+1pp，OS*/UNK各不下降超过1pp；综合post4best/final、K/V、实际新增unknown/误known曝光及独立覆盖，不仅best。新增成员的可靠性是待检验假设，不能靠target真值选择。增加拒识不等于语义分组改善。

## Constraints

- max_iterations:1; pause_every:never; Evaluator: _(none — agent judges manually)_。
- rta_only/reliable_union同commonwarm4各续训6(epoch5–10)，实际12新轮；3600s每arm、7200s总上限覆盖技能默认5分钟。预计8分钟，以真实history近期elapsed重估hook。
- 唯一因素unknown CE入口：原RTA r集合保持原顺序并保留全部，再追加本batch当前IMP几何可靠候选中的未选成员，不重复、不添加teacher-known或unreliable。所有新增成员使用同已有screened IMP簇ID标签。
- source_only每轮重建、无merge、正常CLS BN、原known entropy/alignment和source/virtual/unknown损失系数、argmax、head匹配同；ResNet50/256teacher/source3/frozenencoderBN/C10/seed3/batch64/confidence target阈值同。不额外删原selector，不加新损失/强度/阈值。
- CE仍按原选择集合平均；人口扩大带来平均归一化与选中子batch BN统计变化是此入口改变的后果，不将结果归因成纯簇身份单一中介。
- target真值仅结束评价；一次新接口检查，不hash/checkpoint重评/新seed/70轮/网格/重启销毁。缓存/mount复用，小结果Drive，模型runtime关闭前备份。

## Search Space

只扩充原r集合的可靠IMP成员入口。逐样本teacher_added记录在实际labels调用处，区分真正unknownCE曝光与提案/预热零系数。

## History

预声明。之前标签覆盖实验只在原r内部改资格，不等于本次扩充训练集合。source_only参考97.6774/54.9944/70.3694、unknown独立结构覆盖145只是背景，重新同批控制；不推广失败承接/merge。三任务及整体未知结构目标未完成。
