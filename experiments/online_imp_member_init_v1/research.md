# 每轮IMP承接当前成员中心

## Goal

检验每轮只从source中心重新建簇是否使未知结构的训练支持不稳定。分类头已有匹配承接，不将它误称每轮完全重置。原raw变化率含编号置换，先增加无标签Hungarian稳定性指标，两组相同记录。

## Success Metric

同批source_only对照，final HOS至少+1pp，OS*/UNK各下降不超过1pp；综合post4best/final、匹配后成员保持、实际训练覆盖与K变化，不只选best。Target真值仅评价。

## Constraints

- max_iterations:1; pause_every:never; Evaluator: _(none — agent judges manually)_。
- 两组同commonwarm4各续训6（epoch5–10），实际12新轮，3600秒/arm、7200秒总上限，覆盖技能默认5分钟，预计8分钟，按实际history重估hook。
- 唯一因素是IMP未知起点：source_only原流程；current_members使用上一轮各非空未知簇成员在当前网络下重新提取特征的均值，与当前source anchors一起启动。不是旧坐标EMA、不是新unknown先验强度，不加损失、不冻结K；超过原阈值仍出生，软质量及无硬成员仍剪除。
- 两组原CLS BN、frozenencoderBN/ResNet50/256teacher/seed3/source3/C10/batch64，confidence target阈值、screened结构标签、virtual、source/unknown选择/CE、known entropy/alignment、argmax完全相同，不叠加失败mask。
- known identity保持，unknown头/SGD仍原成员匹配；初次warm-end Kmeans对应重置保留。指标Hungarian只计算，不进训练。
- 一次新接口检查，不hash/重复checkpoint评价/新seed/70轮/网格/重启或销毁。小结果Drive、模型runtime关闭前备份，缓存/mount复用。

## Search Space

仅是否承接当前成员均值；不同时改阈值、轮数、损失或预测。

## History

原控制epoch5–10 raw变率约27–49%，但包含编号置换，不是可用的稳定性结论。head承接仍每轮有0–4新随机行；因此分开测试IMP初始化，不改头策略。预声明，未运行；总体未知结构及三任务目标未完成。

已启动：b2325543，T4复用，exec32，shell13，`/content/online-member-init-console.log`。一次当前坐标/置换不变/出生/剪除与训练代码生成检查通过，无预热重跑。5分钟hook已恢复并替换为本批，不监控旧exec27。

launcher60073。普通结构记录新增每轮成员NPZ用于结束后对齐诊断，不重新评模型。commonwarm4旧记录缺匹配后变率，只比较续训5–10阶段。

实测source_only epoch7 elapsed114.2131s、epoch8 elapsed150.0261s，近期35.81秒/轮。按相近速度本批合计约7.5–8分钟、当前剩约5分钟；current_members改K后重估。控制组已完成epoch8，无错误。
