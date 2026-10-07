# 可靠IMP训练入口的同设置确认

## Goal

第一次同批union final HOS+6.8185pp、known−0.2326pp通过，先确认而非立即推广。过往source_only控制UNK约49–55%跨运行漂移；完整方法收益尚未验证。

## Success Metric

与本批rta_only配对final HOS至少+1pp、OS*/UNK各下降不超过1pp，综合两次完整best/post4best/final，不仅挑更好的重复。若本次不通过，不能丢弃本次结果、挑第一次宣称稳定；至少标为未稳定，不连续无上限重复直到通过。ARI/NMI与新增known污染/独立unknown覆盖同报。

## Constraints

- max_iterations:1; pause_every:never; Evaluator: _(none — agent judges manually)_。
- 独立输出名online-imp-training-support-confirm-v1，复用同commonwarm4，两arm rta_only/reliable_union各续训6(epoch5–10)，实际12新轮，3600秒/arm、总7200秒，覆盖技能默认5min；预计8分钟，实测重估hook。
- 所有训练方法与第一次相同：source_only每轮IMP、无merge、原BN、C10/ResNet50/256teacher/source3/frozenencoderBN/seed3/batch64/confidence阈值/screened标签与virtual；唯一组间因素原r与原r+可靠成员。原r全保留、无重复，已有teacher簇ID训练；原entropy/alignment/源CE/unknownCE系数/argmax/头SGD匹配不改。
- 不是新seed统计或三seed均值；训练数值漂移照实报告，不新增determinism设置作为隐藏因素。
- 不是工程重复smoke/独立checkpoint重评，是有预算的科学确认；稳定训练代码不反复合成测试，不hash、不额外warm/重新下载；不70轮/新loss/阈值网格/重启销毁。
- 小结果Drive、模型runtime关闭前备份，复用mount/cache；target语义仅结果评价，不用于K/阈值/训练，方法选择仍属于探索性target-oracle实验，三任务未完成。

## Search Space

仅重复已声明配对一次；不调参数，不挑最好一次。

## History

预声明。第一次final control97.3441/49.8748/65.9564，union97.1115/58.1918/72.7749；fullbest共同warm4，post4best5/75.3200和5/75.3096。新增194次，其中unknown177/known17；独立unknown结构覆盖136→152，但ARI/NMI未改善，不能宣称恢复真实类别。

已启动：8b92b200，exec43/shell13，`/content/online-training-support-confirm-console.log`；仅控制面新增安全独立run-name，其它训练文件复用稳定版本，不重跑合成接口检查。5分钟hook已恢复为本批，不重复旧exec40通知。

launcher82176。rta_only epoch6 elapsed79.3434s、epoch7 elapsed116.1382s，近期36.79秒/轮，预计整批约8分钟、当前剩约6分钟；union阶段重估。已完整三续训轮，无报错，loss有限。

完成：exec43/collector0，两组各epoch5–10完整，229.20/235.19秒。control final97.6774/54.3523/69.8415，union96.7679/60.6717/74.5819（OS*/UNK/HOS%）。Δ−0.9096/+6.3194/+4.7404pp，第二次也通过；边界余量仅0.09pp，不夸称完全无损。两次都留存，不挑第一次。fullbest两组仍共同warm4，post4best5 HOS75.3200/75.3096。

第二次新增197实际曝光（181unknown/16known），独立76unknown/5known；独立unknown结构监督138→152。ARI .47069→.49456/NMI .69469→.71404，本次增加但首次下降，不声称已稳定提升语义结构。下一阶段方法不变，预声明完整70轮配对验证持续性，当前不做第三次短程确认或新参数网格。
