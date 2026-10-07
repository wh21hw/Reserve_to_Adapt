# 分类头BN坐标一致性单因素

## Goal

多轮候选筛查未同时守住known/unknown。检验分类头对source/fulltarget/unknown小batch分别归一化，是否妨碍IMP结构到分类器训练的一致性。此为待验证假设，不宣称根因已确定；不改变ResNet结构。

## Success Metric

同批batch_bn对照：final HOS至少+1pp，OS*/UNK各下降不超过1pp，综合post4best/final/结构支持判断。Target语义只评价，不用于训练/阈值/K。

## Constraints

- max_iterations:1; pause_every:never; Evaluator: _(none — agent judges manually)_。
- T4/ResNet50/seed3/C10/batch64/source3/frozenencoderBN，复用同完整warm4；每arm续训6（epochs5–10），实际12轮。3600秒/arm、7200秒总上限，覆盖技能默认5分钟，预计8分钟，实测重估hook。
- batch_bn维持原样；fixed_bn每个训练batch前仅令CLS BN使用commonwarm4 running统计，affine仍可训练。encoderBN不改，未知head状态/SGD匹配沿用，warm不重训。
- 每轮256teacher IMP/confidence target阈值/screened簇标签/virtual；两组entropy和alignment全原known权重，不叠加失败veto。source/unknownselector/CE系数/argmax都不改。
- 一次关键功能检查；不hash、不checkpoint重评、不新seed/70轮/参数网格、不重启销毁runtime。
- 小结果Drive、本地模型runtime，关闭前备份。缓存和mount复用，无数据重下载。

## Search Space

仅分类头BN统计量模式batch/fixed，其他因素固定。若失败，保留结果，不自动调BN动量或混合权重。

## History

预声明。当前方案尚未达到三数据集目标，上一径向筛查因known guard失败未推广。
