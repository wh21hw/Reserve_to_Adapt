# 熵取舍候选：完整70轮配对

## Goal

三点10轮发现gamma0.5和0均改善HOS但known/unknown取舍不同，完整预算验证两个候选，保持未知结构训练骨架不变。不以短程最高点当毕业设计终点。

## Success Metric

完整final HOS及OS*/UNK组合为主，half/zero同批直接比较，HOS+1pp为有意义探索收益参考，不按known下降1pp单独否决。报告fullbest/post4best/final和轨迹、ARI/NMI、K/V、权重曝光、污染及覆盖；若被另一点两指标均支配则不选择。综合取舍、known塌缩和结构风险手工判断，不只挑最高best。

## Constraints

- max_iterations:1; pause_every:never; Evaluator: _(none — agent judges manually)_。
- T4/Office31 A→W/seed3/ResNet50/C10/source3/frozenencoderBN/batch64，复用完整commonwarm4，各续训5–70共66新轮，总132，有效70轮/arm。每组5400秒、总10800秒，覆盖默认5min，预计80分钟，按实测与换组重估；超时保留证据，不自动延长或重复直到通过。
- 两组统一reliable_union训练入口/source_only每轮IMP/confidence阈值/screened簇标签/virtual/无merge/原headBN/原alignment/unknownCE/最终argmax/匹配头SGD。唯一组间因素raw IMP unknown候选的known entropy权重gamma0.5 vs0；known身份权重保留1，源CE不变。第10轮后原筛选分支按原代码执行。
- 同warm4重新分叉到70，不从不同epoch10模型继续以免把短程随机漂移当共同起点。新接口已检查，不重跑稳定smoke/hash/checkpoint重评。不修改算法，仅声明两组及长预算。
- 已有gamma1/union完整70轮是独立背景参考，非本批三arm严格配对，不重跑稳定参考；不能拿它的数值差当无噪声因果估计。所有旧结果留存，不用短程best替换完整final。
- 事后seed3/方法候选与epoch选择为target-oracle探索，真值仅结果评价与曝光注释，不进入K/阈值/训练，非多seed统计。额外source预热/BN非原论文完整流程，OfficeHome/VisDA仍待验证。
- 独立目录、同runtime/cache/mount，无额外授权或下载。小结果Drive+电脑，模型runtime关闭前备份；API项目quota无法独立证明挂载同步，不删除数据/重启/销毁。

## Search Space

固定两个已声明候选完整验证，无更密gamma网格，不叠加阈值/BN/对齐/预测因素。

## History

预声明，未启动。短程gamma1/half/zero final OS*/UNK/HOS=97.1382/59.0648/73.4614，94.8535/64.8620/77.0418，92.9232/72.1939/81.2573%；zero有更高HOS/ARI/NMI但known误结构标签更多。完整gamma1独立背景final99.6667/56.9958/72.5200，不能混称同批。

已启动：代码0e051825，exec47/launcher152769/shell13，console /content/online-entropy-dose-70e-console.log，现有T4/mount/cache保留。10分钟hook已替换短程hook，算法接口不重测，观察实际近期轮间elapsed再更新ETA。
