# 有代价判据的候选簇合并

## Goal

上一承接组更稳定但K34不减、known污染增加，不能把连续性当语义恢复。只检验IMP输出上的objective-decreasing候选合并能否使不必要的分裂退出，并改善实际RTA训练支持。

## Theory and Current Approach

[Kulis & Jordan ICML2012](https://icml.cc/2012/papers/291.pdf) Eq1为hard类内SSE+lambda K；论文原算法局部下降，不证明恢复真实类别。这里沿用每轮现有source/无标签target校准的距离阈值lambda，不另扫惩罚。

两组先原source-anchored soft IMP。改组仅随后合并candidate-candidate：以当前硬成员均值提出pair，fixed-partition代价增加与少一簇节约lambda比较，再对所有target按当前已知/新候选中心重新分配，移除空候选，只有实际SSE+lambda K下降才接受。合并阶段known中心/身份不动，source-anchor先验项是常数，因此此条件只保证本次后处理条件目标下降，不保证整个soft IMP/神经训练单调、不宣称完整DP后验/全局最优。已知吸收/释放由nearest assignment决定，不直接合并或删除known身份。

## Success Metric

同批no_merge对照final HOS至少+1pp，OS*/UNK各下降不超过1pp；同时与已完成source_only参考比较，不能仅打败失败承接组就称优于原健康baseline。综合best/post4best/final、K/V、目标下降、实际监督覆盖/known污染，target真值仅评价。

## Constraints

- max_iterations:1; pause_every:never; Evaluator: _(none — agent judges manually)_。
- no_merge/objective_merge都从同commonwarm4各续训6(epoch5–10)，实际12轮；3600s/arm、总7200s，覆盖技能5min默认，预计8分钟，实测重估hook。
- 唯一因素candidate merge on/off；两组current_members初始化、IMP5/source prior5、confidence target阈值/screened标签/virtual、原BN/known entropy/alignment、source/unknownselector/CE系数/argmax和头SGD匹配同。
- T4/ResNet50/256/source3/frozenencoderBN/C10/seed3/batch64；不加损失、不强制语义K，不同时改init/阈值；K0/cap失败保留，不强制K1。
- 一次新接口检查，不hash/checkpoint重评/新seed/70轮/参数网格/重启销毁；reusecache/mount，小结果Drive，模型runtime关闭前备份。

## Search Space

仅是否按条件SSE+lambda候选数接受合并。每次最多删一或更多空簇，有限轮数由当前K决定，不另设调参步数。

## History

预声明，尚未启动。承接无merge已完成95.8790/51.2789/66.8204，source_only97.6774/54.9944/70.3694仅背景，不伪称同批控制或论文复现；整体及三任务目标未完成。

已启动：fafc50f4，exec36/shell13，控制台`/content/online-candidate-merge-console.log`。新合并接口的目标单调/known身份/不强制合并、完整生成代码一次检查通过，现有runtime/mount/cache复用，无重复warm。hook5分钟已替换旧exec32。

launcher65808。实测no_merge epoch6 elapsed78.0972s、epoch7 elapsed118.7820s，近期40.68秒/轮，按此速本批约8–9分钟、当前剩约6分钟；objective_merge改K/合并后重估。原控制组已完成三续训轮，K34，无错误。

objective_merge epoch5 elapsed41.3727s、epoch6 elapsed78.6634s，近期37.29秒/轮，K32，预计剩约两三分钟（观测时），无错误。接受的合并objective_history留存，不能将K减少当分类改善。

完成：exec36/collector0，两组各续训6，229.74/224.32秒。no_merge final96.5456/49.6126/65.5438；objective_merge95.2511/52.1034/67.3601（OS*/UNK/HOS%），Δ−1.2946/+2.4907/+1.8163pp，guard失败，不推广。共接受3次合并，K34→32→31，固定特征目标总下降约0.34503；跨epoch目标不单调也未要求单调。仍低于source_only背景70.3694。未知独立训练覆盖128→140，但printer类别曝光仍0。下一单因素应回到可靠IMP成员的训练入口，不继续扫lambda/BN/保留簇。
