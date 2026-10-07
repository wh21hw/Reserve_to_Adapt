# Known目标分解：未知收益/已知损伤分别来自哪里

## Goal

上一both-veto UNK+15.15pp/HOS+9.41pp但OS*−5.41pp失败；去掉382真实未知和161真实已知加权曝光。分离known entropy与target alignment，不把整套屏蔽晋升。

## Success Metric

相对同设置原known权重参考final HOS至少+1pp、OS*/UNK各不下降超过1pp；人工综合，附full/post4best、轨迹、真实权重取消与曝光、ARI/NMI/K/V。目标标签仅结束注释。

## Constraints

- max_iterations:1；pause_every:never；Evaluator: _(none — agent judges manually)_。
- 两新arm entropy_veto/alignment_veto，各从同现存warm4续训6（有效epoch10），实际12新轮。参考online-imp-known-veto-v1的rta_known/imp_veto同warm同采样规则，不重复跑参考；跨进程GPU数值噪声存在，参考不是统计显著性。
- T4/ResNet50/256teacher/source-only3/frozenencoderBN/C10/seed3/batch64、每轮IMP动态K/confidence校准/screened标签与virtual/原系数预测。
- 唯一因素为同一IMP已知mask放到哪个known目标：仅entropy vs仅target adversarial。原unknown r/unknownCE/source监督不改。两个被修改位置一次路由检查，300秒，不重评checkpoint。
- 每arm3600秒/总7200秒，覆盖技能5分钟；预计8分钟、实际轮间elapsed重估。K0/cap100失败停止不强制K，不新loss/BN/骨干/预测或扫阈值，不70轮/换seed/其他数据集。
- 小结果Drive，模型runtime，关闭前备份，不擅自删旧模型；数据/共同warm复用。

## Search Space

仅known mask的作用模块；参考旧both-veto保留证据，不能从新两arm对比推断独立的完整交互效应。

## History

预声明，尚未启动。整体未知结构目标和三数据集验证仍未完成。

已启动：代码9c9c6103，T4复用，exec20/launcher43715/shell13，控制台`/content/online-known-components-console.log`。模块权重路由四种情况一次检查通过；训练母进程确认running。两新续训6，旧参考不重跑，正常hook5分钟安静检查；不能重复旧exec17结果。
