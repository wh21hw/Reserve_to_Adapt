# Target校准支持资格：共同预热状态分叉

## Goal

完善每轮IMP未知结构建模，降低被错认已知的target未知样本污染阈值校正。上一批每轮结构标签确实用899次，但confidence校准末轮370支持大于295真实已知，至少75未知进入校准；两组独立warm最大偏差7.49pp，因此本批先用一个共同warm checkpoint。

## Success Metric

有效第10轮final OS*/UNK/HOS及第5–10轮轨迹，best仅描述。结构支持校准相对confidence校准final HOS至少+1pp、OS*/UNK各不下降超过1pp作为筛查，人工综合评估。记录校准支持数、K/V、槽训练/最终使用，结束后对保存的支持mask做target标签事后污染/覆盖统计，不用真值选阈值或K。

## Constraints

- max_iterations:1；pause_every:never；Evaluator: _(none — agent judges manually)_。
- 同T4、ResNet50、Office31 A→W、seed3事后选择、source-only3共享prior、C10/batch64/encoderBN冻结/原RTA系数预测。
- 一次共同4轮warm（confidence校准），保存完整网络/判别器/relation bank/GMM/三优化器及schedule步数/GRL/RNG/结构成员。两arm从该状态恢复，各再训练6轮（epoch5–10）；实际新训练16轮，有效每组10轮，不谎称各自重跑4轮。恢复不重新source预热或warm-end初始化。
- 总7200秒；每阶段3600秒，关键接口检查300秒。沿用项目声明预算，覆盖技能默认5分钟。实际几轮后估时、正常安静hook。
- 两arm均每轮当前完整source/target特征IMP、动态K/V、可靠结构标签。唯一因素：confidence校准资格 vs 再要求同一轮source-only初步IMP把该样本分到相同已知身份；初步IMP5步、source99%尺度、prior_strength5。共享当前特征，没有额外图片前向；最终IMP仍采用source/target各半校准，其他阈值/步数/损失不改。
- 初步几何票也不独立、不保证正确；支持不足5仍fallbacksource；如K0/超过100容量则保存证据停止，不强制K、不调参。分类头方向仍不取IMP中心。
- 共同warm状态匹配做一次有针对性的接口检查（实际加载、optimizer/RNG/scheduler），不hash、不重复checkpoint重评。不修改根目录用户训练文件。
- 本批不扩至70轮，不改其他数据集或新损失，不扫描confidence阈值；结束分析再自主选下一单因素。
- 为避免Drive再次接近容量上限，共同warm和两arm完整last暂保留runtime，只把两arm best和小结果复制Drive；关闭runtime前必须备份必要完整状态。当前Drive CLI project API Queries限额未解决，不自动删除或把mounted存在当独立上传验证。

## Search Space

仅target阈值校正支持资格。K/V/后续学习的变化是这个单因素的下游结果，不称同时独立消融K/virtual/labels。

## History

已预声明，尚未启动。训练复用已挂载T4 gpu-t4-s-kkb-ass1c2-1j7bns9e7dhke。研究整体仍含OfficeHome Pr→Rw和VisDA VGGNet口径任务，本批A→W不能代替全目标。
