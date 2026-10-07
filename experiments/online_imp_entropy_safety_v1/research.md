# 熵保护的资格：raw新簇 vs screened新簇

## Goal

分解发现收益主要与known entropy屏蔽相关（final80.65% vs原参考70.26），但raw误伤known−4.95pp。保留domain alignment，只对几何可靠的新候选屏蔽entropy，检验能否保留收益并保护已知。可靠筛查不是正确性保证，实际错误权重仍须量化。

## Success Metric

相对原RTA known权重参考final HOS至少+1pp、OS*/UNK各不下降超过1pp，人工综合；raw/screened实际新同批对照，报告original参考复用/跨进程噪声。已知权重损失、未知去除权重、结构曝光/ARI/NMI/K/V，只结束注释，不进训练。

## Constraints

- max_iterations:1；pause_every:never；Evaluator: _(none — agent judges manually)_。
- 两arm raw_entropy/screened_entropy现存共同warm4各续训6，实际12新轮、有效final10。原参考rta_known98.0108/54.7566/70.2602及entropy-only93.0658/71.1560/80.6493仅背景；本批重新raw作同批控制，不混成三seed均值。
- T4/ResNet50/256teacher/seed3/source-only3/frozenencoderBN/C10/batch64，每轮IMP动态K/confidence校准/screened标签/virtual，source/unknown选择/损失系数和最终预测不改。只entropy mask资格不同；alignment两组全原RTA。
- raw屏蔽全部候选；screened只屏蔽centroid在全部source半径之外的候选，同原已有reliable规则，不调距离阈值。原GMM weight若0仍0，不新label/loss或知类监督。
- 每arm3600/总7200秒、新接口300秒，覆盖技能默认5分钟。预计8分钟、实测重估，安静hook。不70轮/换seed/BN/预测或扫强度。
- K0/100cap失败保留不强制K；小结果Drive、大模型runtime，关闭前备份，不擅自删云盘。复用数据和warm，不再预热。

## History

预声明，尚未启动；整个未知结构和三任务目标仍未达成。

已启动：代码1774a263，T4复用、exec23/launcher49453/shell13，控制台`/content/online-entropy-safety-console.log`。资格接口一次通过，alignment保持原样。两个续训6，新hook已换23，每5分钟安静检查；确认母进程running。实际几轮后再估时间，不重复旧exec20结果。

完成：两arm各epoch5–10完整，collector成功。raw final92.8332/72.9920/81.7256；screened96.7782/56.7397/71.5378（OS*/UNK/HOS，%）。screened相对原权重参考HOS+1.2775但OS*−1.2326pp，guard_violation，不推广。raw删除170已知/431未知entropy权重；screened仅删除2已知/32未知，仍保留763未知错误known权重。转向分类头BN坐标一致性的单因素诊断，不继续阈值网格。
