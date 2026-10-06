# Research: 原固定K2与阶段性自适应K完整预算对照

## Goal

持续优化OSDA未知结构建模与RTA容量，当前检验source先验约束的结构估计是否优于固定K2，而不是只优于偏大K8。总体目标仍含Office31 A→W、OfficeHome Pr→Rw、VisDA Syn→Real；当前单任务结果不代表整体完成。

用户2026-10-06授权“启动。不用找我确认。预算足够”。后续自主按证据选单因素实验并预声明预算，不逐批再问。保持已有source C-only前置/固定与估计K共享初始化原则，超出研究范围或必要Google操作才请求用户。

## Success Metric

完整第70轮final HOS以及OS*/UNK；同时报告full best、post-10 best与epoch及所有轨迹。探索筛查：final HOS至少比fixed2高1pp，且OS*/UNK各不损失超过1pp；助手综合判断，不只取最高best。目标标签只用于评价，不拟合K/阈值或自动调参。单seed事后seed选择及oracle best必须披露，不称显著性或三seed均值。

## Constraints

- max_iterations: 1（本批一个两arm对照；结束后形成结论再自主声明下一批，不未经分析扩网格）。
- pause_every: never；Evaluator: _(none — agent judges manually)_；noise_runs: 1；min_delta: 1pp。
- fixed2/refresh各70完整epoch，合计140新RTA轮；复用C-only source3 checkpoint，不重训source，不从旧不完整训练状态续跑。
- 每arm最多7200秒，总训练最多14400秒。达到上限停止、保留证据，不因观察超时重启；后续决策按用户最新自主授权，不再因本批耗尽重复请求预算批准。
- seed3（历史事后选择），C10/Q20、batch64、ResNet50、RTA lr5e-5、T4旧venv。encoder BN stats冻结/affine可训，head BN不变。这不等于论文完整原始流程。
- 唯一训练因素：第10轮完成后是否应用当前无标签推断K。两组均提特征并推断；不加loss、簇标签训练、原型头初始化、预测门控，不改原RTA伪标签/最终argmax或scheduler/warmup。
- 估计器冻结为source-calibrated-birth-cost-v1，source校准及先验，proposal块64、当前known-head似然匹配身份。K0/匹配不成立保存证据并停止，不强制K1或按target指标调阈值。
- 关键监督仅进程、有限loss、OOM/报错、边界K、逐轮指标与保存。已做一次预算参数build-only，不再做hash、重复初始化或checkpoint重评。
- 每arm完成先保存Drive再开始下一个；目录拒绝覆盖。失败先定位，纯工程修复不改变设置；不自动重试模型训练。
- 保留现有CPU与T4，不擅自销毁/重启；缓存复用不重复下载。根目录用户训练改动不覆盖、不提交。

## Current Approach / Baseline

此前20轮K8→K4相对固定K8 final HOS+8.1101pp，但best仅+0.4004pp；即时删头不改变二元预测。原版available57 K2头与估计4候选簇高二元一致率说明簇数不直接等于必需输出维数。现存原版logs70/weight57差异保留，不冒称final70权重。

本批实际运行固定K2完整70作为控制；同C-only source3、同冻结BN/环境/预算，不能用旧L4控制或发布原版历史结果冒充同设置控制。两条前10轨迹可能因数值/自举分歧不同，collector同时报告前10最大偏差、边界差异及描述性前后差值，不作无偏因果解释。

## Search Space

只比较是否应用第10轮估计K；初始K2，两组其他设置一致。可增加/减少/保持K，无目标真值输入。保持已知和对应未知头/SGD；新增行随机norm匹配，不用IMP方向。若K保持2，视为合法零效果，不挑别的更新时机强求差异。

## Execution / 保存

训练代码afb3331a时对应入口已准备，实际算法来自既有capacity_refresh_boundary.py等稳定模块。T4端点gpu-t4-s-kkb-ass1c1-1lpayjb3tm80f，CPU端点m-s-kkb-use1b1-3hr6nb4xjabhs均在；Drive实际mounted、source与数据缓存存在，训练前确认无其他训练进程和新输出。

background exec31启动venv子进程，父控制台`/content/a2w-capacity-refresh-k2-70e-pair-console.log`，固定组worker PID44599。正常监督用独立shell12（kernel被exec31占用，勿排队新exec检查），各arm console/history与capacity-after-010/estimate.json。结束后exec31自动运行日志collector，不重评checkpoint。

运行目录`/content/imp-runs/a2w-capacity-refresh-k2-70e-v1/{fixed2,refresh}`；Drive `OSDA/runs/a2w-capacity-refresh-k2-70e-v1`。结束下载summary/结果ZIP，绘图并报告；ordinary best/last保留Drive，不重复传大模型。

## History

| # | Change | Metric | Result | Timestamp |
| --- | --- | --- | --- | --- |
| 0 | 同前置固定K2，70轮控制 | final HOS88.9305%，best89.0618% | 完整70轮、exit0、Drive已保存 | 2026-10-06 |
| 1 | 初始K2，第10轮后K2→估计K5，70轮 | final89.0618%，Δ+0.1313pp；best89.0618%，Δ0pp | 近似零效果，未达+1pp筛查，不晋升默认 | 2026-10-06 |

状态：本批实际完成，exec31训练及自动collector成功，两组均70完整轮。完整向量、前10偏差、迁移和保存证据见final_report.md。总体未知建模研究仍未完成，没有新训练启动。
