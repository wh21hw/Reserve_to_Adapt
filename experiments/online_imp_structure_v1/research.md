# 每轮 IMP 与未知结构学习 v1

## Goal

2026-10-07 用户明确：建模未知结构，ResNet不改；允许调整RTA模块，结合未知方向排斥、target分布阈值，每轮重新IMP并调整K。该版本不是K-only，不声称完整DP后验或真实类别发现。

## Success Metric

同预算第10轮OS*/UNK/HOS及全部轨迹；best仅描述。结构监督组final HOS比自举组至少+1pp，OS*/UNK各不下降超过1pp作为探索筛查，助手综合判断。报告每轮K/V、簇迁移、实际伪标签支持与最终槽使用，不只看最高best。target真值只评价，不参与结构、阈值、停早或配置选择。

## Constraints

- max_iterations: 1；pause_every: never；Evaluator: _(none — agent judges manually)_。
- 两组各10轮，共20轮；每组3600秒、整批7200秒训练上限，明确覆盖技能默认5分钟；新接口检查最多300秒。
- Office31 A→W，ResNet50，C10，seed3（历史事后选择），batch64，source C-only3复用checkpoint，encoder BN统计冻结、head BN照常；不是论文完整原流程。
- 两组共同：每个epoch开始提取完整eval/center-crop当前source/target；重新source均值作为已知锚点、可移动，IMP5步/prior_strength5；阈值采用source每类99%距离与target已知支持距离99%分位各半混合（支持不足5个则只source）。target支持需当前已知条件概率>=0.8、全头预测已知、几何最近已知与分类身份一致。它是待验证的启发式，不保证纯净，不用总体target方差。
- 用source残差与target支持残差估计正数方差；每类校正半径按source类别样本数加权成一个全局建簇阈值，并非实现了类别各异的DP-means出生代价。所有新建原型仍是候选。虚拟方向只保留落在所有source锚点覆盖半径之外的候选；V可能小于K或为0。
- 每轮硬分配中有样本的未知候选决定K。以同一target样本跨轮成员重叠匹配分类行，保留known权重、可匹配unknown权重和SGD动量；新增行随机norm匹配，不复制原型；不重置scheduler。K相同也必须传递身份。
- 共用原版warm-end K-means初始化（第4轮后），其后首次刷新按当前预测对齐，不能把warm-end前簇编号直接当新头身份。
- 唯一arm差异：self_label使用原未知槽argmax；structure_label在原RTA筛出的候选中，对IMP可靠候选簇成员替换标签，其余保留原自举。交叉熵系数、筛选、预测规则不变，不加loss。
- K0保存证据并停止，不强制K1；不扩网格，不自动延至70轮。失败允许不改设置的工程修复，不覆盖旧结果。source/target支持的0.8、5、0.5为预声明设计常数，不使用target成绩扫参。
- 同初始化同seed不保证GPU轨迹完全相同，报告前4轮偏差。两组共同骨架不是原版RTA，不能以两组对照判断阈值/virtual/K各自贡献。
- 数据复用Drive压缩包，结果本地+Drive；不hash、不反复checkpoint重评。先观察几轮再估ETA、设置安静hook。

## Search Space

仅比较当前IMP簇身份能否比RTA自举标签提供有效未知监督。后续阈值/virtual/K消融单独声明；本批不同时比较全部模块。

## History

Drive已挂载，环境和缓存恢复完成，针对性身份迁移/当前缓存特征及真实入口编译检查通过。第一次启动在模型加载时因本地控制副本的`weights_only=False`参数与torch1.7不兼容失败，0训练轮；只删除该兼容参数，保留startup-failed目录，修复后启动同设置，不作失败算法重跑。新T4 gpu-t4-s-kkb-ass1c2-1j7bns9e7dhke。

运行入口background exec5；self_label worker PID4595，独立shell13；控制台`/content/online-imp-pair-console-r1.log`。每arm目录`/content/imp-runs/online-imp-structure-v1/{self_label,structure_label}/office31-a2w_seed3`。第1轮前首次当前结构估计K17/V3，刷新8.72秒，target可信支持0，因此本次阈值实际回退source校准，不伪称target校正已生效；后续每轮记录支持数。初始K2是建头占位，实际第1轮已经按IMP调整17，不能沿用旧K2→5口径。

前三轮完成elapsed41.3/78.1/113.8秒，近期35.7–36.8秒/epoch；预计两组总训练约12–15分钟（不是硬截止）。K轨迹17→38→35，第4轮开始K33/V18、target支持268，阈值的target校正此时有数据支持，但支持正确性未证明。原有rta-baseline hook已更新为本批5分钟等待，正常安静，完成收集后报告。不把K抖动当作语义类别数确实变化；跨轮成员传递与最终槽支持需要结果分析。

完成：两arm各10轮exit0，exec5自动collector成功，总训练12.56分钟。final自举75.2402% vs结构76.4282%，Δ+1.1880pp，但前4偏差7.4872pp，不能确证标签收益。监督确实接通，active阶段899/1341次替换；校准支持至少20–23%污染下界。完整向量/边界见final_report.md。下一批共同warm状态分叉，只改target校准支持资格，单独目录预声明，不延长本批。
