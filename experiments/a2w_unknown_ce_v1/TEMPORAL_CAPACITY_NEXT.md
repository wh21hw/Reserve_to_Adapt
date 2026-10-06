# 下一单因素：结构推断的时机，而非重复门控规则

当前完整研究目标未完成。本轮是已完成K8身份对照之后的代码准备，不是新实验成绩；没有创建Colab实例或启动GPU。

## 必须保留的已有负面证据

1. A→W关闭未知CE，final UNK从86.06%降到3.33%，不是修复。
2. A→W K8固定早期簇身份，final HOS比原argmax低0.6990pp；近97%候选用到标签，不是模块没有接通。
3. 纯已知误建簇对应槽全程0伪标签，说明RTA筛选可以挡住部分IMP伪未知，不能为了满槽强行补监督。
4. CONDITIONAL_DPMEANS.md已有门控后聚类探针：A→W先前不同prior/seed下K2候选混合且污染；source留类代理也失败。OFFICEHOME_RTA_10E.md的初始IMP交集训练HOS仅+0.0082pp。不能把这些旧方案重新命名为下一新方法。

## 当前可检验因素

比较source预热结束的状态，与固定预声明RTA节点的状态，在完全相同的结构估计规则下，K、成员分配与已知/未知证据是否更一致。两个状态的表示及分类头均可能变化，不能叫仅encoder变化；这首先是时机/状态诊断，不是容量对训练成绩的因果实验。

先使用本批原argmax组的固定final10（C10+K8），而不是挑best epoch7或某一较好seed。它能直接复用已持久保存模型，不需要重训。source与target都用同一固定网络一次提取，避免跨阶段中心/特征混搭。GMM分量数不是未知语义数，关系权重不是严格Bayes语义后验。

## 已准备的缓存入口

scripts/cache_a2w_current_relation_colab.py：原ResNet50和完整BN→LeakyReLU→fc分类头；eval/no_grad；source标签允许用于source模板；target标签在读列表边界丢弃。保存全量958/564的256维特征、18维logits、source标签、target路径、C维条件分布关系KL及source关系模板。没有GMM拟合、K选择、未知训练标签或新的损失。

它重建的是当前冻结模型下的source概率均值，不是历史训练增广/在线关系库的精确回放。与已保存source3状态比较时必须披露这个边界。每个模型只创建一次缓存，输出存在则拒绝重复前向。缓存实际保存到Drive后才可销毁runtime；当前没有缓存产出，不写成已执行。

本地AST语法检查通过；真实checkpoint接口检查和图像提取尚未执行。下次恢复Colab环境时补传relation_gate.py，复用Drive的argmax-last.pt和Office31压缩包；优先T4。只做一次新接口功能检查，不重复稳定代码smoke或hash。

## 后续如何回到训练验证

只有时机诊断给出值得检验的容量规则后，才声明同checkpoint、共同早期训练预算的固定K/估计K训练对照。重建分类头/optimizer时保留已知身份和可对应未知状态，沿用已有交替K接口；不要顺便加入新标签、门控loss、IMP权重初始化或概率聚合，并称K-only。

当前10轮只覆盖发布代码前段top16筛选；未来完整结论必须跨过筛选切换并按完整训练预算验证。OfficeHome与VisDA外部验证、VisDA backbone口径、论文对齐以及多seed都仍未完成，不以A→W一次小实验代替毕业设计目标。

无人值守循环和统一evaluator/keep policy尚未确认，已按autoresearch技能发出配置问题；不默认启动无限搜索。当前GPU保持关闭，不因自动续行创建闲置实例。

## 离线比较器已完成，尚未跑真实缓存

scripts/compare_a2w_temporal_capacity_colab.py复用已保存的source3 raw K13/身份匹配K8两个无标签artifact，只对final10缓存执行一次原source-calibrated birth-cost规则，再按同一head-likelihood身份匹配。source99%半径/建簇成本根据当前source重新标定，没有target阈值扫描；因此是同规则跨状态比较，不是数值lambda固定的比较。

temporal_partition_report.py输出K之外的划分变化：只在两边都非noise的样本上计算ARI（不受簇编号排列影响），同时报告共同覆盖、noise变化、未知候选身份变化及共同known上的身份一致率。不能将任意簇ID换位当结构失效，或把减少K当改善；不足两个共同样本时ARI返回未定义，不强填一个好分数。若实际有成员的簇数不足C，保留原推断但身份匹配报告不可用，不强造K1。

一次新统计接口功能检查已在本地CPU完成：编号换位ARI1、交叉重新分组ARI−0.5、noise与共同样本不足分支正确。初次断言因浮点严格等号失败，改用容差断言通过，算法没有因此改动。比较入口AST语法通过；没有重新测试稳定聚类核心、没有GPU或真实checkpoint前向。

当前真实final10特征缓存尚未生成，所以没有新的K、漂移程度或模型效果结论。下一实际动作仍是复用Drive final10模型生成一次冻结缓存，然后运行此纯CPU比较器；不再次训练source或RTA，不把准备完成写成实验完成。需上传source_precision_capacity.py、robust_capacity.py、prototype_identity_reconciliation.py、temporal_partition_report.py及relation_gate.py到运行环境，缓存持久保存后及时关闭实例。

## 实际恢复进展

2026-10-06，通过CLI检查确认无在线runtime；Drive直读API仍返回共享client项目202264815644的Queries quota错误，不是用户Drive磁盘容量不足。仅需一次前向，本批改用标准CPU实例m-s-kkb-usw4a1-1tx31818s2ldc，实测0.08 CCU/hr，shell9；挂载exec1已请求用户授权。CPU推理会比GPU慢，但没有新训练或模型设置改变。

原ResNet源码get_mean/get_std强制.cuda()，缓存入口已在推理设备上显式分配相同float32 ImageNet常数；仅设备分配兼容修复，不改变架构、权重或归一化数值。恢复入口scripts/restore_a2w_temporal_snapshot_colab.py只读取既有Drive数据/权重/argmax-last.pt，未重训source。首个setup文件上传fetch failed，已成功重传；之前python因此找不到文件，未发生任何训练/提取，重新运行依赖安装不改变实验设置。

代码/辅助包已上传。当前等待Drive授权与依赖安装完成，尚未提取final10缓存、估计当前K或产生漂移数值；不要从准备过程声称假设已验证。完成后保存小缓存、离线比较结果到Drive并关闭CPU实例。
