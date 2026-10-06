# A→W完整70轮：K2→K5未带来明确收益

## 结论

本批未支持“当前阶段性估K规则显著优于固定K2”。自适应组在第10轮后实际扩容K2→K5，两组best指标完全相同；final HOS仅+0.1313pp，UNK+0.2273pp，OS*相同，未达预声明+1pp探索筛查。不能把小幅末轮波动包装成突破。

这与此前K8→K4/20轮相比固定K8改善8.11pp不矛盾：救回偏大容量的退化，不等于优于原人工K2。当前证据也不证明自动容量永远无效，或5个候选原型等于5个未知语义。

## 设置和完成证据

Office31 A→W，958 source/564 target，C10、Q20、seed3（历史事后选择）、Tesla T4、batch64、RTA lr5e-5、torch1.7.1+cu110。共享已有C-only source监督3轮，encoder BN running stats冻结/affine可训练，head BN不变；额外前置和BN策略不是原版论文完整流程。

两组从K2开始，保持原版RTA损失、warm-end K-means、未知候选筛选、未知槽argmax伪标签及最终C+K argmax。不用簇ID训练、不按IMP方向初始化、不加loss。两组第10轮均提取当前全量特征并按相同source-only规则估K5；固定组不应用，自适应组应用一次后保持K5。

exec31实际done，训练及自动collector成功；fixed2 worker44599与refresh worker59042均exit0、timed_out=false，两组history恰为1..70。各组普通best/last与日志、边界缓存、summary及结果ZIP已实际保存Drive。没有重新评价checkpoint、hash验证或重训。

累计history训练时间：2246.52秒+2265.86秒=75.21分钟（不等同于精确端到端墙钟含所有IO）；此前根据15轮估计整批78.04分钟，数量级吻合。

## 完整指标（%）

| 方法 | 选取 | 完成epoch | K | OS* | UNK | HOS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed2 | full best／post-10 best | 59 | 2 | 96.2016 | 82.9086 | 89.0618 |
| fixed2 | final | 70 | 2 | 96.2016 | 82.6813 | 88.9305 |
| refresh | full best／post-10 best | 61 | 5 | 96.2016 | 82.9086 | 89.0618 |
| refresh | final | 70 | 5 | 96.2016 | 82.9086 | 89.0618 |

两个best都是目标标签选epoch的oracle描述，不是label-free验证；这是事后选seed3的单seed探索，不代表三seed均值或统计显著。full best与post-10 best在本批重合，不省略末轮，也不选择短程峰值替代70轮结果。

final差值：OS*0pp、UNK+0.2273pp、HOS+0.1313pp。best差值三项均0pp。手工判断为近似零效果，候选不晋升默认方法；保留实现和全部结果作负面/边界证据，不回滚用户文件，不删除模型。

## 前10轮偏差

更新前最大三指标绝对差异1.6257pp；第10轮refresh−fixed为OS*−0.3333pp、UNK+0.0586pp、HOS−0.1298pp。final差值相对这个节点的变化为OS*+0.3333pp、UNK+0.1687pp、HOS+0.2611pp，仍很小。

这些是描述性轨迹差异，不是无偏因果估计。两组不是严格相同数值轨迹；单次轻微差异不足以支持容量扩展的稳定收益。

## 变化实际发生了，但收益未发生

状态迁移保留已知10行及原unknown两行10/11和对应SGD动量，新增3个随机norm匹配行；无移除、无warmup/scheduler重置、无原型方向初始化。因此不能把此次零效果解释成“忘了扩K”，扩容确实执行了。

但仅容量更新仍不把IMP簇身份交给RTA未知学习，也不改变已知/未知边界或未知候选选择。新增槽是否得到有效监督、是否承担不同未知结构，本次history无法回答；不能未经诊断说它们从未使用，也不能根据best相同说两组逐样本预测相同。

## 对标与下一问题

论文A→W报告HOS93.0%，本批final分别低4.0695pp/3.9382pp。对标只是背景：这里有额外source预热/冻结BN、事后seed选择及发布代码与论文公式差异，不能将差值只归因于GPU或IMP。

优先把“基线表征/训练流程是否健康”与“容量模块是否有用”分开：固定K2本身已弱于既有原版参考，扩K没有明显救回。下一项应基于新observation先定位，而不是扫K/alpha、继续扩槽或直接加多项loss。可检验方向是新增槽是否被使用，以及当前额外前置/encoder BN策略对完整预算固定K2的影响；它们是不同问题，不能并称K-only改进。

本批已用完140轮，无额外确认训练或新实验启动。按用户最新指令，完整observation现在已得到，恢复总体研究评价/诊断；当前等待hook在报告后暂停，避免重复通知。VisDA后续按用户新决定采用VGGNet，旧ResNet结果保留，具体变体还需核实，不在本批中混入。

## 保存与绘图

Drive：`OSDA/runs/a2w-capacity-refresh-k2-70e-v1`；本地`pipeline-results/a2w-capacity-refresh-k2-70e-v1-{summary.json,results.zip}`。完整日志与无标签边界NPZ在结果ZIP，普通模型留Drive，不重复下载大文件。

训练/汇总代码版本afb3331a及已记录启动4d85d43b，绘图改造bbf66389。绘图仅读取已有日志：本地bundled Python没有matplotlib，改用既有Colab venv；上传技能样式时修正WSL路径，随后显式Agg避免继承notebook inline backend。上述工程问题不影响已完成训练/指标，不重跑任何模型。曲线由实际K和70轮metadata生成，不硬编码旧K8→K4。
