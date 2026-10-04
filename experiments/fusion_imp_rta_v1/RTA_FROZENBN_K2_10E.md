# 保持 RTA BN 统计下的自适应 K 对照

2026-10-04，在新增arm结果产生前记录。现有三个arm分别是更新BN/K4、更新BN/估计K2、保持BN/K4，各source3轮+RTA10轮。新增一个保持BN/估计K2 arm，补上容量与BN因素的交叉对照；不搜索alpha/lambda/seed，不重跑已有控制。

使用既有prior/capacity.json的K2（exec147，无target标签），不重估、不按HOS选择K。共享同一冻结encoderBN的C25 source CE3轮checkpoint、seed1/Q29/ResNet50、原RTA损失/伪标签/argmax/初始化/10轮预算。保持encoder BN统计，权重与affine及head正常训练。与已完成保持BN/K4的唯一研究因素为K；与更新BN/K2的唯一因素为BN模式。

源码入口与BN hook已验证，不重复smoke。新目录/content/imp-runs/officehome-rta-frozenbn-k2-10e-v1，拒绝覆盖。launcher scripts/run_rta_frozenbn_ablation_colab.py --arm estimated；原无参数用法仍为fixed4，绝不在本次调用重跑fixed4。沿用已部署encoder_bn_policy.py和train_legacy_task_entry.py。

完成后报告四组best/final OS*/UNK/HOS及同epoch差值；分别计算BN内的容量对照、同K下BN对照，不把双因素差异伪称单因素收益。best仍target标签选epoch，单seed短程探索，不称显著性或真实未知语义数恢复。实验是看到前三组结果后的模块研究，不称无偏预声明外部验证。若结果不支持容量收益，也保留，不调整K或延长预算寻找好峰值。

正式ResNet不变、DINO只诊断。无新loss、IMP未知头初始化、结构KL或动态改变K；当前推断一次K后固定。三数据集完整外部验证和原论文预算对齐仍未完成。

实现e0fdb9eb，collector ecdf5b21。已单独启动后台exec156，现有L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o；启动日志确认53个encoder BN模块保持统计、affine trainable/head unchanged，以及C25/K2/Q29和同一source-final.pt。当前running，尚无最终结论；没有重新训练source、估计K或重复前三组。

collector scripts/collect_bn_capacity_cross_colab.py已部署，仅在本arm完整10轮后执行。它读取四组已有history，列出同BN下K差值和同K下BN差值、final交互项，并在新arm结束时读取一次统计buffers确认BN因素生效，不做前向或重新评价。输出/content/officehome-bn-capacity-cross-10e-v1-results.zip。

论文口径更正：本轮只读PDF复核确认OfficeHome K4，按论文C+K目标聚类应为Q29；此前两者“未确认”的记录已由PAPER_SETTING_CORRECTION.md更正。K2 arm仍保持Q29以隔离头容量，不在运行中改成论文绑定式Q27。完整预算及发布代码/论文公式差异仍未解决，不能声称严格论文复现。

## 四组完成

exec156正常done，history恰好10轮，loss有限。collector bac74e6c正常收集四组；新hold/K2 encoder统计buffers与prior完全相同，因子生效。该确认不做模型前向或重新评价。最终hold/K2日志未观察到GMM收敛警告；hold/K4记录1条，不把消息数解释为失败拟合数。

| RTA BN / K | best HOS%（epoch） | final OS*% | final UNK% | final HOS% |
| --- | ---: | ---: | ---: | ---: |
| 更新 / 固定4 | 73.0093（10） | 69.1347 | 77.3440 | 73.0093 |
| 更新 / 估计2 | 74.5557（7） | 73.2505 | 75.8897 | 74.5467 |
| 保持 / 固定4 | 75.4183（9） | 72.9730 | 78.0010 | 75.4032 |
| 保持 / 估计2 | 76.6334（10） | 69.4019 | 85.5471 | 76.6334 |

同epoch10的单因素差值：

- 更新BN中K2−K4：OS*+4.1158pp、UNK−1.4543pp、HOS+1.5374pp。
- 保持BN中K2−K4：OS*−3.5711pp、UNK+7.5462pp、HOS+1.2301pp。
- 固定K4中保持−更新BN：OS*+3.8383pp、UNK+0.6569pp、HOS+2.3939pp。
- 估计K2中保持−更新BN：OS*−3.8486pp、UNK+9.6574pp、HOS+2.0866pp。

因此，估计K2在两个BN策略下的final HOS均较固定K4高，但改善来源不同；保持BN/K2显著更多地拒识未知，也牺牲已知准确率。这里“更多”是数值描述，不是统计显著性。容量收益的final HOS交互项为−.3073pp；仅为单seed差分描述，不做显著性检验或独立可加贡献宣称。

保持BN下，K2第5/6轮HOS反而低于K4，第7–10轮才高于K4；不能说所有轮次都支配对照。best选择使用target标签，四组全量记录保留，不选较有利的一组当均值。这个容量规则得到K2不是target标签调参结果，也不代表找回OfficeHome真实40个未知类。

下一步应冻结当前候选定义，用其他任务检验容量与已知/未知取舍，而非继续在OfficeHome搜K、阈值或seed。完整预算稳定性、多seed及VisDA仍未完成；保持BN是单独的训练策略版本，不把联合收益称为仅自适应K。四组共同增加source CE3轮，不能将其成绩直接当原论文无此阶段的预算对等结果。

结果已下载：pipeline-results/officehome-bn-capacity-cross-10e-v1-results.zip、pipeline-results/officehome-bn-capacity-cross-10e-v1-summary.json；含四组普通配置、完整log/history与单因素差值。只有新增arm被训练，前三组全部复用。
