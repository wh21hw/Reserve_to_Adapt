# 保持 RTA BN 统计下的自适应 K 对照

2026-10-04，在新增arm结果产生前记录。现有三个arm分别是更新BN/K4、更新BN/估计K2、保持BN/K4，各source3轮+RTA10轮。新增一个保持BN/估计K2 arm，补上容量与BN因素的交叉对照；不搜索alpha/lambda/seed，不重跑已有控制。

使用既有prior/capacity.json的K2（exec147，无target标签），不重估、不按HOS选择K。共享同一冻结encoderBN的C25 source CE3轮checkpoint、seed1/Q29/ResNet50、原RTA损失/伪标签/argmax/初始化/10轮预算。保持encoder BN统计，权重与affine及head正常训练。与已完成保持BN/K4的唯一研究因素为K；与更新BN/K2的唯一因素为BN模式。

源码入口与BN hook已验证，不重复smoke。新目录/content/imp-runs/officehome-rta-frozenbn-k2-10e-v1，拒绝覆盖。launcher scripts/run_rta_frozenbn_ablation_colab.py --arm estimated；原无参数用法仍为fixed4，绝不在本次调用重跑fixed4。沿用已部署encoder_bn_policy.py和train_legacy_task_entry.py。

完成后报告四组best/final OS*/UNK/HOS及同epoch差值；分别计算BN内的容量对照、同K下BN对照，不把双因素差异伪称单因素收益。best仍target标签选epoch，单seed短程探索，不称显著性或真实未知语义数恢复。实验是看到前三组结果后的模块研究，不称无偏预声明外部验证。若结果不支持容量收益，也保留，不调整K或延长预算寻找好峰值。

正式ResNet不变、DINO只诊断。无新loss、IMP未知头初始化、结构KL或动态改变K；当前推断一次K后固定。三数据集完整外部验证和原论文预算对齐仍未完成。

实现e0fdb9eb，collector ecdf5b21。已单独启动后台exec156，现有L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o；启动日志确认53个encoder BN模块保持统计、affine trainable/head unchanged，以及C25/K2/Q29和同一source-final.pt。当前running，尚无最终结论；没有重新训练source、估计K或重复前三组。

collector scripts/collect_bn_capacity_cross_colab.py已部署，仅在本arm完整10轮后执行。它读取四组已有history，列出同BN下K差值和同K下BN差值、final交互项，并在新arm结束时读取一次统计buffers确认BN因素生效，不做前向或重新评价。输出/content/officehome-bn-capacity-cross-10e-v1-results.zip。

论文口径更正：本轮只读PDF复核确认OfficeHome K4，按论文C+K目标聚类应为Q29；此前两者“未确认”的记录已由PAPER_SETTING_CORRECTION.md更正。K2 arm仍保持Q29以隔离头容量，不在运行中改成论文绑定式Q27。完整预算及发布代码/论文公式差异仍未解决，不能声称严格论文复现。
