# RTA encoder BN 统计保持消融

2026-10-04，在新arm结果产生前记录。由BN_TRANSFER_DIAGNOSIS的统计变化诊断引出，目标是检验预热新颖性几何到RTA的传递；不是“仅自适应K”方法。

唯一研究因素：RTA阶段encoder BN使用running统计（eval模式）；encoder权重、BN affine、分类头与head BN照常训练。每次encoder forward前重新应用该策略，避免原TrainingMode的train()解除冻结。无teacher/DINO/新loss，正式ResNet50不变。

控制组复用已完成 `/content/imp-runs/officehome-frozenbn-capacity-10e-v1/fixed4`。两组共享完整C25 source冻结encoderBN CE3轮prior、K4/Q29、seed1、原初始化和RTA损失、batch64、lr5e-5、10轮预算。保持原warmiter3（前4轮warmup）及C+K目标K-means初始化。新输出 `/content/imp-runs/officehome-rta-frozenbn-10e-v1`，拒绝覆盖。

encoder_bn_policy.py模式接口检查已在legacyPy3.8/Torch1.7的CPU通过：两次train()+forward不更新encoder均值/方差/计数，BN affine有有限非零梯度，head BN更新2次。只做这一次针对性检查，不重复稳定smoke或checkpoint评测。

新组完成后只读history报告best/final OS*/UNK/HOS、与同epoch对照差值；best仍使用target标签选epoch，单seed短程探索，不称显著改善。target标签不决定K/阈值/预算，不因中间结果扩大网格或自动70轮。若失败，保留日志定位工程原因，不改研究设置重跑。

脚本：scripts/run_rta_frozenbn_ablation_colab.py。入口scripts/train_legacy_task_entry.py新增RTA_FREEZE_ENCODER_BN=1可选开关，默认0保持旧行为；launch.json明确该变化，普通配置/history/log/checkpoint照常保存。该记录不表示已完成或有效。

实现1d75b4a0；已启动后台exec154，现有L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o。启动输出确认53个encoder BN module、affine trainable/head unchanged，以及C25/K4/Q29与相同source prior。当前running，尚无最终效果结论。collector为scripts/collect_rta_frozenbn_ablation_colab.py；新模式只在完成时读取一次BN buffers确认研究因素生效，不做前向或checkpoint重评价。

首轮已完成，CE=.655/virtual=.723/adv=.789等loss有限，无OOM/异常终止；当前仍running。首轮warmup的UNK/HOS=0不作为方法失败或成功判断。

只读日志比较：旧普通source fixedK4、新冻结source/RTA正常BN fixedK4、K2三个完整日志均无ConvergenceWarning；新RTA保持BN组目前有一条GMM initialization未收敛警告。只代表观察到的消息数，Python过滤可能隐藏重复消息，不当作失败拟合次数。实际RTA代码用target KL的GMM概率参与已知权重/关系分组，并有epoch级BayesianGaussianMixture；当前警告不能仅凭文本定位到某一调用或证明导致后续性能变化。不改变max_iter/分量数，不自动调参或重跑；完整log保留。

## 正常完成：保持统计改善此短程配对

exec154 done，RTA_FROZEN_ENCODER_BN_ARM_COMPLETE；history恰好epoch1..10，loss有限。collector5bc2ca77正常收集，只读取已有history与一次buffers，不前向/重评checkpoint。318个encoder统计/计数state-dict路径全部与共享source prior相同，确认hold因素生效；该数字包括注册别名，不是独立BN层数。

| RTA encoder BN / 选取 | epoch（1-based） | OS*% | UNK% | HOS% |
| --- | ---: | ---: | ---: | ---: |
| 更新统计 best/final | 10 | 69.1347 | 77.3440 | 73.0093 |
| 保持统计 best | 9 | 72.2454 | 78.8828 | 75.4183 |
| 保持统计 final | 10 | 72.9730 | 78.0010 | 75.4032 |

best差值OS*+3.1107pp、UNK+1.5387pp、HOS+2.4090pp；final差值OS*+3.8383pp、UNK+0.6569pp、HOS+2.3939pp。第5–10轮同epoch HOS均更高，但第5轮OS*较低，不能声称每一轮所有指标都改善。观察到hold组1条GMM警告、update组0条，未发生训练失败，不能把警告直接当作性能下降原因。

这支持当前OfficeHome/共享prior/固定K4/seed1/10轮下，RTA保持预热encoder统计有益；不证明先前所有性能差异都由BN产生，也不证明长期、多seed或跨数据集泛化。best仍target标签选epoch，不按target标签选聚类成本/K。不是严格论文复现或仅K改变版本。

回到主线：已有“BN更新、K4/K2”的容量配对，以及“固定K4、BN更新/保持”的BN配对。不能把保持BN的K4与更新BN的K2直接相比来判断K优劣；若将BN改动加入新版本，需补共享同prior/保持BN下的K2对照，继续使用已估计K2，不重调阈值或按HOS选K。70轮、额外seed及VisDA外部验证仍未完成。

结果已下载：pipeline-results/officehome-rta-frozenbn-10e-v1-results.zip、pipeline-results/officehome-rta-frozenbn-10e-v1-summary.json。没有重新训练控制组、source prior或重新估计K。
