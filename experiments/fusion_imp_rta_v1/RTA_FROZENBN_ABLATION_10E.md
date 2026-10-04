# RTA encoder BN 统计保持消融

2026-10-04，在新arm结果产生前记录。由BN_TRANSFER_DIAGNOSIS的统计变化诊断引出，目标是检验预热新颖性几何到RTA的传递；不是“仅自适应K”方法。

唯一研究因素：RTA阶段encoder BN使用running统计（eval模式）；encoder权重、BN affine、分类头与head BN照常训练。每次encoder forward前重新应用该策略，避免原TrainingMode的train()解除冻结。无teacher/DINO/新loss，正式ResNet50不变。

控制组复用已完成 `/content/imp-runs/officehome-frozenbn-capacity-10e-v1/fixed4`。两组共享完整C25 source冻结encoderBN CE3轮prior、K4/Q29、seed1、原初始化和RTA损失、batch64、lr5e-5、10轮预算。保持原warmiter3（前4轮warmup）及C+K目标K-means初始化。新输出 `/content/imp-runs/officehome-rta-frozenbn-10e-v1`，拒绝覆盖。

encoder_bn_policy.py模式接口检查已在legacyPy3.8/Torch1.7的CPU通过：两次train()+forward不更新encoder均值/方差/计数，BN affine有有限非零梯度，head BN更新2次。只做这一次针对性检查，不重复稳定smoke或checkpoint评测。

新组完成后只读history报告best/final OS*/UNK/HOS、与同epoch对照差值；best仍使用target标签选epoch，单seed短程探索，不称显著改善。target标签不决定K/阈值/预算，不因中间结果扩大网格或自动70轮。若失败，保留日志定位工程原因，不改研究设置重跑。

脚本：scripts/run_rta_frozenbn_ablation_colab.py。入口scripts/train_legacy_task_entry.py新增RTA_FREEZE_ENCODER_BN=1可选开关，默认0保持旧行为；launch.json明确该变化，普通配置/history/log/checkpoint照常保存。该记录不表示已完成或有效。

实现1d75b4a0；已启动后台exec154，现有L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o。启动输出确认53个encoder BN module、affine trainable/head unchanged，以及C25/K4/Q29与相同source prior。当前running，尚无最终效果结论。collector为scripts/collect_rta_frozenbn_ablation_colab.py；新模式只在完成时读取一次BN buffers确认研究因素生效，不做前向或checkpoint重评价。
