# 旧方案alpha/source轮数单因素20轮

用户2026-10-07明确授权调alpha及预热轮数，并认为旧方案20轮可能收敛。这里预热指额外source监督轮数，不改RTA warmiter3（前4轮）或损失系数。

两候选seed3分别alpha.01/source5、alpha.05/source3，RTA20，各从ImageNet完整重训。原配方alpha.05/source5参考来自正在运行三seed筛查的seed3，全部budget20，不能混用旧70轮best。固定任务A→W、C10+2、ResNet50/256/batch64、旧IMP公式/固定known/每轮virtual/原pseudoargmax/原BN/旧优化器，工程边界同legacy_imp_seed_screen_v1。

选择依据是历史10轮结果提示alpha.01可能保护known，source3可能加快unknown上升；不是承诺达到记忆中的成绩。一次两候选不扩密网格，不同时改两因素。每arm1500秒/总4500秒保护，预估本批20–25分钟，按实测history估。已有三seed先跑完，不打断、不争同GPU。新batch等待前batchlauncher234371退出后自动接续，最多等待3600秒，否则停止而非并发启动。

保留每epoch OS/OS*/UNK/HOS、V/耗时与best/last，目标标签只评价不聚类/训练。披露target-oracle配置/epoch选择，非独立最终测试；20轮峰值不证明收敛，需看尾段曲线。依赖未随用户提供，使用冻结RTA依赖，非整个旧环境逐字恢复。全部结果电脑+Drive小记录，关闭runtime前模型备份。失败保留证据，不偷偷改数值公式或重跑；不自动延长70。状态：准备并等待已有三seed，不算训练已经启动。
