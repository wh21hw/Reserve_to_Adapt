# 旧方案硬原型更新单因素对照

用户2026-10-08授权。Office31 A→W、fresh seed3、source5、RTA20、warm2、alpha.05、IMP5步、known固定、分类头C10+2、ResNet50/256、batch64、原BN/virtual/未知argmax/entropy/adv/SGD动量复用全部不变。

唯一变量：IMP迭代的原型更新由Gaussian软责任度加权均值改为最近原型硬分配后的簇内均值。建新簇仍用原平方距离阈值公式；空unknown簇沿用原fit删除，已知锚点保留；每轮重新提取/聚类，不新增EMA或未知软伪标签，不改输出K。固定known中心不因硬分配移动。

对照复用刚完成同seed3/source5/warm2/alpha.05/20轮软更新：best=final第20轮OS*92.0805%、UNK86.9633%、HOS89.4488%。这是fresh同配置对照而非相同checkpoint轨迹fork，存在运行数值噪声。

预算：单arm1500秒、总4500秒保护（仅单arm），根据前批预计11–13分钟，实际观察前几轮后重估。仅一次针对性硬分配功能检查，不重复稳定smoke/hash/checkpoint重评。保留全部20轮指标、best/last模型，target标签仅评价；不改alpha/换seed/延长70或失败静默改公式。

预计root/content/imp-runs/legacy-imp-hard-update-20e-v1，archive/content/legacy-imp-hard-update-20e-v1-results.zip。small日志/配置/summary自动复制mounted Drive并下载电脑，模型留runtime关闭前实际备份。完成报告全部best/final、峰值epoch、V范围和尾5轮及与soft的取舍，不只选更高best；报告后暂停hook，不关闭实例。

状态：准备，复用现有T4与挂载，不重新下载数据。
