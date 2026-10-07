# Research log

同一ResNet，IMP256 vs normalized encoder2048，仅容量/标签视图改变；virtual维数固定256。关键真实接口一次通过，K29，0新训练轮；两arm复用同commonwarm4，各6轮。

exec11完成，HOS+4.1202pp，但OS*−1.1008pp超1pp guard。未知预测ARI.5433→.4378/NMI.7479→.7053，不能说高维更好地恢复未知语义。真实曝光：256有137个独立未知获得结构标签，2048只有61、7类0次数；更多原始候选没有自动进入学习。下一项不是继续换backbone或扫阈值，而是只移除unknown标签的中心筛查、保留RTA候选规则和virtual保护来检验覆盖。

小结果已下载，summary单次fetch失败从ZIP提取，未重复collector/训练。概率readout只读正常评价输出，未知竞争分散不是完整退化解释。target真值用于结束注释，不入校准。保存与披露见final_report。
