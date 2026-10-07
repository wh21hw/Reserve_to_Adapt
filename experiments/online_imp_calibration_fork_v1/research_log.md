# Research log

2026-10-07：从上一批target已知支持污染和独立warm分歧出发，建立共同warm4/全状态恢复及单因素支持资格对照。只增加source-only当前初步IMP身份筛查，核心损失、K/V刷新和结构监督均两组相同。

共同warm4 exit0，155.5秒、K34/V20；restore_check exit0、0新训练，完整模型/优化器/schedule/GRL/relation/GMM/RNG恢复，分类动量shape/device检查通过。exec8继续confidence再structure_support，各6轮。full last/commonwarm暂只保留runtime，best和小结果Drive；不独立重评分数，不hash。

confidence已完成epoch5–7，elapsed117.0秒、近期37.04秒/轮，预计本批总约10–12分钟，随structure_support阶段重新估时。初步结构资格的新接口用已有缓存（非checkpoint重评）验证支持集合不增、维度/有限值正确；327→327仅是旧缓存功能检查，不声称本批过滤有效，实际污染率必须等结束后的支持mask统计。

完整对照结束exec8 done，两续训分支225.4/227.7秒。结构支持finalHOS−0.6357pp，OS*−1.4651pp违反guard；暂不启用，不删除实现。保存mask事后真值注释：115/400未知vs100/379未知，仍26.39%污染。两组UNK由共同73.91%退到约52%，只增加筛查或结构标签数没有解决。下一步改IMP观察的层，不改ResNet本身；同共同warm4续训预算，新目录单因素，不把快照或初始化代理当训练结果。
