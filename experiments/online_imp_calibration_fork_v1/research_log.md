# Research log

2026-10-07：从上一批target已知支持污染和独立warm分歧出发，建立共同warm4/全状态恢复及单因素支持资格对照。只增加source-only当前初步IMP身份筛查，核心损失、K/V刷新和结构监督均两组相同。

共同warm4 exit0，155.5秒、K34/V20；restore_check exit0、0新训练，完整模型/优化器/schedule/GRL/relation/GMM/RNG恢复，分类动量shape/device检查通过。exec8继续confidence再structure_support，各6轮。full last/commonwarm暂只保留runtime，best和小结果Drive；不独立重评分数，不hash。

confidence已完成epoch5–7，elapsed117.0秒、近期37.04秒/轮，预计本批总约10–12分钟，随structure_support阶段重新估时。初步结构资格的新接口用已有缓存（非checkpoint重评）验证支持集合不增、维度/有限值正确；327→327仅是旧缓存功能检查，不声称本批过滤有效，实际污染率必须等结束后的支持mask统计。
