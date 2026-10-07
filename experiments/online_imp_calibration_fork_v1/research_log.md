# Research log

2026-10-07：从上一批target已知支持污染和独立warm分歧出发，建立共同warm4/全状态恢复及单因素支持资格对照。只增加source-only当前初步IMP身份筛查，核心损失、K/V刷新和结构监督均两组相同。

共同warm4 exit0，155.5秒、K34/V20；restore_check exit0、0新训练，完整模型/优化器/schedule/GRL/relation/GMM/RNG恢复，分类动量shape/device检查通过。exec8继续confidence再structure_support，各6轮。full last/commonwarm暂只保留runtime，best和小结果Drive；不独立重评分数，不hash。
