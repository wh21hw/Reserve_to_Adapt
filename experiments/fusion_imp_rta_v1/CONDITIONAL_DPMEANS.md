# 先门控再候选内部 DP-means：冻结探针 v1

独立候选容量建模，不是K-only正式训练结果。复用融合版seed1/source C10监督3轮冻结特征，在现有L4 runtime用colab-cli运行；无新训练、无参数扫描、无目标标签拟合。

与已有relation-prior聚类不同：先按关系门控硬划分集合，候选内部不与已知锚点竞争。关系分数沿用source分布中心||target已知条件分布的KL；BayesianGaussianMixture四成分、max_iter800、random_state2026，最低均值成分为known，known_probability<0.5进入pool。这个规则是预先声明的工程门控，四个混合成分不是语义类数。

候选内部目标J=sum_i min_j ||x_i-mu_j||²+lambda*K。lambda沿用source类内平方残差99%分位0.53685586，没有target调阈值；从集合均值初始化，farthest birth、硬分配均值更新、删除空中心，最多100步/100簇，达到上限报错而不是偷偷截断。每步目标不增加，只保证该过程的局部收敛，不声称全局最优或完整DP贝叶斯后验。源码conditional_dpmeans.py。

工程修复：首次缺relation_gate依赖，第二次既有features.npz没有logits；均在拟合/训练前失败，未产出结果。补上传既有relation_gate.py，用source-final.pt保存的eval BatchNorm+LeakyReLU+fc从缓存的归一化bottleneck恢复logits；没有重新提取图像特征、SGD、参数改动或覆盖历史结果。

## 实际结果

pool371/564张：覆盖253/269未知（94.05%），同时包含118/295已知（已知误入40.00%）；未知纯度68.19%。对比初始IMP候选覆盖20.82%，门控解决了大部分漏选，但污染明显。

DP-means得到K2，14步局部收敛，目标124.1513→112.7304，支持301/70。第一簇含118已知与183未知，混合多个类别；第二簇70张均未知，但仍混合未知20/25/26/29。不能称发现2个真实未知类别，也不能仅因为K2接近历史好容量就宣称方法有效。

结果：pipeline-results/conditional-dpmeans-a2w-v1.json及npz；脚本scripts/probe_conditional_dpmeans_colab.py。标签只在聚类完成后做诊断，不能据此提高门控阈值、缩小lambda、挑参数或用于训练槽位映射。

## 决策

暂不把这个简单版本接入70轮：其两个问题分别是关系门控污染、source尺度下候选结构严重合并。“不被已知吸收”改善候选覆盖，不足以建立语义原型。后续如果检验容量用途，只能作为K2工程容量与同初始化固定K2等价，不重复训练来伪称分类收益。要实现未知结构的真实增量，需要先提出可在source留类控制上检验、而非用target诊断调出来的拒识/结构学习规则。已有相关控制和失败方案需保留并核对，禁止继续目标标签驱动的阈值扫描。
