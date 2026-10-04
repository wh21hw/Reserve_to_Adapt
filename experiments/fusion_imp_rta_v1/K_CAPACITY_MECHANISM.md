# 为什么增加 K 不一定改善 RTA

2026-10-04。以下是对实际运行 `/content/rta-legacy-l4-bridge-v1` 代码的只读核实，不修改正在进行的实验。正式 backbone 仍为 ResNet50，DINO 只作诊断。

K 是未知输出行数，不等于目标虚拟聚类数 Q，也不保证每行对应一个真实未知语义。

1. `main.py` 的未知伪标签在 `predict_prob_otherep[:, shared_classes:]` 中取 max。增加 K 改变候选竞争范围，但不会自动纠正此前把未知样本视为已知的筛选错误。
2. `main.py` 把完整 `fc_source[:, :]` 传给 `virt_forward`。`networks.py` 的该方法按 source 真类权重范数缩放虚拟方向，将虚拟 logits 拼到完整 C+K logits 后做 softmax。因此新增未知行也进入 source virtual CE 的竞争，并非没有梯度影响的空位。
3. 原版 warm-end 初始化采用 C+K 目标 K-means。两组共享初始化策略，但不同 K 的未知权重和聚类划分本来就不同；不能说未知头初始权重完全相同。
4. 评测把所有未知输出行合并为一个 unknown 决策，但训练伪标签仍在 K 行内竞争。更多槽位可能提供更细的表示，也可能产生碎片化和不稳定伪标签；这是待实验检验的机制解释，不是已证实的当前失败原因。

当前真实 OfficeHome 配对只比较固定 K4 与估计 K2，共享 source prior、seed1、Q29、损失、ResNet50、10轮预算。估计值只传整数 K，不传 IMP 分配或中心，避免把容量收益与头方向初始化混在一起。保持同一 seed 不等于两种头维度下所有随机抽样轨迹 bit-identical。

解读结果应同时看同轮次 OS*/UNK/HOS、best 与 final。即使某个 K 的 target HOS 较高，也不能用 target 标签反选聚类阈值或宣称恢复真实未知类数。source 冻结 BN 的诊断改善是否传到跨域任务，仍须由完成的配对结果回答。
