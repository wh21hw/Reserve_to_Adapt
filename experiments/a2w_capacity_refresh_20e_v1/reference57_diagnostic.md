# 原版RTA参考模型：簇数不等于必需输出槽数

## 本次发现（2026-10-06）

当前可用的原版A→W seed3模型实际为epoch57，分类头C10+K2。使用当前不变的source-calibrated-birth-cost-v1估计器和已知身份匹配，得到4个候选未知簇；目标标签未输入估计器、标定或匹配。

全量target564张，其中3张未分配；模型预测unknown265张，原型分配unknown262张。剩余561张中，原型known但模型unknown为1张，原型unknown但模型known为1张，已知/未知判定一致559/561=99.6435%。这是两种规则的一致率，不是对真实标签的准确率。

簇占用数为14，计数为37/21/30/28/27/31/38/35/27/25/121/83/16/42。raw与匹配后候选未知K均为4。不能由此声称恢复4个真实未知语义，或分类器必须有4个未知槽。一个已有2槽的模型可以在特征空间呈现更多候选未知结构。

## 对研究的意义

此前K8→K4的20轮对照改善的是相对固定K8的后续学习与拒识保持，不能推出“槽越多越好”，也不能推出估计簇数必然是最优分类头维数。下一项优先问题是：共享相同source前置训练、相同完整预算下，估计K是否优于固定K2。原型几何、伪标签自举和分类容量是不同层次，不先堆新损失。

本参考模型与当前source C-only3轮、冻结encoder BN的20轮两组不是同预算/同前置设置，不作为干净的K、BN或预热因果消融。不根据这项诊断修改阈值，也不重新挑选best checkpoint。

## 权重与日志轮数不一致

`OSDA/runs/rta-official-a2w-v2/a2w_seed3/last.pt`在当前T4和保留CPU的独立Drive挂载中都读取为epoch57、head shape[12,256]。原归档和当前history/metrics日志仍有70轮。原因尚未确定；不能把当前权重称为final70，也不覆盖历史日志成绩或改写权重epoch。

首次请求final70提取时轮数检查提前拒绝，未进行图像前向。随后明确改用当前可用last57；没有使用目标成绩选checkpoint。一次完整图像提取约7.14秒，source958/target564、256维，之后仅从缓存做一次无标签估计，没有新训练、阈值扫描或重新计算accuracy。

同时CPU独立挂载确认新20轮对照的fixed8与refresh `last.pt`均为epoch20，头分别[18,256]与[14,256]；这次只读元数据，没有重复评分或hash验证。

## 保存

Drive持久缓存：`OSDA/runs/a2w-reference-available57-v1/{features.npz,manifest.json}`；推断结果：`OSDA/runs/a2w-reference-available57-capacity-v1/{summary.json,clusters.npz}`。

本地小汇总：`pipeline-results/a2w-reference-available57-capacity-v1-summary.json`与`pipeline-results/a2w-reference-available57-manifest-v1.json`；缓存与分配NPZ也已下载，二进制不提交Git。脚本为`scripts/infer_a2w_reference_available57_capacity_colab.py`，通用提取器为`scripts/cache_a2w_current_relation_colab.py`。

当前40训练轮预算已经用完；下一批固定K2与阶段性估K各70轮（总140轮、训练上限4小时）仅已提议，尚未批准或启动。CPU与T4按用户要求保留，不声称GPU可原地无损降级为CPU。
