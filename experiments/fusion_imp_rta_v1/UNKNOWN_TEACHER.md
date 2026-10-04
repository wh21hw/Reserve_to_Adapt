# 原型未知伪标签 v1（独立融合变体，不是K-only）

假设：未知槽的正反馈伪标签缺少独立几何依据，多槽分工不稳定；用原型分配代替分类器自举，可能更好利用估计容量。尚未证明。

只替换未知训练伪标签。保持seed1、相同source3轮prior、估计K7固定、IMP虚拟更新、原版KL/GMM选样、损失系数、平坦argmax预测、优化器和10轮预算。对照为fusion-capacity-diagnostic-v1/K7的10轮。

第4轮末用原版warm-end C+K K-means找到的K个未匹配目标中心初始化teacher。它们在归一化bottleneck特征坐标，独立于BN后的分类器权重。第5轮起，对RTA选中的疑似未知样本按最近teacher中心赋予槽标签，沿用原版未知CE；对应中心用batch成员均值进行EMA（momentum0.9，预声明，不扫参数）。没有新损失，不调整门控、不把IMP几何中心直接写入头。

原型只从原门控选中的样本更新，仍可能受已知/未知误分流和域偏移影响。EMA不可保证对应语义；固定K，不进行birth/death。teacher未初始化的预热期保持原版伪标签（未知CE系数0）。空槽保持中心，不强制均衡。

teacher-history记录逐轮真实原型分配次数与分类器argmax一致率；mechanism-history中的selected槽计数仍表示分类器argmax，而非teacher标签，应结合teacher-history解释。

先做一次构造/接口检查再跑10轮，目标标签不进入teacher。若改善，仅算事后best-seed短程探索，不能替代70轮和另外两个数据集验证。

状态：K2/K7容量诊断已完成。构造与teacher接口检查通过，L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o的exec27已启动10轮，前1轮loss有限。teacher第4轮末初始化、第5轮开始生效，预热结果不能代表改动有效。运行目录/content/imp-runs/fusion-unknown-teacher-v1，对照K7 HOS82.9817%，K2 HOS86.9174%仅作容量背景。代码版本见本轮Git提交。

## 结果（exec27已完成）

各10轮best均在第10轮：自举K7 OS*90.2738%、UNK76.7796%、HOS82.9817%；teacher OS*90.0950%、UNK80.2421%、HOS84.8836%，HOS+1.9019pp。支持仍不均匀：teacher总分配[188,113,386,342,1,248,66]，第5槽仅1次；与分类器一致率87.57%，最终96.88%。该改动未明显解决容量利用，不足以排除重复运行波动。

普通日志已下载pipeline-results/fusion-unknown-teacher-v1-results.zip并解压。另一次有针对性的final checkpoint推理消融exec30，不训练、不扫阈值、不重新评价所有epoch：未知槽概率求和后与各已知类概率竞争。目标标签仅评分，保留两个预声明规则结果。

| 第10轮模型 | 单槽argmax OS* / UNK / HOS | 合并未知概率 OS* / UNK / HOS |
| --- | --- | --- |
| 自举K7 | 90.27 / 76.78 / 82.98 | 87.12 / 82.24 / 84.61 |
| teacher K7 | 90.10 / 80.24 / 84.88 | 87.53 / 85.73 / 86.62 |

teacher合并后17个真实未知被纠正，13个真实已知被误拒绝。规则变化的差异来自同一checkpoint同一批概率，可隔离解码因素；跨训练的teacher增益仍有随机波动。未达到原版RTA的已报告水平，不据此扩大70轮实验。评分口径保留11个未知语义类各自召回再取平均，与官方一致；合并类后概率argmax的普通准确率决策解释不保证宏平均HOS最优。边缘化结果不是原版RTA规则的复现成绩。

精确结果：pipeline-results/fusion-unknown-teacher-marginal-decode.json。下一步仅检验低支持候选是否虚增K，见SUPPORTED_CAPACITY.md；不同时叠加teacher或新预测。
