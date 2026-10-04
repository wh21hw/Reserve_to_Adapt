# 原型未知伪标签 v1（独立融合变体，不是K-only）

假设：未知槽的正反馈伪标签缺少独立几何依据，多槽分工不稳定；用原型分配代替分类器自举，可能更好利用估计容量。尚未证明。

只替换未知训练伪标签。保持seed1、相同source3轮prior、估计K7固定、IMP虚拟更新、原版KL/GMM选样、损失系数、平坦argmax预测、优化器和10轮预算。对照为fusion-capacity-diagnostic-v1/K7的10轮。

第4轮末用原版warm-end C+K K-means找到的K个未匹配目标中心初始化teacher。它们在归一化bottleneck特征坐标，独立于BN后的分类器权重。第5轮起，对RTA选中的疑似未知样本按最近teacher中心赋予槽标签，沿用原版未知CE；对应中心用batch成员均值进行EMA（momentum0.9，预声明，不扫参数）。没有新损失，不调整门控、不把IMP几何中心直接写入头。

原型只从原门控选中的样本更新，仍可能受已知/未知误分流和域偏移影响。EMA不可保证对应语义；固定K，不进行birth/death。teacher未初始化的预热期保持原版伪标签（未知CE系数0）。空槽保持中心，不强制均衡。

teacher-history记录逐轮真实原型分配次数与分类器argmax一致率；mechanism-history中的selected槽计数仍表示分类器argmax，而非teacher标签，应结合teacher-history解释。

先做一次构造/接口检查再跑10轮，目标标签不进入teacher。若改善，仅算事后best-seed短程探索，不能替代70轮和另外两个数据集验证。

状态：K2/K7容量诊断已完成。构造与teacher接口检查通过，L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o的exec27已启动10轮，前1轮loss有限。teacher第4轮末初始化、第5轮开始生效，预热结果不能代表改动有效。运行目录/content/imp-runs/fusion-unknown-teacher-v1，对照K7 HOS82.9817%，K2 HOS86.9174%仅作容量背景。代码版本见本轮Git提交。
