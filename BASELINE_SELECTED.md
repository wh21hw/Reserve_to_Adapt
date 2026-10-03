# 当前选定 baseline 与模块对照

2026-10-03。用户目标：先尽量取得能对标论文的 best seed，再固定该 seed 做模块实验。

## A→W：选定旧环境 L4 seed3

原T4工程代码、Python3.8、torch1.7.1+cu110、torchvision0.8.2、numpy1.23.4、sklearn1.1.2、faiss1.7.4；GPU L4。K=2、虚拟聚类Q=20、batch64、70完整epoch，不改损失或学习率。

exec134已完成。当地指标文件 `pipeline-results/rta-legacy-l4-seed3-metrics.json`；runtime目录 `/content/imp-runs/rta-legacy-l4-bridge-v1/a2w_seed3`。

| 口径 (%) | OS* | UNK | HOS |
| --- | ---: | ---: | ---: |
| 论文 A→W | 92.2000 | 93.8000 | 93.0000 |
| 选定seed3 best，第50轮 | 95.1705 | 95.7032 | 95.4361 |
| 选定seed3 final，第70轮 | 94.7472 | 95.4002 | 95.0726 |

best是目标标签HOS选epoch，seed3也是依据既有结果选出的种子。这是可对标/超过论文的单seed结果，不是三seed均值，不声称精确复现论文完整指标向量。此前T4三个seed和新环境L4三个seed均保留。

同L4新环境seed3 best HOS90.0455%，旧环境/旧工程代码桥接得到95.4361%；说明不能将此前差距直接归因GPU性能。它支持软件/实现版本的重要性，但没有单独分离每一个库及工程改动的因果贡献。

best.pt已通过13个小分块下载并合并保存到本地 `pipeline-results/rta-legacy-l4-seed3-best.pt`，原best.pt和last.pt仍在runtime。此前整文件下载取消的只是本地CLI客户端，没有删除云端文件。last.pt尚未保存到本地。没有SHA/重复checkpoint审计。

## 同seed模块路线

1. seed3，固定3epoch，纯C输出source监督；冻结网络，提取source/target特征与source中心。exec137在首次scheduler回调失败，尚未执行SGD（initial_lr关键字与lr形参不匹配）；保留v1目录和console。exec138修正参数名后在v2新目录完成，同设置不调参。source 958×256、target 564×256；第3轮source loss0.68630，训练准确率93.53%。
2. 在冻结特征上推断K。当前第一版预先设置：source类内平方距离99%分位数为DP-means风格建簇阈值，source类内方差为高斯软分配尺度，source先验强度5，迭代5次，已知中心可移动，无RTA关系gate和质量≥5筛簇。新增非空组件数作为K。数值方差下界1e-8；容量100若达到上限报错，不截断美化K。不读取target标签选K。
3. 原版RTA同seed3，双方共享上述source初始化，固定K=2 vs估计K。这个固定K对照应单独训练，不能拿前面的ImageNet初始化baseline直接冒充纯K对照。
4. 后续加入分段交替版本，K可增可减，总RTA训练预算一致。暂不加入层次未知合并、Dirichlet分类权重或结构KL。

exec139完成首次移动已知中心的估计，K=9（总19原型），5次迭代都为19。阈值0.5108345，方差0.0009093746；新增簇有约1个样本的簇，未事后过滤，也不能解释为已经恢复9个未知语义类。本地结果 `pipeline-results/konly-seed3-estimate-v1.json`。

exec141已启动同source初始化固定K=2的70epoch对照，目录 `/content/imp-runs/konly-rta-v1/fixed2/a2w_seed3`，入口 `scripts/train_konly_rta_entry.py`。最新检查已完成15epoch，损失有限，尚未完成。估计K=9的RTA训练还未启动，不能报告收益。下一臂入口 `scripts/run_konly_estimated_seed3.py` 只在固定K完成后启动；结果保存/汇总用 `scripts/collect_konly_results_colab.py` / `scripts/summarize_konly_seed3.py`，不独立重算checkpoint或hash审计。

baseline权重保存完成：独立shell将best.pt分为8MiB片段，`scripts/download_selected_baseline.ps1` 13片下载并合并完成。未改动云端原权重，也不使用hash检查。

### 已知中心固定/移动：冻结特征层消融

同一source初始化、同一阈值/方差/迭代数，只切换 `known_centers_fixed`，CPU推断已完成；移动/固定两组都得到K=9，总19个原型，5轮不变。原型责任质量如下（共564个target样本）：

| 已知中心 | 已知原型总质量 | 新增原型总质量 | K |
| --- | ---: | ---: | ---: |
| 可移动，source锚点先验强度5 | 397.7448 | 166.2552 | 9 |
| 完全固定 | 140.8716 | 423.1284 | 9 |

这说明中心更新改变了责任分配，但当前只向RTA传K、不传原型/责任，两者下游设置相同，不重复训练相同K臂。质量不是语义准确率，也不能用其证明哪种更好。固定中心结果已存 `pipeline-results/konly-seed3-estimate-fixed-v1.json`，无target标签或优化器更新。

该估计器是source校准的DP-means/IMP-inspired版本，不是IMP原论文的端到端方差学习，也不是完整DP后验。source阈值校准与先验强度是明确的建模选择，不按targetHOS调整。

另外两个任务OfficeHome Pr→Rw、VisDA Synthetic→Real仍属于项目范围，尚未完成正式baseline/模块对照；VisDA backbone口径待确认。
