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

交替实现准备中（未正式运行）：预先固定在完成20/40/60轮后重新提取当前source/target表示并推断K，仍用source距离99%分位数/方差/先验5/5轮。新增类头按随机方向和RTA式权重范数尺度初始化，不安装IMP原型方向。旧未知头通过当前未知预测与新责任矩阵的重叠做匈牙利对应；保留匹配头和其SGD动量，已知头不动，调度器不归零。K不变时不重新排列任何头。CPU实际CLS接口一次检查已通过3→5→2→2；这不是完整交替训练通过，更不是效果结论。先完成固定一次K的两臂，再启动此版本。

exec139完成首次移动已知中心的估计，K=9（总19原型），5次迭代都为19。阈值0.5108345，方差0.0009093746；新增簇有约1个样本的簇，未事后过滤，也不能解释为已经恢复9个未知语义类。本地结果 `pipeline-results/konly-seed3-estimate-v1.json`。

exec141已完成同source初始化固定K=2的70epoch对照，目录 `/content/imp-runs/konly-rta-v1/fixed2/a2w_seed3`，入口 `scripts/train_konly_rta_entry.py`。best第66轮与final第70轮指标相同：OS*=91.8587%、UNK=95.5965%、HOS=93.6903%。相对论文HOS+0.6903pp；相对原ImageNet初始化baseline，best HOS−1.7458pp、final−1.3822pp，说明额外source预训练不能忽略。普通配置/70轮日志/指标已打包下载至 `pipeline-results/konly-rta-v1-fixed2-results.zip` 并解压；云端best/last权重保留。

exec143已完成估计K=9的同source初始化70epoch臂，目录 `/content/imp-runs/konly-rta-v1/estimated/a2w_seed3`。best第17轮OS*=96.9797%、UNK=81.5706%、HOS=88.6103%；final OS*=95.9017%、UNK=77.7575%、HOS=85.8817%。对照K2，best差+5.1210/−14.0259/−5.0801pp，final差+4.0430/−17.8390/−7.8086pp（OS*/UNK/HOS）。这是完整预算的单seed负结果，不能把已知类提升写成总体方法提升。普通日志/配置/指标已下载ZIP并解压；汇总 `pipeline-results/konly-seed3-comparison-v1.json`。没有独立重算checkpoint或hash审计。

exec145已完成交替K版本70轮，目录 `/content/imp-runs/konly-alternating-v1/seed3/a2w_seed3`。K轨迹9→9→11→11，20/40/60轮后刷新；40轮保留9个匹配未知头及其SGD动量，新增2个随机方向头，已知头保持，调度器不重置。三次阈值1.0746273/0.9701381/0.8631213，由当前source特征重新校准，未用target标签。best第17轮（K9）：OS*=96.9797%、UNK=81.2002%、HOS=88.3913%；final（K11）：97.1128/76.1557/85.3669%。对照一次K9，best/final HOS−0.2190/−0.5149pp；对照固定K2，−5.2991/−8.3235pp。当前交替未改善结果，不依据负结果重新挑阈值/seed/刷新频率。

共享source-final已完整保存到本地 `pipeline-results/konly-source-prior-seed3.pt`；source配置/3轮日志/冻结特征也已存 `pipeline-results/konly-source-prior-seed3-features.tar.gz`。固定K对照best权重已完整保存 `pipeline-results/konly-fixed2-seed3-best.pt`，估计K9的best已保存 `pipeline-results/konly-estimated-seed3-best.pt`，原云端文件保持不动。交替完整入口的源代码拼接/编译已通过，启动脚本在一次性K两臂都完成70轮后执行。

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

跨任务准备：新增旧环境任务入口 `scripts/train_legacy_task_entry.py`，不替换现有A→W训练；类别映射/目标训练标签sentinel/宏平均UNK按任务适配，virtual Q必须显式给出，不伪称未知的作者设置。`task_protocol.py`改为Python3.8可用的relative_to路径检查，去掉例行list hash，5个协议测试通过。VisDA官方原ID已确认0..11，RTA已知六类为1/2/3/6/10/11；train.tar官方HEAD200、7698031104字节，现通过独立shell下载到/content/osda-visda-syn2real-v1（未解包/未训练）。来源见experiments/visda_protocol_v1/README.md；仅非商业研究教育，不发布图像。

OfficeHome旧环境实际CPU数据入口检查通过，source1785/target4357、source头29维、target训练sentinel25、评价65原始类别，未创建模型/SGD。Q29是预先声明的C+baselineK选择，不是作者已确认配置；不是完整聚类/训练通过。结果已下载 `pipeline-results/legacy-officehome-loader-check-v1.json`。

OfficeHome正式baseline已启动exec147，seed3/K4/Q29/70epoch/旧环境L4，入口 `scripts/run_legacy_officehome_colab.py`，目录 `/content/imp-runs/legacy-officehome-v1/seed3/officehome-pr2rw_seed3`。seed3是首个测试种子，不称为该任务已选best；70轮是移植发布代码预算，并非已确认作者OfficeHome预算。论文Pr→Rw参照OS*=82.1%、UNK=77.2%、HOS=79.5%，不是12任务平均70.9%。

VisDA数据阶段已完成：官方train/validation下载并解包所需图片，source已知六类79765张，target55388张。原始类别ID和无标签路径分开保留，准备报告已下载 `pipeline-results/visda-data-preparation-v1.json`。这不解决backbone或论文训练预算，不代表开始正式训练。
