# VisDA 全量跨任务验证：冻结当前候选规则

2026-10-04，在本任务训练/估计/评价结果产生前声明。当前source archive正在恢复；没有把历史数据准备结果当当前就绪。

任务Synthetic→Real validation：source原ID1/2/3/6/10/11，共79765图；target全12类55388图。ID依据官方列表与既有论文协议，source映射连续0..5，target语义仅协议/评测使用，不进入特征或K拟合。

第一阶段独立运行source监督：ResNet50/ImageNet、C6头、seed1、全量3轮，batch64/drop_last/workers4；同OfficeHome的SGD encoder/head lr5e-5/5e-4、momentum.9/nesterov/wd5e-4、inverse decay、Resize256x256/RandomCrop224/HorizontalFlip。encoder BN统计保持，权重/affine/head正常训练，无teacher/新loss/DINO。3轮约3738步，因数据规模不同，不声称跨任务同一步数预算。使用target-unlabeled-paths.txt，训练后eval/CenterCrop提取全量256维瓶颈。

第二阶段单独估计：同已冻结的source-calibrated-birth-cost-v1，source70%原型/15%成本校准，seed2026、source99%半径、source负对照最大初始gain*(1+1e-6)、source类计数先验、R类平均计数、moving known/birth-first。使用proposal_block_size256的完整候选分块接口，无采样、无target标签、无阈值网格。内存改善不解决二次计算时间；耗时/收敛仍须实际验证。

第三阶段在source/容量均完成后才能独立启动：固定K2与估计K共享该prior、seed1、ResNet50、Q8（C6+论文固定K2）、保持encoder BN、原发布代码损失/伪标签/argmax/初始化，初期各10轮。这是跨任务短程候选验证，不是已确认论文VisDA总预算或解决VGG表标题矛盾。K仅传整数，不用IMP中心初始化头；估计组保持Q8以隔离头容量，不按估计K自动改Q。

K0原样记录失败，不强制K1；K2与固定组方法输入相同，仅运行一个arm，不伪称独立容量提升；保护上限或未收敛必须保留失败，不当作真实未知数。不同于原论文增加的source3轮及BN模式明确保留在版本描述。

不按target指标挑seed/阈值/K或扩预算。最终保留全部best/final OS*/UNK/HOS及已知/未知取舍；best若使用target标签选epoch则明确oracle探索，非无偏验证或统计显著性。先按阶段验证，失败定位直接原因，不自动更改研究设置重跑。

source launcher：scripts/run_visda_frozenbn_source_colab.py（1fd9885f）；训练入口单独部署为/content/train_visda_frozenbn_source_v1.py，未替换历史OfficeHome入口。日志间隔200步只影响输出，默认log_every0保持旧行为。计划输出/content/imp-runs/visda-frozenbn-capacity-10e-v1。当前只有脚本已部署，source/容量/RTA均未开始。

推断入口已准备scripts/infer_visda_source_cost_colab.py：全量shape与C6/source3检查后调用既定分块接口，保存普通capacity.json及中心/分配capacity.npz，以便必要诊断，不做checkpoint前向。核心将隔离部署到/content/visda-capacity-code-v1，避免误用历史OfficeHome的旧模块；当前推断尚未运行。K结果如为0或不收敛仍保留失败，不据此改规则。

实际进展：source3为exec164，首轮完成、第二轮训练中；全量数据已恢复、无标签target列表准备完成。推断入口及分块核心已隔离部署，容量/RTA未启动。

后续RTA独立入口scripts/run_visda_frozenbn_rta_colab.py已准备并部署：fixed2/estimated单arm、共享source3、seed1、Q8、保持encoder BN、原版初始化/损失/筛选/预测、10轮；不传簇身份。必须先取得完整source summary和收敛正K，K2相同时不重复同一arm；K0/失败不强制改K。类别映射使用仓库class-map.json，ResNet口径和原总预算不伪称已解决。当前source正在第3轮，容量及RTA均未执行。结构版OfficeHome为另一个实验，不能混称K-only。

实际进展：exec164完整完成source3（3738步，最终source CE0.080924、训练accuracy0.97492）及全量79765×256 source/55388×256 target特征，不能当target成绩。普通配置/日志/summary已保存并下载pipeline-results/visda-frozenbn-source3-v1-results.zip。CPU容量推断独立exec166已启动，日志capacity-console.log，尚无K结果；GPU交给OfficeHome结构对照，不同时运行两项GPU训练。VisDA RTA尚未启动。

排程修正（未见VisDA K或目标成绩前）：OfficeHome结构10轮已完整结束，GPU释放；fixed2控制只依赖完整source prior，不依赖估计K，因此允许CPU推断仍在运行时独立启动该控制，避免无谓空置GPU。研究参数不变，不串联两项GPU任务。estimated仍须完成收敛正K，K2则复用等价控制，不强制K1；该排程不表示估计已成功。
