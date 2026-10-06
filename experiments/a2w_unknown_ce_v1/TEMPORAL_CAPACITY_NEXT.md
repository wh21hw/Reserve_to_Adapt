# 下一单因素：结构推断的时机，而非重复门控规则

当前完整研究目标未完成。本轮是已完成K8身份对照之后的代码准备，不是新实验成绩；没有创建Colab实例或启动GPU。

## 必须保留的已有负面证据

1. A→W关闭未知CE，final UNK从86.06%降到3.33%，不是修复。
2. A→W K8固定早期簇身份，final HOS比原argmax低0.6990pp；近97%候选用到标签，不是模块没有接通。
3. 纯已知误建簇对应槽全程0伪标签，说明RTA筛选可以挡住部分IMP伪未知，不能为了满槽强行补监督。
4. CONDITIONAL_DPMEANS.md已有门控后聚类探针：A→W先前不同prior/seed下K2候选混合且污染；source留类代理也失败。OFFICEHOME_RTA_10E.md的初始IMP交集训练HOS仅+0.0082pp。不能把这些旧方案重新命名为下一新方法。

## 当前可检验因素

比较source预热结束的状态，与固定预声明RTA节点的状态，在完全相同的结构估计规则下，K、成员分配与已知/未知证据是否更一致。两个状态的表示及分类头均可能变化，不能叫仅encoder变化；这首先是时机/状态诊断，不是容量对训练成绩的因果实验。

先使用本批原argmax组的固定final10（C10+K8），而不是挑best epoch7或某一较好seed。它能直接复用已持久保存模型，不需要重训。source与target都用同一固定网络一次提取，避免跨阶段中心/特征混搭。GMM分量数不是未知语义数，关系权重不是严格Bayes语义后验。

## 已准备的缓存入口

scripts/cache_a2w_current_relation_colab.py：原ResNet50和完整BN→LeakyReLU→fc分类头；eval/no_grad；source标签允许用于source模板；target标签在读列表边界丢弃。保存全量958/564的256维特征、18维logits、source标签、target路径、C维条件分布关系KL及source关系模板。没有GMM拟合、K选择、未知训练标签或新的损失。

它重建的是当前冻结模型下的source概率均值，不是历史训练增广/在线关系库的精确回放。与已保存source3状态比较时必须披露这个边界。每个模型只创建一次缓存，输出存在则拒绝重复前向。缓存实际保存到Drive后才可销毁runtime；当前没有缓存产出，不写成已执行。

本地AST语法检查通过；真实checkpoint接口检查和图像提取尚未执行。下次恢复Colab环境时补传relation_gate.py，复用Drive的argmax-last.pt和Office31压缩包；优先T4。只做一次新接口功能检查，不重复稳定代码smoke或hash。

## 后续如何回到训练验证

只有时机诊断给出值得检验的容量规则后，才声明同checkpoint、共同早期训练预算的固定K/估计K训练对照。重建分类头/optimizer时保留已知身份和可对应未知状态，沿用已有交替K接口；不要顺便加入新标签、门控loss、IMP权重初始化或概率聚合，并称K-only。

当前10轮只覆盖发布代码前段top16筛选；未来完整结论必须跨过筛选切换并按完整训练预算验证。OfficeHome与VisDA外部验证、VisDA backbone口径、论文对齐以及多seed都仍未完成，不以A→W一次小实验代替毕业设计目标。

无人值守循环和统一evaluator/keep policy尚未确认，已按autoresearch技能发出配置问题；不默认启动无限搜索。当前GPU保持关闭，不因自动续行创建闲置实例。

## 离线比较器已完成，尚未跑真实缓存

scripts/compare_a2w_temporal_capacity_colab.py复用已保存的source3 raw K13/身份匹配K8两个无标签artifact，只对final10缓存执行一次原source-calibrated birth-cost规则，再按同一head-likelihood身份匹配。source99%半径/建簇成本根据当前source重新标定，没有target阈值扫描；因此是同规则跨状态比较，不是数值lambda固定的比较。

temporal_partition_report.py输出K之外的划分变化：只在两边都非noise的样本上计算ARI（不受簇编号排列影响），同时报告共同覆盖、noise变化、未知候选身份变化及共同known上的身份一致率。不能将任意簇ID换位当结构失效，或把减少K当改善；不足两个共同样本时ARI返回未定义，不强填一个好分数。若实际有成员的簇数不足C，保留原推断但身份匹配报告不可用，不强造K1。

一次新统计接口功能检查已在本地CPU完成：编号换位ARI1、交叉重新分组ARI−0.5、noise与共同样本不足分支正确。初次断言因浮点严格等号失败，改用容差断言通过，算法没有因此改动。比较入口AST语法通过；没有重新测试稳定聚类核心、没有GPU或真实checkpoint前向。

当前真实final10特征缓存尚未生成，所以没有新的K、漂移程度或模型效果结论。下一实际动作仍是复用Drive final10模型生成一次冻结缓存，然后运行此纯CPU比较器；不再次训练source或RTA，不把准备完成写成实验完成。需上传source_precision_capacity.py、robust_capacity.py、prototype_identity_reconciliation.py、temporal_partition_report.py及relation_gate.py到运行环境，缓存持久保存后及时关闭实例。

## 实际恢复进展

2026-10-06，通过CLI检查确认无在线runtime；Drive直读API仍返回共享client项目202264815644的Queries quota错误，不是用户Drive磁盘容量不足。仅需一次前向，本批改用标准CPU实例m-s-kkb-usw4a1-1tx31818s2ldc，实测0.08 CCU/hr，shell9；挂载exec1已请求用户授权。CPU推理会比GPU慢，但没有新训练或模型设置改变。

原ResNet源码get_mean/get_std强制.cuda()，缓存入口已在推理设备上显式分配相同float32 ImageNet常数；仅设备分配兼容修复，不改变架构、权重或归一化数值。恢复入口scripts/restore_a2w_temporal_snapshot_colab.py只读取既有Drive数据/权重/argmax-last.pt，未重训source。首个setup文件上传fetch failed，已成功重传；之前python因此找不到文件，未发生任何训练/提取，重新运行依赖安装不改变实验设置。

代码/辅助包已上传。当前等待Drive授权与依赖安装完成，尚未提取final10缓存、估计当前K或产生漂移数值；不要从准备过程声称假设已验证。完成后保存小缓存、离线比较结果到Drive并关闭CPU实例。

随后实查：依赖安装完成，但mount exec1在2m2s以Authorization timed out终止，未读取模型或数据。不存在新提取结果需要保存；既有checkpoint/数据/结构artifact仍在Drive，代码在Git。为避免空耗，已请求销毁该CPU实例，不连续刷新授权或自动新建runtime。上述授权URL与端点不再可用；下一次需要人在线及时完成新挂载授权后才能实际提取缓存。这是访问阻碍，不是模型失败或新结果。

## 授权等待窗口的直接原因已修复

只读检查发现本机Murphy CLI的src/daemon/server.ts及dist/daemon/server.js将后台授权轮询窗口硬编码AUTH_POLL_TIMEOUT_MS=120_000。超时由daemon主动发送Authorization timed out，因此Python _message的900秒设置不足以延长它。CLI exec --help也没有授权等待时长选项。

现已仅将这两处本机常量改为900_000（15分钟），保留原5秒轮询、凭证校验、人工同意流程和其他既有CLI修改。Node --check通过；重新读取两处常量确认修改。现有后台进程没有被粗暴终止，新启动daemon使用新设置；尚未以真实Google授权验证15分钟完整流程或Google侧URL有效期，不能声称OAuth链接保证15分钟有效。

保留最小修复记录scripts/colab-background-auth-wait.patch，便于重建CLI时恢复。没有修改RTA模型、数据、训练设置或已有结果，也未新开实例来测试等待计时。真实缓存提取仍需人完成新的Drive授权；本次进展是消除过短的本地等待窗口，不是产生模型实验成绩。

## 2026-10-06 人在线恢复成功

新标准CPU端点m-s-kkb-use1b1-3hr6nb4xjabhs，mount exec1在用户完成同意后实际返回Mounted at /content/drive及IMP_DRIVE_MOUNTED。依赖已安装，代码、Drive缓存Office31数据与预训练权重恢复完成；真实argmax-last.pt的final10/C10/K8接口检查通过，无图像前向或重复评估。shell10已启动一次CPU冻结缓存提取，日志/content/a2w-current-cache-console.log；尚未产生当前K/漂移结果，没有重训。

用户最新明确要求保留实例，空闲希望CPU，不要反复关闭导致授权；当前已是CPU，将保留该实例及挂载，不执行先前文档中本小诊断结束即销毁的安排。CLI runtime命令当前没有原地切换硬件接口，不能承诺GPU转CPU保留原VM/挂载。新约定同步AGENTS.md。

## 实际诊断完成：K与成员归属都变了，但不是全方向改善

冻结缓存已完整生成，CPU前向406.24秒；958×256 source、564×256 target、完整18维logits、source关系模板均有限。没有训练、重评checkpoint准确率或GMM拟合。缓存先实际复制到Drive `OSDA/runs/a2w-current-relation-snapshot-v1`，然后只读取缓存进行容量推断；小缓存与manifest亦已下载本地pipeline-results。CPU实例与挂载按用户最新要求保留。

预热source3和RTA final10遵循同一source-calibrated-birth-cost规则：source分层随机划分seed2026、667个anchor样本、144个birth-cost样本、相同类先验计数/总质量、移动已知中心、birth-first、全target564张、proposal_block_size64。没有target标签参与阈值、K、身份匹配或配置选择。初次结果保存Drive后，独立事后评估才读取target truth。

| 模型状态 | 原始新建簇K | 匹配已知后K | 有成员的总簇数 | 已知误入未知候选 | 未知进入未知候选 | 未知被已知簇吸收 | 已知身份准确率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| source3 | 13 | 8 | 18 | 26.1017% | 92.9368% | 7.0632% | 71.1864% |
| RTA final10 | 4 | 4 | 14 | 7.7966% | 87.3606% | 12.6394% | 91.8644% |

两边全564张共同有分配，noise为0，划分ARI=0.673826（忽略任意簇编号置换）。匹配已知身份后87/564=15.4255%改变已知/未知候选状态；共同known样本身份一致率97.3684%。这说明早期结构并非在训练后原封不动，但不证明它是固定簇监督退化的原因：本次提取的是argmax控制组final10，不是identity组的完整训练轨迹。

当前4个候选未知簇大小68/82/79/29，分别含0/9/5/9个真实已知样本。第一个簇混合了未知语义20、25、26；另外三个也有多种未知语义或已知混入。10个已知身份均匹配到原已知锚点，不再需要像source3那样用新簇补回空锚点。然而34/269个未知被已知簇吸收，source3是19/269。K4不是恢复了真实未知语义数，也不是无代价的已知保护。

这些是簇结构诊断，不是新的OS*/UNK/HOS训练成绩；不要把87.3606%当RTA的UNK准确率。目标真值仅用于事后解释已经固定的结果，不回流到K估计。

## 新发现：数值标定本身强烈改变容量

同规则跨状态比较并不固定数值lambda。source半径从0.566554变为0.938595，建簇成本从0.189747变为0.368069。因此不能把K减少直接归因为更好对齐或未知只剩4个语义。

追加一个明确单因素离线控制：保持当前final10的source/target特征、当前source anchors/先验、完整样本、当前分类头匹配规则不变，仅将半径/建簇成本这一个标定块换成已保存source3数值；不扫阈值、不新增前向/训练、不看目标HOS选配置。

| 特征及分类头状态 | 数值标定块 | 原始K | 匹配后K |
| --- | --- | ---: | ---: |
| source3 | source3标定 | 13 | 8 |
| final10 | source3标定（离线控制） | 9 | 9 |
| final10 | final10 source重新标定 | 4 | 4 |

同一final10空间上K9与K4的区别说明标定块对推断结果有直接影响；不能将二者差值称为未知语义数量变化。控制K9有14个noise，共同分配550/564，ARI=0.867131；它不是自动晋升的替代方案，也没有做其训练或事后质量择优。当前代码没有显式保证归一化bottleneck的source类内距离在RTA过程中缩小；当前半径上升提示这一几何假设需要实证，不等于分类头已知准确率必然下降。

## 与研究目标的关系与下一步

1. 继续保留原未知CE：先前关闭CE使拒识消失，本轮没有改变该结论。
2. 不再把source3的簇身份当永久正确监督。当前表示下已知身份更清楚，值得单独检验更新时机；但未知仍有合并/被吸收，不能只追求K下降。
3. 若训练验证，最干净的下一对照是相同初始K8、相同source/早期训练预算，原机制继续固定K8 versus 预声明节点按同规则更新K；只改变容量，不同时加簇伪标签、保护loss、概率聚合或IMP头权重初始化。当前节点得到K4只能作为这一规则的实际输出，不按真实11类改成K11，也不因离线控制得到K9就择优调阈值。
4. 仅将argmax final10 checkpoint当共同起点不自动等于精确训练续跑：虽然已保存三个optimizer和discriminator，训练关系库/GMM、调度步数与随机状态也需有明确恢复口径。应优先在一次新配对训练中共享同一早期状态再分支，或者完整披露共同近似恢复，不能静默重置后声称严格反事实。
5. OfficeHome/VisDA与论文完整预算验证仍未完成。本次不是提升证明，不能以A→W一次缓存诊断完成整个研究目标。

实际输出：`pipeline-results/a2w-temporal-capacity-v1-summary.json`、`a2w-temporal-capacity-posthoc-v1.json`、`a2w-temporal-calibration-v1-summary.json`，对应clusters以及完整final10缓存本地/Drive均保留。CPU与之前CUDA提取可能存在数值差异；同空间K9/K4控制都使用同一CPU缓存，不受跨设备前向差异影响。代码入口分别为compare_a2w_temporal_capacity_colab.py、score_a2w_temporal_structure_posthoc_colab.py、probe_a2w_temporal_calibration_colab.py。

正式无人值守autoresearch循环仍未启动。为后续训练明确询问逐组人工review/无人值守预算，以及人工综合评价/已有机械evaluator；在答复前不自行扩大矩阵或长时自动搜索。本次已完成诊断和保存不受此限制。
