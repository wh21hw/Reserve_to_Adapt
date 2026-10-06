# 从新簇数到未知容量：必须先核清身份

## 当前假设与范围

CE=0实验已完成且拒识严重退化，不能删掉未知监督。下一步沿用source3共享C10/frozen-BN/seed3/ResNet特征，在CPU上检验现有容量推断是否把domain shift当成unknown。没有新训练、图像前向、参数网格或目标标签选阈值。

这次既有规则为source-calibrated-birth-cost-v1（source70%锚点、15%校准建簇代价，RNG2026、类内距离99%上界、source样本数作为先验强度、移动已知中心）。它是已有版本，不是原粘贴IMP的alpha公式或最初融合版prior_strength5版本。不能把不同规则/BN/seed下的K变化称为单因素算法改善。

## 1. 原身份规则的实际问题

同一份刚完成的A→W预热特征，推断K13，source中心中5个没有target成员；共18个实际有成员的target簇。已知候选率79.3220%，未知候选率100%，已知身份准确率20.6780%。候选簇并非语义未知：例如簇19含24张已知类3、没有未知图；簇16含33张已知和1张未知；簇18含27张已知和2张未知。

原规则中的“新原型出生”说明旧锚点不能解释当前特征，不足以说明它是source没有的类别。强source先验下，即使允许中心移动，仍可能保留空锚点，同时另建目标已知类的域偏移簇。

实现scripts/probe_a2w_imp_structure_colab.py。先保存capacity.json/clusters.npz，再隔离读取真实target标签生成事后组成诊断。target标签不反馈估计器；没有按诊断结果手动删除簇或改K。

## 2. 一个明确假设下的身份重识别候选

假设：全部C个已知类出现在target，一个已知语义对应一个target簇。对于非空簇，用source C维分类器的平均负对数条件概率做C×簇数代价矩阵，Hungarian一对一分配已知身份。没有成员的source锚点不重复算target结构；未匹配簇作为未知容量候选。算法只接收cluster IDs、C维logits；不接收target truth。

实现prototype_identity_reconciliation.py、scripts/probe_a2w_identity_reconciliation_colab.py。用已缓存normalized bottleneck和真实head的BN→LeakyReLU→fc→temperature恢复logits，CPU运行一次实际接口检查；没有遗漏BN，没有重提图像特征。

| 身份口径 | K | 已知被分入未知候选 | 未知进入候选 | 已知身份准确率 |
| --- | ---: | ---: | ---: | ---: |
| 新建簇直接当候选未知 | 13 | 79.3220% | 100.0000% | 20.6780% |
| 非空簇先匹配已知身份 | 8 | 26.1017% | 92.9368% | 71.1864% |

后者未知被匹配为已知7.0632%。这是簇身份诊断，不是RTA分类成绩；K8仍不是恢复真实11个未知语义的证明。表格比较的是同一簇划分下的身份规则，没有重新聚类、source训练或改损失。

## 未解决的风险

- 仍有26.1%的已知落入未知候选，不能把所有剩余簇直接提供强未知监督。
- 一对一假设会把同一个已知类的碎裂簇留下为unknown：类3有旧锚点簇3的3张及候选簇19的24张，匹配选择前者，后者仍落入未知。类7也存在类似碎裂。
- 不保证source类全部在target中，或每类真的只有一个几何簇；换成部分/通用域适应时强制匹配会错误吸收未知簇。
- 同一source网络提供几何和分类知识，不是两份独立证据；未知分类器可能仍自信预测某个已知类。
- 没有按target组成选matching cost、阈值、K或训练标签。不能从事后百分比直接挑方法当无偏验证。

## 下一项训练应怎样解耦

保留原未知CE，不同时加保护门控、新损失或改预测。若继续训练，先预声明两个同seed/source/BN/Q20/10轮的K8 arm：仅使用候选K8的原argmax对照，与K8+固定簇身份桥接候选。这样区分容量和身份监督；已有K2控制仅帮助描述容量变化，不能与身份版本的双因素差异混称单因素。

artifact已保存但未输入任何训练。匹配在source3空间完成，后续warm-end仍需将未知簇排列对应RTA未知槽；不能用真实标签对齐。若未知簇漏进warm-end批次，应明确报错，不强造身份。固定离线簇会随训练过时，后续再单独检验交替重估，不在首个结构对照里同时加入。

更根本的路线仍是known条件对齐→目标结构发现→known身份保留→unknown结构监督。当前证据说明“更多空间”不是充足条件，身份与域偏移处理必须一起被解释。训练前还需明确新的GPU批次预算；无人值守autoresearch循环的pause_every/max_iterations与评价规则未配置，本次没有开启无限循环。

## 实际保存

运行根目录/content/imp-runs/a2w-unknown-ce-10e-v1下两个probe目录；无标签clusters.npz和普通报告已实际复制到Drive OSDA/runs/a2w-unknown-ce-10e-v1。CPU推断结果下载到pipeline-results/a2w-imp-structure-{capacity,posthoc}-v1.json和a2w-identity-reconciliation-posthoc-v1.json。已有完整RTA控制/关闭CE结果均保留，不重复训练或重评checkpoint。

2026-10-05后续：两个K8 arm的隔离launcher和日志collector已准备（scripts/run_a2w_reconciled_identity_colab.py、collect_a2w_reconciled_identity_colab.py），默认prepare-only，明确--run才会训练。WSL Python已做一次AST语法检查，尚未在真实runtime编译身份patch或跑训练。用户已同意后续做实验，但要求先关闭所有实例节省积分；当前不启动这两组。之后恢复同一source checkpoint与无标签artifact，不重新训练source或根据目标标签改K。

当前GPU策略由用户更新为能用T4就用T4，不再强制至少L4。恢复时先依据实际显存决定设备；这会与历史L4结果产生环境差异，需披露。K8两组必须用同一种GPU，不能一组T4一组L4后声称干净消融；若T4不够，不通过偷偷减batch改变研究设置。

2026-10-06：用户明确恢复研究，开始恢复本批T4，端点gpu-t4-s-kkb-usw4a1-1aub4wll4xfhh、shell7。Python3.8旧版依赖安装完成（与前批相同版本），尚未训练或测试实际显存。mount exec1因Google授权超时而终止，不是OOM/模型失败，不据此换GPU。待完成挂载后用scripts/restore_a2w_reconciled_inputs_colab.py恢复Drive缓存与共享prior，无需重训source。两组manifest新增实际GPU/软件版本；旧K2是L4结果，只作背景，不能与新T4 K8构成干净容量消融。collector保存best及last再释放runtime。

用户随后完成授权，mount exec2正常done。完整缓存、共同source-final.pt、固定无标签artifact恢复成功；identity实际入口的build-only编译检查通过一次，没有模型训练/重复smoke。已开始argmax的正式10轮训练（shell7），随后同一T4运行identity，不自动改K、loss系数或batch。训练实现29c0f6f0；尚无本批最终结果。

## T4配对10轮已完成：本版未改善

2026-10-06，两组各恰好10轮，loss有限，无OOM。相同Tesla T4、Python3.8.20、torch1.7.1+cu110、numpy1.23.4、batch64、seed3、共同source3/frozen BN/K8/Q20；只切换covered unknown候选的簇身份伪标签。未扩大预算、改阈值或挑新seed。

| 标签机制 | 选取 | epoch | OS*% | UNK% | HOS% |
| --- | --- | ---: | ---: | ---: | ---: |
| 原未知argmax | best=final | 10 | 96.3023 | 78.2380 | 86.3354 |
| 固定簇身份 | best | 7 | 93.8480 | 78.8850 | 85.7184 |
| 固定簇身份 | final | 10 | 95.0805 | 77.8989 | 85.6364 |

final身份−argmax：OS* −1.2218pp、UNK −0.3391pp、HOS −0.6990pp；best HOS −0.6169pp。单seed短程结果没有支持该版改善，保留为负面探索，不晋升默认方法或删除记录。best仍用目标标签选择epoch，seed3事后选择的历史披露保留。生效前四轮最大评价差异0.7273pp，不能将这次小差值称统计显著或严格确定性因果效应。

簇标签实际覆盖1304/1344次未知标签曝光（97.0238%）；回退40次。8个未知槽累计标签曝光为[220,115,3,4,407,445,0,150]。该统计计重复训练曝光，不是唯一图像数。原argmax未记录同口径槽计数，不宣称身份版改善相对均衡。

### 原筛选机制拒绝了部分误建簇

训练完成后，scripts/probe_a2w_identity_usage_posthoc_colab.py只读取无标签artifact、训练计数和独立target标签，生成事后风险报告，没有重新前向模型或反馈改配置。

- 候选身份15实际含24张已知类3、零未知，映射未知槽16；这个槽全程获得0次伪标签。该实例说明RTA筛选挡住了这一误建簇，不能据IMP误候选率直接宣称这些已知都被未知CE拉走。
- 候选身份17含28张已知、2张未知，映射槽12累计3次标签；候选身份13含15张未知、映射槽13累计4次。混合簇内曝光的真实组成无法由汇总槽计数精确恢复，不按总体组成比例伪造训练误标签数量。
- 纯已知候选簇的保守错误标签曝光下界为0；这不证明所有候选都干净。计算只对纯已知簇对应槽计数，减掉全部fallback作为最坏情形，混合簇不计入下界。

因此，当前问题不只是“缺簇标签接口”。增加并传递离线结构没有带来收益，部分容量不符合RTA实际采信的unknown证据；冻结在source3空间的簇还可能随特征学习过时。本批不能区分这两种机制的全部贡献，更不能简单归罪所有未知筛选都错误。

### 短程预算边界与下一假设

核对发布代码：前段使用高KL候选，候选较多时每batch保留top16；`epoch<=10`为这一规则，随后改为最高KL GMM簇。10轮只覆盖第一段，不能据此断言70轮完整训练也一定无效，或无视后续筛选切换。

下一假设应让容量推断与实际学习的表示/unknown证据一致：在一个共同、预声明的早期节点重新提取当前source/target特征，用RTA的无标签已知兼容性作为候选证据，再估K；与同checkpoint/预算固定K对照分开检验。GMM兼容性只是经验置信度，不能包装成严格贝叶斯语义后验。候选不足返回0/未确定，不能按真实target标签删类或强造K；动态K和簇监督仍分步，不一次加门控、初始化、新loss和新预测。

当前没有据此自动开启新实验或延长到70轮。两组best/last及普通日志ZIP已经实际保存Drive OSDA/runs/a2w-reconciled-identity-10e-v1；ZIP、summary与事后风险JSON已下载pipeline-results。T4已销毁，CLI确认无在线runtime、消耗0 CCU/hr，余额223.1991 CCU。完整研究目标仍未完成，本批完成不等于毕业设计方案已验证。
