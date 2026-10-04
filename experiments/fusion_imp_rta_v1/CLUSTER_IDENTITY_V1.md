# 簇身份传递：与只传 K 解耦的新增版本

用户2026-10-04授权继续。目标：固定相同K，检验让RTA学习已发现的簇身份是否优于仅分配容量。不把此版本称为“仅自适应K”。

首个对照计划复用OfficeHome Pr→Rw frozen-BN K2 seed1、source C25 CE3和RTA10轮的已完成控制。Q29、原版虚拟方向、head warm-end K-means初始化、未知筛选、CE权重、最终argmax全部不变。禁止IMP原型初始化分类头、未知概率聚合或新损失一起加入。

新增机制：固定source预热特征下的未知簇样本身份；训练数据携带样本索引。原版warm-end初始化未知头后，用该轮现有target特征按初始簇身份求均值，与未知head方向做余弦Hungarian一对一匹配，只确定簇ID到槽ID的排列，不写head参数。之后仅对RTA原机制筛出的候选，用其未知簇ID替代未知槽argmax伪标签。初始被分到已知簇或noise的候选仍使用原版argmax，不增删候选，不能声称解决未知筛选错误。K2簇固定不交替，特征漂移导致簇过时是本版待检验问题。

这是固定离线簇身份的结构注入，不是语义真标签。目标标签只作评价；簇身份与head对应完全无目标语义标签。匹配开始前的warmup保持原行为。保存映射、每轮覆盖/回退数量、各槽伪标签次数及best/final OS*/UNK/HOS，不能只报HOS。

OfficeHome旧容量结果未保存assignments，因此需要从同一份已完成source features重建一次同规则簇身份；必须得到已有K2和相同设置，否则保留差异并停止该对照，不据目标成绩改估计规则。此步骤为补齐缺失研究输入，不重评checkpoint、不扩参数网格。artifact携带target路径顺序，不含目标标签。

当前：实现已准备，尚未功能检查、重建簇artifact或启动训练。VisDA source exec164正在运行，不排队启动结构对照。接口变化仅做一次针对性检查，成功后在GPU空闲时分阶段执行。

进展：一次CPU针对性检查通过，验证样本索引随shuffle/cycling传递、簇到槽排列、已知/noise回退及entry插入。实际task-adapted入口集成编译也通过，无模型训练。旧OfficeHome同source features恢复得到K2、未知支持1152/62、noise57，源半径0.5785163178276425及成本0.1281749691242145，与已有记录一致；已保存/content/imp-runs/officehome-cluster-identity-10e-v1/clusters.npz（固定身份与无标签路径顺序）。未修改历史控制入口，结构版隔离部署于/content/cluster-identity-code-v1。结构训练尚未启动。

汇总入口scripts/collect_cluster_identity_colab.py已准备：读取两组现有日志，记录best/final与逐轮差值、映射/覆盖率/槽计数；不重新前向checkpoint。旧argmax控制没有槽计数，因此不可宣称结构组改善了相对槽均衡。报告前四轮warmup指标差异，以暴露新增模块生效前可能的控制偏差，不把未见偏差当确定性证明。训练未完成时不执行collector。

部署路径修正：隔离entry的PYTHONPATH包含/content，以加载既有task_protocol与encoder_bn_policy，不写原历史入口或改研究设置。训练仍未启动；等待exec164释放GPU。

实际启动：exec164预热/特征提取完整结束后，结构版独立shell4启动scripts/run_officehome_cluster_identity_colab.py，输出/content/imp-runs/officehome-cluster-identity-10e-v1/rta。已观察训练入口进程及GPU99%工作，尚无完整epoch指标，不能提前判断改进。独立shell3可读日志/状态；VisDA容量exec166仅CPU，不抢GPU。失败先保留证据并诊断，不自动改簇标签或重跑。

中途诊断（非选配置）：warm-end成功一对一匹配[26,25]。第5轮候选1088、覆盖728、槽计数64/1024，HOS73.815%（控制同轮71.843%），不是最终结论。第6轮覆盖717、槽计数0/1088，说明传身份不保证所有簇得到未知监督。

为解释小簇未获监督，独立读取评测标签作事后组成诊断，未输入估计/训练/阈值选择：簇25支持1152，其中已知150、未知1002，混合许多语义类；簇26支持62，其中已知61（原类0有60）、未知1。保存posthoc-cluster-composition.json及本地pipeline-results/officehome-cluster-identity-posthoc-composition-v1.json。小簇未被RTA未知筛选接受，不能直接归咎筛选漏检；估计器将已知域偏移解释成新类的可能性得到直接支持。K2依然是容量，不是两个真实未知类别。此观察不授权按目标标签删除簇、强制小簇未知监督或中途改参数；当前10轮继续完成。

## 完整10轮结果

两组best均在第10轮，与final相同。原argmax：OS*69.4019%、UNK85.5471%、HOS76.6334%；传簇身份：OS*71.8911%、UNK84.2510%、HOS77.5819%。差值+2.4892/-1.2961/+0.9485个百分点。仅一个seed的探索性短程对照，best使用目标标签；生效前四轮最大指标差异0.1840个百分点，非逐位确定性复现或统计显著性证明。

实际候选6528，簇标签4398（67.3713%），累计两个槽伪标签64/6464；第6至10轮仅一个槽获标签。支持结构传递可能改善已知/未知权衡，但不能声称均衡槽、恢复语义类别或解决误建簇。完整普通日志/summary下载到pipeline-results/officehome-cluster-identity-10e-v1-results.zip与对应summary.json。

无标签已知兼容性诊断初版遗漏head的BN与LeakyReLU，旧unlabeled-known-compatibility.json及其预测/概率/KL数值无效，已撤回，保留旧报告为错误证据；不能据此声称分类器近均匀。该错误不涉及训练或直接计数的簇组成，也不改变10轮对照结果。

修正版本使用已有瓶颈特征及source checkpoint的head BN统计/affine，依实际BN→LeakyReLU(.2)→fc→softmax计算；8行已保存source特征与实际head的一次针对性等价检查通过，无图像/encoder前向、无目标语义标签输入。head-v2报告：小簇91.9355%预测已知类0，大簇最大集中度15.5382%；平均max概率0.2410/0.1225，平均KL0.31043/0.52206。源域描述性KL99%界0.63695以内分别98.3871%/67.3611%，说明小簇已知兼容性有无标签信号，但简单全局KL界仍接受大量未知成分；源域同样本分位数不是目标覆盖保证。未据该报告删簇、调阈值或启动新训练。

修正结果保存unlabeled-known-compatibility-head-v2.json及pipeline-results/officehome-unlabeled-known-compatibility-head-v2.json。当前簇到槽匹配是预head瓶颈均值与fc方向的余弦启发式，沿用RTA原型/权重口径；它并不等同于完整head输出的最佳对应，未来若改成head-aware匹配需独立消融，不偷偷改已完成版本。
