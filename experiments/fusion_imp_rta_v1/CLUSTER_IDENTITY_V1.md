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
