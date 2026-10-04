# OfficeHome 容量配对实验（探索性10轮）

同source prior：C25监督3轮、seed1，/content/imp-runs/source-precision-officehome-v1/seed1/source/source-final.pt。ResNet50，L4，Python3.8/torch1.7.1；batch64，SGD backbone5e-5、分类器/判别器5e-4，原发布代码warmiter3、损失/未知自伪标签、argmax预测保持；warm-end未知头仍为原版C+K目标K-means初始化，不使用估计簇中心初始化。

配对：fixed4（C+K=29）vs estimated（source规则K1，C+K=26）。两组Q29固定、原版Hungarian未匹配方向V机制，前置source checkpoint和训练预算完全共享，只K不同。原始RTA随机头构造及warm-end聚类策略一致，不声称不同头尺寸的随机数轨迹逐位相同。总训练各10轮，不和70轮best混比。

K估计不使用target标签；指标使用target原标签做macro OS*/UNK/HOS评测及oracle-best epoch，报告best和final。这只是OfficeHome工程协议：Q29/K4与70轮原预算口径未确认作者设置，不能称严格论文baseline复现。新估计K1不是未知语义类数恢复成功；source控制有误聚/合并，仍需检验是否是有用容量。

2026-10-04：入口构建检查通过，修复task_protocol Python3.8路径兼容，取消任务列表hash字段，不新增审计门槛。已在既有L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o启动fixed4，后台exec90；console.log已确认TASK_KONLY_START C25/K4/Q29及共享source路径。尚未观察首轮loss。estimated未启动，不能排队新exec。输出/content/imp-runs/officehome-capacity-10e-v1/fixed4/，训练子目录officehome-pr2rw_seed1。

后续：先观察fixed4 loss有限/实际epochs，再独立启动estimated；两组完成后下载逐轮指标与launch.json，汇总best/final，不自动重跑或改设置。正常训练用独立shell读取console.log/history.jsonl或exec attach90，不排队kernel检查。

最新进度：exec90仍running；console已输出原训练epoch0（history记为完成epoch1），OS*=0.817、UNK/HOS=0，ce0.859、virtual0.944、ce_ep2.220、adv0.788，loss通过有限值保护。尚处warmup，不能以UNK0判失败或提前调整。collector collect_officehome_capacity_colab.py已准备，要求两组均完整10轮，仅打包普通日志/配置/指标，不传checkpoint或重评价；estimated尚未启动。

后续检查：fixed4已完成4/10轮；发布代码warmiter3配合零基epoch<=3，实际前4轮warmup，第4轮末初始化未知头。console epoch3已得到OS*79.8%、UNK29.1%、HOS42.7%，先前三轮UNK0不代表失败。进程仍running，L4利用率100%，显存13440MiB，未见OOM或非有限loss。该值只是进行中指标，不是最终对照结论；estimated仍未启动。

最新状态：exec90正常结束，returncode=0，history确认恰好epoch1..10。fixed4的best和final均为完成第10轮：OS*=70.58996%、UNK=77.76500%、HOS=74.00397%。best按target评测HOS选epoch，仅为单seed短程探索结果。

已独立启动estimated，后台exec92，使用原先声明的同source prior/seed1/Q29/10轮设置，K从source-cost-target-capacity.json读取（既有估计为1），未调参或改变损失。需观察启动与首轮，再等待完整10轮，运行collector做配对汇总；目前尚不能判断估计K优劣。

exec92启动日志确认C25/K1/Q29和共享source路径，现已完成2/10轮，history第2轮OS*=82.37572%、UNK/HOS=0，elapsed230.21秒。console前两轮loss项有限，训练仍处原版4轮warmup，暂不能把UNK0解释为K1失败。训练子进程与exec92均在运行，未重跑或排队新kernel exec。

进行中第5轮配对：fixed4 OS*=64.55527%、UNK=76.84999%、HOS=70.16813%；estimated K1 OS*=62.23596%、UNK=86.48600%、HOS=72.38392%。K1相对fixed4分别为-2.31931、+9.63602、+2.21579个百分点。两组elapsed约526/530秒，均完成相同5轮。K1第4轮末UNK8.991%到第5轮86.486%，loss项有限，exec92仍running。这支持“一个未知槽也能学拒识”的观察，但不证明最终改善、真实类别恢复或概率分散是因果；需等完整10轮配对，不据中途结果选参。

## 完成结果与下一步

exec90/92均正常结束（returncode0）。collector确认每组完整epoch1..10、指标有限，直接读取已有history，未重评价checkpoint。日志/配置/指标/summary下载至pipeline-results/officehome-capacity-10e-v1-results.zip，并解压至同名目录。未重启runtime、重跑或改参数。

两组best均为第10轮，因此best与final相同：

| 设置 | K | best/final epoch | OS* | UNK | HOS |
| --- | --- | --- | --- | --- | --- |
| 固定容量 | 4 | 10 | 70.58996% | 77.76500% | 74.00397% |
| source估计容量 | 1 | 10 | 66.45630% | 85.74736% | 74.87931% |
| 估计−固定（百分点） | — | — | -4.13366 | +7.98236 | +0.87533 |

结论：本次单seed短程实验中，source规则K1并未导致未知拒识失败，而是提升UNK、牺牲部分OS*，HOS略高。不能宣称显著提升、恢复40个未知语义类、证明概率分散因果或对齐论文成绩；best为target标签选epoch。本实验只支持容量可能影响已知/未知权衡，不能推广到其他seed/完整70轮。

下一步优先机制诊断而非按target成绩调整K：检查已知→未知与已知类间混淆的分解，以及RTA未知筛选是否将已知样本纳入伪标签训练。若需要推理读取，仅分析一份每组final checkpoint作为新诊断，不重复逐checkpoint评分或训练。诊断指标只作解释，不回流source估计规则。先定位主要误差，再声明一次单因素优化；不要把IMP头初始化、改解码、额外损失同时叠加成K-only。

### Final误差分解（exec95完成）

每组final仅一次推理，4357张target、原center-crop/argmax，无训练或调参。结果保存pipeline-results/officehome-capacity-10e-v1/final-error-diagnostic.json。以下均为已知25类macro比例，三项相加100%：

| 设置 | 已知正确 | 已知→未知 | 已知→其他已知 |
| --- | --- | --- | --- |
| K4 | 70.58996% | 25.07948% | 4.33056% |
| K1 | 66.45630% | 31.69955% | 1.84415% |

K1已知被拒识增加6.62008个百分点，而已知类间混淆减少2.48642个百分点，二者净解释OS*下降4.13366个百分点。仅在已知25行取argmax（不是正式推理规则），K4/K1已知识别为84.13830%/85.13190%：不存在“已知类区分能力明显变差”的证据，主要观察是已知/未知竞争偏向拒识；尚不证明训练中伪标签污染的因果。

K4四个未知槽获胜样本数757/306/684/615，非空槽。未知总概率超过最大已知概率、但单未知槽未超过的样本仅87张（31真实已知、56真实未知，微观计数），不能从本诊断认定概率分散是主要原因，也不把合并概率解码自动作为修复。K1没有槽间分散。

后续优化假设应优先围绕source校准的已知保护/未知伪标签筛选，而非盲目扩大K。需先获取当前relation筛选和source兼容性之间的交叉证据，再声明单因素实验；不会用这次target标签结果挑阈值或重新估计K。任何新增gate需作为K-only之外独立方法版本，保留当前完整对照。

### 关系筛选机制诊断

exec96因实际bridge checkpoint缺少source_relation_bank直接KeyError，在target推理前失败；证据保留，无训练失败或模型改动。实际last.pt仅model/discriminator/epoch/metrics/optimizer三项状态，不能精确回放历史关系中心/GMM。工程修正为独立重建诊断：exec98在final模型center-crop source输出上重建25类关系均值，target分数拟合4分量BayesianGaussianMixture（random_state2026），用一次固定label-independent shuffle2026重现64 batch/top16机制。此为冻结当前机制快照，不代表历史训练增广或实际筛选记录。

| 设置 | 筛选总数 | 真实已知数 | 真实未知数 | 筛选中已知比例 | 全部已知被选比例 |
| --- | --- | --- | --- | --- | --- |
| K4 | 1088 | 144 | 944 | 13.23529% | 7.95141% |
| K1 | 1088 | 128 | 960 | 11.76471% | 7.06792% |

结果不支持“K1筛选污染更多”：K1误选已知比例反而较低。K1误选的128个已知均被final分类器拒识，K4为135/144；这只是同一final模型的关联，不证明训练因果。两组均存在关系筛选误选，值得独立优化，但不足以解释K1额外6.62pp的已知拒识差异。

保存relation-gate-diagnostic.json；远端保存每组final-relation-snapshot.npz，后续统计可复用不用重复推理。目标标签仅评价诊断，不参与拟合、shuffle或选阈值。下一步可以声明source-only关系兼容性保护的单因素实验，但不能因看到上述target误选率而挑保护阈值或宣称已找到因果。
