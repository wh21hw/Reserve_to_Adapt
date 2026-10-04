# 真实 Pr→Rw：冻结BN预热下的容量配对

2026-10-04。前两source留类块支持冻结encoder BN的预热改善，现在检验真实跨域RTA性能。不是严格论文复现：OfficeHome Q29/K4及原训练预算尚未确认作者设置。

新prior：完整Product25已知类1785张、C25头、seed1、source CE3轮。encoder BN统计固定为ImageNet，仿射参数/encoder权重与分类模块照常训练；head BN正常更新。其余原source warmup设置不变：batch64/drop_last/workers4，SGD momentum.9/nesterov/weight_decay5e-4，encoder/head lr5e-5/5e-4，原inverse decay，Resize256x256/RandomCrop224/HorizontalFlip。无teacher/额外loss/DINO。

冻结网络后提取source/RealWorld target归一化256维瓶颈；target只读取文件名，原语义标签不进入特征或容量拟合。沿用source70%建已知原型、15%已知校准建簇成本公式，lambda=source99%类内平方残差，beta=source负对照最大初始建簇增益*(1+1e-6)，κ_c=source训练各类计数，R=均值，birth-first、移动已知中心。source剩余15%不参与本次真实target拟合。估计一次K后固定，不调整beta/lambda或用target标签选K。

配对：固定K4 vs估计K，完全共享新prior、seed1/Q29/ResNet50/原RTA损失与伪标签/argmax/warm-end C+K目标K-means初始化，各10轮。只传K，不传IMP簇中心或标签。进入RTA后恢复原net.train/BN更新，不把预热的BN冻结延伸至域适应。原代码warmiter3实际前4轮warmup，两组一致。

K0显式记录估计失败，不强制K1；K4则两组输入相同，无需重复训练伪称独立改进。与旧普通BN的固定K4/K1结果仅作预热因素背景，不混报成纯K改变或同一prior。若新估计与旧估计不同，两种预热的estimated结果不作为单因素BN性能对照。

报告各10轮内best和第10轮final OS*/UNK/HOS，best为target标签选epoch，单seed探索，不称三seed均值/显著提升。无70轮或参数扩网格授权；先判断同预算性能与已知/未知权衡。

L4原Py3.8/Torch1.7环境，不重启runtime、不更新原RTA依赖。阶段独立运行并检查输出：source、infer、fixed4、estimated。脚本`scripts/run_frozenbn_officehome_colab.py`；输出`/content/imp-runs/officehome-frozenbn-capacity-10e-v1`。仅保存普通配置/history/log/checkpoint，不做hash或重复checkpoint评价。

## Source与容量完成，RTA配对启动

实现4fe8b704，collector1e224b57。exec146 source3轮正常完成、81次更新，CE2.71874/1.57049/1.04724，无NaN/OOM；source/target特征分别1785x256/4357x256，target标签不在特征中。

exec147容量估计正常收敛K2，15步目标18.75601→17.88952；source anchors1236张、建簇成本校准267张，lambda=.5785163178、beta=.1281749691、R49.44。两个未知候选支持1152/62，噪声57；支持不均衡，不称真实未知类恢复。

后续配对固定K4(C+K29) vs估计K2(C+K27)，Q29不变，两个arm都共享此新prior。旧普通BN下的估计K1不与新K2混成同K的BN消融。

fixed4已启动，需确认TASK_KONLY_START与loss；estimated尚未启动，禁止排队kernel检查/训练。collector为`scripts/collect_frozenbn_officehome_colab.py`，完整配对后只读已有history收集best/final、普通日志，不重评价checkpoint。

容量结果：`pipeline-results/officehome-frozenbn-capacity-10e-v1-estimate.json`。当前不具备最终RTA性能结论。

最新只读检查：exec148仍running，fixed4已完成8/10轮（console Epoch7）；最近loss有限，无报错。第8轮显示OS*约66.1%、UNK79.0%、HOS71.9%，只是中间观察，不作完整预算比较。estimated K2尚未启动，待fixed4正常终止后单独启动。

随后exec148正常done，wrapper输出FROZENBN_RTA_ARM_COMPLETE fixed4；history确认epoch1..10恰好10轮。fixed4的best和final均为第10轮：OS*=69.134668%、UNK=77.344037%、HOS=73.009306%。best使用target标签选epoch。

已单独启动estimated K2（exec149），TASK_KONLY_START确认C25/K2/Q29与同一source-final.pt，当前running。保持10轮预算、原ResNet50与RTA目标，不传IMP中心初始化头。当前只完成一组，尚不能判断容量配对优劣；也不能以旧普通BN K1对比代替此配对。
