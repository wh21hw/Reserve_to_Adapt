# 真实 Pr→Rw：冻结BN预热下的容量配对

2026-10-04。前两source留类块支持冻结encoder BN的预热改善，现在检验真实跨域RTA性能。不是严格论文复现：OfficeHome Q29/K4及原训练预算尚未确认作者设置。

新prior：完整Product25已知类1785张、C25头、seed1、source CE3轮。encoder BN统计固定为ImageNet，仿射参数/encoder权重与分类模块照常训练；head BN正常更新。其余原source warmup设置不变：batch64/drop_last/workers4，SGD momentum.9/nesterov/weight_decay5e-4，encoder/head lr5e-5/5e-4，原inverse decay，Resize256x256/RandomCrop224/HorizontalFlip。无teacher/额外loss/DINO。

冻结网络后提取source/RealWorld target归一化256维瓶颈；target只读取文件名，原语义标签不进入特征或容量拟合。沿用source70%建已知原型、15%已知校准建簇成本公式，lambda=source99%类内平方残差，beta=source负对照最大初始建簇增益*(1+1e-6)，κ_c=source训练各类计数，R=均值，birth-first、移动已知中心。source剩余15%不参与本次真实target拟合。估计一次K后固定，不调整beta/lambda或用target标签选K。

配对：固定K4 vs估计K，完全共享新prior、seed1/Q29/ResNet50/原RTA损失与伪标签/argmax/warm-end C+K目标K-means初始化，各10轮。只传K，不传IMP簇中心或标签。进入RTA后恢复原net.train/BN更新，不把预热的BN冻结延伸至域适应。原代码warmiter3实际前4轮warmup，两组一致。

K0显式记录估计失败，不强制K1；K4则两组输入相同，无需重复训练伪称独立改进。与旧普通BN的固定K4/K1结果仅作预热因素背景，不混报成纯K改变或同一prior。若新估计与旧估计不同，两种预热的estimated结果不作为单因素BN性能对照。

报告各10轮内best和第10轮final OS*/UNK/HOS，best为target标签选epoch，单seed探索，不称三seed均值/显著提升。无70轮或参数扩网格授权；先判断同预算性能与已知/未知权衡。

L4原Py3.8/Torch1.7环境，不重启runtime、不更新原RTA依赖。阶段独立运行并检查输出：source、infer、fixed4、estimated。脚本`scripts/run_frozenbn_officehome_colab.py`；输出`/content/imp-runs/officehome-frozenbn-capacity-10e-v1`。仅保存普通配置/history/log/checkpoint，不做hash或重复checkpoint评价。
