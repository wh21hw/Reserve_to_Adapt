# OfficeHome 容量配对实验（探索性10轮）

同source prior：C25监督3轮、seed1，/content/imp-runs/source-precision-officehome-v1/seed1/source/source-final.pt。ResNet50，L4，Python3.8/torch1.7.1；batch64，SGD backbone5e-5、分类器/判别器5e-4，原发布代码warmiter3、损失/未知自伪标签、argmax预测保持；warm-end未知头仍为原版C+K目标K-means初始化，不使用估计簇中心初始化。

配对：fixed4（C+K=29）vs estimated（source规则K1，C+K=26）。两组Q29固定、原版Hungarian未匹配方向V机制，前置source checkpoint和训练预算完全共享，只K不同。原始RTA随机头构造及warm-end聚类策略一致，不声称不同头尺寸的随机数轨迹逐位相同。总训练各10轮，不和70轮best混比。

K估计不使用target标签；指标使用target原标签做macro OS*/UNK/HOS评测及oracle-best epoch，报告best和final。这只是OfficeHome工程协议：Q29/K4与70轮原预算口径未确认作者设置，不能称严格论文baseline复现。新估计K1不是未知语义类数恢复成功；source控制有误聚/合并，仍需检验是否是有用容量。

2026-10-04：入口构建检查通过，修复task_protocol Python3.8路径兼容，取消任务列表hash字段，不新增审计门槛。已在既有L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o启动fixed4，后台exec90；console.log已确认TASK_KONLY_START C25/K4/Q29及共享source路径。尚未观察首轮loss。estimated未启动，不能排队新exec。输出/content/imp-runs/officehome-capacity-10e-v1/fixed4/，训练子目录officehome-pr2rw_seed1。

后续：先观察fixed4 loss有限/实际epochs，再独立启动estimated；两组完成后下载逐轮指标与launch.json，汇总best/final，不自动重跑或改设置。正常训练用独立shell读取console.log/history.jsonl或exec attach90，不排队kernel检查。

最新进度：exec90仍running；console已输出原训练epoch0（history记为完成epoch1），OS*=0.817、UNK/HOS=0，ce0.859、virtual0.944、ce_ep2.220、adv0.788，loss通过有限值保护。尚处warmup，不能以UNK0判失败或提前调整。collector collect_officehome_capacity_colab.py已准备，要求两组均完整10轮，仅打包普通日志/配置/指标，不传checkpoint或重评价；estimated尚未启动。
