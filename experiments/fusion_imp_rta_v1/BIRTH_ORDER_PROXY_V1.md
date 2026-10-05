# 已知中心先更新的 source-only 探针

动机：VisDA 容量 K8 中96.1%的已知 target 分入新簇，已知中心几乎不移动。检查 birth-before-update 是否剥夺了中心适配机会。只比较更新顺序，不改变 prior、source校准半径/成本或RTA训练。

复用 C20 source frozen-BN CE3 seed1 的1458个已知特征，以及原 source proxy split seed2026：已知231、隐藏类别51（raw0..4）。前对照复用 prior-mass-probe-v1 的 source_counts；后对照调用原 robust_capacity 的 after_update，所有中心/特征以float64计算。无真实 target，无目标标签调参。已知探针图片参加过encoder训练，因此不证明跨域泛化。

结果：纯已知 K0、误建簇0%；已知+隐藏 K3、已知误建簇0%、隐藏候选召回58.8235%、隐藏被已知吸收39.2157%、noise1。与 before_update 对照的这些指标相同，均收敛；已知身份准确率均92.6407%。当前证据不支持仅换顺序就能改善语义分配，不因此启动昂贵VisDA重聚类或替换正在运行的K8对照。

工程备注：修正版上传失败，第一次远端运行仍为float32均值的旧脚本，保留birth-order-probe-v1.json、不作为严格单因素结果。随后exec19以float64执行相同探针并写新birth-order-probe-float64-v1.json，以上报告使用后者；仅CPU聚类，没有重跑source/RTA。脚本为scripts/probe_birth_order_c20_colab.py，原RTA训练不变。归档下载状态以实际CLI成功为准。
