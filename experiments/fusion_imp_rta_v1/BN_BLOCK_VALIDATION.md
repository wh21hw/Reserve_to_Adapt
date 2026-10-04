# 冻结encoder BN：额外 source 留类块配对验证

2026-10-04，已固定方法之后，选择原source类0–4作为额外留类块，不按它的结果选网络设置、epoch、块或容量公式。不是新数据集验证，也不是完全未接触语义的盲测：此前source几何探索涉及原25类，但此次C20表示网络不会用0–4的任何图像训练或更新BN。原压力块10–14的全部结果保留，不挑较好块来当整体成绩。

两组均重新由相同官方ImageNet ResNet50初始化，seed1，source其他20类监督3轮，C20分类模块。普通CE vs 冻结encoder BN统计的CE；只改变encoder BN模式，其仿射/权重均可训练，分类BN正常训练。无teacher/保留损失、无DINO。原L4/Py3.8/Torch1.7环境。

预算/优化器固定为此前source预热：batch64/drop_last/workers4、Resize256x256/RandomCrop224/RandomHorizontalFlip；SGD(momentum.9,nesterov,weight_decay5e-4)、encoder初始lr5e-5、head5e-4、inverse decay gamma10/power.75/max_iter10000。只用final3，不延长训练、不扫描参数。两组共享名单、seed及代码；普通组也保存backbone，故保存方式一致但不改变训练。

评分沿用Random2026的每类70/15/15；source已知训练部分建原型，已知校准99%门限，最后15%评分。容量主规则沿用κ=source每类计数、R=均值、source99%lambda、source校准最大初始增益beta、birth-first、移动已知中心；不因留类评分改变公式。原始成本和全已知负对照一并保存。

隐藏类标签只在最终诊断阶段使用，不进入网络、阈值/建簇成本拟合；known评分图像曾参加source监督，不能宣称独立known泛化。无RealWorld真实target、正式K/RTA改动或70轮训练。

实现：`scripts/run_source_frozenbn_validation_colab.py`；复用已检验source trainer，不重复稳定smoke。输出：`/content/imp-runs/source-frozenbn-block0to4-v1/source-ce`、`source-frozenbn`。
