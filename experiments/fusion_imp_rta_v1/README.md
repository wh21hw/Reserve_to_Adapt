# 融合版 v1：自适应未知容量 + 自适应虚拟方向

2026-10-04 用户授权结合新旧方案，并进一步要求完整70轮、3个seed。独立方法版本，不称 K-only，不覆盖历史实现或结果。

## 当前正式实验

A→W，seeds1/2/3，分别 C=10 source监督3轮，再70轮RTA，L4、Python3.8/torch1.7.1。每seed重新学习自己的source prior；K由该seed初始IMP决定，训练期固定。保留原版warm-end K-means未知头初始化，不加入IMP头初始化。

端点 gpu-l4-s-kkb-ass1a1-2xg3a949wz74o，source阶段exec15。目录 /content/imp-runs/fusion-imp-rta-v1/seed{1,2,3}；source和rta分目录。原C-only prior在已回收runtime，不复用旧方案C+2模型。普通loss/结构/指标/checkpoint保存；不做hash或重复评价。三seed best/final均报告，均值和样本标准差；best为目标标签选epoch。

## 流程

1. 共享 source C 维监督预训练；使用该 checkpoint 对应的完整冻结 features.npz。
2. SourceAnchoredIMP 初始已知中心来自 source 类均值，已知身份保留但中心可移动；source 99% 类内平方距离定阈值、类内均方残差定方差，prior_strength=5、5步、farthest birth。沿用新主线，不引入旧 alpha 公式。
3. 初始候选数同时给 K（未知输出槽数）和 V（虚拟方向数）。继承 source 已知头；新增未知行仍为随机初始化，不把 IMP 方向复制进分类器，不做原版初始随机行匹配排列。
4. 保留发布代码 RTA 的所有损失、系数、关系判断、未知伪标签、平坦 argmax 预测。预热结束仍保留原版 C+K K-means 未知头初始化，明确它不是 IMP 头初始化。
5. 每轮结束 eval/center-crop 提取完整当前 source/target 特征，重新从当前 source 中心运行 IMP，以候选中心直接更新 virtual CE 的虚拟方向。不再运行 Q=20 的虚拟 K-means/匈牙利匹配，因为 IMP 已保留已知身份。
6. K 固定为初始估计；V 可变化，允许 V=0。不扩缩头，不需要跨轮未知原型身份匹配。动态 K 留待第二版。

新增候选是容量/虚拟结构假设，不宣称真实语义类别。目标标签不参与阈值、聚类或 K 推断。

## 必须披露的变化

相对 K-only：虚拟方向改为 IMP，完整冻结特征代替训练增广 batch 汇总，初始已知头不做随机行重排。相对旧方案：source 阶段仅 C 维、移动已知中心并有 source 锚点、source 定阈值、稳定 softmax、初始 K 自适应；不是旧 alpha 调参的延续。

## 最小实验

共享同一 source checkpoint、seed3、A→W、软件环境和10轮预算：K-only / 融合版，另保留固定 K=2 + 同样 IMP 虚拟更新对照以隔离容量因素。旧 alpha.05/source3 结果仅作背景，预训练/估计设置不同，不当干净消融。尚未启动，结果待填，不混比70轮best。

代码：fusion_imp_rta.py、scripts/train_fusion_imp_rta_entry.py。部署需现有 train_konly_rta_entry.py、source_anchored_imp.py、alternating_konly.py 与原版依赖；KONLY_SOURCE_PRIOR 指 source-final.pt，FUSION_FEATURES 指同一阶段 features.npz；FUSION_EPOCHS 默认10，FUSION_FIXED_K=2 可运行共享 IMP 虚拟更新的固定容量对照。入口使用现有 CLI 参数与日志/checkpoint机制，单独新输出目录。当前版本仅支持 source 标签连续0..C-1。

本地针对性检查：以官方 main.py 构造融合代码并编译，确认 Q20 虚拟聚类被替换、原版预热结束 C+K 初始化保留、预算10轮。该检查不代表 GPU 训练验证或效果改善。
