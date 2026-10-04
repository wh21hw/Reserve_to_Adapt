# 每轮虚拟更新模块对照

相同seed1、source3轮prior、支持度推断K（5）、A→W、L4、10轮预算、原版自举未知伪标签/损失/预测。保持初始IMP虚拟方向不变。仅第1轮末起，将每轮IMP更新替换为Q20 K-means和source/target中心匈牙利匹配，固定V=10。

两组均用相同完整current eval center-crop source/target特征、局部DataLoader随机生成器；不把表示提取差异混入本对照。保留原版warm-end C+K未知头初始化。

这不是逐行原版RTA：发布代码虚拟聚类使用训练增广batch累计特征；初始方向及source头继承规则也与发布代码不同。本组隔离每轮虚拟更新模块，不称原版baseline复现，不称纯K-only。

先一次构造检查、再训练10轮。与fusion-supported-capacity-v1比较相同预算；target标签仅用于报告best epoch，不用于K或虚拟聚类。单seed存在随机波动，不扫描Q、支持阈值、loss权重。

状态：支持度过滤组已完成，L4端点gpu-l4-s-kkb-ass1a1-2xg3a949wz74o的exec35已启动；构造检查通过。目录/content/imp-runs/fusion-q20-virtual-control-v1，K由相同支持度推断得到5。对照IMP虚拟更新K5：best HOS84.2939%、final84.0751%。仅当稳定观察到改善，再进一步固定K2或完整预算对照，不从本组直接宣称原版RTA复现或纯K-only成功。
