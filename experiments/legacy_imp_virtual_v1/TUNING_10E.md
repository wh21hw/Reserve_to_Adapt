# 旧方案 alpha / source 微调轮数：10轮快速筛选

2026-10-04用户要求调alpha和source微调轮数，并明确将RTA预算改为10轮。

固定Office31 A→W、已选seed3、L4/旧Python3.8+torch1.7.1环境、分类器10+2、已知中心固定、每轮IMP5步、原旧损失与max_iter10000调度不变。只更改alpha或source微调轮数，不改变阈值公式/软分配。先单因素扫描，非完整3×3网格。

| alpha | source微调轮数 | RTA轮数 | 执行方式 |
| ---: | ---: | ---: | --- |
| .05 | 5 | 10 | 复用旧方案v3前10轮history，不重跑，不使用其70轮best |
| .01 | 5 | 10 | 新配置，先单独观察运行 |
| .1 | 5 | 10 | 待首个新配置完成，再串行执行 |
| .05 | 3 | 10 | 同上 |
| .05 | 7 | 10 | 同上 |

source阶段分别按3/5/7轮执行，10轮指RTA适应阶段，不把source预算混入同一个10轮计数。每组从同seed ImageNet初始化开始，不从目标标签best权重继续训练。已有70轮结果只截取前10轮，报告10轮内best和第10轮final。评价使用目标标签HOS；按此选配置属于探索性/oracle调参，不是无偏最终测试结果，也不称论文复现成功。

新增4组目录`/content/imp-runs/legacy-imp-tuning-v1/alpha*-ft*`；入口`run_legacy_imp_tuning_arm_colab.py`。首组完成后才允许运行`run_legacy_imp_tuning_remaining_colab.py`，后者串行执行余下3组，失败即停止，无自动重试/调参。每组完成自动保存普通日志/配置/history/summary及小ZIP，best/last留在runtime。

与v3一致的空虚拟集合处理保留；另补初始IMP总原型少于12时保留未匹配的既有分类头行，以保持固定12维，不制造新簇。已有v3初始12个原型，因此该边界不会改变它的初始行为。原始用户快照仍不修改。

alpha变小按当前公式会提高建簇阈值；“更多/更少原型”与HOS均由实际结果判断，不提前断言更好。此轮不展开70轮网格，不擅自挑新seed。
