# ResNet 权重 × BN 统计冻结诊断

2026-10-04。Source 表征保留正则3轮未改善，故先做局部机制定位，不扫描系数或重复原训练。

固定四组：原始 ImageNet 权重/统计、微调权重+原始统计、原始权重+微调统计、微调权重/统计。BN因子仅指encoder的running_mean/running_var/num_batches_tracked；BN仿射参数归权重因子。全部使用同一个 source CE final3 checkpoint 的 C20分类模块（含分类BN），不是为混合encoder重训的分类器。

全部在原Py3.8/Torch1.7环境与原RTA preprocessing下提取同一 Product source1785张特征，避免此前native/legacy提取差异。Source隐藏10–14、70/15/15划分不变；已知训练建原型、已知校准99%距离门限、最后15%评估。报告backbone/瓶颈距离AUROC/AP/误报/覆盖，以及固定分类头的已知准确率。

这是冻结checkpoint的局部干预，不是“训练时冻结BN”的性能验证，不是RTA正式成绩，也不是独立未知泛化证明。压力块事后选定、known测试参加过监督的限制保留。没有真实target数据、梯度更新、target标签调参或正式K/RTA变更。DINO不进入本实验。

脚本`scripts/diagnose_resnet_bn_factorial_colab.py`。先真实8图检查四个分支，再完整提取/独立评分；保存新目录`/content/imp-runs/resnet-bn-factorial-v1`，不覆盖已有checkpoint与特征。

## 冻结诊断结果

| encoder 权重 / BN统计 | backbone AUROC / 覆盖 | 瓶颈 AUROC / 覆盖 | 固定头已知准确率 |
| --- | --- | --- | --- |
| 原始 / 原始 | .90250 / 21/66 | .90264 / 14/66 | 91.20% |
| 微调 / 原始 | .90194 / 19/66 | .90481 / 17/66 | 91.20% |
| 原始 / 微调 | .88314 / 14/66 | .86862 / 11/66 | 94.44% |
| 微调 / 微调 | .88468 / 14/66 | .87072 / 14/66 | 94.44% |

单独替换BN统计能复现大部分AUROC下降；单独替换微调权重在此工作点影响较小。仅恢复encoder统计会同时影响已知头兼容性，不能把冻结干预中的更高排序直接当方法提升或改好正式模型。各组source门限各自校准，误报不完全一致：backbone4/3/3/3，瓶颈2/2/3/3。该机制定位是局部证据，不保证冻结BN训练一定成功。

结果 `pipeline-results/resnet-bn-factorial-v1.json`。

## 下一组预先声明训练对照

固定 source 预热的 encoder BN running统计为初始ImageNet统计（每次net.train后仅将encoder BN置eval），仿射参数及其余encoder权重仍训练；C20分类模块BN照常训练。无teacher、无保留损失，不加新RTA损失。唯一新训练因素是encoder BN模式。

复用此前CE seed1/3轮/63步对照；新臂相同1362张known、同增广/初始化/优化器/学习率/seed/预算（详见SOURCE_RETENTION_V1.md），retention_weight=0。隐藏source类10–14仍不参与梯度或BN更新，真实target未读取。仅评分final3，source校准99%门限不扫描。

先一次真实batch检查冻结统计与有限梯度，再完整3轮训练。输出`/content/imp-runs/source-frozenbn-officehome-v1/seed1/source/`；报告已知分类与backbone/瓶颈新颖性。不能根据隐藏类标签选择冻结层范围、epoch或系数；不能把更换预热BN设置称纯K-only。

## 3轮冻结 encoder BN 训练完成

实现83e8279a，exec138正常完成3轮63次更新；loss2.50867/1.38610/.87124，无NaN/OOM。仿射参数及encoder权重仍训练，头BN正常更新，retention_weight=0。

| final3 | 瓶颈 AUROC / AP | 瓶颈误报 / 覆盖 | backbone AUROC / AP | backbone误报 / 覆盖 | 已知分类 |
| --- | --- | --- | --- | --- | --- |
| 普通 source CE | .87072 / .69848 | 3/216 / 14/66 | .88468 / .71961 | 3/216 / 14/66 | 94.44% |
| 冻结encoder BN的CE | .91842 / .81548 | 1/216 / 15/66 | .90979 / .76870 | 3/216 / 20/66 | 94.91% |

瓶颈排序改善，而source99门限的覆盖只小幅增加；仍不能宣称问题解决。backbone覆盖增加但不因此临时挑backbone作为主估计特征。

## 原容量规则迁移：保持公式，不选新阈值

对新模型默认256维瓶颈运行既有 `score_source_leaveclass_colab.py`。source70%建原型，κ_c=各已知训练样本数，R=其均值，lambda=source类内平方残差99%分位；beta=source已知校准负对照最大初始建簇增益*(1+1e-6)，birth-first、已知中心允许移动，其余目标/噪声/删除规则不改。原始成本对照和校准成本对照均保留，不选择较好成本来改变规则。新lambda=.5403349530、beta=.1201874129，其数值由既定source公式重算，不是新超参数搜索。

| 校准成本规则 | 全已知负对照K / 误入 | 混合K | 混合已知误入 | 留类候选覆盖 | 一簇一类匹配/全留类 |
| --- | --- | --- | --- | --- | --- |
| 普通 source CE 特征 | 1 / 2/216 | 2 | 1/216 | 25/66 = 37.88% | 24/66 = 36.36% |
| 冻结encoder BN特征 | 0 / 0/216 | 3 | 0/216 | 41/66 = 62.12% | 39/66 = 59.09% |

新簇的隐藏类计数：类10=[0,12,0]、11=[1,0,0]、12=[15,0,0]、13=[0,1,0]、14=[0,0,12]。主要覆盖10/12/14，11/13仍大多没有脱离已知结构；K3不能称恢复真实五类。噪声5张；所有arm正常收敛。原始lambda建簇成本规则仍K0/留类覆盖0%，未用失败结果改公式。

这项结果有机制和容量的初步支持，但依然是同一seed/事后压力块，不是独立验证或正式target成绩。下一步先在另一预先固定的source留类块、同预算CE/冻结BN两组检验，再决定真实target的固定K/估计K训练；不自动启动70轮矩阵。

结果 `pipeline-results/source-frozenbn-officehome-v1.json`、`pipeline-results/source-frozenbn-officehome-v1-capacity.json`；标准对照为既有 `pipeline-results/source-leaveclass-officehome-v1-controls.json`。评分与估计各执行一次，未重复checkpoint审计。
