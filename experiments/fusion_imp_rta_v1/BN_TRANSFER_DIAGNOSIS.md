# Source 冻结 BN 到 RTA 的统计变化

2026-10-04。仅读取已有 source-final 与 RTA last checkpoint 的 encoder BN buffers，不运行模型、不重新评测、不训练，不用 target 标签。比较普通 source CE 与冻结 source BN 两个已完成 fixedK4/seed1/RTA10 arm；估计K1/K2不加入本诊断，避免容量混淆。

| source prior | 均值绝对偏移/预热标准差 q50/q90/q99 | 方差绝对log比 q50/q90/q99 | RTA增加的批计数 |
| --- | --- | --- | ---: |
| 普通 CE | .00930 / .02799 / .05612 | .03077 / .09719 / .23786 | 1496 |
| 冻结 encoder BN | .01754 / .06161 / .15339 | .07424 / .22296 / .49023 | 1496 |

每组统计106个 encoder running-mean state-dict路径；注册别名可能重复，**不称106个独立BN层**。量化分位数也是按这些路径中的通道统计，不是独立层的显著性检验。

可以确认：冻结仅在source预热生效，后续RTA更新了这些统计量；相对于自身prior，冻结组的统计变化更大。不能确认：这些变化有害或导致HOS下降。普通RTA更新统计可能有助于域适应；两种prior的权重、head与尺度也不同，buffer诊断不是因果消融。尤其不能据此宣称K推断必然错误。

下一项针对性假设是“保留预热统计是否改善迁移”，应以一个单独方法版本对照：共享同一冻结source prior、固定K4/Q29/seed1/10轮，仅改变RTA阶段encoder BN训练/评估模式，继续训练encoder权重与BN affine，不冻结head BN，不加loss或改变K。已有普通RTA fixed4可复用；新增组不按target成绩改预算或阈值，报告best/final以及已知/未知取舍。该组尚未启动，不能提前声明有效；也不属于仅自适应K的版本。

脚本：`scripts/diagnose_officehome_bn_transfer_colab.py`。结果：`pipeline-results/officehome-frozenbn-bn-transfer-diagnosis.json`。首次输出将路径数误标为layers，已只改元数据字段为buffer_paths并说明别名，没有重载checkpoint或重算数值。
