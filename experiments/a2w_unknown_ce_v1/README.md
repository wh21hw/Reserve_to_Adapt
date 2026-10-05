# A→W：未知伪标签 CE 单因素对照

用户批准改用 Office-31 A→W，替代尚未启动的 VisDA CE 消融。仅两个 arm，各 RTA 10 轮；不跑新的 VisDA。

共同设置：L4、ResNet50、seed3、C10/K2/Q20、batch64、RTA learning_rate .00005、source C维监督3轮共享同一 checkpoint。source及RTA保持 encoder BN running statistics，affine可训练，分类头BN正常更新。原版warm-end K-means头初始化、虚拟方向、候选筛选、未知槽argmax、其余损失和最终预测都相同。

唯一研究因素：warmup后未知伪标签CE系数，control=1，off=0。off仍执行候选筛选与head forward，避免额外改变BN。该实验检验未知自举监督的作用，不是最终未知建模方案，也不是仅自适应K。A→W结果不能直接证明VisDA崩塌原因。

当前runtime没有可直接复用且初始化/BN/预算完全匹配的A→W控制；重新生成一次共享source prior，分别运行两个短程arm。不用历史70轮best或不同配方代替对照。seed3来自此前事后seed选择；报告两组全部10轮、目标标签选出的best以及第10轮final，不当三seed均值，不用target训练标签改模型或选阈值。

数据只从已持久保存的Drive `OSDA/datasets/office31_images.tar`复制到本地解压，不重复公网下载。输出 `/content/imp-runs/a2w-unknown-ce-10e-v1`。

实现版本 a40458d4，已推送 module_imp。source3已完成，共享958×256 source与564×256 target特征；target标签不进入训练或特征。系数切换build-only通过一次，无重复模型smoke。控制组已在现有L4端点 `gpu-l4-s-kkb-ass1a1-28xqgmbpgb0gj` 的shell6启动，前两轮loss有限；尚无完整对照结论。后续off复用同一prior，失败不自动重跑。普通结果压缩包及最终checkpoint保存到Drive `OSDA/runs/a2w-unknown-ce-10e-v1`。

## 两组完成

source3和两个RTA arm均正常完成，无NaN/OOM。各arm恰好10轮；收集器只读日志，没有重复checkpoint评价。下方epoch按history的1-based编号（训练console打印0-based）。

| 未知CE | 选取 | epoch | OS*% | UNK% | HOS% |
| --- | --- | ---: | ---: | ---: | ---: |
| 1（control） | best HOS | 9 | 89.0662 | 87.2522 | 88.1499 |
| 1（control） | final | 10 | 89.6799 | 86.0565 | 87.8309 |
| 0（off） | best HOS | 4 | 94.3346 | 25.5782 | 40.2444 |
| 0（off） | final | 10 | 98.6774 | 3.3309 | 6.4442 |

final off−control：OS* +8.9975pp、UNK −82.7257pp、HOS −81.3867pp。关闭未知CE没有解决问题，而是几乎失去了拒识：未知样本被已知输出吸收。结果支持这项监督承担未知槽学习的关键作用，不能简单删除。它不证明原伪标签都正确，也不排除其在VisDA上推动错误自举；数据任务改变后的单seed短程对照不能解释VisDA全部退化原因。

前3轮两组评价指标相同；第4轮warm-end初始化后已有0.3788pp的UNK小差异，未施加全局确定性，不能宣称数值轨迹完全相同的确定性反事实。研究设置和共享source初始化匹配，之后系数差异下的巨大取舍仍需按本次单seed机制证据解释，而不是统计显著性。

下一方向：保留未知监督，研究如何让可靠的未知簇提供监督、避免已知误选；不是以CE=0作为最终方法，也不把本次改损失称为仅自适应K。没有因此自动新增实验、调系数或延长预算。

本地结果：`pipeline-results/a2w-unknown-ce-10e-v1-summary.json`、`pipeline-results/a2w-unknown-ce-10e-v1-results.zip`（两组完整日志/配置/history、source日志与汇总）。Drive目录已实际保存同名结果包、共享source-final.pt、control-last.pt和off-last.pt；无需下一runtime重新生成本次模型。
