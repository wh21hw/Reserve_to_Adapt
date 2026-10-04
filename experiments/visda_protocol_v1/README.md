# VisDA protocol preparation (no formal training yet)

Official category IDs and download locations:
https://raw.githubusercontent.com/VisionLearningGroup/taskcv-2017-public/master/classification/README.md

RTA Table III known classes map to original IDs **1,2,3,6,10,11**, not0..5.
Source: synthetic train, these six known classes only. Target: real validation,
all12 classes; raw labels retained only for evaluation/protocol preparation.
Training and K estimation must not receive target semantic labels.

Data terms: non-commercial research/education only, no image redistribution.
Do not upload images to GitHub. Downloading the official archives does not
settle the unresolved ResNet-50 text vs VGGNet table backbone choice.
Actual lists and class/sample counts are now prepared; no formal training yet.
# 当前数据状态（2026-10-03）

官方train/validation压缩包已在L4的`/content/osda-visda-syn2real-v1`下载。source仅解包已知六类，79765张；target解包validation全12类，55388张。`source-known-6.txt`保留原ID1/2/3/6/10/11，`target-real-12.txt`供协议映射/评价，`target-unlabeled-paths.txt`仅文件路径供无标签处理。无图像上传GitHub、无hash检查、无模型训练。准备脚本保留已有目录，不覆盖数据。

VisDA正式训练尚未启动：正文ResNet50/Table III VGGNet口径尚待确认；论文实现段落未明确该任务epoch预算，不将Office31发布代码70epoch伪称为作者VisDA预算。

## 当前runtime恢复（2026-10-04）

上述2026-10-03准备报告是历史证据，不代表现有runtime仍有数据。只读核实现有L4 gpu-l4-s-kkb-ass1a1-2xg3a949wz74o中该数据目录/列表不存在，所检查的/content数据目录也无VisDA存档，OSDA Drive目录未找到可复用VisDA目录。不能把历史报告当作当前数据就绪。

已重新核实官方classification README仍提供BU train.tar/validation.tar及原类别ID；数据只用于非商业研究/教育、不上传GitHub图像。现有runtime磁盘使用55.4/235.7GB，L4空闲；账户余额248.7422CCU、速率1.54CCU/hr（这一时点快照，不是未来余额保证）。

官方train.tar HEAD返回200、Content-Length7698031104，与既有下载脚本一致；正在独立恢复source archive，不链式自动解包/训练。下载完成后核查列表与已知六类映射，再恢复真实target；目标语义标签只供协议/评测，source训练和K推断使用无标签路径列表。

后续正式模型继续用户明确要求的ResNet50。计划冻结当前候选设置（source C6 CE3轮、source encoder BN统计冻结，容量规则与OfficeHome相同），全量特征推断使用已验证分块接口，不采样替代全量。source与RTA各阶段分开启动；训练和推断尚未开始，预算/表标题差异不伪称论文解决，也不直接复用OfficeHome监督模型。
