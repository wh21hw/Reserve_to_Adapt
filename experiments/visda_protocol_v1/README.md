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
