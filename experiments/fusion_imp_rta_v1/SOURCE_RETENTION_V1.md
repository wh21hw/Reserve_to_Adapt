# Source ResNet 表征保留：短程探索 v1

2026-10-04，独立预热版本，不是 K-only；正式 RTA 尚未改动，DINO 不进入此版本。

假设：source C 类交叉熵可能让 ImageNet 表征向已知类专化；在预热中保留原特征方向，可能减少未知结构信息损失。该假设不是已经验证的语义发现或 Bayes 后验结论。

固定目标：L=CE_source + η mean(1-cos(fθ(x), f_ImageNet(x)))，η=1，仅此一个预先声明系数，无扫描。f 为原 ResNet50 的 2048 维 avgpool 表征；teacher 为初始 ImageNet ResNet 的冻结副本、eval 模式，无梯度。两者使用同一增广图像。student 为 train 模式，因而约束也包括 train/eval BatchNorm 表征差异；不能称只约束参数漂移。

teacher 与 RTA 原始权重相同，无外部模型或额外预训练信息。约束没有直接保护 256 维瓶颈，故 backbone 和瓶颈分别报告。默认 η=0 仍保持原 source CE 行为，不影响已有实验。

预算固定：OfficeHome Product，隐藏原 source 类10–14不参与训练；1362 张其余20类source监督，seed1，C20分类头，3轮；batch64、drop_last、workers4，Resize256x256/RandomCrop224/RandomHorizontalFlip。SGD momentum0.9、Nesterov、weight_decay5e-4；ResNet lr5e-5，分类模块lr5e-4，原 inverse decay gamma10/power.75/max_iter10000。原 Py3.8/Torch1.7/CUDA11.0环境、L4，不改原 RTA 依赖。每臂63次优化更新。

对照复用既有同预算 source CE seed1 checkpoint，不重跑稳定对照；原模型source loss约2.67024/1.72452/1.16536。teacher为deepcopy，不引入重新初始化随机数；其推理无dropout，无新的数据顺序选择。

训练后只提取原 Product source25类1785张固定特征，不读取真实RealWorld target；既有70/15/15划分建已知原型、source已知校准99%门限，最后15%测试已知分类/新颖性排序、已知误报及留类覆盖。只用第3轮final，不按隐藏类标签选epoch、系数或模型。隐藏类标签仅用于事后diagnostic；known测试图像参加过source网络监督，压力块是事后选定，结论范围必须披露。

先进行一次真实batch的损失/梯度/teacher冻结接口检查，再启动3轮新臂；失败保留日志，不自动调系数或延长轮数。完成后决定是否继续容量估计及真实target/RTA；本协议不授权自动扩大网格或70轮矩阵。

运行输出：`/content/imp-runs/source-retention-officehome-v1/seed1/source/`。实现：`scripts/train_source_prior_konly.py` 的可选 retention 参数。
