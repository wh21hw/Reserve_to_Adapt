# ResNet 表征诊断与下一假设（2026-10-04）

正式 RTA base 保持 ResNet50；DINO 只做诊断，不进入正式容量估计或 RTA。当前全目标仍未完成：尚无跨三数据集验证的有效未知建模增量。

## 当前证据

OfficeHome Product source-only：隐藏 10–14 五类，沿用既有 70/15/15 分层划分；942 已知样本建原型、204 已知样本校准分数 99% 分位，216 已知 + 66 留类样本评分。不用真实 RealWorld target，不调分位数或尺度。

| 冻结特征 | AUROC | AP | 已知误报 | 留类覆盖 |
| --- | ---: | ---: | ---: | ---: |
| 原始 ImageNet ResNet50，2048维 | 0.90250 | 0.74942 | 4/216 = 1.85% | 21/66 = 31.82% |
| source 微调 ResNet50 backbone，2048维 | 0.88468 | 0.71961 | 3/216 = 1.39% | 14/66 = 21.21% |
| source 微调分类瓶颈，256维 | 0.87072 | 0.69848 | 3/216 = 1.39% | 14/66 = 21.21% |
| DINOv2，仅外部诊断 | 0.98071 | 0.91049 | 7/216 = 3.24% | 53/66 = 80.30% |

原始 ResNet 使用原 RTA 的现有官方 torchvision 权重 `resnet50-19c8e357.pth`，没有重新下载其他 ResNet 预训练。固定 Resize256x256/CenterCrop224/ImageNet normalize，输出 L2 归一化。source 微调特征复用旧缓存，无重复提取。L4 原生 Torch 提取原始特征，原 RTA Py3.8/Torch1.7 环境不改。首次读取旧 tar 权重时 native weights_only=True 不兼容，在任何提取前报错；对既有可信官方权重明确 legacy 加载后成功，无设置变化或训练重跑。

差异支持 source 微调/投影可能损失部分新颖性信息，但不是因果证明：单一事后压力块、已知测试图像参与过 source 网络监督、native/legacy 提取版本不同，误报率也不完全一致。DINO 的额外预训练和不同预处理必须单列；上述不是 RTA HOS 或语义 K 恢复结果。

类别尺度归一化另有失败控制：覆盖 18/66、误报 8/216，AP 反而从 .69848 降到 .64004。它改变了吸收位置，未可靠解决未知漏检，不继续沿尺度扫参数。

## 研究决策

1. “K 太小”不能解释所有失败。大量未知未进入候选池，多槽不能凭空把这些样本从已知特征结构中分离。
2. 仅把分类瓶颈换成同一已训练 backbone 不足；原始 ImageNet backbone 有部分改善，也仍漏检多数留类。
3. 下一独立假设：source 监督预热同时保留原 ImageNet ResNet 的表征结构，减少仅适配 C 个已知类别造成的信息损失。它属于 source 表征学习新版本，不冒称仅改变 K 来源。
4. 可检验增量是 source warmup 的表征保留，随后仍冻结特征、用 source 中心初始化 IMP、固定估计 K，RTA 损失/伪标签/预测不变。不能把 DINO 当 teacher 偷渡入正式方法。
5. 先 source 留类短程对照，再决定是否真实 target/RTA；不直接启动三 seed 70 轮，不根据 target 标签选正则系数。未知候选内部语义结构及跨域收益仍待检验。

下一实验的预算、优化器、正则/冻结范围须在启动前写明，固定对照共享所有其他设置；本次只有冻结诊断，没有启动新训练或修改正式估计器。

脚本：`scripts/diagnose_resnet_pretraining_colab.py`；结果：`pipeline-results/resnet-pretraining-diagnostic-v1.json`；类别尺度结果：`pipeline-results/resnet-class-scale-diagnosis-v1.json`。完整原始 ResNet 特征保存于 runtime `/content/imp-runs/resnet-pretraining-diagnostic-v1/features.npz`。
