# 外部冻结表征诊断：已授权，仅诊断（2026-10-04）

用户最新限定：DINOv2 可以用于诊断，但 RTA 的 base 仍是 ResNet。该限定覆盖下文早期提案中的“用于容量估计”：本次不把 DINO 特征或估计结果输入正式 K 估计、RTA 分类头或训练。正式方法仍须在 ResNet 上成立。

执行协议：冻结 ViT-S/14，对 Product source 的 1785 张图像提取特征；固定既有留类 10–14 和 70/15/15 划分。首先只比较最近已知原型距离的新颖性 AUROC/AP，以及 source 校准 99% 阈值下的已知误报/未知覆盖，不扫描阈值、不运行新 RTA 训练。保留旧环境，使用 Colab 原生 PyTorch 推理。相关脚本：`scripts/diagnose_dinov2_source_colab.py`。

动机：当前source分类瓶颈的关系KL/距离存在一定新颖性排序信息，但单分数、相对混合门控、两分数联合source代理学习均未形成可靠的低误报/高未知覆盖与类别结构。仅换同一个已训练ResNet的2048维backbone也已有失败控制，不重复作为新方案。

获准的独立诊断路线：DINOv2 ViT-S/14冻结表征，仅用于 source 留类结构诊断；RTA 的 ResNet50、损失、预测和前置训练不变。不把外部特征用于正式容量估计，不把外部方向写入 RTA 分类器，不增加 RTA 损失。

官方仓库支持torch.hub加载dinov2_vits14，模型约21M参数。官方说明模型以142M图像无标签预训练。额外预训练数据及模型信息是实质研究因素，不能将性能差异归为仅自适应K、不能与原论文同预训练条件混报。模型卡及论文不能保证本项目真实未知类别发现或排除所有数据重叠。

来源：https://github.com/facebookresearch/dinov2 ，https://arxiv.org/abs/2304.07193 ，https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md 。

先仅提取OfficeHome Product原source25类1785张固定特征，使用既有source留类10–14控制、相同70/15/15划分及已知原型/校准规则。比较新颖性排序、误报/覆盖；不因target成绩调阈值，不启动RTA70轮。先不在RealWorld真实target上选模型或设置。

诊断若支持表征问题，后续仍研究 ResNet 下的表示/未知结构学习或原机制修正；新增训练目标必须另列版本，不能称 K-only。DINO 的改善本身不构成 RTA 方法改善。

执行记录见本文件后续结果；仅下载官方模型并做冻结推理，无新 GPU 训练、无 RTA 环境安装或变更。

## 已完成的 source-only 诊断

Colab L4，冻结 DINOv2 ViT-S/14，原生 Torch 2.11.0+cu130；1785 张 Product source 图像，384 维归一化特征。ResNet 对照复用既有 C20 source 模型的 256 维归一化瓶颈特征。无梯度、无新损失、无真实 target 数据、无正式 K 或 RTA 改动。

同一划分：测试 216 张已知 + 66 张留出类（10–14）。已知训练部分建原型，已知校准部分的最近原型距离 99% 分位数决定门限；没有扫描门限。

| 表征 | 新颖性 AUROC | AP | 已知误报 | 留类覆盖 |
| --- | ---: | ---: | ---: | ---: |
| RTA ResNet50 瓶颈 | 0.87072 | 0.69848 | 3/216 = 1.39% | 14/66 = 21.21% |
| 冻结 DINOv2 | 0.98071 | 0.91049 | 7/216 = 3.24% | 53/66 = 80.30% |

解释：外部表征的排序及覆盖明显更好，支持当前 ResNet 表征/距离几何是重要限制的诊断；不能把失败仅归因于 K 太小。相同 source 校准分位数不等于测试误报完全一致，因此这不是严格相同误报率下的比较。也尚未测试候选未知内部能否分成语义类别，不能宣称正确估计 K。

限制：留类压力块是此前事后选定；ResNet 已知测试图像参与过 backbone 的监督；DINO 有额外预训练及不同预处理，不能隔离出单一网络结构因果，也不保证预训练数据与本任务无重叠。这些数字不是 RTA HOS 或方法提升。

结果：`pipeline-results/dinov2-source-diagnostic-v1.json`；配置：`pipeline-results/dinov2-source-diagnostic-v1-config.json`。完整特征留在 runtime `/content/imp-runs/dinov2-source-diagnostic-v1/features.npz`，未反复传输大文件。

后续仍在 ResNet 上检验：未知候选覆盖、候选内部结构，再评估改变 K；若需要改变表征学习目标，必须另立版本并取得实验授权。本次没有启动这些实验。
