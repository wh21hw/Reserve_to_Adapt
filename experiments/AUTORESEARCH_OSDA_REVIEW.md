# Autoresearch 与 OSDA 理论导向的优化路线

检索日期：2026-10-05。技能wjgoarxiv/autoresearch-skill已安装到用户Codex skills目录；本文是初步文献筛查和实验设计，不是完整系统综述、SOTA排行榜或已执行的新闭环。

## 技能在本项目的用法

使用假设—单因素实验—机械评价—保留证据—更新假设的循环。项目AGENTS和用户指令优先：不对数小时训练套五分钟timeout，不用git reset覆盖用户修改，不因失败删证据，不扩大预算/重跑，不用target标签筛K或阈值。已有目标持续保留，技能的局部iteration耗尽不代表整个毕业设计目标完成。

技能要求运行前定义预算、无人值守方式和评价器；已询问新的小批次模式，尚未启动控制脚本、后台无限循环或新GPU训练。不新增自动化与重复Goal。当前VisDA K8十轮仍按原协议完成。

## 第一性原理

target是已知与未知的混合。已知风险与未知风险应分别控制；总体域不可分不代表语义对齐。跨域已知需要身份一致，target-private结构不能被拉向source。unknown内部结构又不是单一二元拒识能够完全表达。

拟议研究结构是选择性已知适应、未知结构发现、可靠结构反馈；以下三个部分不是新定理，也不直接把文献的保证移植到RTA。

### 理论与方法证据

1. **OSDA理论界（2019）**：[Open Set Domain Adaptation: Theoretical Bound and Algorithm](https://arxiv.org/abs/1907.08375)。论文引入open-set difference，把未知风险纳入适应界。对本项目的启示是不能只最小化两个域总体差异；本文未据摘要复述完整常数/假设。
2. **UADAL（2022）**：[作者全文](https://arxiv.org/html/2206.07551v2)。明确分source、target-known、target-unknown，采用不同对齐/分离信号。其对齐结论包含support不相交等假设；后验划分可能错，不能视为无需条件的保证。可作为RTA已知/未知门控分析参考，不直接整体移植新损失。
3. **JAUM（Pattern Recognition 2026）**：[期刊正文页面](https://www.sciencedirect.com/science/article/pii/S0031320325016760)，[作者代码](https://github.com/sentaochen/Joint-Distribution-Alignment-and-Unknown-Risk-Minimization)。其摘要/引言给出source风险、source与target-known联合分布χ²差异、target未知风险构成的风险界。可作为分解研究目标的近期理论参考。尚未完整审查PDF中的估计器、定理条件与公平协议，不宣称已验证对RTA适用或SOTA。
4. **SCDA（2022）**：[Open Set Domain Adaptation By Novel Class Discovery](https://arxiv.org/abs/2203.03329)。已提出发现隐含类别、重组分类器、更新域适应特征交替进行。因此“动态K+交替特征学习”不能独立当作新颖性；需要比较其类别发现、身份保留和训练接口，找出我们的可验证增量。
5. **DTE（CVPR 2025）**：[正式论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_Distinguish_Then_Exploit_Source-free_Open_Set_Domain_Adaptation_via_Weight_CVPR_2025_paper.pdf)。通过partially unbalanced optimal transport区分未知，再用稀疏匹配得到已知伪标签。source-free协议不同于我们可访问source的OSDA；借鉴“不强制全部样本匹配已知”的机制，不混比成绩。
6. **ProtoGCD（2025）**：[作者预印本](https://arxiv.org/abs/2504.03755)。已知/新类原型学习及偏置问题是相关方向；GCD与OSDA跨域协议不同，不能直接把其类别估计当我们跨域K的解决方案。当前仅初筛，精确算法需阅读全文后才实现。
7. **TALON（2026）**：[作者预印本](https://arxiv.org/abs/2603.08075)。把语义原型更新与encoder适应结合，针对类别碎裂/爆炸，并在离线阶段促进margin与紧凑性。它属于在线category discovery，不是RTA同协议baseline；可参考“表示与原型共同学习、避免一个类别碎成多簇”，不能用其榜单证明OSDA收益。

以上近期工作不是穷尽2026全部文献。Foundation模型/CLIP相关OSDA另成路线，本项目ResNet基线不因追热点更换backbone。

## 对我们当前失败的解释与优先级

原K8和共享K7把大量已知target分进新簇，显示冻结source几何的簇数包含域偏移；固定K2与K8都后期过度拒识，显示仅改善容量不解决RTA自举。簇身份桥接与时序更新需在同预算、共同前置阶段下分别测，不能同时加KL、头初始化、预测边际化并叫K-only。

优先实验顺序建议：

- E0：完成现有VisDA十轮及保存结果；固定节点诊断当前表示和头，禁止使用best.pt拟合K。
- E1：固定预适应阶段后提取source/target，用相同聚类器估计结构；与source-stage结构比较。预适应不一定成功，要检查已知身份/未知吸入，不能因经过RTA就称已统一空间。
- E2：使用相同K/共同前置特征，原版未知argmax伪标签 versus可靠簇身份反馈。只改伪标签接口，不改分类头初始化、预测或新增损失。
- E3：在E2接口明确后，再比较固定容量 versus分阶段重估容量/原型，并保留已知及可对应未知头状态。阶段点预声明，不按target HOS挑。

E1首先改变表示/时序，E2改变结构反馈，E3改变容量动态；每项需要明确共享控制组、总更新数与初始化状态。上述是待定协议，不是已开始的实验矩阵。

## 评价：不能把官方target HOS当自动搜索奖励

target标签保留用于终局/事后OS*、UNK、HOS及簇混合/碎裂分析，不输入优化器、threshold/K估计或自动keep/revert评分。若继续以target best epoch/seed作为探索展示，必须标明事后oracle选择，不能宣称无标签模型选择。

机制筛查使用事先划定的source已知/隐藏类代理与source伪域，报告已知误入、隐藏吸入、隐藏结构覆盖及稳定性，不仅K。其分数只支持代理筛查，不替代真实跨域成功。最终保留方法须同预算、多个任务验证；单一分数提高不构成用户研究目标完成。

需要为新闭环单独明确评价器/预算/停止规则。现有普通日志收集器可机械报告best/final，但没有合适的无目标标签自动选方法评价器；不假称已有autoresearch可直接无人值守科学选优。
