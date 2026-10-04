# 论文设置复核与更正

2026-10-04，读取用户提供的RTA PDF并检查第4页方法、第6页Implementation Details原页；只读，不改训练。

需要更正先前记录：OfficeHome固定K4不是未经确认的探索值。论文第6页明确Office-Home K4，Office31/VisDA/ImageCLEF K2；同页OfficeHome已知25类、未知40类。第4页规定目标C+K K-means，并匹配C个已知中心，其余K个作为虚拟方向。因此按论文基线C25/K4应有Q29、V4，当前基线Q29有直接依据。

仍须区分版本：当前发布代码桥接版在估计K2组固定Q29，为隔离未知输出容量而不同时改变虚拟方向生成。这不等于将论文中所有K位置一起改成2；若严格沿论文绑定，Q应变27、V变2，属于不同对照。我们现有实验仍以共享Q29的发布代码训练机制为准，不中途修改。

第6页还确认ResNet50/ImageNet、256维瓶颈、SGD momentum.9/weight_decay.0005、分类/判别器lr.0005、encoder小10倍。固定总训练epoch数仍未在本次文字核查中找到；Algorithm1只有MAX_EPOCH符号，不能把当前10轮或发布代码70轮冒充已确认的论文预算。VisDA表标题与实现文字的backbone口径矛盾也未因此自动解决，正式实验继续用户指定ResNet。

发布代码与论文公式存在另外的口径差异，不能因K4/Q29确认便宣称严格复现：

- 论文式(1)的virtual softmax分母是C个已知logits加K个虚拟方向；实际运行networks.py把完整C+K分类logits再拼虚拟logits。
- 论文式(6)是KL(target prediction || source soft prototype)；实际main.py调用kl_div(log target, source prototype)，得到反向KL(source || target)。
- 第6页正文lambda3=.4等系数与实际main.py适应loss `ce + .01 virtual_ce + .3 adv_loss + entropy + ce_ep` 不可只按相同数字直接一一对应成论文式(18)。不在当前K/BN消融中修正这些项，否则新增研究因素。

还有与研究创新直接相关的区别：论文已对**一维KL关系分数**使用DPGMM推断混合分量M（第4页），但M不是未知语义类别数，也不是分类器K。我们的方案试图在冻结ResNet特征空间、以source中心为锚，估计未知容量K；不能声称RTA没有使用任何DP思想，也不能把原DPGMM的M直接当未知头维度。

此前ZIP/config保留当时“Q29/K4未确认”的历史表述，不覆盖结果；本更正以及后续cross-summary中的paper字段为最新核实口径。K4/Q29已确认，不等于完整baseline已与论文对齐。

来源：根目录 Reserve_to_Adapt_Mining_Inter-Class_Relations_for_Open-Set_Domain_Adaptation (1).pdf，PDF页4/6（印刷页1385/1387）。
