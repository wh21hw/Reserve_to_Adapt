# VisDA backbone与论文对标口径（2026-10-06现场复核）

## 一手证据与结论

用户提供的RTA论文PDF第6页（印刷1387）Implementation Details写ResNet-50、ImageNet预训练、瓶颈256；同段VisDA K2。第8页（印刷1389）Table III标题却明确VGGNet。已完整提取相关页并渲染检查，不把两栏抽取错位当作者设置。

Table III的RTA Synthetic→Real报告OS*73.6%、UNK83.7%、HOS78.3%；这些是该表的结果，不是已经确认ResNet50下的直接目标。表内报告已知类列为Bicycle/Bus/Car/M-cycle/Train/Truck，与既有原ID1/2/3/6/10/11映射一致。

2026-10-06重新读取作者[公开仓库](https://github.com/PRIS-CV/Reserve_to_Adapt)：README明确说明Office31 A→W/ResNet50；公开[networks.py](https://raw.githubusercontent.com/PRIS-CV/Reserve_to_Adapt/main/networks.py)提供ResNet系列特征提取器，但并不提供独立VisDA实验配置或能解释Table III的VGG实现。GitHub只读issues全状态查询返回Dataset issues（关闭，无评论）及Excessive memory usage（打开，无评论）；这次检查未找到backbone澄清。此为有限公开材料核查，不证明不存在私人配置/作者说明。

结论：公开材料不足以判定表标题笔误，或VisDA实际用了VGG。不得将“我们采用ResNet50”写成“论文VisDA表确认为ResNet50”，也不未经说明改为VGG重新跑。

## 对当前研究的执行决定

1. 按用户此前明确指定继续ResNet50。当前A→W在跑，本文没有修改其模型或设置，也未开启新的VisDA训练。
2. VisDA主证据是同ResNet50、同source前置/BN策略/预算/seed的固定K与自适应K配对；未知容量贡献由这一配对说明。
3. 与论文Table III的差值可以保留作背景，但必须注明backbone未完全确认，不能用该差值宣称严格复现、胜过或失败复现表中设置。
4. 若未来得到明确作者配置/公开勘误，再单独补论文对齐版本；它不替代、覆盖已有ResNet模块对照，不同时调K规则/loss弥补指标差异。
5. 不向作者发消息或开issue；本次仅阅读，不进行外部协调。无需因此停止已授权的同设置研究，但完整论文对齐仍是不确定项，不伪称解决。

## 检查范围

只读论文、公开README/network代码和issues列表，没有hash、重复训练或checkpoint评分。WSL缺少pdftotext后使用本地bundled pdfplumber；本地缺fitz及stdout编码错误仅影响文献读取，修正工具后成功读取/渲染，不属于模型失败，不改训练环境或Colab runtime。

本项目三数据集计划保留：Office31 A→W、OfficeHome Pr→Rw、VisDA Synthetic→Real。当前容量配对先在A→W完整70轮形成判断，再自主声明后续单因素/跨任务预算，不为等文献澄清浪费GPU重复实验。
