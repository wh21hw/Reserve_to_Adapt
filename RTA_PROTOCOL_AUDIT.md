# RTA 三任务协议与复现口径审计

2026-10-03最新约定：SHA/重复checkpoint审计不再是本项目门槛，后文哈希要求为历史记录。当前新主线为共享C类source前置训练→IMP估计K→原版RTA仅改K；其后允许分段交替K增减，不继续旧结构KL矩阵。L4旧软件环境原版seed3桥接已启动，待完整结果再判断环境影响。论文公式忠实实现与published-code应另列配置，不为了凑分静默改系数。

记录日期：2026-10-03。正式范围固定为 Office-31 A→W、Office-Home Pr→Rw、VisDA Synthetic→Real，每任务 baseline/IMP 同协议对照，训练 seeds 1/2/3 全部报告。不得以短程 pilot 替代完整实验。用户后来允许挑最好 seed：可作明确标注的事后选择补充结果，同时保留三种子统计；不得把补充结果写成无偏均值或据此调参。

## 复现口径

现有 baseline 与 IMP pilot 是发布代码基点 `8a4cd890b341a0dd4ed7c239533a1b6a9583fe8b` 的工程化实验，不应写成完全等价于论文公式。以下论文条目来自此前对项目下 TIP 扩展版 PDF 第4–6页和第8页的逐页核对；代码条目再次由本地 `git show` 核验。

| 项目 | 论文描述 | 发布代码 | 对实验的影响 |
| --- | --- | --- | --- |
| 适应期损失 | Eq18: CE + .01 virtual + .3 unknown + .4 (adv + entropy) | CE + .01 virtual + 1 unknown + .3 adv + 1 entropy | 不是同一组损失系数，不能静默修正后与旧结果混用 |
| 目标熵 | Eq17 已知 C 类概率 | 全部 C+K 输出概率 | 扩大未知槽会改变平坦熵的语义 |
| 加权损失归一化 | Eq14/16/17 按相关指示权重和归一化 | utilities.py 除以 batch N | 权重变化同时改变梯度尺度 |
| 关系 KL | Eq6 target || source soft prototype | F.kl_div(log target, source prototype)，即 source || target | 非对称 KL 的方向不同 |
| 虚拟模板 | Eq2/3 对 C+K 聚类，匹配 C 后余 K | Office-31 K_cluster=20，匹配10后余10 | 虚拟模板数不等于输出未知槽数2 |
| VisDA backbone | 正文实现段落 ResNet-50；Table III 标题 VGGNet | 公开 main.py 实际使用 ResNet-50；未实现可用的 VGG 路径 | 尚不能宣称精确复现 VisDA Table III |

原版平坦推断为 C+K argmax 后合并未知类；IMP 分层边缘化推断是方法变更。两者必须分别命名，不把离线换决策规则产生的增益当成新增训练收益。

后续主对照保留发布代码口径，以保证既有实验可追溯。若加入论文公式忠实实现，需建立独立配置与基线，不能覆盖旧实验。跨数据集入口重构仅更换任务协议时，也必须逐项披露未由官方实现证实的选择。VisDA 的精确 backbone 尚未解决。

## Office-Home Pr→Rw：本地文件审计

路径根应为项目 `data/`，而不是项目根；列表中采用 `Product/...` 和 `RealWorld/...` 相对路径。

| 文件 | 样本 | 标签集合 | 缺失图片（相对 data/） | SHA256 |
| --- | ---: | --- | ---: | --- |
| data/product_0-24_train_all.txt | 1785 | 0–24，25类 | 0 | 867c9539d2dc11dd9096bf59d69293917b4de7c9b0aaae2a656e15ffc5c5ee5c |
| data/real_world_0-64_test.txt | 4357 | 0–64，65类 | 0 | d4d6a3dd28681b3bc943cfa52317540463d781971ca6d448b92243c79a853f39 |

两个列表在官方基点 Git tree 中存在。源0–24与目标0–24的类别名一致：Alarm_Clock, Backpack, Batteries, Bed, Bike, Bottle, Bucket, Calculator, Calendar, Candles, Chair, Clipboards, Computer, Couch, Curtains, Desk_Lamp, Drill, Eraser, Exit_Sign, Fan, File_Cabinet, Flipflops, Flowers, Folder, Fork。目标25–64为协议中的40个未知类别。

这只是输入文件与类别协议就绪，不代表 Colab 数据已传输、训练入口已支持该任务或复现已完成。需进一步固定完整目标类别映射、逐类样本数、图像完整性和传输校验。目标语义标签仅用于协议检查和评价，不传入 IMP 聚类、先验估计或结构损失。

## 仍须完成的门槛

1. 任务参数化：删除训练/评价入口中的 Office-31 已知10、未知20–30、目录路径等硬编码；对三任务分别验证标签映射。
2. VisDA：取得真实文件列表，核验6个已知类别为 Bicycle、Bus、Car、M-cycle、Train、Truck，不能直接取数字0–5；澄清 backbone 口径。
3. 结构模块：固定无标签先验及条件结构学习规则；先做数值与组件分裂一致性检查，再隔离模块消融，不根据目标 HOS 挑系数或容量。
4. 完整训练：同环境、同预算 baseline/IMP；seed1/2/3；固定 final 与目标标签 oracle-best 分别报告均值和样本标准差。
5. 每阶段立即保存配置、代码版本、逐轮指标、checkpoint 和哈希；GitHub 同步必须验证成功，不能将本地 commit 视为已推送。

本轮通过 colab-cli auth status/runtime list 确认现有 L4 `gpu-l4-s-kkb-ass1a1-13n8bs33o8id1` 在线。未启动训练、未改变运行时、未改动用户根目录训练代码。
