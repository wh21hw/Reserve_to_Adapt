# 旧方案：IMP 自适应虚拟原型，固定未知分类槽

## 完整结果（2026-10-04，最新状态）

exec7/v3已完成source5轮+RTA70轮，L4、seed3，固定未知输出槽2；普通日志/配置/70轮history/summary已保存并下载到`pipeline-results/legacy-imp-virtual-v3-seed3-results.zip`及同名解压目录。原快照未修改，实际入口含上述空虚拟集合处理。云端best/last保留，本轮未传输大权重。

| 口径（%） | OS* | UNK | HOS |
| --- | ---: | ---: | ---: |
| 旧方案best，第65轮 | 92.4461 | 88.4671 | 90.4128 |
| 旧方案final，第70轮 | 92.4461 | 88.0718 | 90.2060 |
| 选定原版baseline best | 95.1705 | 95.7032 | 95.4361 |
| 选定原版baseline final | 94.7472 | 95.4002 | 95.0726 |

旧方案相对选定baseline的best/final HOS为−5.0233/−4.8666个百分点，未改善原版。相对论文A→W HOS93.0%，旧方案best低2.5872pp。相对一次K-only best88.6103%和交替best88.3913%更高，但各配方source预训练/初始化等不同，不能归因为纯IMP模块收益。

虚拟原型V每轮范围0–7，最终4（总原型Q_imp14），分类器仍2个未知输出槽。best由目标标签HOS选epoch，seed3来自此前baseline事后选择；此结果是单seed旧配方效果，不是三seed均值，不证明发现真实未知类数。下方“正在运行/尚未完成”为历史阶段记录。

## 2026-10-04：用户授权独立效果实验

计划A→W、此前选定seed3、旧Python3.8/torch1.7.1环境、L4，source微调5轮+RTA70轮；alpha=.05、已知中心固定、IMP5步、每轮更新虚拟原型、输出未知槽固定2。保留原粘贴IMP数值规则，暂不改成source分位数/稳定softmax或移动先验版本。此实验是旧方案整体配方对比，不是单因素模块消融（source前置阶段/优化器等也不同）。

原runtime已被Colab回收，已创建新L4 `gpu-l4-s-kkb-ass1a1-2xg3a949wz74o`，准备恢复数据/环境。尚未启动训练，不把准备成功当效果结果。前次OfficeHome exec147因runtime消失，目前无完整结果可确认。

启动记录：环境/数据恢复完成。exec5在训练前因新增配置记录引用未导入Path而失败，未执行SGD；失败目录`/content/imp-runs/legacy-imp-virtual-v1/seed3`保留。仅将该记录改为路径字符串，新目录改为`legacy-imp-virtual-v2/seed3`，不修改旧IMP数值算法或超参数。

修正后exec6完成source5轮，最后一轮loss0.3769；初始IMP总原型12（10已知+2新增），初始化通过。但第1轮RTA更新后IMP仅保留10个已知原型，nomatch空列表触发np.stack错误，未完成第1轮评价，无完整结果。v2失败目录保留。

必要的空集合处理：nomatch为空时使用shape(0,256)矩阵，允许虚拟原型为0，不强制新簇、不改alpha/阈值/分配规则。此时virt_forward没有额外方向，virtual CE退化为原分类CE。这是原代码未定义边界的工程补全，不称逐字原代码运行。修正版使用新目录`legacy-imp-virtual-v3/seed3`，保持5+70预算；由于v2无checkpoint，需同seed重放前置阶段，不挑其他seed。原快照与两次失败证据保留。

当前正式进程exec7正在运行v3：source5轮完成，前3轮RTA完成，loss有限，无OOM/NaN；每轮结束Q_imp=10/12/13，V=0/2/3，分类器K_out仍为2。预热期UNK=0，不据此判定最终失败；未完成70轮，不报告最终效果。普通逐轮指标及best/last已由入口保存。完成后运行`collect_legacy_imp_virtual_colab.py`，下载v3结果ZIP，报告OS*/UNK/HOS的best/final以及Q/V范围，对比选定原版baseline95.4361/95.0726% HOS，并明确不是纯IMP消融。

独立入口 `scripts/train_legacy_imp_virtual_entry.py` / `scripts/run_legacy_imp_virtual_colab.py`；仍依赖已选baseline工程归档的data/networks/utilities等。仅改路径、固定seed、防覆盖、普通逐轮指标/best+last保存、评价no_grad，以及移除浏览器自动下载。原快照保持不动，根目录用户修改不动。运行目录 `/content/imp-runs/legacy-imp-virtual-v1/seed3`，若失败先保留证据，再决定必要的工程修复，不自动调alpha/预算。

记录日期：2026-10-03。依据用户本次提供的旧 main.py 附件与聊天中贴出的 IMPClusterer。此目录是历史方案记录，不是当前训练入口，不表示已经跑过本快照，也不覆盖根目录用户代码。

代码快照：`main_user_snapshot.py` 保存附件正文；`IMPClusterer_user_snapshot.py` 保存本次粘贴的聚类逻辑（排版整理，不修复行为）。原附件来源：`C:/Users/46025/.codex/attachments/4e022986-cb0a-4862-998e-341f95fa4dd4/已粘贴的文本.txt`。依赖仍引用项目 data/utilities/networks/centroid/domain_bus 等；这些依赖的历史版本未随附件提供，不能据此宣称完整可复现实验环境。

## 研究直觉与实际作用

原始直觉：先用 source 已知类监督改善特征，再以 source 类中心初始化 target 聚类；距离超过阈值则新增原型，通过每轮聚类自适应建模 target 结构。

实际实现是 **IMP 替代 RTA 的目标虚拟聚类，更新虚拟损失中的原型集合**，并未让分类器未知输出槽数自适应。记：C为已知类数，K_out为分类器未知槽数，Q_imp为IMP总原型数，V为未匹配虚拟原型数。

- 默认 `shared_classes=10`、`all_classes=12`，因此 K_out=2；分类器在IMP之前已经建立，后续未扩展。
- `K_cluster=len(t_centroids)` 是Q_imp，不是K_out。已知中心固定保留且匹配成功时，V通常为Q_imp−C。
- `nomatch` 传入 `cls.virt_forward(...)`，因此改变虚拟方向及虚拟损失，而不是分类器输出维度。
- 初始化的匈牙利匹配选择/排列现有分类头权重，不把IMP中心直接安装成更大分类器。
- 预热结束的未知权重初始化仍用 `faiss.Kmeans(..., args.all_classes)`，固定C+2，而不是IMP数量。

## 原代码流程与设置

1. 默认任务W→D（不是当前A→W）；ResNet50、256维瓶颈、12维分类头。batch64，骨干学习率5e-5、分类器5e-4。
2. source微调5轮：只取前C维logits计算已知类CE；SGD momentum .9、nesterov、weight decay5e-4，固定学习率。之后复用这些优化器及其动量进入RTA。
3. 在训练模式下用source/target训练loader和DomainBus提取特征（随机裁剪/翻转、drop_last）；`no_grad`不等于`eval`，BN仍可能更新。
4. 计算source类中心，调用IMP：alpha=.05、5次迭代、已知中心固定。阈值和分配尺度由target全局方差计算。
5. source/target中心做匈牙利匹配，未匹配target中心构成nomatch；原型引导已有头的排列，并供虚拟损失使用。
6. RTA训练70轮；warmiter=3，`epoch<=3`即前4轮CE+virtual。随后CE+.01virtual+.3adv+1entropy+1unknown（命令行lambda可改）。
7. 每轮结束，对训练期间累积的source/target表示再次IMP聚类，重置source中心为当轮中心并固定，更新nomatch。不保留跨轮未知组件身份，不改变分类头维度。
8. 保留RTA GMM关系判别、未知槽argmax伪标签、最终C+2 argmax后合并未知的评价。best按目标标签HOS选epoch。

## IMP 具体规则（照录行为，不称论文忠实实现）

设target特征维度D，`rho=features.var(dim=0).mean()`、`sigma=sqrt(rho)`；alpha=None时实际回退.05，而不是自动估计alpha。

```text
lambda = -2*sigma*log(alpha) + D*sigma*log(1 + rho/sigma)
最近中心的平方距离 > lambda → 将当前样本作为新中心
p_ik = exp(-||z_i-mu_k||²/(2*sigma²)) / (sum_j exp(...) + 1e-8)
已知中心固定；其他中心用target软分配加权均值更新
责任质量 < 1e-8 的非固定中心删除；重复5次
```

若`fix_source_centroids=False`，已知中心也直接用target加权均值更新；没有持续source锚点/伪计数约束。此过程没有随机采样，是确定性扫描与软更新，不是Gibbs采样，也没有5次必然收敛的证明。顺序改变可能改变建簇结果。

## 与当前 K-only 方案的边界

| 项目 | 本旧方案 | 当前已执行方案 |
| --- | --- | --- |
| IMP对RTA的主要作用 | 改虚拟原型集合V/Q，未知槽固定2 | 将新增原型数作为K_out，保留原RTA虚拟聚类Q20 |
| source前置阶段 | 5轮、C+2头只用前C维CE、固定LR | 3轮纯C输出CE，共享checkpoint与source调度 |
| 阈值/尺度 | target全局方差+alpha公式 | source类内距离99%分位数/类内残差方差 |
| 已知中心 | main中固定 | 默认可移动，source锚点伪计数κ=5；另做固定消融 |
| 特征提取 | 训练增强/BN模式，逐batch变化的特征 | eval固定裁剪、完整数据、同一网络坐标 |
| 建簇顺序 | 输入顺序逐样本扫描 | 确定性最远点优先 |
| 更新时机 | 每epoch更新虚拟原型 | 一次估K固定；交替版20/40/60轮更新分类头K |

当前方案仅传整数K，不传IMP中心方向或责任矩阵作为RTA损失监督。因此两条路线检验不同假设，不能用当前K-only结果替代旧方案实验结果。记录旧方案不等于授权重启旧实验。

## 保留的已知问题（未在快照修复）

- `max_clusters`参数未被使用，没有实际容量上限。
- 按所写公式，其他量固定时alpha越小，lambda越大，建簇越难；“越小聚类越多”的注释与实现相反。
- 指数可能全下溢为0；分母+1e-8得到全0责任，不是合法归一化概率。没有rho/sigma正数下界，退化特征可能产生NaN。
- source中心数量/空类、N=0等缺少输入保护；`nomatch`为空时`np.stack`报错。
- 原型匹配已有头不能扩容；原型总数少于头行数时，copy存在维度不匹配风险。
- main部分强制`.cuda()`，虽有use_cuda分支，仍不是真正支持CPU。
- main使用Office31类别与评价硬编码，不能原样用于OfficeHome/VisDA。
- 文件没有可归属本快照的seed、配置/checkpoint和完整实验指标；不把其他版本的成绩归到这里。

## 后续若比较这条路线

应独立命名“adaptive virtual prototypes”，与“adaptive unknown slots”分开。控制任务、seed、source预训练、特征提取、优化器及预算，才能区分改变虚拟原型与改变未知容量的效果。当前仅归档，不启动新训练，不替换现有主线。
