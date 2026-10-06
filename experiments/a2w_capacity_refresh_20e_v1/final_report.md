# A→W：阶段性容量更新的20轮对照

## 结论

本批初步支持“在RTA当前表示下阶段性更新未知容量”：第10轮之后由K8调整为无标签估计的K4，final HOS比始终K8高8.1101pp、UNK高11.7097pp，OS*低0.3226pp。主要改善是后半程拒识保持，不是best峰值显著提高。

这是相对固定K8的单seed短程结果，尚未证明优于原版固定K2或达到论文成绩；也没有恢复真实未知语义数。当前unknown宏平均漏检仍约25.6%。整个研究目标未完成。

## 设置与唯一因素

- Office31 A→W，958个source/564个target，C10/Q20，seed3（此前事后选择），两组各完整20轮。
- 同一Tesla T4、Python3.8.20、torch1.7.1+cu110、numpy1.23.4、batch64、RTA lr5e-5。
- 两组复用同一个C-only监督3轮source checkpoint；encoder BN running stats冻结、affine可训练，head BN不变。此前单独source预热/BN策略不等于论文原始完整设置，不能把本批称为论文复现。
- 同样初始K8、原版warm-end K-means、原RTA损失/virtual Q20/候选筛选/unknown槽argmax/最终argmax。
- 两组都在10轮完成后提取当前source/target并按同一source-calibrated-birth-cost-v1+已知身份匹配推断。fixed8推断K4但不应用；refresh推断K4并应用。
- 只将容量/无标签状态对应用于head resize；不把簇ID当训练标签，不用IMP方向初始化，不加新loss、门控或未知概率求和预测。
- 两组正常exit0，无NaN/OOM。每个20轮arm结束后模型/日志/边界缓存实际保存Drive，再收集日志；没有重新评价checkpoint。

## 指标（全部epoch为history的1-based）

| 方法 | 选取 | epoch | K | OS*% | UNK% | HOS% |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fixed8 | best HOS | 7 | 8 | 95.4139 | 77.9881 | 85.8254 |
| fixed8 | 更新节点后best | 12 | 8 | 97.0108 | 74.3927 | 84.2094 |
| fixed8 | final | 20 | 8 | 99.6667 | 62.7251 | 76.9941 |
| refresh | best HOS（也是节点后best） | 12 | 4 | 97.0108 | 77.5990 | 86.2259 |
| refresh | final | 20 | 4 | 99.3441 | 74.4347 | 85.1042 |

full best HOS差值仅+0.4004pp；final差值+8.1101pp。不能只报一个最高best就称大幅突破。两组的best都使用目标标签选epoch，属于oracle描述；final固定20轮。单seed不代表三seed均值或稳定性已经证明。

## 更新前的偏差

两组从相同source状态、相同设置开始，但不是严格同一训练轨迹。前10轮三指标最大绝对差异1.7045pp；更新前第10轮refresh−fixed：OS*−0.7877pp、UNK+1.6747pp、HOS+0.7337pp。

final相对差值较第10轮差值又扩大：OS*+0.4651pp、UNK+10.0350pp、HOS+7.3764pp。这个“差值的变化”只作描述，不能称无偏因果估计或显著性检验。它说明final优势不只是照搬更新前约0.73pp的HOS优势，但单次轨迹仍不能排除所有随机/自举分歧。

预声明人工筛查条件均满足（final HOS至少+1pp、OS*/UNK损失各不超过1pp）。保留实现作为下一轮验证候选，不修改默认训练/宣称正式方法已获证实。需确认重复/完整预算；本批max_iterations1与40轮已用完，没有自动追加确认训练。

## 机制：不是删头瞬间增加拒识

更新只保留了原未知头14/10/15/17对应的4行及SGD状态，已知10行不变，删掉另4行，不添加随机行。独立无标签脚本从已存第10轮logits及row_mapping重建即时pruning，不重跑图像/模型。

564张target，更新前预测unknown223张，删头后仍223张；known→unknown和unknown→known都是0，全部known/unknown身份保持。因此收益出现在后续学习中，不是softmax分母变小或立即改变预测规则。

纯删减未知行时最大未知logit只能不变或下降，已知最大logit固定，不能凭删头增加即时拒识。这一点与“少几个槽就自动更容易拒识”的直觉不同。已有4个保留槽覆盖了当前eval的全部unknown赢家，但不能由此宣称被删槽在训练增广/BN状态下从未参与或从未收到监督。

## 尚未解决

1. K4仍是建模容量，不是4个真实未知语义；当前结构/标定可以合并未知或吸收未知。此前同空间旧标定K9、当前标定K4的证据保留，未根据target成绩选标定。
2. 当前聚类使用normalized bottleneck欧氏空间，分类头经过BN/LeakyReLU后打分；度量一致性值得后续分析，但不是本批已证实的退化原因，也未在本批更换。
3. 尚未与同预算固定K2比较，不应把救回偏大K8的退化等同于超越原版RTA。
4. 70轮完整训练、其他seed、OfficeHome/VisDA及论文设置对齐都仍待完成，VisDA backbone口径仍未解决。

## 下一步与资源

建议先做共享同样source前置的固定K2与初始K2、阶段性估K的70轮配对，保留原RTA其他机制，不先堆新loss；这才能更接近回答是否优于原人工容量，而非仅固定K8。下一批预算已询问用户，尚未启动，不能将其写成已批准或已执行。

按用户要求未关闭CPU或T4。现场usage余额约222.36 CCU，CPU+T4约1.15 CCU/hr；这不是固定长期余额。下一批若不训练，T4仍会计费，需根据用户明确选择安排释放，不能伪称CLI可保留VM原地降级CPU。

## 保存与工程记录

Drive：`OSDA/runs/a2w-capacity-refresh-20e-v1`，包含每arm的best/last、普通日志、完整逐轮指标及无标签边界缓存，summary.json、结果ZIP、resize-rejection-diagnostic.json。本地：`pipeline-results/a2w-capacity-refresh-20e-v1-{summary.json,results.zip,pruning.json}`。

代码训练实现a2cb4669（普通Git版本记录，不做hash审计）。collect脚本追加描述性前后差值时有一次字符串续行IndentationError，修复后只重新读取日志打包，未重训/重评模型；失败日志保留/content/a2w-capacity-refresh-summary-console.log，成功为console-v2.log。raw指标与普通checkpoint未改动。
