# 熵力度确实改变已知—未知取舍：保留两个候选继续验证

## 同批结果（%）

统一reliable_union/source_only每轮IMP/confidence/screened/virtual/原alignment/unknownCE/ResNet50/C10/source3/frozenencoderBN/seed3/batch64。仅raw IMP unknown候选的原known entropy权重乘数gamma改变；known身份成员权重保留，不是全局关熵。

| gamma | full best epoch/OS*/UNK/HOS | post4best epoch/OS*/UNK/HOS | final OS* | final UNK | final HOS |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | 4 / 86.4022 / 73.9103 / 79.6696 | 5 / 91.1287 / 64.1702 / 75.3096 | 97.1382 | 59.0648 | 73.4614 |
| 0.5 | 同共同warm4 | 10 / 94.8535 / 64.8620 / 77.0418 | 94.8535 | 64.8620 | 77.0418 |
| 0 | 10 / 92.9232 / 72.1939 / 81.2573 | 同fullbest | 92.9232 | 72.1939 | 81.2573 |

gamma0.5相对1：OS*−2.2848/UNK+5.7972/HOS+3.5804pp；gamma0相对1：−4.2150/+13.1291/+7.7959pp。gamma0相对0.5再牺牲known1.9302pp，换UNK7.3319/HOS4.2155pp。三个点在本批OS*/UNK平面互不支配；0组HOS最高但不是证明普适sweet point。按新评价政策保留0.5和0两个候选，不按旧known1pp否决，不默认抛弃保护known的重要性。

## 结构与监督

ARI/NMI：gamma1 .48528/.71636，0.5 .52956/.73784，0 .56684/.76955；本批拒识与分组都向好。各K范围17–38、V3–20，final K/V=23/8、20/5、19/9，仍不能把K当真实unknown数。

实际known entropy减弱权重：gamma0.5真known89.5/真unknown232.5，剩真unknown496.5；gamma0真known148/真unknown375，剩136。alignment两组都未屏蔽，剩unknown729/511；模型下游轨迹使这些计数改变，不把跨分支差全部当直接熵删除效应。它们是加权曝光，不是loss大小。

独立unknown结构监督146/140/170；误known结构标签曝光30/25/55。teacher新增170/176/268次，其中known16/12/22（9.41%/6.82%/8.21%污染），独立unknown65/72/98。0组结构覆盖更多，但known误结构标签也更多；不能只看HOS忽略这个风险。类别21新增三组均0、实际结构标签仅2/0/2次，未知类别覆盖缺口仍在。

## 下一步判断

本批支持熵目标存在可调取舍，而非“known下降>1pp必然无效”。half与zero值得完整预算验证，不继续密集扫gamma，也不以短程最高点宣称类别结构已解决。

下一批优先gamma0.5与0同warm4续训5–70、方法固定，直接比较两个候选；已有gamma1/union完整70轮仅作独立参考，不伪称本批三组严格配对。这样不重复稳定gamma1完整训练，也不把不同预算的10轮最高结果与70轮final混比。具体预算与启动另行预声明，此报告不声称已启动。

## 完成、保存与限制

exec46/collector0，三组process-status exit0、timed_out=false，各own5–10加同shared4，共18新轮。elapsed224.32/229.53/229.77秒，合计683.62秒≈11.39分钟。ZIP/summary/posthoc/图已下载，小结果mounted Drive由collector保存；云端独立同步仍受API项目quota限制。模型runtime保留，关闭前备份，不hash/CRC/checkpoint重评。

训练代码aa20128b、记录ab8bd01c、绘图38d92f4e。best/配置事后target-oracle选择、seed3事后选择、额外source预热/encoderBN均披露；target真值仅正常评价与记录后注释，未用于K、阈值或训练。这是单seed短程探索，不是统计显著性、完整论文复现或三个数据集验证；整体目标仍未完成。
