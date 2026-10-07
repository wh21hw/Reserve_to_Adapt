# 完整70轮：更高未知收益伴随更大已知代价

## 指标（%）

同commonwarm4，各续训epoch5–70共66新轮，总132；T4/Office31 A→W/ResNet50/C10/source3/frozenencoderBN/seed3/batch64。统一reliable_union入口、每轮source_only IMP/confidence/screened/virtual/原alignment/unknownCE/argmax/头SGD匹配。唯一因素raw IMP unknown候选的known entropy样本权重gamma0.5 vs0；known身份保留1。

| gamma | 选取 | epoch | OS* | UNK | HOS |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.5 | full best / post4 best | 12 | 96.0538 | 72.0037 | 82.3078 |
| 0.5 | final | 70 | 97.4086 | 67.3161 | 79.6136 |
| 0 | full best / post4 best | 20 | 93.4784 | 87.3198 | 90.2942 |
| 0 | final | 70 | 85.6462 | 89.2638 | 87.4176 |

zero−half final：OS*−11.7624pp、UNK+21.9477pp、HOS+7.8039pp。两个final在OS*/UNK平面互不支配；zero HOS更高、half明显保护known，不按旧1pp门槛一票否决，也不忽视zero的已知代价。zero从best20到final70：known−7.8322pp、UNK+1.9441pp、HOS−2.8766pp，后期继续偏向未知。它是有价值的偏未知工作点，不是已经找到稳定普适最优点。

此前独立gamma1/union70 final99.6667/56.9958/72.5200仅背景，不是这批严格三组配对。不能以跨运行差值当无噪声因果估计，不能将10轮最高点与70轮末轮混比。论文A→W HOS93.0%仅参考；本批zero oracle best低2.7058pp，仍非原论文流程或已对标成功。

## 结构和训练支持

final ARI/NMI：half .58167/.78819，zero .73572/.86247；zero分组指标也较好，但ARI/NMI不能抵消已知误拒风险或证明K等于真实类别数。K范围两组17–38，final31/32；V范围3–24/3–34，final23/32。

独立真unknown结构监督206→245；unknown CE独立覆盖223→248。结构标签误known曝光276→2968，原/扩展selector选中known655→3510。zero确实使更多unknown获得监督，也更容易把known送入unknown结构。这是实际记录后的标签注释，不是模型重评或完整因果分解。

teacher新增曝光half1269=1227unknown+42known（3.31%污染），独立141unknown/7known；zero2061=1821unknown+240known（11.64%污染），独立199unknown/39known。这些独立人数是各组新增入口覆盖，不是相对另一组净新增数。unknown类别21总结构曝光仍仅93/97次，新增入口2/8次，覆盖仍不均衡。

entropy削弱记录：half移除true-known1317/true-unknown1760加权曝光，剩unknown7428；zero移除952/1336，剩unknown2353。alignment没有屏蔽，unknown权重9188/3689。不同学习轨迹和原known posterior改变了总曝光，不能把zero移除数较少解释为干预更弱，也不能把权重次数直接当loss幅度。结构快照明确为第70轮开始，非final70特征。

## 完成与保存

exec47/collector0，两arm process-status exit0/timed_out=false。本地summary两组history均完整1–70；elapsed2452.71/2490.83秒，合计82.39分钟，接近最初约80分钟估计。ZIP/summary/posthoc/取舍图/progress已下载。posthoc原URL两次fetch失败，改传输文件名成功；只处理传输，没有重训、覆盖collector或checkpoint重评/hash/CRC。

mounted Drive小结果summary893KiB、ZIP1.4MiB可见，独立云同步仍受API项目quota限制，不伪称已验证；模型保留runtime，关闭前需备份。训练代码0e051825、记录fbd468ab，普通训练日志用于全部评价。用户根目录原修改保留。

## 披露与停止

seed3事后选择、方法工作点/epoch使用target标签作探索性选择，非多seed统计；target真值没有参与K/阈值/训练。额外source3/冻结encoderBN非发布版RTA完整流程，OfficeHome/VisDA尚未验证。

用户已明确暂停主动研究以节省token。本报告只完成当前批次必要收集和评价，不提出或启动下一实验，不自动恢复目标。等待hook已暂停，当前实例不关闭、不重启；研究目标保持paused，等待用户主动恢复。
