# 旧方案硬原型更新单因素对照

用户2026-10-08授权。Office31 A→W、fresh seed3、source5、RTA20、warm2、alpha.05、IMP5步、known固定、分类头C10+2、ResNet50/256、batch64、原BN/virtual/未知argmax/entropy/adv/SGD动量复用全部不变。

唯一变量：IMP迭代的原型更新由Gaussian软责任度加权均值改为最近原型硬分配后的簇内均值。建新簇仍用原平方距离阈值公式；空unknown簇沿用原fit删除，已知锚点保留；每轮重新提取/聚类，不新增EMA或未知软伪标签，不改输出K。固定known中心不因硬分配移动。

对照复用刚完成同seed3/source5/warm2/alpha.05/20轮软更新：best=final第20轮OS*92.0805%、UNK86.9633%、HOS89.4488%。这是fresh同配置对照而非相同checkpoint轨迹fork，存在运行数值噪声。

预算：单arm1500秒、总4500秒保护（仅单arm），根据前批预计11–13分钟，实际观察前几轮后重估。仅一次针对性硬分配功能检查，不重复稳定smoke/hash/checkpoint重评。保留全部20轮指标、best/last模型，target标签仅评价；不改alpha/换seed/延长70或失败静默改公式。

预计root/content/imp-runs/legacy-imp-hard-update-20e-v1，archive/content/legacy-imp-hard-update-20e-v1-results.zip。small日志/配置/summary自动复制mounted Drive并下载电脑，模型留runtime关闭前实际备份。完成报告全部best/final、峰值epoch、V范围和尾5轮及与soft的取舍，不只选更高best；报告后暂停hook，不关闭实例。

状态：已启动。T4 gpu-t4-s-kkb-usw4a2-10aku92peratz、shell14、launcher26344/worker26345，console/content/legacy-hard-update-console.log。代码91592a7e。一次硬分配与均值针对性检查exec6完成；source前3轮loss2.0283/1.1280/.6490有限，无OOM。复用现有T4与挂载，不重新下载数据。5分钟hook已启用，首个检查按RTA近期elapsed重新估算，正常保持安静。

## 完成结果：2026-10-08

20轮完整（epoch1–20）、source5、warm2、imp_update=hard；process0/timed_out=false、collector0。含source耗时688.59秒，约11.48分钟。小结果已复制mounted Drive（summary可见），电脑ZIP为pipeline-results/legacy-imp-hard-update-20e-v1-results.zip，未独立确认云同步。best/last模型仍runtime，关闭前必须实际备份。等待hook暂停，不追加实验或关闭实例。

| 更新规则 | 选取 | epoch（1-based） | OS*% | UNK% | HOS% |
| --- | --- | ---: | ---: | ---: | ---: |
| soft | best=final | 20 | 92.0805 | 86.9633 | 89.4488 |
| hard | best | 18 | 91.1813 | 87.3962 | 89.2487 |
| hard | final | 20 | 92.0805 | 86.5681 | 89.2393 |

hard相对soft：best HOS−0.2001pp；final OS*相同、UNK−0.3953pp、HOS−0.2095pp。V均0–4、final2，分类头unknown槽始终2。未超过论文HOS93.0或原RTA复现best95.2172。这次无明显改善，不足以断言hard在所有设置都更差；同seed fresh而非checkpointfork，有数值噪声，尚非多seed统计。

hard最后5轮HOS：88.3620、89.2066、89.2487、89.0939、89.2393%。第3/4/5/6轮UNK35.3278/67.8634/86.5998/93.1917%；第6轮OS*80.9469%。软版第6轮OS*80.6021/UNK92.7964%，两者都发生前期拒识上升、known下降、后期known回升的现象。硬更新没有消除这一模式，不证明其唯一原因。

保留每epoch日志，峰值epoch以history顶层18为准，嵌套best.epoch17为0-based。target真值仅评测，事后seed/config/epoch选择披露；旧用户snapshot依赖未全提供，不声称完整历史环境逐字复原。
