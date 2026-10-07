# IMP看哪层特征：ResNet不改，容量/结构标签视图对照

## Goal

当前结构支持校准未解决未知吸入：污染仍26.39%，UNK约52%。检验已知分类256维bottleneck是否限制未知结构建模。**只是待检验假设，不宣称源域监督已经证明丢失未知信息或2048维天然更好。** 同一个当前ResNet输出也可能带来更多域/背景噪声。

## Success Metric

共同warm4后的有效第10轮final OS*/UNK/HOS和epoch5–10轨迹；HOS至少+1pp、OS*/UNK各不下降超过1pp筛查，人工综合。报告K/V、teacher可靠候选、真实伪标签曝光、最终槽活跃、target支持污染；增加未知预测ARI/NMI和概率/语义对保存，仅正常评价/结束事后诊断，不进入训练或阈值/K/配置选择。

## Constraints

- max_iterations:1；pause_every:never；Evaluator: _(none — agent judges manually)_。
- 两arm从现存online-imp-calibration-fork-v1共同warm4完整checkpoint恢复，各续训6轮（epoch5–10）。实际新训练12轮，不重跑warm，不source预热；有效每组10轮。
- 总7200秒、每arm3600秒、关键接口检查300秒，覆盖技能默认5分钟。预计约8分钟、先观察实际轮间elapsed再估ETA。
- T4/ResNet50/A→W/C10/seed3历史事后选择/source-only3/frozenencoderBN/batch64/原RTA系数与预测。保持每轮IMP、动态K、结构标签；calibration使用confidence模式，上一失败结构支持模式不启用。
- 唯一因素为IMP容量/结构标签输入：归一化256瓶颈 vs 同一当前encoder的L2归一化2048输出。两者同一次图片前向已经得到，不改ResNet架构/权重初始化，不加loss。
- 两组virtual方向始终由256维IMP决定；2048中心绝不直接用于256头或virtual。backbone分支额外在当前256特征上拟合virtual，其变化只能由训练的下游变化产生，不独立改virtual算法。teacher可靠筛查用于label，V不必等于teacher可靠簇数。
- 新接口只做一次共同warm实际2048提取/IMP/头与virtual维数检查，0梯度/0新增训练轮，不作成绩重评或能否学会的代理结论；通过后再启动正式两arm。
- source中心和source/target支持尺度均在各自teacher当前坐标计算，同99%/.8/5/各半规则；维数改变带来的几何变化是该因子的下游，不单独调阈值。K0或容量100失败停留证据，不强制K、不扫参。
- 新增逐样本selected/overridden曝光计数和评价输出仅日志，不影响训练。warm计数不是实际unknownCE支持（前4权重0）。ARI/NMI是结构诊断而非替代OSDA评价，不能用高维多簇高纯度证明语义发现。
- 只best/小结果复制Drive，完整last/commonwarm留runtime，关闭前备份；当前Drive独立API限额未确认解除，不自动删旧文件。根目录用户训练文件不覆盖。
- 不延长70轮/换seed/其他数据集，完整observation后再下一单因素。整体三任务目标仍未完成。

## Search Space

仅IMP所看层。与上一批跨方法成绩仅背景，不把独立数值轨迹差异当同控制结果。

## History

已预声明，尚未启动；复用挂载T4 gpu-t4-s-kkb-ass1c2-1j7bns9e7dhke和共同warm4，不重复下载。
