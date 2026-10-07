# 未知结构标签覆盖：保留RTA筛选，单独移除标签中心筛查

## Goal

从真实曝光证据解决“发现了簇但没有充分学习”。上一批256模式11类虽都有标签，但printer(21)仅3次、tape(29)5次；2048模式7类0次。原始未知候选覆盖78–87%，中心筛查后只16–28%。不改ResNet或阈值、不继续换层，检验标签筛查是否过于保守。

## Success Metric

有效第10轮OS*/UNK/HOS与5–10轨迹，HOS至少+1pp/OS*和UNK各不下降超过1pp筛查，人工综合。报告实际独立未知覆盖、各语义类曝光、错误已知曝光、最终未知预测ARI/NMI、槽使用及校准污染。不让高覆盖替代保护已知；target真值仅结束评价，不决定标签/K/阈值。

## Constraints

- max_iterations:1；pause_every:never；Evaluator: _(none — agent judges manually)_。
- 复用online-imp-calibration-fork-v1共同warm4，两arm screened/all_candidates各续训6（epoch5–10），实际新训练12轮、有效各10，不重复warm/source或下载数据。
- 每arm3600秒/总7200秒；新接口300秒，覆盖技能5分钟默认；预计8分钟，按实际几轮重估并hook。T4/ResNet50/256teacher/A→W/C10/seed3/source-only3/frozenencoderBN/batch64/原RTA损失系数与预测。
- 两组同每轮当前IMP、target参与confidence阈值、动态K/头状态迁移、**virtual继续筛查**。唯一因素：原RTA筛出的样本中，仅可靠候选簇替换unknown槽标签 vs 所有occupied新候选簇都替换。known结构成员仍原unknown argmax，不改变样本的known/unknown监督类别或增加被选样本；只改unknown槽身份。后续模型不同会造成下游RTA选择不同，不能说样本集合永远相同。
- 不把全部target直接送unknownCE；原GMM/top16规则保持。raw候选会有误判已知，本批必须量化实际错误曝光与OS*guard，不假设安全。
- 两组同工程修正：new_fc构造用CPU fork_rng隔离，K/头随机初始化不能额外改变DataLoader/augmentation随机流。随机方向仍用当前状态，SGD/known/匹配unknown和schedule保留；不reset整个头。与旧批数值变化不作直接干净比较。
- K0/cap100失败保存停下，不强制K或扫阈值；不加loss、不BN/预测门控改造、不延70轮或换seed/数据集。
- 为避免再增加Drive大模型占用，**本批只小记录/ZIP上传Drive，best/last留runtime**，关闭前备份需要模型。云端独立API Queries限额未确认解除，不擅自永久删旧文件。

## Search Space

只标签覆盖范围。独立SEM覆盖和已知错误曝光是诊断，不能依据target真值筛候选或挑最佳配置。

## History

已预声明；复用T4 gpu-t4-s-kkb-ass1c2-1j7bns9e7dhke。上一特征层试验不晋升（OS*−1.10pp guard）；本批仍256，并修正采样RNG耦合，两组共享修改。

完成：代码f7563ac7/exec14/launcher30342，两个续训6均exit0，collector done。finalHOS67.7184→68.7055%，Δ+0.9871pp未达+1筛查；独立未知结构标签138→190，但错误已知覆盖7→96次，ARI近乎不变。不能说更多标签有效解决。下一因素是known目标权重与IMP身份协调，单独新目录，不延长本批。
