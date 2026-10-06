# 使用情况诊断记录

观察起点：70轮扩K2→5相对固定K2，best相同，final HOS仅+0.1313pp。迁移日志确认保留两行并新增3个随机norm匹配行；训练history不记录逐槽正伪标签。不能用这些日志直接断言新增行从未被使用。

测试计划：一次final70/K5无目标真值的冻结cache，统计完整target和source的全头/unknown-only赢家，另从缓存裁去新增行观察是否改变当前target二元拒识。没有训练、重评分checkpoint或K搜索；hypotheses.md中两种解释均先记录。

## Iteration1 observation（2026-10-06）

复用稳定cache入口，声明expected-epoch70、expected-K5、实际完整last.pt；一次提取source958及target564，图像前向约9.05秒。缓存已保存Drive，未使用target真值/训练/GMM拟合/K重估。随后diagnose_a2w_new_slots_colab.py只读缓存及epoch10迁移对应，exit0。

target预测unknown237张，原unknown行10/11分别104/133张；新增12/13/14各0张。仅在5个unknown行内部argmax，旧行240/324张，新行仍0。source958张的全头预测均为known；其unknown-only赢家为556/402/0/0/0。

每个新增行相对最佳旧unknown行的最大target logit差依次−2.1973/−1.0869/−1.4306，所以不仅是全头竞争被known压过，新行连旧unknown行都没有赢过。删去三行对target564/source958的任何最终预测与二元拒识均0次改变。

epoch10的五个候选未知结构，当前最终模型的描述性对应为：candidate10有76张全落旧unknown10；candidate11有134张，其中23被判known、111落旧unknown；candidate12有53张，其中4被判known，余49被两旧unknown拆分；candidate13/14的15/21张全被判known。这里只是候选ID与模型预测对应，没有真实语义输入，不能将后两簇称已证明的误建已知类。

## 能确认与仍不能确认

确认的最小现象：当前最终K5的功能未知赢家仅使用两个旧槽，新增三槽没有功能使用；从缓存删除新增三行可复现全部预测不变。它解释为何“头维数增加”本身没有带来实际额外拒识，但尚不是整个训练退化机制的根因证明。

训练中的正伪标签、增广特征及train-mode head BN没有记录，因此不能说新增槽训练期从未有梯度/正监督；随机新行的冷启动、自举强化旧赢家是后续待验证解释。后续若要改变新行初始化，需要独立版本和单因素预算，不偷改K-only版本、追加损失或把簇语义当已知事实。

预算已完成，保留partial root-cause report，不宣称全部根因已确认。下一项优先验证新槽冷启动/初始化与实际RTA头输入空间的兼容性，再决定训练对照；不继续盲目增加K。

## 保存与复现入口

cache：Drive `OSDA/runs/a2w-capacity-refresh-k2-70e-final-v1/{features.npz,manifest.json}`；统计：`OSDA/runs/a2w-capacity-refresh-k2-70e-slot-usage-v1/summary.json`。本地小汇总和manifest在pipeline-results，feature NPZ可复用，不重复提取。

命令入口：cache_a2w_current_relation_colab.py指定上述last.pt、epoch70/K5及独立output/drive-output；随后diagnose_a2w_new_slots_colab.py指定cache、refresh的capacity-after-010、独立output/drive-output。诊断不接收target语义标签或训练参数，不加载模型，不改变任何checkpoint。
