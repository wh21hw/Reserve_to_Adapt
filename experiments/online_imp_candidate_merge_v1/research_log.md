# Research log

先行工作核实：Kulis/Jordan ICML2012 Eq1与Theorem3.1为hard SSE+lambda K及局部下降，https://icml.cc/2012/papers/291.pdf。这里不把原softIMP直接宣称DP-means，增加条件后处理objective merge；known中心/先验项在合并阶段不变，candidate对合并后对全部target重新分配，记录并确认实际目标下降。

预声明only merge因素，两组都是current_members。fafc50f4/exec36，T4/shell13/Drive缓存复用，同warm4各6轮，最多3600秒每arm/7200秒总。接口检查目标下降/known身份/no-forced-merge和生成训练代码通过。lambda沿用原校准距离阈值，没有扫参/语义K，没有额外损失。仍需和source_only背景比较，不能只打败失败承接组就推广。

完成：exec36/collector0，实际454.06秒；ZIP首次fetch失败重试下载成功，无重训。no_merge final96.5456/49.6126/65.5438，objective_merge95.2511/52.1034/67.3601。已知−1.2946pp guard失败，HOS+1.8163不采用。相对source_only背景97.6774/54.9944/70.3694，三项均更低，不能声称回到原强baseline。两组fullbest共享warm4，post4best均5，HOS76.1255/74.6532。

实发生合并仅epoch5两次、epoch7一次：objective117.881805→117.619853、103.957233→103.874156，合计约0.345030；其它轮未接受。K34→32→31，并没有充分消除过分割，且目标跨epoch变化因为features/lambda变化，不能拿跨轮曲线证明单调。未知预测ARI .35190→.35881，NMI .67773→.67525，结构改善未证明。

实际独立unknown结构监督覆盖128→140、误known覆盖24→0，说明合并影响了标签可靠资格和训练支持；不过原r选中known总曝光123→129，错误known entropy中unknown权重741→829。类别21曝光仍0，29只有1。不能把缺少训练误称不需要分类空间。下一步从恢复原source_only/无merge/正常BN的骨架出发，只扩展当前几何可靠IMP成员进入未知CE的入口，与原selector配对，不额外删原selector/改entropy/调阈值。该计划待另批预声明和实现；本批budget结束，整体三任务未达成。
