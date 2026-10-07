# Research log

先行工作核实：Kulis/Jordan ICML2012 Eq1与Theorem3.1为hard SSE+lambda K及局部下降，https://icml.cc/2012/papers/291.pdf。这里不把原softIMP直接宣称DP-means，增加条件后处理objective merge；known中心/先验项在合并阶段不变，candidate对合并后对全部target重新分配，记录并确认实际目标下降。

预声明only merge因素，两组都是current_members。fafc50f4/exec36，T4/shell13/Drive缓存复用，同warm4各6轮，最多3600秒每arm/7200秒总。接口检查目标下降/known身份/no-forced-merge和生成训练代码通过。lambda沿用原校准距离阈值，没有扫参/语义K，没有额外损失。仍需和source_only背景比较，不能只打败失败承接组就推广。
