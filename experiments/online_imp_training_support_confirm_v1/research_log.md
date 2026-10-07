# Research log

CONFIRMATION RUN: verifying the guard-passing unknown-support improvement. First control/union final HOS65.9564/72.7749，known97.3441/97.1115、unknown49.8748/58.1918。背景控制跨进程漂移，确认一次、两次全留，不选最好重复。

预声明同commonwarm4各6轮，无新参数/seed/determinism/BN/损失/预测变化。8b92b200/exec43，T4/shell13/mount缓存复用，只新增安全输出run-name防覆盖，无新算法重复smoke/hash/预热/数据下载。每arm3600/总7200秒，实测重估hook。整体三任务未知结构尚未完成。

完成：exec43/collector0，训练464.39秒。ZIP/summary/posthoc正常下载；summary等待过程读同ZIP提取，不重训。control final97.6774/54.3523/69.8415；union96.7679/60.6717/74.5819。guard通过：known−0.9096pp（仅0.0904pp余量）、unknown+6.3194、HOS+4.7404；第一批Δ−0.2326/+8.3170/+6.8185也通过。两次同seed3共同warm重复，不是multi-seed显著性结果，不能只挑第一次或用均值掩盖guard。

fullbest两组均共享warm4，post4best都5 HOS75.3200/75.3096，收益体现在后期final。K最后22/V9 vs18/V6。第二次ARI .470685→.494557、NMI .694685→.714042，但第一次下降，不宣称语义分组稳定改善。teacher新增197（181unknown/16known、独立76/5），独立unknown结构监督138→152；误known结构标签7→31、新增污染8.12%，类别21新增仍0。known entropy中的unknown权重800→822，原权重没有改，风险仍在。

两次收益方向一致，允许进入完整70轮配对而非再做短程第三次。完整训练仍从同warm4接续5–70，保护warm公平，不变seed/阈值/BN/预测/损失；预算与运行目录另批预声明。保留source3/frozenencoderBN非原论文口径、target-oracle方法选择/best及事后seed caveat，仍需与原RTA独立真实baseline和三数据集验证。整体目标未完成。
