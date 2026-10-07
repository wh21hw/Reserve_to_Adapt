# 实验记录

2026-10-07：预声明三点gamma1/0.5/0，共享warm4各续训6，18新轮，每arm3600/总10800秒预计12分钟。只有候选unknown known entropy权重改变，其余统一reliable_union。接口一次检查通过。代码aa20128b，exec46/launcher146130，5分钟hook运行，模型/挂载保留。

entropy1 epoch6/7 elapsed77.6948/114.8971，37.20秒/轮，首组剩3轮约2分钟，另两组约8分钟，当前全批预计余10分钟。三点取舍图工具仅读summary记录，展示三指标轨迹及final OS*/UNK平面，不沿用1pp known guard自动否决。gamma1原算法控制也有数值漂移，因此只作本批同预算比较，不挑以前更弱control制造收益。

完整结果三组exit0/collector0，683.62秒，final HOS73.4614/77.0418/81.2573；known97.1382/94.8535/92.9232，UNK59.0648/64.8620/72.1939。raw候选熵干预改变取舍，half/zero保留为待完整验证候选，不改旧guard记录，不挑best隐去final。ARI/NMI向好，zero误known结构曝光55仍是风险。小结果下载成功，图根据真实记录生成，修复一处文字布局，不重新评价模型。下一步完整预算比较half/zero，gamma1已有70轮作非配对背景，暂未启动。
