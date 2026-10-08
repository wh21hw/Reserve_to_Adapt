# 旧方案加快RTA预热

用户2026-10-08授权试缩短RTA预热、尽早训练未知。唯一因素RTA前4轮预热缩为前2轮：warmiter3改1（内部epoch从0起），未知CE/entropy/adv从第3轮开启，预热末尾C+2 K-means未知头初始化也提前到第2轮。其余损失计算与BN前向保留，零权重前向仍可能影响BN，不称预热完全没有target影响。

固定alpha.05/source5、seed3、RTA20，freshImageNet完整源微调动量带入RTA，C10+2/ResNet50/256/batch64/原BN/旧IMP5步fixedknown/virtual/argmax及原损失系数，不新版动态K/union/gamma。参考已完成旧方案同seed3/source5/alpha.05/RTA20，仅预热进程变化。不同运行仍有数值噪声，不称严格同轨迹fork。

每arm1500秒总4500预算（仅一个arm），预计11–13分钟，当前参数v2两arm完成后接续，不打断或同GPU抢资源。等待launcher8396最多4500秒，只有其results.zip成功才启动；失败保留证据暂停不追加。

新warmiter接口做一次compile/build-only针对性检查，验证loss开关与未知头初始化共用变量，不重复训练smoke/hash/checkpoint重评。保留每epoch所有评测和best/last，比较UNK上升速度、OS*代价、HOS峰值/final/尾段稳定性，不承诺超过论文93。target真值仅评价，事后seed/方法/epoch选择探索披露。小结果Drive+电脑，关闭runtime前模型必须实际备份。

状态：准备，等待参数v2完毕。

已启动：参数v2 exec5完成collector0后，等待driver自动启动warmup launcher17227、worker17228。console /content/legacy-warmup2-console.log，shell14，运行目录/content/imp-runs/legacy-imp-warmup2-20e-v1。只本单arm，不再追加新的实验。
