# 旧方案加快RTA预热

用户2026-10-08授权试缩短RTA预热、尽早训练未知。唯一因素RTA前4轮预热缩为前2轮：warmiter3改1（内部epoch从0起），未知CE/entropy/adv从第3轮开启，预热末尾C+2 K-means未知头初始化也提前到第2轮。其余损失计算与BN前向保留，零权重前向仍可能影响BN，不称预热完全没有target影响。

固定alpha.05/source5、seed3、RTA20，freshImageNet完整源微调动量带入RTA，C10+2/ResNet50/256/batch64/原BN/旧IMP5步fixedknown/virtual/argmax及原损失系数，不新版动态K/union/gamma。参考已完成旧方案同seed3/source5/alpha.05/RTA20，仅预热进程变化。不同运行仍有数值噪声，不称严格同轨迹fork。

每arm1500秒总4500预算（仅一个arm），预计11–13分钟，当前参数v2两arm完成后接续，不打断或同GPU抢资源。等待launcher8396最多4500秒，只有其results.zip成功才启动；失败保留证据暂停不追加。

新warmiter接口做一次compile/build-only针对性检查，验证loss开关与未知头初始化共用变量，不重复训练smoke/hash/checkpoint重评。保留每epoch所有评测和best/last，比较UNK上升速度、OS*代价、HOS峰值/final/尾段稳定性，不承诺超过论文93。target真值仅评价，事后seed/方法/epoch选择探索披露。小结果Drive+电脑，关闭runtime前模型必须实际备份。

状态：完成。20轮完整，process-status exit_code=0、timed_out=false，collector与launcher均0。逐epoch记录已下载电脑pipeline-results/legacy-imp-warmup2-20e-v1-results.zip。小记录按runner复制mounted Drive，不能独立保证云同步；模型仍在runtime，关闭前需实际备份。

已启动：参数v2 exec5完成collector0后，等待driver自动启动warmup launcher17227、worker17228。console /content/legacy-warmup2-console.log，shell14，运行目录/content/imp-runs/legacy-imp-warmup2-20e-v1。只本单arm，不再追加新的实验。

## 完成结果（2026-10-08）

best与final均第20轮：OS*=92.0805%、UNK=86.9633%、HOS=89.4488%。V范围0–4，最终2；分类头未知槽始终2。耗时690.67秒（约11.51分钟，包含source阶段）。相对同seed3/warm4/20轮final，OS*+1.9570pp、UNK+1.5982pp、HOS+1.7690pp；相对其best HOS+1.1195pp。仍低于论文A→W HOS93.0%，也未超过原版RTA复现best95.2172%。

warm2第3/4/5/6轮UNK分别34.6676/68.5617/85.8093/92.7964%，第6轮OS*降至80.6021%，后期回升。最后5轮HOS为88.2052、89.4110、89.2487、88.8684、89.4488%，峰值出现在预算末轮，不能断言完全收敛。结论仅支持缩短预热改善本次20轮工作点，不证明未知语义发现或稳定超过RTA。

属于目标标签选择seed/配置/epoch的探索，target标签没有进入训练或IMP阈值。旧用户两个snapshot的历史依赖未完整提供，固定bridge依赖而非完整旧环境逐字复原。等待hook已暂停，不追加实验，不关闭实例。
