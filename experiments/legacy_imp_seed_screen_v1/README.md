# 用户旧方案原配方seed筛查

2026-10-07用户明确要求严格按刚提供的main (2).py、IMPClusterer (2).py试几个seed。运行Office31 A→W，seed1/2/3，各重新ImageNet初始化、source5轮固定LR微调、RTA最多20轮，总source15+RTA60。不是共享seed3 warm后改RNG。每seed1500秒、总4500秒硬保护，预计25–40分钟，观察实际轮间耗时更新ETA。

固定C+2头/C10/batch64/ResNet50/256/α.05/IMP5步/固定已知中心/每轮nomatch virtual方向/原伪标签argmax/原熵和对抗权重。source SGD LR5e-5、head5e-4，momentum.9/nesterov/wd5e-4，动量复用进入RTA；warmiter3意味着RTA前4轮CE+virtual。不使用新版动态K、标签桥接、可靠union、熵屏蔽、BN冻结或source-only3 prior。

引用用户2026-10-07下载文件，新独立snapshot目录。依赖data/networks/utilities/centroid/domain_bus使用已固定RTA依赖/content/rta-legacy-l4-bridge-v1（用户未提供历史依赖，因此不能声称整个环境逐字相同）。工程修改仅路径、seed、预算、日志checkpoint/no-overwrite/evalno_grad/关闭浏览器自动下载；复用已声明空nomatch与头匹配行数边界处理，不改IMP数值公式。出现NaN/OOM先保留证据定位，不静默重写softmax/阈值。

目标是保留三个seed全部逐轮结果，允许展示最好seed/epoch，但标注target-oracle和单任务短程探索，不伪称70轮收敛或多seed均值。评估best/final OS*/UNK/HOS和V范围，论文A→W HOS93.0、本地原RTAseedbest95.2172是不同参照。训练目标标签不进入IMP/阈值/损失。小结果Drive+电脑，模型关闭runtime前备份。已有exec51运行不干扰、不并行争同GPU，不在训练kernel排队新exec。

状态：准备中，尚未启动。
