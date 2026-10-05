# A→W：未知伪标签 CE 单因素对照

用户批准改用 Office-31 A→W，替代尚未启动的 VisDA CE 消融。仅两个 arm，各 RTA 10 轮；不跑新的 VisDA。

共同设置：L4、ResNet50、seed3、C10/K2/Q20、batch64、RTA learning_rate .00005、source C维监督3轮共享同一 checkpoint。source及RTA保持 encoder BN running statistics，affine可训练，分类头BN正常更新。原版warm-end K-means头初始化、虚拟方向、候选筛选、未知槽argmax、其余损失和最终预测都相同。

唯一研究因素：warmup后未知伪标签CE系数，control=1，off=0。off仍执行候选筛选与head forward，避免额外改变BN。该实验检验未知自举监督的作用，不是最终未知建模方案，也不是仅自适应K。A→W结果不能直接证明VisDA崩塌原因。

当前runtime没有可直接复用且初始化/BN/预算完全匹配的A→W控制；重新生成一次共享source prior，分别运行两个短程arm。不用历史70轮best或不同配方代替对照。seed3来自此前事后seed选择；报告两组全部10轮、目标标签选出的best以及第10轮final，不当三seed均值，不用target训练标签改模型或选阈值。

数据只从已持久保存的Drive `OSDA/datasets/office31_images.tar`复制到本地解压，不重复公网下载。输出 `/content/imp-runs/a2w-unknown-ce-10e-v1`。状态：准备中，尚无新结果。
