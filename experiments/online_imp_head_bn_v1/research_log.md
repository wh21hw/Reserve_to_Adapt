# Research log

DEEP PIVOT：多个教师/筛查改动未同时满足known/unknown guard，停止增添阈值与候选门槛，回查训练坐标。原CLS BN在各前向batch使用独立统计，IMP eval使用running统计；是否实际导致退化尚待单因素验证，不把代码差异当成已证明原因。

预声明batch_bn vs fixed_bn，共同warm4后各6轮，保持每轮IMP、原loss和预测，encoder不变。a0991367/exec27/launcher54725，T4/mount缓存复用。接口功能检查通过，初次代码build缺环境变量已补齐，无训练设置改变或重复预热。预计8分钟，按真实history重估。总体未知结构建模及三任务尚未完成。

完成：batch/fixed共454.30秒，ZIP与summary下载成功。两组fullbest共同warm4 HOS79.6696；post4best分别5/75.3200、9/79.5166；final98.3441/53.4242/69.2365 vs92.3548/62.2183/74.3487。Known−5.9892pp，guard_violation，不保留为默认BN模式，未调保护门槛。

记录诊断：source已知被原unknownselector选中总曝光106→147，IMP身份覆盖known7→16；unique未知结构覆盖134→132，未增加覆盖。未知预测ARI .47583→.37102/NMI .70733→.64936，不能将拒识率上升称为语义结构改善。known entropy含unknown权重787→956，没有屏蔽或修改过这些权重。当前结果不支持“固定BN足以修复”，也不能排除归一化有影响。K最终19→29，多槽非充分条件。开始epoch10结构不是最终网络结构。下一方向回到每轮簇重建及身份承接，保留正常BN，不叠加失败mask。
