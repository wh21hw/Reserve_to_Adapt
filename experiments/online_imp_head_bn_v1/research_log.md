# Research log

DEEP PIVOT：多个教师/筛查改动未同时满足known/unknown guard，停止增添阈值与候选门槛，回查训练坐标。原CLS BN在各前向batch使用独立统计，IMP eval使用running统计；是否实际导致退化尚待单因素验证，不把代码差异当成已证明原因。

预声明batch_bn vs fixed_bn，共同warm4后各6轮，保持每轮IMP、原loss和预测，encoder不变。a0991367/exec27/launcher54725，T4/mount缓存复用。接口功能检查通过，初次代码build缺环境变量已补齐，无训练设置改变或重复预热。预计8分钟，按真实history重估。总体未知结构建模及三任务尚未完成。
