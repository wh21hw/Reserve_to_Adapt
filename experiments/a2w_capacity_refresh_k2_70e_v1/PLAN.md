# A→W：固定K2与阶段性估K，完整预算对照草案

状态：用户2026-10-06明确“启动。不用找我确认。预算足够”，本批已启动。背景exec31运行串行配对，固定组worker PID44599；尚无完整结果或收益结论。初始声明预算140轮/4小时保持。

## 要回答的问题

从域适应的本质出发，容量需要服务于已知类跨域判别和未知拒识，而不是只追求更多聚类。K8→K4短程对照只证明值得继续调查；当前原版K2模型也能出现4个未知候选簇，故不能把“簇更多”当分类维数更多的充分理由。

假设：在共同已知类source监督前置下，RTA适应10轮后更新source/target共同表示的容量，比一直人工固定K2有更好的完整训练拒识与已知/未知平衡。允许数据估计增加、减少或保持K，结果不预设必须扩容。

## 固定设置

- Office31 A→W，C10、source958/target564、seed3（历史事后选取），Q20。
- 两组共享当前已保存C-only监督3轮source-final.pt；不重新预热。encoder BN running stats冻结、affine可训练，head BN不变。
- 两组均从C10+K2开始，使用原版warm-end K-means初始化、RTA损失、未知候选筛选、未知槽argmax伪标签和最终C+K argmax。
- 完成第10轮后两组均做一次当前全量特征提取及同规则无标签推断。fixed2仅记录；refresh应用估计K并保留已知和可对应未知头/SGD状态，不重置warmup/scheduler。
- 原型分配只用于未知头状态对应，不作为训练标签，不用原型方向初始化、不加新损失、不改Q/V、预测门控或未知概率求和。
- 同一T4旧环境、ResNet50、batch64、RTA lr5e-5；不为显存/积分偷偷减配置。

估计器保持source-calibrated-birth-cost-v1，source-only标定，source99%半径/先验、birth-first、proposal块64、当前C维head条件似然匹配已知身份。目标真值不用于决定K、阈值或匹配。若没有定义的正整数K或已知匹配不成立，保存证据并停止，不强制K1或扫阈值。

这是同前置、同预算的K模块比较，不是论文原始流程复现：额外C-only预热及冻结BN与发布原版不同。原版70轮历史成绩和当前可用57轮权重只能作背景，不替代fixed2控制。

## 预算、停止与保存

- 一个串行配对，两组各70完整RTA epoch，总140；不扩网格、重跑、换seed或额外确认训练。
- 每组上限7200秒，总训练上限14400秒；触顶停止并保留已完成结果，不延长预算。
- 无NaN/OOM/异常时监督关键日志与边界K；观察超时不等于训练失败，不因轮询失败重启。
- 每arm完成先复制普通模型、日志、逐轮指标及边界缓存到Drive，再进入下一arm。已有目录拒绝覆盖；训练前确认持久路径可用。
- 数据和权重继续复用现有Drive缓存及T4本地环境，不重新公网下载、不重复授权或重启实例。
- 按最新用户要求保留CPU与T4；不声称CLI能原地无损切换GPU到CPU。训练完成后的实例安排另遵用户选择。

## 报告与判断

报告每组完整best/post-10-best/final OS*/UNK/HOS、best epoch、初始/推断/最终K及变化；展示70轮轨迹，特别看后期未知是否被已知吸收。full best是目标标签选epoch的oracle描述，单seed不代表三seed均值。

探索筛查继续使用final HOS相对fixed2至少+1pp且OS*/UNK各不损失超过1pp，但由助手综合best/final及轨迹人工判断，不自动按target成绩调算法。K未变是有效零效果结果，不偷改更新节点找收益。前10轮偏差必须报告；即使同seed/初始化，也不宣称轨迹严格相同或单次显著性。

后续问题只在本对照回答后考虑：若扩K反而弱，研究“原型几何容量→判别容量”的映射；若无变化，不认为IMP发现真实语义已获验证；若改善，也需其他seed/论文三个数据集任务确认。不在本批同时加结构损失或改伪标签。

## 已准备入口（尚未执行）

现有脚本新增显式preset，旧20轮默认路径及预算保持；70轮用独立`a2w-capacity-refresh-k2-70e-v1`目录。

批准后，固定venv从shell运行：

```bash
/content/rta-py38/bin/python -u /content/run_a2w_capacity_refresh_colab.py --preset fixed2-70 --run
```

结束读取日志：

```bash
/content/rta-py38/bin/python /content/collect_a2w_capacity_refresh_colab.py --preset fixed2-70
```

训练和收集命令仅作入口说明，现在未运行。2026-10-06已完成本地三个修改入口的py_compile；现有T4通过colab-cli执行一次`--preset fixed2-70 --build-only`，两个arm均输出`LEGACY_TASK_SOURCE_BUILD_COMPLETE: no model/training executed`，总入口输出`A2W_CAPACITY_PAIR_BUILD_COMPLETE`。检查编译后的70轮/第10轮回调接口，不创建模型、不训练或评价checkpoint。没有hash、额外数据前向或稳定接口重复检查。

新preset、collector与wrapper已上传现有T4的/content。runtime list同时确认原T4和CPU仍保留，不销毁、不创建新实例。此前build-only成功不代表方法有效；本次随后由用户明确授权并实际启动训练，当前状态以research.md及日志为准。
