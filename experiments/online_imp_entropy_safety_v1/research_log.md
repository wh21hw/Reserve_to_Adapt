# Research log

known目标分解显示收益来自entropy veto，但raw误伤known；只比较entropy资格raw vs现有screened，alignment保留，不扫强度。

代码1774a263/exec23/launcher49453启动，资格接口验证原alignment不变。共用warm4，实际12新轮，T4/Drive数据和环境复用，正常关键监督。整体研究和三任务未完成。

完成记录：raw full/post4best/final均epoch10，92.8332/72.9920/81.7256；screened fullbest共同warm4，post4best5为91.1287/64.1854/75.3200，final96.7782/56.7397/71.5378。screened−raw OS*+3.9450、UNK−16.2523、HOS−10.1878pp。跨进程原参考比较screened known−1.2326pp，不能放宽1pp guard。最终结构快照为epoch10开始（epoch9特征），不伪称最后参数结构。曝光总量为权重样本次数，不等同损失大小：raw删170known/431unknown，screened删2/32；alignment全部保持。后者不是证明unknown都可靠，只说明径向筛查与原known权重冲突的交集太小。target语义仅事后分析。
