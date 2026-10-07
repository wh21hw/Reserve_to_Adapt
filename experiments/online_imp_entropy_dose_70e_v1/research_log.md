# 运行记录

2026-10-07：完整预算配对half/zero启动，code0e051825/exec47/launcher152769/shell13。同warm4、各66新轮，总132；5400/arm、总10800秒预计80分钟。短程三点和完整gamma1参考保留，后者仅独立背景。10分钟hook运行，不重跑稳定接口或重复gamma1完整训练。

首组epoch5/6 elapsed41.1244/78.0651秒，近期36.9407秒/轮，当前余64+66轮约80分钟。换组后再实测更新，ETA不是预算截止。前两续训轮完成无报错。

用户明确暂停主动研究目标，保留hook仅监测当前批次。最新完成证据exec47/collector0，两arm完整1–70及exit0/timed_out=false。训练82.39分钟。half/zero final HOS79.6136/87.4176、known97.4086/85.6462、UNK67.3161/89.2638；zero best20 HOS90.2942，已知后期下降。ARI/NMI zero更好但known结构误标签276→2968。结果/图下载保存，posthoc传输更换名字解决fetch错误，无训练重跑。hook暂停，目标继续paused，不开展下一实验或销毁实例。
