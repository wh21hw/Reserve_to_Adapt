# 恢复旧方案参数20轮探索

## Goal

用户2026-10-08明确恢复实验，检验旧方案是否可超过论文A→W HOS93.0，允许best seed/epoch但保留全部曲线。当前已完成原配方seed1/2/3各20轮，不重跑。

## Success Metric

全部best/final OS*/UNK/HOS、最后5轮趋势及V范围，HOS>93作为论文数字比较，原RTA复现95.2172是另一个参考。target-oracle配置/epoch探索不是无偏测试，不把最好seed当均值。

## Constraints

- max_iterations:1，pause_every:never，Evaluator:none(agent judges manually)。已授权无人值守，禁止无限网格或反复直到通过。
- 两候选alpha.01/source5、alpha.05/source3，固定seed3，各RTA20。从ImageNet重训，源域固定LR与SGDmomentum带入RTA。被用户中断v1只有模型无optimizer断点，不能无缝续训，旧证据恢复保留且本批独立v2命名。
- 依赖旧Python3.8/torch1.7.1/numpy1.23.4/faiss1.7.4；固定C+2、原BN、原virtual/entropy/adv/unknownargmax，不新版K或union或gamma。只已声明空nomatch及head匹配边界补全，不改IMP公式。
- 每组1500秒总4500保护，预计训练20–25min（环境准备另计）；原5min技能限时被本项目明确轮数预算覆盖，不改变预算偷偷训练。
- 使用T4 gpu-t4-s-kkb-usw4a2-10aku92peratz，授权mount已成功exec2。数据复用Drive Office31压缩包，本地解压，不重复官网下载。setup独立shell14，恢复exec4。
- 每epoch评测日志、best/last模型保存，正常只监督进程/finite/OOM/指标。稳定代码不重复smoke/hash/checkpoint重评。失败保留证据定位，不自动算法重跑。
- target真值只指标和事后分析，不进入聚类/K/阈值/训练。不声称整个历史环境逐字相同（用户配套依赖未提供）。
- 小结果Drive与电脑，模型关闭runtime前备份。当前连续实验不关闭/重启，结束报告hook暂停，不自动更多batch。

## Search Space

仅补完上次明确授权的两组单因素对照，参考原配方seed3同20轮。先恢复旧日志确认未完成，不覆盖v1。没有更密alpha或source轮数组合。

## History

准备：mount第一次failed，第二次15min脚本成功。缓存与旧目录可见，环境旧依赖已安装；恢复历史小结果进行中，尚未启动训练。

已启动：2026-10-08 exec5，console /content/legacy-param-v2-console.log，root/content/imp-runs/legacy-imp-param-screen-v2，独立shell14。历史归档从Drive恢复到电脑pipeline-results/legacy-recovered-oct8.zip，三seed各20轮且process0，参数v1 alpha.01/source5只3轮、process−15/user stop，未自动重试而按新用户授权独立v2重新训练。原数据与预训练权重复用缓存成功，无下载数据副本。

完成：exec5/collector0，两arm各恰好20轮、process0/timed_outfalse，source轮数保持5/3。结果ZIP已下载电脑pipeline-results/legacy-imp-param-screen-v2-results.zip。总训练682.293+673.632=1355.925秒（22.60分钟）。alpha.01/source5 best epoch18 OS*/UNK/HOS93.7687/85.2124/89.2860%，final94.6679/83.7730/88.8878%；alpha.05/source3 best epoch19 87.1098/90.2273/88.6411%，final86.7764/90.2273/88.4682%。V范围0–1/0–3，final1/2。两点known/unknown互不支配，均未超过论文93。详见final_report.md。参数batch不追加；已授权warm2单arm接续，hook暂保留。
