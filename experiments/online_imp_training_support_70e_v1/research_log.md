# 运行记录

2026-10-07：从已存在commonwarm4启动完整70轮配对，exec45/launcher109227，代码9241b007。原始短程及确认全部保留，不再重复短程。仅轮数/预算/收集范围参数化，未改模型和方法。每组66新轮，共132；5400秒/arm、总10800秒。epoch5已完成loss有限，等待近期history间隔更新ETA。

结果整理工具去除epoch10硬编码：图按实际最大轮数，离线当前快照阶段从structure history读，最后曝光按数值轮数排序，且不得把epoch-start几何标为final特征。用已完成确认ZIP做一次针对性的记录解析检查，两组snapshot_epoch=10、解析成功；无模型重评、无重复训练，既有报告不覆盖。exec45仍running，第一组已epoch11；epoch10→11为35.74秒/轮，总剩预计约75分钟，后期/换组继续更新。

用户进一步指出需寻找已知/未知sweet point，不能以1pp已知损失单独否决。已记录于experiments/OSDA_TRADEOFF_POLICY.md：本批训练和原预声明gate保留，结果另作完整取舍解释；后续风险提示与方法价值区分，不回写历史通过状态或盲目叠加新因素。

换组：rta_only process-status exit_code0/timed_out=false；reliable_union已完成epoch5/6，elapsed41.3027/78.9892秒，近期37.6865秒/轮，余64新轮约40分钟。仍同exec45串行执行，无额外授权、预热或新设置。

完整结果：两arm66+共享4、collector0，final HOS67.5535→72.5200、OS*99→99.6667、UNK51.2685→56.9958%，gate通过。新增1148=1117unknown+31known，独立结构覆盖171→187；已知目标仍有大量unknown权重，语义覆盖不均衡。保留训练入口模块，不声明完整方法有效或论文对齐。fullbest仍共同warm4，post4best5，全部结果保存。ZIP下载fetch暂失败，分片处理，仅修复传输，不重复训练/collector。
