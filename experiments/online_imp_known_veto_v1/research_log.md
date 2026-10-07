# Research log

前几批证据：校准支持仍被污染，高维不恢复语义，扩大标签覆盖无实质拒识收益。检验当前IMP新候选与RTA known支持冲突，先算unknown r、再只vetoknown entropy/target alignment；原source监督和unknownCE不改。

代码b2272ab8/exec17启动，两arm续训6从同warm4恢复。新接口检查通过，weight0按batch N安全。正常只监督关键训练日志，等待完整结果。共同T4和Drive缓存复用，模型暂runtime、小记录Drive。
