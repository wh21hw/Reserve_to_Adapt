# Research log

2026-10-07：按用户最新研究方向实现online_imp_structure.py和独立入口，不修改根目录用户main/networks等训练文件。两组共同更新当前特征、target支持校准、动态K/V；只改变unknownCE伪标签来源。针对性检查覆盖K相同的簇编号置换、扩容、known权重/动量、可靠簇标签替换；实际缓存特征推断和真实入口编译通过。

首次模型加载因torch1.7不接受weights_only关键字失败，0epoch。只修工程兼容、保留local/Drive startup-failed目录，同研究设置再次启动。exec5运行；第1轮结构K17、V3、target校准支持0，是必须披露的fallback而非target标定成功。
