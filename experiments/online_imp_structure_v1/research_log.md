# Research log

2026-10-07：按用户最新研究方向实现online_imp_structure.py和独立入口，不修改根目录用户main/networks等训练文件。两组共同更新当前特征、target支持校准、动态K/V；只改变unknownCE伪标签来源。针对性检查覆盖K相同的簇编号置换、扩容、known权重/动量、可靠簇标签替换；实际缓存特征推断和真实入口编译通过。

首次模型加载因torch1.7不接受weights_only关键字失败，0epoch。只修工程兼容、保留local/Drive startup-failed目录，同研究设置再次启动。exec5运行；第1轮结构K17、V3、target校准支持0，是必须披露的fallback而非target标定成功。

自举组10轮完成exit0、elapsed378.7秒，final OS*98.43%/UNK60.89%/HOS75.24%，K24/V12；结构组已启动，前2轮82.2秒、近期37.5秒/轮。仍待完整配对结果，不据单组提前评价标签收益。训练代码b786451c；后续badfe15c只新增结果绘图，不改变正在运行算法。

Drive挂载复制第一arm正常返回；独立Drive CLI查询两次都被Google OAuth项目API Queries配额限制（project202264815644），这不是用户存储已满的证据，也不能把挂载文件存在当独立云端验证。保留runtime原模型，完成后先下载小结果，云端模型可见性待API恢复或独立方式确认；不反复查询、不自动删除。
