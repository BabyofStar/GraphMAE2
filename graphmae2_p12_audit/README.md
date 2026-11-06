# P12服务器独立只读审计

无需GitHub同步或备份，不训练、不安装依赖、不读取模型权重或test_metrics.json。只使用现有graphmae2环境的NumPy和Torch CPU私有随机重放；不加载模型，不分配CUDA训练张量。

ZIP上传至/home/jovyan/work/GraphMAE2后：

```bash
cd /home/jovyan/work/GraphMAE2
unzip -n graphmae2_p12_audit.zip
bash graphmae2_p12_audit/start.sh
```

脚本对固定批次outputs/p_development12_by02jzbr审计，使用另写的复算代码，不调用训练队列的validate函数，也不导入源码快照。核对12完整运行及4组100/5诊断、114项测试证据、48文件源码快照、原始输入/配置/环境指纹、采样和反馈时序、H/P/PS权重数值多重集、PS私有重放、测量筛查及已记录CPU重载证据。

只在新建outputs/p12_independent_audit_*内写报告、摘要和证据哈希；成功后在本工具目录写audit_receipt.json，作为后续新实验的审计关卡。不会覆盖历史记录或修改模型源码。公共实验锁被占用时停止。

终端成功标志P12_INDEPENDENT_AUDIT=PASS，并打印SEEDS、PAIRED、MEASUREMENT和AUDIT_COUNTS。请回传这些末尾行；失败时保留P12_AUDIT_DIR与报错，不启动新实验。

这里的权重检查范围是已有保存读回/哈希/CPU重载日志与元数据；未读取权重本体。n=3、Cora单图结果不作显著性或推广结论。测量筛查通过不代表真实可学习性或下游收益。
