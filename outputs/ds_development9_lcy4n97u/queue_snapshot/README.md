# DS组件交互：一次短程验收＋完整开发9次

日期：2026-11-05，北京时间。先有R9独立审计，再固定本方案；DS完整开发结果尚未出现。

## 目的与固定边界

R9产物已从GitHub固定提交705cdc51f63763996975f0305dec9f9806b11597完成独立核验。S高于这份固定随机分组R，但S相对B2的等权验证增量仅+0.0667个百分点；Citeseer的S低于B2。旧开发实验中，Citeseer的D均值比S高0.6667个百分点，D在Cora/PubMed又低于S。这只构成检验组件交互的动机，不保证组合有效。

本包进行一次有限预算的DS探索：保持S真实密度层配额，在层内使用已有D的重构误差课程。主比较DS−S，支持比较DS−B2、DS−D、DS−B0。三个数据集、种子0/1/2、官方完整轮数全部保留。只读取best_val_acc，不读取test_metrics.json，不调参数或扩大种子。

沿用已验收的34文件R版本源码和record_full_batch_r.py。该记录器本已支持DS；random_group_seed参数对DS无效，DS不使用R随机分组。此包只增加队列文件，不修改模型、YAML、旧补丁或已有运行目录。

## 用户执行

将ZIP上传到/home/jovyan/work/GraphMAE2：

```bash
cd /home/jovyan/work/GraphMAE2
unzip -n graphmae2_ds_development9.zip
bash graphmae2_ds_development9/start.sh
```

脚本激活graphmae2、导出原CUDA库路径并校验包内哈希。后台派发前取得公共锁，重新核对已验收源码、N2、B0/B2/S/D历史参照及R9完整状态和产物。

同一公共锁连续传给后台队列、训练及CPU重载进程；仍为单GPU串行执行。终端断开后可继续运行，实例重建不保证存活。启动回执会阻止重复派发第二批，失败时保留回执和日志先定位原因。

执行顺序固定：

1. Cora DS种子0，5轮预训练＋5轮探测；核对真实密度配额、误差课程、节点预算及源码，保存读回并独立CPU重载。短程失败则不开完整训练。
2. 三数据集×种子0/1/2，共9次官方完整训练；逐次核对配置、数据/特征/边/划分指纹、环境、源码快照、实际配额、课程、节点数组和CPU重载。
3. 所有运行完成后写验证汇总和等权配对差值，不因某个数据集退步而删除它。

模型和协议未改变，因此不重新运行整套80项测试。已有80项包含DS真实前后向、反馈、权重读回及与固定v2模型的回归检查；本次增加DS记录器完整5/5轮短程，仍须在服务器实际通过后进入9次训练。

## 状态与产物

新批次为outputs/ds_development9_*。启动器打印DS9_BATCH_DIR、DS9_PID及查看命令。batch_status.json的stage依次为preflight、smoke、development、complete；completed只计完整训练，不包含短程。

短程成功标志DS_SMOKE=PASS；最终完成须状态complete、completed=9、total=9，并有ALL_DONE 9/9和BATCH_EXIT_CODE=0。失败保留status、日志和已完成次数，不能覆盖旧成功结果或宣称按epoch断点续训。

权重仍保存到项目外GraphMAE2_artifacts/checkpoints。模型和测试JSON不纳入开发汇总，主要结果是validation_summary.json及逐次证据。

## 分析与停止条件

先报告DS−S、DS−B2、DS−D的全部逐种子差值、均值与ddof=0标准差；主比较固定DS−S。n=3只作探索，不宣布显著或稳定提升。若没有相对S的额外开发收益，保留负结果并停止DS扩张；即使有小幅正差，也需冻结后另行设计独立确认，不继续无限调参。

本地仅检查队列、参数选择、汇总与源码契约。真实DS短程、NPZ核对及9次训练尚未执行，需以服务器日志和状态为准。
