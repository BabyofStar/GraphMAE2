# C：渐进启用密度分层，9次探索

日期：2026-11-05，北京时间。用户选择预算为9次新方案探索，不运行18次的配套顺序对照。GitHub同步故障和DS独立审计继续作为待办，不阻塞本轮；本包无网络下载、安装依赖或GitHub操作。

## 假设与方法

已有结果中S在Cora/PubMed有局部收益，Citeseer退步；D/DS没有形成额外整体改进。本轮只检验：先使用均匀随机掩码、再逐步引入S，完整方案能否改善验证结果。这是假设，不是对Citeseer退步原因的既定解释。

C的input_mask_mode为density_curriculum。每个预训练轮次只选择一种掩码方式：

```text
u = epoch / max(total_epochs - 1, 1)
p_density = clip((u - 0.2) / (0.8 - 0.2), 0, 1)
route[epoch] = 1 if private_uniform_draw[epoch] < p_density else 0
0: 原始全局均匀无放回抽取K个节点
1: S的真实密度4层固定配额，层内均匀无放回抽取，总数仍为K
```

前20%轮次为均匀随机，最后20%为S，中间概率逐步从0升至1。这个概率与度数/误差引导alpha不同；C不使用D，不做度数加权，诊断alpha恒为0。每轮总掩码率仍为Cora/Citeseer 0.5、PubMed 0.9。

路线由私有CPU torch.Generator一次产生，Cora/Citeseer/PubMed的路线种子固定为2026110501/2026110502/2026110503；每图在训练种子0/1/2下使用同一份路线。生成路线不改变训练全局CPU/CUDA随机状态，不使用标签或验证成绩。保存每轮概率、0/1路线、实际两类轮数及SHA256，可独立重放核对。

**范围限制：C的S预期使用量约为完整轮数的一半，S组则从第一轮到最后一轮都分层。** 本次没有同使用量、打乱顺序对照，因此C−S只能评价完整方案，不能把差值单独归因于课程顺序。只有结果值得继续时再讨论对照，不在本包自动追加训练。

## 版本与安装

独立版本graphmae2_v3_density_dev1，记录器record_full_batch_c.py，版本3.0.0-density-dev1。对已验收R源码仅修改models/__init__.py、models/edcoder.py、utils.py，新增C模块、数学辅助、测试、固定R参考模型及新记录器，形成41文件源码清单。

旧B0/B2/D/S/DS/R分支保持原定义。原main_full_batch.py、YAML、各旧采样器和旧记录器保持原字节，旧record_full_batch_r.py不会被改哈希。安装C后旧记录器拒绝新源码属于预期行为；历史复核使用其source.tar或安装前备份，不绕过校验。

安装程序先在公共锁下核对R的34文件、检查补丁，再将旧源码备份到唯一outputs/c_dev1_before_install_*，最后安装并核对41文件。已完成的DS/R等运行目录及外置权重不覆盖；DS尚未上传时，其source.tar和日志仍留在原目录，可后续审计。

## 执行

将ZIP上传到/home/jovyan/work/GraphMAE2后执行：

```bash
cd /home/jovyan/work/GraphMAE2
unzip -n graphmae2_v3_density_dev1.zip
bash graphmae2_v3_density_dev1/start.sh
```

入口会自动激活graphmae2、使用原CUDA库路径和已有DGL缓存，不需要联网。流程为：

1. 包校验、R版本核对与备份，安装C独立补丁。
2. 在服务器运行94项测试，无跳过。其中保留旧采样/数学/模型/记录器检查，加入5项路线数学和9项C真实CPU/CUDA测试，旧记录契约检查更新到新C契约。
3. Cora C种子0做5轮预训练＋5轮探测、保存读回与独立CPU重载，验证路线路径、配额和节点预算；短程不用于选性能。
4. 只有上述关卡全部通过，才后台派发3数据集×种子0/1/2的完整9次。主比较C−S，支持C−B2、C−B0；官方模型、数据、参数和完整轮数不变。

测试或短程失败即停止，不会进入完整9次。成功验收生成stage_receipt.json；相同源码下重启可复用已完成验收，不必重复94项测试。已有launch_receipt.json时只显示已有批次，避免误开第二批。

安装、工程验收和完整队列使用同一公共outputs/.formal_record_v1.lock；锁描述符继承给子进程。保持单GPU串行，不并发无锁手动训练或安装。后台队列可应对终端断开，不保证实例重建后存活。

## 产物与判断

工程目录为outputs/c_dev1_stage1_*，完整批次outputs/c_development9_*。启动器打印C9_BATCH_DIR和C9_PID；完整成功须status=complete、completed=9、total=9，且日志ALL_DONE 9/9、BATCH_EXIT_CODE=0。

每次保存density_route_plan.json及其哈希、配置、环境、源码、资源、反馈、节点统计、权重哈希和CPU重载证据；同图3个训练种子必须使用完全相同的路线。真实密度诊断保留；只有density路线要求该轮满足S层配额，uniform路线只要求原始均匀抽样和精确K。

开发只汇总最终编码器best_val_acc，不读取test_metrics.json，不挑数据集、种子或末轮指标。n=3不支持显著或稳定结论。没有额外开发收益就保留负结果，不自动扩种或搜索时间参数。权重仍在项目外保存，不支持按epoch断点续训。

本地仅可运行纯数学、记录契约、队列和安装检查，没有Torch/DGL/NumPy环境。94项真实环境测试、5/5轮短程和9次完整训练尚未执行，以服务器实际输出为准。
