# GraphMAE2 R随机分组对照：N1交付与N2验收

日期：2026-11-03，北京时间。版本：graphmae2_v2_r_dev1。

## 已核对和已完成的范围

用户回传的服务器N0为PASS：环境前缀、28文件哈希与已验收v2-dev1一致，三批历史运行完成。HEAD仍为9f693f9，分支feat/structure-curriculum；Git工作目录内v2尚未提交，安装程序保留已有文件和历史结果，不切分支、不清理工作区、不提交或推送。

新代码在独立本地副本实现。本地6项记录器契约测试通过；实际Git补丁安装、34文件后验哈希、重复安装不覆盖、拒绝改动的安装前源码均通过。参考夹具只去掉行尾空白，已核对AST与旧v2相同。

本地系统Python没有NumPy、Torch/DGL验收环境。尝试数学测试时因缺少NumPy导入失败，没有安装依赖；数学测试与真实模型测试均留待服务器执行。本包不声称80项测试已经通过，也没有R完整训练成绩。

## R方法

- 模式random_group_stratified。私有CPU torch.Generator固定节点排列，按S实际组大小切为4组。Cora/Citeseer/PubMed分组种子分别2026110301、2026110302、2026110303；训练种子仍为0、1、2。
- 每轮复用S的无放回抽样、最大余数配额及随机API形式，alpha始终为0，无难度课程。总掩码K、组大小、组配额与S相同；抽中的节点不同。
- 真实density_layer和local_feature_density保留，随机组另存sampling_group、sampling_group_node_order和种子；日志同时保留density_layers与sampling_groups。random_group_plan.json记录组大小、配额及有序成员指纹，metrics.json记录该文件哈希，独立CPU重载前校验。
- 模型、损失、教师EMA、YAML、原main_full_batch.py及旧记录器保持原定义。修改4个文件：models/__init__.py、models/edcoder.py、utils.py、models/adaptive_masking.py。最后一个只增加默认空的诊断扩展钩子；原方法的损失、权重、随机状态和反馈需通过与固定v2的回归比较。
- 新record_full_batch_r.py独立维护源码契约。旧record_full_batch_v2.py及旧契约保持原字节，安装R后旧记录器拒绝当前源码是预期行为，不应改旧哈希；复验旧结果应使用旧源码快照。

## 用户执行N2

将ZIP上传到/home/jovyan/work，解压后执行：

```bash
cd /home/jovyan/work
unzip -n graphmae2_v2_r_dev1.zip
bash graphmae2_v2_r_dev1/acceptance.sh
```

acceptance.sh自动激活graphmae2、导出已验收CUDA运行库路径、检查包内文件哈希，然后安装和工程验收。无需安装Torch/DGL或重装旧v2补丁。

安装会在公共outputs/.formal_record_v1.lock保护下核对旧v2源码，先git apply --check，再创建唯一r_dev1_before_install_*源码备份，最后应用新补丁并验证34文件哈希。已有训练持锁、源码不匹配或新文件冲突时停止；保留错误现场，不绕过哈希。

工程入口要求指定Conda环境、Torch2.1.2+cu118/DGL1.1.3+cu118/NumPy1.26.4以及可用CUDA。它使用同一公共锁，锁描述符传给测试、短程和重载子进程；新R记录器的训练命令要求继承该锁，拒绝直接无锁启动。

顺序为：

1. 80项测试，无跳过。其中21项旧掩码、12项数学、16项v2掩码、16项旧记录器、7项更新到新源码契约的R记录器、8项新R测试。原v2记录器测试文件保留；其旧源码契约检查由新记录器测试替代，不关闭旧记录器的校验。
2. Cora R种子0，5轮预训练＋5轮线性探测，保存权重和反馈。
3. 独立CPU权重重载、节点预算、实际随机组配额、真实密度诊断、成员指纹及NPZ核对。

成功标志为R_DEV1_ALL_TESTS_PASSED、SMOKE_R=PASS、R_DEV1_STAGE1_PASSED。唯一日志目录由R_DEV1_STAGE1_DIR给出；失败则保留目录、status和各步骤日志，控制台显示失败日志末尾。请回传终端输出；若失败，附对应tests.log或smoke_R.log末尾。

## 尚未执行的工作

此包仅用于N2工程验收，完整9次R开发队列尚未启动。本次不运行DS，也不扩大种子。工程验收通过后再准备完整队列和匹配参照核对；只汇总最终编码器best_val_acc，三数据集与负结果全部保留，不读取测试JSON选方法。

保存的是最终模型和两种分类器权重，不含完整优化器、调度器、随机或自适应状态，不支持按epoch断点续训。项目外权重继续单独保存，Git提交不等于权重备份。
