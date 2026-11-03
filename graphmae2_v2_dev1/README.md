# GraphMAE2 v2-dev1：实现与服务器验收

日期：2026-10-30。本包基于你上传到 GitHub 的
[`9f693f995993ed33b9372b48cd2322b956eaf14c`](https://github.com/BabyofStar/GraphMAE2/tree/9f693f995993ed33b9372b48cd2322b956eaf14c)。

当前阶段是工程验收。真实 PyTorch/DGL/GPU 测试必须在服务器完成；5轮冒烟结果不能当作性能提升证据。

## 本包包含什么

- `graphmae2_v2_dev1.patch`：修改3个现有源码文件，新增7个文件。
- `manifest.json`：修改前后逐文件SHA256和补丁SHA256。
- `install_v2.py`：核对原始文件、保存源码快照、检查并应用补丁；重复完整安装可识别。
- `run_stage1.py`：回归测试通过后，执行两次Cora种子0的5轮预训练＋5轮线性探测，并验证权重重载。
- `local_validation.md`：本地验收范围及尚待服务器执行的测试。

旧 `record_full_batch.py`、原21项掩码测试、原16项记录器测试、三个官方YAML、`main_full_batch.py`及v1.2采样器保持原文件字节。旧记录器会继续拒绝不符合其旧源码契约的工作目录；v2训练使用新入口 `record_full_batch_v2.py`。查看/复验旧实验请使用其保存的源码快照，不应绕过旧哈希检查。

## 五个开发组的准确含义

| 方法 | 输入掩码 | 目的 |
| --- | --- | --- |
| B0 | 原随机掩码 | 本轮参照 |
| B2 | 原v1度数课程 | 判断是否超过v1 |
| D | 重构误差课程 | 检验实际难度能否替代度数 |
| S | 密度分层、层内随机 | 单独检验覆盖机制 |
| DS | 密度分层、层内重构误差课程 | 检验互补性；有单项改善线索后再做 |

本包没有自动启动36/45次完整开发训练的队列。

### 难度D

仅从已经发生的被掩码节点特征重构取得逐节点SCE误差。误差统计使用 `no_grad` 和 `detach`；不替换原训练损失，不多做一次模型前向，不改变原教师EMA调用。

每个节点首次观察直接记录误差，之后按beta=0.9更新滑动平均。未被观察的节点使用已观察误差的中位数作为中性评分；全未观察时评分相同。使用0.95分位数截断上端异常误差，并保留正数下限。

沿用v1的课程：前20%为随机掩码，20%～80%渐增，后20%最大引导强度0.5。采样权重为随机均匀项与误差评分项的混合，再无放回抽取固定K个节点。权重不是精确的边际入选概率，不声称重加权训练目标是原目标的无偏估计。

### 密度S

使用原始节点特征，计算节点与唯一的一、二跳邻居的平均余弦相似度；去除自环并合并重复边，邻域按无向图处理。孤立节点密度为0；同分按节点索引稳定排序。按密度从低到高划为4组，小图不足4节点时减少组数。

每轮根据各组节点数量分配配额，用最大余数法处理整数舍入，确保总掩码数严格等于 `floor(mask_rate * N)`。S从第一轮开始使用配额；DS前20%在各层内随机，之后逐渐加入层内难度引导。

分层保证密度层配额，不能保证每个被掩码节点都有可见邻居。日志单独记录一、二跳可见邻居数量及零可见比例。

密度和邻域只在每个进程准备时计算一次；通过分块点积避免完整N×N相似度矩阵。邻域列表在密集图上仍可能很大，本版本范围限定为当前三个小图数据集，不是任意大图实现。

### 共同约束与产物

开发阶段沿用官方训练预算：Cora/PubMed预训练2000轮，Citeseer预训练500轮；其余模型、特征、掩码率和线性探测设置由原YAML加载。保留原每200轮的中途线性探测，不挑选中途编码器。

五组使用相同的诊断开关，避免仅对新方法增加记录流程。CPU密度与邻域准备计入预训练墙钟时间，并另记子项；不能把总时间称为纯GPU算力耗时。诊断本身有开销，需要与方法开销区分。

除原记录产物外新增：

- `mask_feedback.jsonl`：逐日志步的重构误差、观察次数、度数相关性、密度层覆盖和可见邻居统计。
- `mask_node_statistics.npz`：节点级误差EMA、观察次数、密度、密度层和非自环入度；使用 `numpy.load(..., allow_pickle=False)`读取。
- `dataset_info`增加特征矩阵和边索引的SHA256，用于核对各组输入相同。

模型权重仍保存到项目外。自适应状态仅用于本次训练，并以NPZ导出供分析；模型 `state_dict`不增加这些临时buffer，保存的权重不能用于恢复完整训练。

## 第一步：上传并解压

将完整 `graphmae2_v2_dev1.zip` 上传到服务器 `/home/jovyan/work/GraphMAE2/`。

先在本次聊天给出的命令中校验压缩包SHA256，再解压。不要把补丁当作Python运行，不要再次应用B3补丁。

解压后的目录应为：

```text
/home/jovyan/work/GraphMAE2/graphmae2_v2_dev1/
  README.md
  graphmae2_v2_dev1.patch
  manifest.json
  install_v2.py
  run_stage1.py
  local_validation.md
```

## 第二步：激活环境，安装并执行第一阶段

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /home/jovyan/work/envs/graphmae2
cd /home/jovyan/work/GraphMAE2

GRAPHMAE2_NVIDIA_DIR="$CONDA_PREFIX/lib/python3.10/site-packages/nvidia"
GRAPHMAE2_CUDA_LIBS="$GRAPHMAE2_NVIDIA_DIR/cuda_runtime/lib:$GRAPHMAE2_NVIDIA_DIR/cublas/lib:$GRAPHMAE2_NVIDIA_DIR/cusparse/lib"
export LD_LIBRARY_PATH="$GRAPHMAE2_CUDA_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export DGLBACKEND=pytorch
export DGL_DOWNLOAD_DIR=/home/jovyan/work/GraphMAE2/data/dgl
export WANDB_MODE=disabled MPLBACKEND=Agg PYTHONUNBUFFERED=1

(
  set -euo pipefail
  python -c "import sys; assert sys.prefix == '/home/jovyan/work/envs/graphmae2'; print('运行环境:', sys.executable)"
  python graphmae2_v2_dev1/install_v2.py --repo /home/jovyan/work/GraphMAE2
  python -u graphmae2_v2_dev1/run_stage1.py --repo /home/jovyan/work/GraphMAE2
)
```

安装前检查失败时不要加 `--ignore-whitespace`、不要执行 `git reset --hard`。反馈报错中的文件名及输出即可，安装器会保留已有文件。快照是修改前源码备份，不是整个项目/数据/环境备份。

安装器和第一阶段脚本都会锁定原记录批次使用的锁文件；发现该锁被占用时会停止。未使用该锁的手动训练不受此机制管理，安装时应保持当前项目没有其他训练进程。

## 成功后反馈哪些输出

预期包含：

```text
V2_SOURCE_INSTALL=PASS
V2_STAGE1_DIR=/home/jovyan/work/GraphMAE2/outputs/v2_stage1_...
V2_ALL_TESTS_PASSED; skipped=0
SMOKE_D=PASS (5 pretrain / 5 probe epochs)
SMOKE_S=PASS (5 pretrain / 5 probe epochs)
V2_STAGE1_PASSED
```

把以上控制台输出发回来。若失败，反馈错误和该目录的 `tests.log`或相应`smoke_*.log`末尾；无需打开 `test_metrics.json`。

下一步才接入18次参照、9次D、9次S的完整开发训练。组合DS是否进行由单项开发结果决定。使用验证集筛选；现有固定测试划分已被查看，最终推广性还需未参与选择的数据集或预先固定的新划分确认。

## 文献与本实现的关系

- [AUG-MAE，AAAI 2024](https://ojs.aaai.org/index.php/AAAI/article/view/29479)：提供困难掩码与渐进训练的参考。本实现采用误差EMA评分，不是论文的可学习对抗生成器，也不是已被该文证明有效的GraphMAE2补丁。[作者代码](https://github.com/CRIPAC-DIG/AUG-MAE)
- [C2-GMAE，ICML 2026](https://proceedings.mlr.press/v306/tang26l.html)：提供局部特征密度分层的参考。本实现单独移植采样思想，并保持精确掩码数量；没有移植完整位置编码或额外损失。[作者代码](https://github.com/gengtang1204/C2-GMAE)

两个方向的性能均待实验验证。
