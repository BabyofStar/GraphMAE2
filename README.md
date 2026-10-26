# GraphMAE2 复现与改进：结构引导掩码与课程调度

本仓库用于《机器学习》课程的论文复现与改进实验，基于 GraphMAE2 官方代码开展转导式节点分类研究。

项目仓库：[BabyofStar/GraphMAE2](https://github.com/BabyofStar/GraphMAE2)。

实验阶段记录截至 **2026-10-25**，文档整理日期为 **2026-10-26**。

> 当前状态：三数据集官方20种子基线、课程参数筛选和静态组筛选消融已完成；B3打乱评分对照尚未完成服务器验收；改进方法的正式10种子实验尚未开展。当前不能宣称测试集性能已得到稳定提升。

## 1. 原论文与代码来源

- 论文：**GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner**。
- 会议：The ACM Web Conference 2023，WWW 2023。
- 论文地址：[arXiv:2304.04779](https://arxiv.org/abs/2304.04779)。
- 官方代码：[THUDM/GraphMAE2](https://github.com/THUDM/GraphMAE2)。
- 本项目固定的上游基准提交：`bc473e8b750d6e90b7cba1c6bb20d858f617d90c`。

本仓库是课程复现与改进项目，不是 GraphMAE2 官方发布。上述提交指上游基准版本，不代表课程改动已经包含在该提交中。

GraphMAE2通过掩码节点属性的自监督学习获得节点表示，并结合解码端重掩码和潜在表征预测增强训练。下游使用冻结编码器后的线性分类器评估节点分类能力。本项目只改进输入端“选择哪些节点进行掩码”的策略，不同时改动原有损失、教师更新或解码器机制。

## 2. 当前进度

- [x] 创建独立环境并验证 PyTorch CUDA、DGL GAT 前向与反向传播。
- [x] 固定官方提交，保存原始源码和环境版本快照。
- [x] 完成 Cora、Citeseer、PubMed 各20个种子的官方配置基线，共60次完整训练。
- [x] 输入掩码 v1.1 的11项单元测试通过，包含 CPU 与 CUDA 检查。
- [x] 关闭新功能后，完成一次完整 Cora 回归检查，报告指标与原基线同种子一致。
- [x] 完成课程组两个数据集、三个强度、三个种子的18次参数筛选。
- [x] 完成选定强度0.5下两个数据集、三个种子的6次静态组筛选消融。
- [ ] B3增量补丁 v1.2 已准备，但尚未完成服务器应用与21项测试的运行验收。
- [ ] B3完整训练、三个数据集正式种子10–19的对照实验。
- [ ] 完善训练检查点、资源统计、正式逐种子结果导出及最终报告。

GPU运算可能存在非确定性。一次回归中报告指标一致，不代表每个训练步骤或最优分类器轮次逐位相同。

## 3. 已验证环境

| 项目 | 实际版本或配置 |
| --- | --- |
| 操作系统 | Ubuntu 24.04.4 LTS |
| Python | 3.10.22 |
| NumPy | 1.26.4 |
| PyTorch | 2.1.2+cu118 |
| DGL | 1.1.3+cu118 |
| GPU设备名称 | NVIDIA A800 80GB PCIe |
| 本实例实际可见显存 | 约15.62 GiB，而不是80GB全部显存 |

CUDA动态库补充包版本：

- `nvidia-cuda-runtime-cu11==11.8.89`
- `nvidia-cublas-cu11==11.11.3.6`
- `nvidia-cusparse-cu11==11.7.5.86`

环境路径为 `/home/jovyan/work/envs/graphmae2`。当前验证环境与上游原始 `requirements.txt` 不完全相同，完整依赖应参考实际环境导出；环境版本清单不等于完整离线安装包备份。

在当前服务器打开新终端后，先执行：

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

python -c "import sys, torch, dgl; assert sys.prefix == '/home/jovyan/work/envs/graphmae2'; assert torch.cuda.is_available(); print(sys.executable); print(torch.__version__, dgl.__version__); print(torch.cuda.get_device_name(0))"
```

以上路径针对当前服务器，换机器时需调整。`nvidia-smi`中的CUDA版本不等于项目实际使用的PyTorch CUDA运行时；本项目实际使用cu118。已知的`pkg_resources`弃用警告不等于训练失败，不应仅为消除警告而升级整个环境。

## 4. 数据集与评估协议

本项目使用DGL加载的公开引用网络数据集。节点是论文，边表示引用关系，节点特征用于表示论文内容，标签为论文类别。

| 数据集 | 节点数 | 预处理后边数，含自环 | 特征维度 | 类别数 | 训练/验证/测试节点数 |
| --- | ---: | ---: | ---: | ---: | --- |
| Cora | 2708 | 13264 | 1433 | 7 | 140 / 500 / 1000 |
| Citeseer | 3327 | 12431 | 3703 | 6 | 120 / 500 / 1000 |
| PubMed | 19717 | 108365 | 500 | 3 | 60 / 500 / 1000 |

采用固定公开划分和转导式评估：预训练可使用整张图的结构与特征，但标签不参与预训练评分或采样；训练标签仅用于线性分类器训练，验证集用于选择分类器及筛选参数，测试集不用于调整改进方案。

上表来自本项目实际数据加载记录，不将其默认视为与论文中所有数据统计口径完全一致。

## 5. 改进方法

### 5.1 结构评分与采样

以去除自环后的入度作为结构重要性的简单代理，采用对数变换减弱极高度节点的影响：

```text
q_i = (log(1 + degree_i) + 1e-8) / sum_j(log(1 + degree_j) + 1e-8)
w_i(t) = (1 - alpha(t)) / N + alpha(t) * q_i
K = floor(mask_rate * N)
```

每轮按权重无放回选取恰好K个节点。权重不是最终节点入选的精确边缘概率。只缓存一维评分，不构造N×N相似度矩阵；评分不读取真实类别标签。

`alpha`表示结构引导强度，不是总掩码率。各组总掩码率保持官方配置：Cora/Citeseer为0.5，PubMed为0.9。结构重要性也不能直接等同于重构难度。

### 5.2 课程调度

当前固定参数为：

```text
structure_alpha_max = 0.5
structure_warmup_ratio = 0.2
structure_ramp_end_ratio = 0.8

progress = epoch / max(total_epochs - 1, 1)
alpha(t) = 0.5 * clip((progress - 0.2) / (0.8 - 0.2), 0, 1)
```

即前20%训练均匀随机掩码，中间60%逐步增加结构引导，最后20%保持最大强度。课程进度通过独立接口传递，不改变原模型forward的epoch/EMA行为。

### 5.3 实验组

| 编号 | 输入掩码策略 | 当前状态 |
| --- | --- | --- |
| B0 | 原始均匀随机，`random` | 官方20种子基线完成 |
| B1 | 从开始就使用固定强度，`structure_static` | alpha=0.5的三种子筛选消融完成 |
| B2 | 渐进结构引导，`structure_curriculum` | 参数筛选完成 |
| B3 | 固定打乱评分对应关系，`structure_curriculum_shuffled` | v1.2补丁待服务器验收 |

B3拟保留评分数值集合和B2调度，只打乱评分与节点的对应。置乱使用私有CPU随机数生成器，固定`structure_shuffle_seed=20261025`，不额外消耗训练全局CPU/CUDA随机数。该模式仅在v1.2实际应用并验收后可用。

B1与B2的平均引导强度不同，因此两组对比不能单独证明调度顺序的因果作用；B3用于进一步检查结构对应关系，而不是保证得到正向结果。

## 6. 官方配置基线结果

三个数据集均完成种子0–19的完整预训练和线性探测，采用官方`--use_cfg`。下表为测试准确率，单位为百分比，标准差使用`ddof=0`。

| 数据集 | 末轮分类器测试准确率 | 验证集选分类器后的测试准确率 |
| --- | --- | --- |
| Cora | 84.27±0.54 | 84.25±0.57 |
| Citeseer | 73.26±0.32 | 73.39±0.37 |
| PubMed | 81.55±0.52 | 81.21±0.56 |

第二个指标对应官方日志中的`early-stopping_acc`，表示用验证准确率选择线性分类器，而不是用测试集选择编码器，也不是提前停止预训练。

`--use_cfg`加载YAML后可能覆盖同名命令行参数，实验应以打印的Namespace为准。当前实际主要配置为：

| 数据集 | 预训练轮数 | 线性分类器轮数 | 隐藏维度 | 注意力头数 | 输入掩码率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 2000 | 300 | 1024 | 8 | 0.5 |
| Citeseer | 500 | 500 | 1024 | 4 | 0.5 |
| PubMed | 2000 | 500 | 512 | 2 | 0.9 |

各组继续保留官方小图配置的fixed解码重掩码、损失、学习率及教师动量，不自行统一成其他值。

## 7. 改进方法的验证集筛选结果

筛选仅使用Cora、Citeseer的种子0、1、2；以下均为最终编码器的最终线性探测Best ValAcc，不是测试准确率，也不使用中途`# IGNORE`评估选择编码器。

### 7.1 参数筛选

固定预热比例0.2、爬坡结束比例0.8，比较三个预先约定的强度：

| 课程组alpha上限 | Cora验证均值 | Citeseer验证均值 | 两数据集平均验证增量 |
| --- | ---: | ---: | ---: |
| 原始基线 | 80.53% | 72.73% | — |
| 0.3 | 80.80% | 73.07% | +0.30个百分点 |
| 0.5 | 80.80% | 73.20% | +0.37个百分点 |
| 0.7 | 80.67% | 73.07% | +0.23个百分点 |

按两个数据集平均验证增量选择并固定alpha上限0.5。它仅比0.3高约0.07个百分点，不能称为可靠的最优值；后续不依据测试集结果重新选择参数。

### 7.2 选定强度下的对照

单位为百分比，均值±总体标准差，n=3。

| 方法 | Cora验证准确率 | Citeseer验证准确率 |
| --- | --- | --- |
| B0 | 80.53±1.09 | 72.73±1.64 |
| B1，alpha=0.5 | 79.00±0.28 | 72.80±1.41 |
| B2，alpha上限=0.5 | 80.80±1.07 | 73.20±1.70 |

配对原始验证准确率如下，顺序均为种子0、1、2：

| 数据集 | B0 | B1，0.5 | B2，0.5 |
| --- | --- | --- | --- |
| Cora | 80.2 / 79.4 / 82.0 | 78.8 / 79.4 / 78.8 | 80.6 / 79.6 / 82.2 |
| Citeseer | 72.0 / 75.0 / 71.2 | 71.8 / 74.8 / 71.8 | 72.2 / 75.6 / 71.8 |

B2在当前筛选中分别比B0增加0.27、0.47个百分点。静态组在Cora上退步、Citeseer上基本持平。静态负结果保留。三种子筛选不能证明稳定或显著的测试集提升。

## 8. 复现入口

以下为后续复现实验示例，收尾备份时不需要重新训练。先完成第3节环境激活。

单元测试入口：

```bash
python -u -m unittest discover -s tests -p test_input_masking.py -v
```

v1.1已经通过11项测试；只有实际应用v1.2后才会包含21项测试。集成测试读取Git中的上游基准源码，应在带Git历史的项目根目录运行，而不是在无Git的归档解压目录中运行。

筛选阶段先定义输出过滤函数；它只隐藏测试指标，不改变原程序内部的评估调用和训练随机数管理：

```bash
graphmae2_hide_test_metrics() {
    python -u -c '
import re
import sys
for line in sys.stdin:
    line = re.sub(r"(test_loss|test_acc|early-stopping-TestAcc|TestAcc):\s*[^,\s]+", r"\1:[HIDDEN]", line)
    line = re.sub(r"(# (final_acc|early-stopping_acc):).*", r"\1 [HIDDEN]", line)
    sys.stdout.write(line)
'
}
```

Cora课程组单种子示例：

```bash
(
    set -euo pipefail
    mkdir -p outputs
    GRAPHMAE2_RUN_DIR="$(mktemp -d outputs/readme_example_XXXXXX)"
    printf '日志目录：%s\n' "$GRAPHMAE2_RUN_DIR"

    python -u main_full_batch.py \
        --dataset cora --device 0 --use_cfg --seeds 0 \
        --input_mask_mode structure_curriculum \
        --structure_alpha_max 0.5 \
        --structure_warmup_ratio 0.2 \
        --structure_ramp_end_ratio 0.8 \
        --mask_log_interval 100 \
        2>&1 | graphmae2_hide_test_metrics \
        | tee "$GRAPHMAE2_RUN_DIR/train.log"
)
```

将模式改为`random`可关闭新策略，改为`structure_static`可运行静态组。`random`模式不使用结构引导强度，`alpha_max=0`也保留原始randperm路径。替换数据集名称时仍使用对应官方YAML，不手工降低网络规模或训练轮数。

正式实验使用与筛选不同的种子10–19，计划覆盖三个数据集的B0/B1/B2/B3；配置匹配的B0逐种子结果可从已有20种子基线中核对后复用。改进方法及B3的正式10种子实验尚未开展。

## 9. 关键文件与原始记录

- `main_full_batch.py`：小图全批次训练入口及独立课程进度传递。
- `models/edcoder.py`：GraphMAE2模型及输入掩码调用。
- `models/input_masking.py`：新增的结构评分、采样和课程模块。
- `models/__init__.py`、`utils.py`：模型构造与参数解析。
- `configs/`：沿用官方数据集配置。
- `tests/test_input_masking.py`：单元测试和集成回归。
- `data/dgl/`：当前服务器的数据缓存。
- `outputs/`：日志、环境记录和源码快照。

服务器当前项目根目录为`/home/jovyan/work/GraphMAE2`。关键记录位于其`outputs/`下：

```text
baseline_snapshot_R7iaPr/       原始代码归档与environment.yml
official_baseline20_h8sk8q/     三数据集20种子基线
mask_v1_1_tests_AgJliQ/         v1.1单元测试
cora_patchcheck_s0_1m6aqu/      关闭新功能的完整回归
cora_static_a03_s0_Om4sAP/     静态0.3的Cora种子0
cora_curriculum_a03_s0_hZMjhp/ 课程0.3的Cora种子0
cora_a03_seeds12_AXos4Z/       上述两组的种子1、2
curriculum_grid15_Z25tq9/       剩余15次课程参数筛选
static_a05_seeds012_343Etq/     选定0.5下的6次静态组消融
```

这些是服务器上的原始记录路径，不表示目录中的文件已经全部纳入Git。当前full-batch流程未自动保存每次训练的编码器权重，源码和日志快照不能当作模型检查点。

## 10. 备份与后续工作

备份需要同时考虑源码版本与未追踪的实验文件：

- Git保存源码、测试、配置、说明、环境版本清单和适当的结果摘要，并保留原LICENSE。
- 数据缓存、全部原始日志和源码归档应另做项目文件归档，记录范围、排除项和校验和；不要把仅推送源码称为完整工作目录备份。
- 原上游`.gitignore`忽略`*.log`；直接`git add .`不能证明实验日志已备份。
- 不上传Token、密码、SSH私钥或包含凭据的配置；备份前审查敏感文件和大文件。
- 环境目录在`/home/jovyan/work/envs/graphmae2`，不在项目目录内。项目目录归档不包含整个Conda环境。

后续顺序：完成B3运行验收，冻结并记录完整协议，开展正式种子10–19实验，补充检查点及资源统计，进行配对结果分析并撰写报告。若没有稳定收益，应如实报告，不通过选测试集结果、删负结果或挑种子得到正向结论。

## 11. 引用与许可证

请保留官方代码来源和原许可证。上游代码的LICENSE为MIT；数据集的使用条件需分别遵循其来源说明。

```bibtex
@inproceedings{hou2023graphmae2,
  title={GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner},
  author={Hou, Zhenyu and He, Yufei and Cen, Yukuo and Liu, Xiao and Dong, Yuxiao and Kharlamov, Evgeny and Tang, Jie},
  booktitle={Proceedings of the ACM Web Conference 2023},
  year={2023}
}
```