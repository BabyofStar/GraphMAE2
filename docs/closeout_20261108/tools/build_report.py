"""Rebuild current report/tables from frozen evidence. No training or model loading."""
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
import zipfile

sys.stdout.reconfigure(encoding='utf-8')
ROOT=Path(__file__).resolve().parent.parent
P_STATUS='服务器独立审计PASS；完整报告与回执已本地核对，未重审全部原始排名数组'
J_STATUS='现有原始产物有限独立核验PASS；未重放Torch随机流或模型'


def load(name):return json.loads((ROOT/name).read_text('utf-8-sig'))
def write(name,text):(ROOT/name).write_text(text,encoding='utf-8')
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,row))+' |' for row in rows])
def replace_once(text,old,new):
    assert text.count(old)==1,old[:100]
    return text.replace(old,new,1)


def main():
    small=module('small_rebuild',ROOT/'tools/rebuild_small_tables.py')
    small.LABELS['P12']=P_STATUS;small.LABELS['J12']=J_STATUS;small.main()
    audit=load('sources/arxiv_A/audit.json');a=load('sources/arxiv_A/A_results.json')
    assert audit['status']=='PASS_FINITE_A_ARTIFACT_AUDIT'
    assert load('sources/J12_audit/audit.json')['status']=='PASS_FINITE_ARTIFACT_AUDIT'
    u=load('tables/unified_results.json');u['date']='2026-11-08 Asia/Shanghai'
    u['arxiv_A']=dict(dataset='ogbn-arxiv',method='B0',metric='val_selected_test_acc',independent_pretrain_runs=1,
       pretrain_seed=0,linear_probe_seeds=list(range(10)),mean_percent=audit['test_mean_percent'],population_sd_percent=audit['test_population_sd_percent'],
       sd_ddof=0,scope='Ten classifiers of one encoder; resource-adapted batch256 and zero loader workers',
       evidence_status='Existing artifacts finite audit PASS; no fresh local weight replay',source='sources/arxiv_A/audit.json')
    write('tables/unified_results.json',json.dumps(u,ensure_ascii=False,indent=2)+'\n')
    for source,dest in [('probe_results_10.csv','arxiv_A_probe_results_10.csv'),('epoch_summary_60.csv','arxiv_A_epochs_60.csv'),
                        ('validation_curves_10000.csv','arxiv_A_validation_curves_10000.csv'),('server_asset_receipts.csv','arxiv_A_server_asset_receipts.csv')]:
        shutil.copyfile(ROOT/'sources/arxiv_A'/source,ROOT/'tables'/dest)
    with (ROOT/'tables/run_ledger_all.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        rows=u['run_ledger']+[dict(batch='arxiv_A',full_pretrain_runs=1,scope='ogbn-arxiv B0 seed0 x60 epochs; ten classifiers separately counted',
             metric='val_selected_test_acc (10 probes / 1 encoder)',evidence_status='Finite artifact audit PASS')]
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    with zipfile.ZipFile(ROOT/'sources/legacy_closeout_20261106.zip') as archive:
        report=archive.read('GraphMAE2_closeout_20261106/课程报告初稿.md').decode('utf-8')
        old_results=archive.read('GraphMAE2_closeout_20261106/02_统一结果总表.md').decode('utf-8')
    report=report.replace('课程报告初稿 · 2026年11月6日（北京时间）','课程报告 · 内容版 · 2026年11月8日（北京时间）')
    report=report.replace('以未遮蔽图生成目标表示','以未遮蔽输入特征的教师视图生成目标表示')
    report=replace_once(report,'本报告同时保留负结果、工程核验范围和固定划分下的统计限制。',
       '另在ogbn-arxiv完成1次独立B0资源适配预训练及10个固定种子的线性探测，测试准确率为71.542±0.054%（总体标准差来自同一编码器的分类器随机性），完整任务耗时约15.14小时。未进行B2大图迁移比较。本报告同时保留负结果、工程核验范围和固定划分下的统计限制。')
    report=replace_once(report,'本项目完成三个小图上的官方配置复现和输入掩码对照，固定上游提交为',
       '本项目完成三个小图上的官方配置复现和输入掩码对照，并补充ogbn-arxiv上的B0资源适配复现。固定上游提交为')
    report=replace_once(report,'267是运行账本，历史与正式批次还使用了部分相同种子，不能解释为267个独立统计样本。',
       '267是原小图运行账本，另登记本次1个arxiv独立编码器，合计268次主要完整预训练；10次arxiv分类器另计。历史与正式批次还使用了部分相同种子，268也不能解释为268个独立统计样本。')
    report=replace_once(report,'本项目遵循官方小图配置，其实际解码设置为`remask_method=fixed`，保留原有潜在预测分支及教师更新。',
       '本项目遵循官方小图配置，其实际解码设置为`remask_method=fixed`，保留原有潜在预测分支及教师更新。arxiv单独沿用官方大图设置，使用random重掩码3次，不能将大小图解码配置混写。')
    report=replace_once(report,'可见48线程。设备名称含“A800 80GB”，实际可见显存为15.625 GiB；不能按80GB安排预算。主机显示内存不代表已经核实容器内存配额。',
       '虽然可见48线程，容器CPU配额实际等效4核、内存上限32 GiB。设备名称含“A800 80GB”，实际可见显存为15.625 GiB；不能按80GB安排预算。高内存记账主要含可回收文件缓存，GPU按共享条件记录，不能据设备名或宿主机信息假定独占容量。')
    report=replace_once(report,'完整审计报告及原始产物尚未在本地重新读取，不能将回传摘要表述为本地完整重审。',
       '完整服务器审计报告、回执及证据哈希现已在本地读取并核对，审计器和源码身份吻合。本地没有对全部P原始排名数组再做一轮重审，不能将报告核对改写成新的原始产物全量审计。')
    report=replace_once(report,'回传摘要中，S/NA/NR/NC的平均配对错开比例约为',
       '现有原始产物有限核验中，S/NA/NR/NC的平均配对错开比例约为')
    report=replace_once(report,'这些数字目前只支持摘要层面的结构操纵描述，完整`pair_plan.json`、全轮`joint_trace.npz`、55文件运行源码快照、输入/环境和重载证据尚未独立核验。报告不据此建立强机制或因果结论，NA/NR暂不扩大。',
       '现已独立核对16次运行（12完整＋4诊断）、16×55＝880项源码文件、24,400轮掩码/层配额/配对算术、累计次数、输入与环境指纹及保存重载回执，有限产物核验PASS。未从原始特征重新生成Torch私有配对随机流，未下载模型重新预测；理论边际仍依赖设计与数学检查。操纵关系核验通过不等于性能提升或因果机制证明，NA/NR仍不扩大。详见[sources/J12_audit/audit.json](sources/J12_audit/audit.json)。')
    start=report.index('## 7. 复现工程');end=report.index('## 参考文献')
    probe_table=mdtable(['探测种子','最佳验证轮次（从0计）','最佳验证准确率%','对应测试准确率%'],
       [[row['probe_seed'],row['best_val_epoch'],f"{row['best_val_acc']*100:.4f}",f"{row['val_selected_test_acc']*100:.4f}"] for row in a['results']])
    arxiv=f'''## 7. ogbn-arxiv：B0资源适配复现

### 7.1 目的、数据与冻结预算

用户在工程测量后选择方案A，仅补齐GraphMAE2的大图B0复现经历；没有选择B0/B2各3个预训练种子的迁移比较。大图在此相对Cora/Citeseer/PubMed而言，ogbn-arxiv在OGB自身分类中属于Small，不能称为百万节点图。官方数据包含169,343节点、1,166,243原始有向边、128维特征和40类别，采用时间划分。[OGB官方数据说明](https://ogb.stanford.edu/docs/nodeprop/)。本次沿用90,941/29,799/48,603个训练/验证/测试节点；双向化、去重和自环后实测2,484,941条存储边。

原作者预计算聚类分享链接已失效，本次按固定上游ACL节点列表过程，在独立环境用LocalGraphClustering 0.6.1生成所有根节点的局部子图，上限256节点；完整文件哈希、上游抽样精确一致及串行/双进程一致记录保留，不声称与不可取得的作者文件逐字节相同。模型训练继续使用原冻结graphmae2环境。

容量测量中，512批次的Python PASS伴随HAMI分配器OOM，两加载子进程还触发SIGBUS；/dev/shm实际仅64 MiB。正式运行前将训练/嵌入批固定为256、加载子进程固定为0，之后不依据准确率调整。模型为GAT隐藏1024、4层、8头、residual、layernorm；AdamW学习率0.0025、权重衰减0.06、60轮余弦调度；随机解码重掩码3次、输入与重掩码率0.5、边扰动0.5、SCE指数6、latent系数10、EMA系数0.996且延迟40轮。所有参数及57文件源码身份已冻结。

一次独立预训练使用种子0，从头完整跑60轮，未把工程测量的一轮权重续成正式成绩。每轮662批，共39,720步；减半批大小改变更新及EMA次数，属于资源适配复现，不声称严格等同原批512。无标签预训练使用全部根节点的结构与特征，标签只进入后续分类器与验证选择。

### 7.2 最终编码器与分类器评价

第60轮编码器固定后，按官方train/valid/test根节点顺序提取169343×1024维LC根嵌入。这里是局部子图根嵌入，不是全图全邻居推断。固定分类器种子0～9，每个完整训练1000轮；线性头使用Adam、学习率0.005、权重衰减0.0001、批大小5120。每轮只按验证准确率选权重，平分时保留较晚轮次，最后计算该分类器的测试准确率；全部10次保留。

{probe_table}

测试结果为**{audit['test_mean_percent']:.3f}±{audit['test_population_sd_percent']:.3f}%**，总体标准差ddof=0；独立预训练数是1，分类器数是10。这不是10个独立编码器的均值，不能据较小标准差宣称预训练跨种子稳定，更不能用它证明B2在大图上有效。固定批256与自生成聚类也限制了对论文数值的直接等同判断。

![arxiv预训练损失与线性探测验证轨迹](figures/arxiv_A_training_and_probe.png)

图3：左图为60轮B0平均训练损失，虚线标记从第41轮开始的原配置EMA更新；右图为同一编码器的10个分类器逐轮验证准确率，蓝线为均值，灰线为最小/最大值，均非置信区间。图仅记录训练和探测过程，不用于选择预训练轮次、方法或扩种；测试成绩不进入该图。来源见tables/arxiv_A_epochs_60.csv及arxiv_A_validation_curves_10000.csv。

### 7.3 实测成本与核验边界

| 阶段 | 实测墙钟 | 范围 |
| --- | --- | --- |
| 完整预训练阶段 | {audit['pretrain_phase_hours']:.3f}小时 | 60轮、记录、断点及最终保存重载 |
| 全范围嵌入阶段 | {audit['inference_phase_minutes']:.2f}分钟 | 载入、169,343根节点提取、保存核对 |
| 10个线性分类器 | {audit['classifier_phase_minutes']:.2f}分钟 | 每个1000轮及验证选模/最终测试 |
| 完整任务 | {audit['whole_job_hours']:.3f}小时 | 含阶段启动和管理；不是独占GPU内核时间 |

任务从北京时间2026-11-07 22:57运行至2026-11-08 14:05，约15小时8分钟；比工程测量外推的13.1小时长，采用实测值报告。预训练分配峰值约{audit['pretrain_peak_allocated_GiB']:.2f} GiB，保留峰值约{audit['pretrain_peak_reserved_GiB']:.2f} GiB，二者含义不同；未见HAMI或cgroup OOM。共享GPU负载和文件缓存影响计时，不能将它写成隔离的算法复杂度或性能保证。

本地独立核对168个清单文件、7个运行工具、57个源码文件、全部39,720个训练批、662个嵌入批和10,000条分类器epoch记录；学习率、EMA epoch参数、固定掩码数、轮次/数量、模型选择及成绩汇总一致。完整模型和嵌入留在服务器，本地没有重新加载权重推断；保存读回、CPU重载与根节点唯一性依赖已逐字节核对的服务器入口回执及散列/计数。有限核验通过不等于本地重新测得准确率。详细边界见[sources/arxiv_A/audit.json](sources/arxiv_A/audit.json)。

## 8. 复现工程、证据状态与保存范围

正式120次、原开发36次及R/DS/C已有独立产物核验PASS；本次整理从保存指标独立复算小图配对区间与Holm校正，保留25份原来源字节快照和303条选读指标。P12完整服务器审计报告和回执已在本地核对，未重新审计全部P原始排名数组；J12现有原始产物有限核验PASS，未重放Torch随机流或模型。arxiv A的60轮与全部10个分类器也已完成有限产物核验。

历史证据快照保留当时状态，当前报告、README及统一机器表使用更新后的证据等级，不覆盖旧结果、旧源码或旧哈希。选B2作为小图主方案不要求把当前服务器J源码回退；实际运行身份以每次源码快照与哈希为准。

历史R/DS/C提交分别为705cdc51f63763996975f0305dec9f9806b11597及40de08f6b42c050e94ed65444beeb4f2b6520a06。2026-11-06的服务器本地HEAD记录为8062dd851552c4f519924d75a28e85dc068c58ab，有限导出中200文件匹配该HEAD，64文件未列入；这不是2026-11-08的最新远端备份证明。本次公共GitHub页面只返回较旧main页面，目标分支/提交无法读取，仍不声称最新P/J及arxiv结果已同步远端。

服务器项目为`/home/jovyan/work/GraphMAE2`；小图外置权重位于`GraphMAE2_artifacts/checkpoints`，arxiv正式资产位于`GraphMAE2_artifacts/arxiv_A_TxrQqJzj`。本地交付保留arxiv原始小返回包、指标/轨迹、源码与审计依据；完整权重、完整嵌入、原始数据及Conda二进制不包含在报告包。服务器保存回执不等于独立第二份备份，详见[保存范围清单](04_保存范围清单.md)。

## 9. 局限与结论

当前结论条件于固定图、公开划分、实现及预算。增加训练种子不使已查看的测试集重新成为未见数据；多轮开发可能适应同一验证集。小图正式n=10、开发n=3仍小，不能推广到任意图。arxiv只有1个独立编码器，10个分类器只衡量该编码器的探测随机性，未检验B2迁移。

B1/B2平均引导强度不同，B3和R各仅一份固定计划，C缺少同使用量顺序对照，稀疏反馈及配对诊断不支持强因果归因。P12完整报告核对不等于全部原始数组重审；J/arxiv有限核验也没有在本地重新推断大权重。原始数据统计差异、资源适配、非隔离计时及独立备份未确认均如实保留。

本项目支持的结论是：对数度数输入掩码课程在Cora获得有限且经主要比较校正支持的收益，PubMed观察到正均值但校正后证据不足，Citeseer未获提升支持；后续探索没有形成可升级为主方案的稳定额外收益。arxiv补充完成了B0资源适配复现，提供可追溯的71.542±0.054%结果与实测成本。用户选定的实验范围已完成，不以追求更高准确率为理由默认追加实验；后续只做交付一致性、课程实际格式和保存范围核实。

'''
    report=report[:start]+arxiv+report[end:]
    report=report.replace('本初稿尚未套用课程指定模板或提交格式；现有材料没有给出页数、截止日期或答辩要求，不对此作假定。结果与证据状态已冻结，后续可根据实际课程要求排版。',
       '本内容版已纳入选定的大图实验和当前可取得的核验证据。现有材料没有给出课程指定封面、模板、页数或答辩要求，未对此作假定；最新远端同步及外置资产独立副本仍未确认，均在保存清单中标明。')
    report=report.replace('## 附录：历史基线与本轮交付','6. Open Graph Benchmark. Node Property Prediction，ogbn-arxiv官方数据与时间划分说明，查阅于2026-11-08. [文档](https://ogb.stanford.edu/docs/nodeprop/)。\n\n## 附录：历史基线与本轮交付')
    report=report.replace('本次公共GitHub页面只返回较旧main页面','本次[公共GitHub页面](https://github.com/BabyofStar/GraphMAE2)只返回较旧main页面')
    write('课程报告.md',report)
    old_results=old_results.replace('2026-11-06（北京时间）','2026-11-08（北京时间）')
    old_results=old_results.replace('证据状态：服务器独立审计脚本回传PASS，完整报告及原始产物未在本地重新读取。','证据状态：'+P_STATUS+'。')
    old_results=old_results.replace('证据状态：保存摘要已回传复算，原始产物独立核验待完成。','证据状态：'+J_STATUS+'。')
    old_results=old_results.replace('本文件由同一份冻结JSON生成','本文件以冻结来源复算，并更新证据等级；数值见统一JSON')
    old_results=old_results.replace('tools/rebuild_results.py','tools/rebuild_small_tables.py')
    old_results+='\n\n## ogbn-arxiv方案A：独立预训练1次、固定分类器10次\n\n'+probe_table+f'\n\n测试均值±总体标准差：**{audit["test_mean_percent"]:.3f}±{audit["test_population_sd_percent"]:.3f}%**。不与开发验证表或小图正式组混成排名，不对B2迁移作结论。原小图267次另加本次大图独立预训练1次，10个分类器不计为10个预训练。\n'
    write('02_统一结果总表.md',old_results)
    write('README.md',f'''# GraphMAE2课程报告与收尾交付

更新：2026-11-08（北京时间）。用户选定的实验范围已完成：原小图267次主要完整训练，另加ogbn-arxiv B0独立预训练1次、10个固定分类器；本地报告内容、统一结果和核验证据已更新。不再默认追加实验。

阅读[课程报告](课程报告.md)或[HTML阅读版](课程报告.html)。大图结果为**{audit['test_mean_percent']:.3f}±{audit['test_population_sd_percent']:.3f}%**（同一编码器10个分类器，总体标准差），实际耗时约15小时8分钟。独立预训练仅1次，不代表B2迁移或预训练稳定性。

小图正式主方案仍为B2：Cora 84.54±0.61、Citeseer 73.30±0.23、PubMed 81.70±0.42%，相对同批B0分别+0.35/−0.10/+0.46个百分点；只有Cora通过三主要比较Holm校正（0.0092/0.4090/0.0932）。开发负结果、P/J不利对照和证据边界均保留。

| 计划段 | 当前状态 | 位置 |
| --- | --- | --- |
| A/B：范围、统一结果 | 完成；正式测试/开发验证/大图单列 | [范围清单](01_范围与证据清单.md)、[结果总表](02_统一结果总表.md)、tables/ |
| C：有限核对 | J12有限核验PASS；P12完整报告回执已读；A完整返回核验PASS | [核验状态](03_核验状态与剩余事项.md)、sources/ |
| D：报告与图 | 内容版完成，含三张图和大图逐种子结果 | 课程报告.md/html、figures/ |
| E：交付与保存 | 本地数字、链接、哈希和版面检查完成；最新远端同步、外置独立备份及课程指定格式未确认 | [保存清单](04_保存范围清单.md) |

本包含arxiv小返回ZIP、审计脚本与数字来源，可从保存记录复核60轮、39,720个训练批及10,000个探测epoch；没有下载全部大权重或在本地重新推断。历史快照保留旧状态，当前状态以本README和报告为准。`sources/legacy_closeout_20261106.zip`是原始工作稿，不覆盖其中字节或旧哈希。

复建与校验见[复算说明](tools/复算说明.md)。服务器数据、权重、嵌入与Conda二进制不等于本包内容，也不因普通Git推送自动备份。本轮不自动提交、推送、清理服务器或重新训练。最终课程模板/封面/页数尚未提供，报告为可继续排版的完整内容版。
''')
    write('01_范围与证据清单.md',f'''# 范围与证据清单

2026-11-08。主方案固定B2，停止新增小图方法、调参或扩种；原J版本和历史源码保留。arxiv按用户所选A完成B0一次独立预训练60轮＋10个固定种子各1000轮线性探测，未选择B2三种子迁移预算。

| 人群/批次 | 预训练数 | 指标与证据 |
| --- | --- | --- |
| 历史小图 | 60 | 历史摘要保留，不替换正式同批B0 |
| 正式小图v1 | 120 | 三图B0/B1/B2/B3×种子10～19；val_selected_test_acc；既有完整产物核验 |
| 原开发v2 | 36 | best_val_acc；既有产物核验 |
| R/DS/C | 9/9/9 | best_val_acc；固定提交下既有产物核验 |
| P12 | 12 | {P_STATUS} |
| J12 | 12 | {J_STATUS} |
| arxiv A | 1 | 本地有限核验60轮/39720批/10个分类器10000轮；权重回执未本地重推断 |

原小图267＋arxiv1＝268次主要完整预训练账本，不是268个独立统计样本。10个arxiv分类器另计，工程测量、筛选、冒烟和诊断不混入主账本。

小图确认性主比较是每图B2−同批B0；总体标准差ddof=0，配对t区间用差值样本标准差ddof=1，三项p值统一Holm校正。开发n=3仅描述，保留Citeseer负结果及P/J不利对照。arxiv仅1个编码器，不宣称B2迁移或跨预训练种子稳定。

本地报告和交付一致性完成；课程特定格式、最新远端提交及完整外置资产独立备份尚未确认，见保存清单，不将其写成已经完成。
''')
    write('03_核验状态与剩余事项.md','''# 核验状态与剩余事项

2026-11-08。选定实验范围已结束，不再默认新增训练。

| 内容 | 结论 | 边界 |
| --- | --- | --- |
| 正式120、开发36、R/DS/C | 既有产物核验PASS，当前转录与统计重算一致 | 历史60只转录摘要，未再次逐种子审计 |
| P12 | 完整服务器独立审计报告、回执及审计器/源码身份已本地核对 | 未对全部原始排名数组重审 |
| J12 | 有限原始产物核验PASS：16运行、880源码项、24400轮配额/配对 | 未重放Torch私有随机流，未本地重推断权重 |
| arxiv A | 有限完整返回核验PASS：60轮、39720训练批、662嵌入批、10000分类器epoch | 10分类器共享1编码器；完整权重/嵌入留服务器 |
| Git与独立备份 | 旧本地HEAD及有限导出范围可查，最新远端未核实 | 公共main旧页面不证明目标分支或新A已同步 |

详细机器核验见[sources/J12_audit/audit.json](sources/J12_audit/audit.json)、[sources/arxiv_A/audit.json](sources/arxiv_A/audit.json)。P12完整报告见sources/J12_audit/P12_records/。

剩余事项仅为实际课程格式/封面/提交要求，以及用户实际提交或备份后的范围确认。当前没有这些操作的成功证据，保留未确认状态；不为填补文档状态重新跑实验，不自动Git推送或删除历史文件。
''')
    write('04_保存范围清单.md','''# 保存范围与备份清单

2026-11-08。本清单区分本地交付、服务器保留和独立备份，不把目录存在或回执等同第二份副本。

本地报告包包含当前报告MD/自包含HTML、三张图的PNG/SVG、统一CSV/JSON、25份历史来源字节快照及原始工作稿ZIP、J12有限核验报告/P12完整服务器审计回执、大图L1与正式A的小返回ZIP和固定启动包，以及复算/核验脚本。SHA256SUMS.txt列出当前包文件哈希；大模型、完整嵌入、原始数据、全部历史原始数组及环境二进制不在本包。

| 资产 | 已知位置/身份 | 保存与核实范围 |
| --- | --- | --- |
| 正式120原始归档 | C:/Users/18870/Desktop/GraphMAE2/formal120_x6CfgE_results.tar.gz | 本地原状，SHA256 2f9a797f5e15eb833ae4f612d1e0f56243030657e7025bf495ba97c753dbb036 |
| v2原36原始归档 | 同目录v2_development36_results.tar.gz | 本地原状，SHA256 e523cba3998593b9a95050a42088dfde6533de82fee39f371d9e840a46884199 |
| J12/P12原导出 | 同工作区outputs/GraphMAE2_J12_audit_20261106/inputs/existing_J12_P12_evidence.zip | 本地原包保留，SHA256 d6ab30b581035badbcc184c887747bef8bc61fea4157b9506b70200b3d733581；本报告包仅含核验及关键回执 |
| arxiv A原返回 | sources/arxiv_A/A_return_a0001.zip | 本包实际含原字节，SHA256 e622b5c4599f14535ba592a784741c1ec5b6bab6247b360f8cde021562e845e2 |
| 小图外置权重 | /home/jovyan/work/GraphMAE2_artifacts/checkpoints | 服务器记录位置；未下载全部文件或确认独立备份 |
| arxiv A完整资产 | /home/jovyan/work/GraphMAE2_artifacts/arxiv_A_TxrQqJzj | final_B0.pt、最近两轮断点、10个最佳线性头、完整嵌入及根映射；服务器SHA/CPU读回回执，未下载大文件 |
| arxiv原始数据/聚类 | GraphMAE2_artifacts/arxiv_prepare_jcmKhIqS | 原始ZIP/8压缩文件与完整聚类SHA在L1/A契约中固定；未确认第二份副本 |
| arxiv工程测量资产 | GraphMAE2_artifacts/arxiv_l1_measurement_v2_kISgbDss | 单轮诊断权重及嵌入仍与正式资产分开，不计正式预训练 |
| 原Conda环境 | /home/jovyan/work/envs/graphmae2 | 版本前后一致有记录，不等于二进制离线备份 |

arxiv正式大文件逐项SHA与状态见[tables/arxiv_A_server_asset_receipts.csv](tables/arxiv_A_server_asset_receipts.csv)。较早的本次断点按入口约定轮换，仅保留第59/60轮与最终编码器，已轮换文件的历史哈希和日志仍在，不将旧断点写成尚存权重。

Git范围：上游固定bc473e8b750d6e90b7cba1c6bb20d858f617d90c；历史R对应705cdc51f63763996975f0305dec9f9806b11597，DS/C对应40de08f6b42c050e94ed65444beeb4f2b6520a06。2026-11-06回传的服务器本地HEAD为8062dd851552c4f519924d75a28e85dc068c58ab，有限264文件中200匹配该HEAD、64未列入，详见sources/J12_audit/file_inventory.json。

2026-11-08公开页面读取只得到较旧main README，目标feat/structure-curriculum分支和8062提交页面无法读取，未取得可证明最新远端同步的材料。arxiv本次57个运行源码不变不等于最新Git状态干净，更不等于新日志或外置权重已推送。当前没有自动提交/推送。

课程封面、页数、模板与最终提交渠道未提供；已交付内容版及阅读版，按实际要求排版/提交的操作尚未发生。独立备份是否完成以真实第二份资产及校验清单为准，不能靠修改README制造已备份状态。
''')
    checks=load('validation_checks.json')
    checks.update(date='2026-11-08 Asia/Shanghai',scope='Frozen small-graph numeric reconstruction plus current P/J and arxiv finite audit receipts',
       J12_status=J_STATUS,P12_status=P_STATUS,arxiv_A_status=audit['status'],arxiv_A_pretrain_runs=1,arxiv_A_probe_runs=10,
       archive_sha256_checks=0,historical_raw_archive_hash_receipts_preserved=2,
       latest_github_sync_verified=False,limitation='Remote branch sync, complete independent asset backup, and course-specific format remain unconfirmed. Full model predictions were not replayed locally.')
    write('validation_checks.json',json.dumps(checks,ensure_ascii=False,indent=2)+'\n')
    write('tools/复算说明.md','''# 复算与核验

本文档生成不会启动GPU或加载模型权重。原小图来源、25份快照及303条选读指标保持原始字节；sources中的历史文档可能记载已过时状态，当前证据等级由build_report.py在不改变成绩的前提下更新。

1. Python标准库运行`python tools/build_report.py`：从冻结来源重算小图正式/开发数字与区间，生成当前MD/结果表，纳入已核验arxiv表。
2. 若需重审arxiv原返回，运行`python tools/recheck_A.py`，临时展开本包冻结启动包并调用有限审计器，核对全部逐批/探测记录；不执行返回包源码、不加载.pt或重新测准确率。
3. ReportLab运行`python tools/render_arxiv_figure.py`；该图只读60轮与10000个验证记录，不读开发测试值。
4. Node运行`node tools/render_preview.cjs`生成PNG与自包含HTML，需要Sharp/marked。最后`python tools/verify_delivery.py`核对数字、链接、来源与资产。生成者的库来自Codex已捆绑运行环境，未安装或升级实验环境。

本报告包不包含全部历史原始数组。旧小图产物审计及J/P审计保留原证据范围，当前数字复算不冒充重新审核所有旧权重。大图arxiv完整权重/嵌入留服务器，核验基于已匹配字节的运行入口、轨迹、散列及保存回执，边界见audit.json。
''')
    print('Rebuilt current report, tables, evidence levels and save-scope documents.')


if __name__=='__main__':main()
