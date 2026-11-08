# GitHub同步与外置权重独立备份：准备包

2026-11-08。用户已指定备份到GitHub，课程排版暂缓。既有配置为`course_backup`→`ssh://git@ssh.github.com:443/BabyofStar/GraphMAE2.git`，本地`feat/structure-curriculum`推送GitHub `main`；不是origin同名开发分支。此前对后者的检查不能作为实际备份成功/失败的依据。

准备阶段收集实际Git状态/候选文件、全部已知外置模型权重哈希、空间及现有GitHub登录。只写独立准备目录，不暂存、提交、合并、推送、创建Release、加载模型或训练。用户已授权同步/备份；这些结果用于把后续执行范围做实，不是再次申请操作许可。

上传此ZIP到服务器`/home/jovyan/work/GraphMAE2`，运行：

```bash
cd /home/jovyan/work/GraphMAE2
unzip -n GraphMAE2_github_prepare_20261108.zip
bash GraphMAE2_github_prepare_20261108/run.sh
```

这是前台任务，逐步显示哈希数量。若读取main时提示`Enter passphrase for key ...`，在终端输入原SSH密钥口令，输入不回显是正常现象；不要把口令/Token/私钥发到聊天中。现有SSH认证只能用于Git操作，不能据此假设Release API已经登录。本包会检查现有`gh`登录或GH_TOKEN/GITHUB_TOKEN的存在及只读API访问，只在内存使用，返回文件不保存或打印凭据。

全部命令结束后，下载最后打印`RETURN_FILE`对应的`github_prepare_return.zip`回传。输出位于`/home/jovyan/work/GraphMAE2_backup_staging/prepare_<编号>`，在项目和权重目录之外，避免清单递归包含自己。若被项目执行锁拦截，不启动新的任务，回传终端提示即可。

源码候选覆盖当前普通源码、测试、日志及结果；未跟踪的preprocess_env、虚拟环境、pip/绘图缓存、native_source、__pycache__等分开列出，已经跟踪的文件不因这些规则被省略。存在未解决冲突、已有暂存、缺失跟踪文件、大文件、符号链接或敏感路径/文本时均记录，后续据实际情况处理，不通过清空用户暂存区或盲目整目录强制提交解决。文本检查有大小/格式边界，不宣称能发现任意二进制或压缩包中的秘密。

权重范围是`GraphMAE2_artifacts/checkpoints`内全部普通文件，以及`arxiv_A_*`、`arxiv_l1_measurement_*`中的模型文件。正式arxiv A的13个.pt逐一匹配已审计SHA256。原始数据、局部聚类.pt、全量嵌入.npy和Conda环境不是模型权重，单独列出未纳入范围，不把它们写成已备份。文件若在哈希过程中变化或跳出已知资产根目录会停止，不删除源文件或其他任务。

普通Git会阻止超过100 MiB的单文件；[GitHub官方大文件文档](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)建议大二进制采用Release等方式。[Release文档](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)规定每个附件小于2 GiB。本次后续将按实测权重规模生成分卷、文件清单与校验信息，通过Release API上传并核对远端副本；服务器上多放一份压缩包不算GitHub独立备份。

`README.sync_candidate.md`是下一次源码同步的README候选；`closeout_content_20261108.zip`是已经完成的报告/结果内容原包，只同步现有内容，不做课程排版。准备阶段不把候选覆盖到项目根目录。后续将依据实际远端/main差异确定合并与提交，保留原分支和既有工作，不强推或重写历史。

本地逻辑检查覆盖环境排除与已跟踪保留、权重范围/符号链接、秘密检测不回显、URL脱敏、文件变化检查及13个A权重身份契约。服务器总权重大小、当前main和Release登录尚需本次回传确认；在真正上传与校验前，不宣称同步或独立备份完成。
