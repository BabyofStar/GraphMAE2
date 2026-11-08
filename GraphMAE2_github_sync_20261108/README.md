# GraphMAE2：GitHub同步和外置权重独立备份

用户已授权同步 GitHub，并指定把外置权重备份到 GitHub。课程排版暂缓。本包承接 github_prepare_return.zip 的实际检查结果，不启动实验。

## 本次范围

- 仓库：BabyofStar/GraphMAE2，服务器既有 `course_backup` SSH 通道，目标 `main`。
- 源码和实验记录：预检选定 3766 个文件，约 532 MiB；另附已经冻结的报告内容版。排除未跟踪的环境、缓存和准备包；已跟踪内容保持记录。
- 权重：235 个文件，原始共 8,790,751,492 字节（约 8.19 GiB）。包含 checkpoints 下全部预检文件、arxiv A 编码器及线性头、一次大图测量检查点。13 个正式 arxiv A 权重均已匹配原验收哈希。
- 不包含原始数据、局部聚类数据、完整嵌入及环境二进制。具体范围由 weight_inventory.json 固定；这次不声称完成整个服务器的备份。

## 执行

将本 ZIP 上传到服务器 `/home/jovyan/work/GraphMAE2`，在 JupyterLab 终端运行：

```bash
cd /home/jovyan/work/GraphMAE2
unzip -n GraphMAE2_github_sync_20261108.zip
bash GraphMAE2_github_sync_20261108/run.sh
```

无需切换当前 conda 环境，脚本使用既有环境的 Python 标准库。请保持前台终端打开。

1. 如果提示 SSH 密钥口令，输入原来密钥的口令。
2. 如果出现 GitHub 登录网址和一次性代码，在自己的浏览器打开网址，使用 **BabyofStar** 完成登录。终端若提示按 Enter，可按 Enter 后自行复制网址打开。无需向聊天发送 Token、口令或密钥。
3. 结束或中途报错后，下载终端打印的 `RETURN_FILE=.../github_sync_return.zip` 并回传。

再次运行同一个 run.sh 会恢复同一个任务、同一个 Release；已经完成并通过验证的资产会复用。不要删解压目录里的 `.launch_receipt.json`，也不要重新解压到另一个目录发起第二个任务。若中断造成一个分卷只上传了一部分，该分卷会重新上传，其他完整分卷保留。

## 会做什么

先在服务器项目之外建立独立 Git 工作目录，复制已经预检的文件、保存旧 README、加入报告内容，获取远端 main 并正常合并和推送。实验目录保持原来的分支、提交和暂存状态。仅 README 冲突会按已经准备的内容更新，并保留双方旧版；其他冲突保留现场并返回，不强制推送。

权重在 `GraphMAE2_backup_staging` 内按 1 GiB 分卷生成无损 tar，预计 9 卷。归档后逐文件读回，核对全部原始 SHA256。原权重不移动或删除。

脚本从官方 cli/cli 固定版本 v2.103.3 的发行包安装一个独立登录工具，验证官方校验和；不安装到实验环境。登录配置存放在项目外的私有目录，不进入 Git 或回传包。

Release 先作为草稿上传。每个资产的大小和 GitHub 计算的 SHA256 必须与本地一致；如果 API 没有摘要，则流式下载重算。全部通过后发布 Release，再把验收清单、恢复说明及 Release 链接提交到 main。Release 标签绑定源码快照提交；main 后续还有一笔验收记录提交，两者完整哈希分别保存在回执中。

上传耗时由服务器出口决定。8.19 GiB 按持续有效速度 1 MiB/s 约需 2 小时 20 分，5 MiB/s 约 28 分，10 MiB/s 约 14 分；还需加上 Git 推送、归档读回和登录时间。这里只是流量换算，不是实测速度预测。

## 验收条件

- `source_synced.json`：初次源码推送后，远端 main 与同步工作目录完整提交哈希相同。
- `release_verified.json`：235 个原文件的归档已逐文件读回，所有远端资产的内容校验通过，并已发布 Release。
- `final_sync_verified.json`：补充验收记录后，main 再次与最终提交哈希相同。
- 三者缺一，都不宣布整个任务完成。本机测试只验证工具逻辑，不代替服务器实际执行及 GitHub 远端回执。

`github_sync_return.zip` 只含小型回执和输入身份，不含权重分卷或登录凭据。上传失败保留本地归档和已完成远端资产；遇到停止信息请回传返回包供针对性处理。

GitHub 官方依据：[Release 容量说明](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)、[普通 Git 大文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)、[Release 资产接口及 digest](https://docs.github.com/en/rest/releases/assets)、[浏览器登录](https://cli.github.com/manual/gh_auth_login)。
