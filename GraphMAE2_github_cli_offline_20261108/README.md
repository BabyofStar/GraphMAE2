# 恢复现有GitHub备份任务

原任务 sync_u0Vtv5j5 停在下载 GitHub CLI 官方校验文件的连接阶段。此处尚未执行 Release 上传。源码同步和权重归档步骤已先返回成功；其实际细节仍需 github_sync_return.zip 回执核验。

本包附官方 cli/cli v2.103.3 的 Linux x86_64 发行包及官方校验文件，均已匹配 GitHub API 资产 SHA256，发行包同时匹配官方校验文件。只在项目外安装这个独立工具，不安装或改动实验环境。不修改原执行包及其身份清单，不创建新备份任务、重跑实验或重新生成已有权重分卷。

上传本ZIP到服务器项目目录，运行：

```bash
cd /home/jovyan/work/GraphMAE2
unzip -n GraphMAE2_github_cli_offline_20261108.zip
bash GraphMAE2_github_cli_offline_20261108/resume.sh
```

离线安装通过后会继续原来的 run.sh。它跳过已经成功的源码步骤，重新校验已有归档，然后浏览器登录 GitHub、上传并核验权重资产、推送验收记录。保持终端打开，按提示输入原 SSH 密钥口令，在自己的浏览器用 BabyofStar 完成 GitHub 登录，无需发送 Token 到聊天。

结束或停止后回传终端 RETURN_FILE 指向的 github_sync_return.zip。离线包只解决下载登录工具超时，后面的 GitHub 登录、API 和资产上传仍需服务器网络；如果这些端点也超时，回传错误和返回包以便按实际失败位置处理。

再次运行相同 resume.sh 仍使用原任务；不要删除原包的 .launch_receipt.json。官方资产来源与哈希见 official_provenance.json。

官方来源：[v2.103.3 发行页](https://github.com/cli/cli/releases/tag/v2.103.3)、[GitHub 资产下载说明](https://docs.github.com/en/rest/releases/assets#get-a-release-asset)。
