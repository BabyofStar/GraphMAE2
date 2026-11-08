# 外置权重恢复

下载本 Release 的全部 weights.tar.partNNNN、weights_manifest.json、parts_manifest.json、SHA256SUMS.weights.txt 和 restore_weights.py 到同一目录。

运行：

```bash
python restore_weights.py --manifest weights_manifest.json --parts parts_manifest.json --directory . --dest /new/path/GraphMAE2_weights_restore
```

目标目录必须不存在。工具先核对所有分卷 SHA256，再逐个写入普通文件并验证原文件 SHA256；拒绝未知成员、链接、绝对路径和目录穿越。整个过程不加载模型、不执行 pickle 或训练。

恢复目录内为 GraphMAE2_artifacts 原相对结构，包含 235 个本次已列明文件。原始数据、局部聚类、完整嵌入和环境二进制未纳入本次权重备份。源码及实际实验记录见本 Release 绑定的 Git 提交，10 个 arxiv 线性头共享一个编码器。
