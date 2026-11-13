# 当前论文主线复现包 repro-v0.1

本版本发布OP、AMS、LCL主线的数据、神经权重、逐点预测和固定版本实现，支持结果复算、权重回放、原始输入重建和神经重训。

| 附件 | 内容 |
|---|---|
| data-repro-v0.1.tar.gz | 冻结输入、标签、逐点预测与统计记录 |
| models-repro-v0.1.tar.gz | 神经权重、局部输出头、树检查点及配置 |
| raw-repro-v0.1.tar.gz | OP与AMS原CSV、LCL已用48条序列的两字段/时期投影 |

三个包共813个文件、约225 MiB。下载命令会核对包和逐文件SHA256。

```bash
python -m pip install -r requirements-reproduction.txt
python scripts/fetch_reproduction.py all
python scripts/reproduce.py scores
python scripts/reproduce.py replay
python scripts/rebuild_inputs.py
```

使用Python 3.12。安装、重训和离线下载方法见[复现指南](https://github.com/yishimengjue/charging-load-history-fusion/blob/repro-v0.1/REPRODUCE.md)。

验收结果：四个评价范围指标/区间复算一致；60组神经权重回放逐点一致；8,274,780个有限值与原始记录核对通过；20个输入文件重建通过；三数据源各一例完整重训与原预测逐点一致。详见[验收报告](https://github.com/yishimengjue/charging-load-history-fusion/blob/repro-v0.1/REPRODUCTION_VERIFICATION.md)。

数据按原发布方CC BY 4.0许可署名再分发，LCL投影不是完整源ZIP。来源、变换及TimeXer代码署名见[许可说明](https://github.com/yishimengjue/charging-load-history-fusion/blob/repro-v0.1/DATA_LICENSES.md)。

历史会话分解、价格和其他预训练模型支线尚未形成同等可运行包；本版本不声称历史方法表中的全部方法都已完成复现。
