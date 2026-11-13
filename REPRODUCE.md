# 复现当前论文主线

已执行的核验范围和数值见[复现验收报告](REPRODUCTION_VERIFICATION.md)。

本版本覆盖OP、AMS、LCL的10个月份、3个随机种子、有摘要/无摘要神经分支和固定等权组合。方法表中的早期会话分解、价格与其他预训练模型支线仍是历史记录，尚未整理为同等可运行的数据包。

## 下载与环境

使用Python 3.12。数据和模型放在同一仓库的 [repro-v0.1 Release](https://github.com/yishimengjue/charging-load-history-fusion/releases/tag/repro-v0.1)，不进入Git历史。

```bash
git clone https://github.com/yishimengjue/charging-load-history-fusion.git
cd charging-load-history-fusion
python3.12 -m venv .venv-repro
source .venv-repro/bin/activate
python -m pip install -r requirements-reproduction.txt
python scripts/fetch_reproduction.py all
```

Windows激活命令为 `.venv-repro\Scripts\activate`。Torch/LightGBM需要适配操作系统的运行库；原验证平台为macOS、CPU、Python 3.12。严格固定包版本不能保证不同硬件逐位一致。

| 包 | 压缩体积约 | 内容 |
|---|---:|---|
| data | 61 MiB | 冻结训练/查询输入、真值、缺失标记、逐点预测、统计记录 |
| models | 145 MiB | 神经权重、局部输出头、树检查点及配置 |
| raw | 19 MiB | OP和AMS原CSV、LCL已用序列的字段/时期投影 |

下载器核对包体积、SHA256、成员清单与逐文件SHA256，拒绝路径穿越、链接和覆盖已修改文件。离线下载三个压缩包后可以运行：

```bash
python scripts/fetch_reproduction.py all --archive-dir /path/to/downloads
```

如果网络环境的证书配置阻止Python下载，可用浏览器下载Release附件后使用离线模式；不要关闭TLS验证。数据来源、署名与变换见 [DATA_LICENSES.md](DATA_LICENSES.md)。

## 1 重算论文摘要消融

```bash
python scripts/reproduce.py scores
```

从逐点预测及真值重算四个合并范围的总体/分负荷状态MAE、种子MAE、误差贡献及7/14日循环/非循环日期块区间，与已发布结果逐项比较。输出 `reproduced/scores.json`。此步骤只需data包，主要计算依赖numpy。

## 2 加载权重重新预测

```bash
python scripts/reproduce.py replay
```

运行10个月份×3种子×2种摘要条件的60组神经预测，并与原逐点预测对照。当前验证环境中60组全部逐点一致；跨环境验收容差为atol=5e-5、rtol=1e-5，同时报告最大差值和是否逐位一致。输出 `reproduced/replay.json`。

## 3 从原始记录重建输入

```bash
python scripts/audit_raw_inputs.py
python scripts/rebuild_inputs.py
```

第一条核对已存输入/标签中的有限值是否来自公开的计量记录。第二条在已发布的固定计量对象、起报时刻与样本划分上，重新计算近期168小时历史、训练标签和三池27维统计，再与参考输入比较。查询标签始终置为未知；不会用未来真值训练。

输出在 `reproduced-inputs/`。样本身份、时间划分以及参考元数据沿用冻结记录；这一步不是重新选择对象和评价月份。LCL的树参照字段保留在文件中，但当前直接拼接神经分支不把它作为预测输入。

## 4 重新训练

```bash
python scripts/reproduce.py train --source ams --month 2019-11 --seed 42 \
  --without-summary --output reproduced-training-ams

python scripts/reproduce.py train --source op --month 2022-07 --seed 42 \
  --input-dir reproduced-inputs --output reproduced-training-op-raw

python scripts/reproduce.py train --source lcl --month 2013-11 --seed 42 \
  --input-dir reproduced-inputs --output reproduced-training-lcl-raw
```

每次运行完整80轮编码器和20轮输出适配。去掉 `--without-summary` 使用摘要。`--input-dir`使用从原始记录重建的输入；不传则使用data包中的冻结输入。为每个来源、月份、种子和摘要条件使用不同输出目录，脚本不会覆盖既有训练。

OP月份为2022-01/02/03/07，AMS为2019-11/12和2020-01，LCL为2013-11/12和2014-01；种子为42/43/44。对这些组合重复训练即可重建神经对照；当前Release附带全部对应已保存权重，不要求先重训才可核查结果。

统计树和部分历史对照的权重/预测包含在包中；尚未提供重训全部历史对照的一键调度器。不要将神经重训验收解释为方法表中所有模型都已从头重训。

## 版本与安全边界

- `reproduction-manifest.json`列出813个包内文件与实验代码的校验值。
- `repro/src`和`repro/diagnostics`保留依赖链实现；历史脚本中的旧freeze/protect入口用于原工作区，不作为本发布包入口。使用上述scripts命令及新公开清单校验。
- 本地路径在JSON中被替换，因此原工作区acceptance/freeze文件没有作为可执行验收凭据发布。新清单核验公开版本，不伪装为原哈希未改变。
- 当前结果支持已评价时期上的点预测比较，不等于全年、成本或光伏协调收益验证。
