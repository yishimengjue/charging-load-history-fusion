# 充电负荷预测研究图解

[在线阅读](https://yishimengjue.github.io/charging-load-history-fusion/) · [复现实验](REPRODUCE.md) · [数据与模型下载](https://github.com/yishimengjue/charging-load-history-fusion/releases/tag/repro-v0.1) · [方法与结果正文](research/METHODS_AND_RESULTS_v0.5.md) · [Word](research/METHODS_AND_RESULTS_v0.5.docx)

## 研究内容

近期168小时序列由TimeXer编码，与7/28/84天同类时段的历史统计及可用性标记直接拼接，再按计量序列适配输出。统计树与神经输出的平均作为独立比较。

| 同预算神经对照 | 无摘要MAE | 有摘要MAE | 降幅 | 单位 |
|---|---:|---:|---:|---|
| OP 2022年1—3月 | 4.938161 | 4.660014 | 5.63% | kW |
| OP 2022年7月1—25日 | 3.597243 | 3.555579 | 1.16% | kW |
| AMS 2019年11月—2020年1月 | 3.239926 | 3.183752 | 1.73% | kWh |
| LCL 2013年11月—2014年1月 | 0.154418 | 0.152106 | 1.50% | 来源原始单位 |

OP七月差值区间跨零；当前匹配消融为回顾性对照。OP与AMS的固定平均误差低于两个分支，LCL神经单分支低于平均。当前模型未完成全年验证，也未测量成本或光伏消纳收益。

## 阅读顺序

论文主线 → 三层泳道 → 贯穿示例 → 取数与融合 → 实验结果 → 方法尝试表 → 研究文件。方法表含74条总览和117条细项，细项可能展开总览，不相加为独立模型数量。绿色表示当前路线采用，不代表部署或全部数据最优。

## 文件说明

- index.html：自包含单页，支持目录、术语卡片、结果切换与方法筛选。
- src/：网页模板、样式和脚本；修改后运行 node build.mjs。
- research/：v0.5第3—5章、Word、附录、指标和参考文献。旧前言未并入当前稿。
- research/evidence/：56份结果报告的脱敏文本与阅读页；保留各自历史版本。
- research/HISTORICAL_LITERATURE_REVIEW_2026-10-16.md：早期综述，研究重心与当前版本不同。
- publication.json：公开文件来源与文本哈希。

## 公开范围

当前OP、AMS、LCL主线的数据、模型和逐点预测已整理为三个Release包，共约225 MiB；仓库包含固定版本实现、依赖和下载/校验/复算/回放/输入重建/重训入口，见[复现指南](REPRODUCE.md)。历史方法表中的会话分解、价格及其他预训练模型支线尚未整理为同等可运行包。

OP和AMS提供原CSV；LCL提供已用48条计量序列的两个字段及使用时期投影。问卷、身份映射、授权文件及凭据不发布。数据按原发布方CC BY 4.0署名与许可再分发，变换见[数据许可说明](DATA_LICENSES.md)。不将数据许可扩展为所有第三方代码和文献的许可。

```bash
python scripts/fetch_reproduction.py all
python scripts/reproduce.py scores
python scripts/reproduce.py replay
python scripts/rebuild_inputs.py
```

运行前使用Python 3.12安装requirements-reproduction.txt。快速复算只需data包；完整操作顺序见复现指南。

## 网页更新

需要Node.js 22或更高版本，无第三方依赖。运行 node build.mjs 后提交 index.html，GitHub Pages从main分支根目录发布。
