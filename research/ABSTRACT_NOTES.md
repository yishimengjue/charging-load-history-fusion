# 摘要编写依据

2026年11月27日。正文位于`src/abstract.mjs`，网页为`abstract.html`，置于研究总览之后。

## 写作参考

- Nature，How to construct a Nature summary paragraph：背景与问题、主要发现、与已有认识的关系、研究含义。已核读[官方PDF](https://www.nature.com/documents/nature-summary-paragraph.pdf)。借鉴组织顺序，不照搬其篇幅要求。
- Lim等，Temporal Fusion Transformers for interpretable multi-horizon time series forecasting，International Journal of Forecasting，2021：摘要按输入问题、架构、组件作用和实证验证展开。已核读[作者稿](https://arxiv.org/pdf/1912.09363)第一页摘要，[正式题录](https://doi.org/10.1016/j.ijforecast.2021.03.012)。
- Applied Energy官方作者指南此次返回403，未作为已核读依据。

## 事实核对

方法与验证设置来自`CURRENT_RESULTS.md`及现有方法页面。摘要中的MAE、相对变化与分组贡献来自`current-evidence.json`。6.17%和7.55%均以OP一至三月的仅近期输入为参照；0.35%与1.21%对应近7天重组；SOFIE增加2.09%对应近远期输入相对仅近期输入。摘要不将单日案例、旧流程组合收益或MAE变化写作经济收益。

三来源、四范围的神经输入比较保持同来源组间设置一致；LightGBM和既有组合是其他学习器与流程的参照，不属于相同网络结构的输入消融。
