# 历史输入与预测效果图件

更新日期：2026年11月27日。全部图件由 Matplotlib 读取已有实验数据绘制，未新增训练。

| 图件 | 内容 | 网页位置 |
|---|---|---|
| 图1：近期历史的整理 | 真实168小时曲线及其中前7天的同小时记录 | 方法 → 历史输入 |
| 图2：历史输入消融 | 近期重组、增加远期记录及总改动的误差变化 | 结果 → 输入消融 |
| 图3：日负荷预测曲线 | 三个日期的真实负荷、预测与逐小时误差变化 | 结果 → 日负荷曲线 |
| 图4：连续日期与预测步 | 逐日收益及未来第1至24小时的平均误差 | 结果 → 连续评价 |
| 图5：使用前后对照 | 7天历史延长到预测日，左右比较两种输入，放大全天曲线并标出18点误差 | 方法 → 历史输入 |

## 数据与读图

图1使用OP编号7在2022年3月23日协调世界时零点起报的历史输入，示例目标为18小时后的功率。选中的七个值按时间顺序为5.66、5.31、10.55、11.73、5.34、2.76、3.40 kW，均与原始输入逐值核对。缺失观测保留为断线。

图2的点为三个随机种子分别计算损失后的平均差值。正值表示误差降低。横线为2000次非循环7日配对日期块重采样的95%区间，条件于已有拟合预测。四个评价范围保留各自单位。近期重组的MAE降幅依次为6.17%、0.35%、1.21%、0.0022%。

图3固定使用种子42。编号7按九条序列的观测平均负荷中位位置选取；三个日期从35个完整日中按日均负荷、日峰值、充电小时占比选择，不按预测误差选择。OP真值为整点前300秒内最后一条有效功率记录。每天协调世界时零点起报，第24小时对应次日零点。

图4汇总三个种子的损失。一至三月有67/90天的日均误差下降，七月有12/25天。总体MAE按有效评分目标数加权，而非直接平均每日MAE。

## 文件与复核

每图提供600 dpi PNG、矢量PDF及保留文本的SVG；`paper_figures.pdf`为五图合集。网页直接嵌入SVG，图中文字以UTF-8保存，可选中复制。图题、坐标轴、图例和说明均使用中文，模型名、数据代号与计量单位保留通用写法。字体明确采用Heiti SC简体中文成员，PDF嵌入字体；网页同时配置简体中文备用字体。

图5沿用图1的历史输入和图3中2022年3月23日的种子42预测。两列使用相同时间与功率范围，浅蓝区域为预测日，右上空心点为7个同小时历史值。18点真实值为1.38 kW，使用前后预测为11.300904、5.984122 kW，绝对误差为9.920904、4.604122 kW。下排显示全天曲线与该时刻的误差线段；最后一个预测点为次日零点。

图5可运行仓库脚本`python scripts/plot-history-before-after.py`重绘，需要matplotlib、numpy、pymupdf、fonttools。脚本读取本目录的CSV，核对168个历史位置、24个预测位置、7个参照值及逐小时误差。非macOS环境可通过`CHINESE_FONT`指定简体中文TTF字体。核验结果保存为`Fig05_history_before_after_verification.json`。

配套CSV保存绘图数值。公开汇总为`research/current-evidence.json`；逐点来源位于本地研究目录`reproduced-history-attribution-replication-v1/analysis/`与`reproduced-history-attribution-v1/prepared/2022-03_query.npz`。本次发布图件和绘图数值，不另行上传整套预测文件。绘图脚本核对来源哈希、预测指标与七个历史值；独立检查脚本复算逐小时、逐日、逐步误差以及加权总体误差。

- 图1：[连续历史](Fig01_continuous_history.csv)、[选中记录](Fig01_selected_records.csv)。
- 图2：[消融数值](Fig02_source_data.csv)。
- 图3：[真实值与预测](Fig03_source_data.csv)。
- 图4：[逐日数值](Fig04_daily_source_data.csv)、[逐步数值](Fig04_horizon_source_data.csv)。

## 制图参考

- [Nature科研制图指南](https://research-figure-guide.nature.com/figures/preparing-figures-our-specifications/)：轴标签、单位、字体嵌入、矢量导出。
- [Elsevier图件规范](https://www.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions)：颜色与线型配合、图像处理。
- [Lim等，2021，Temporal Fusion Transformers](https://doi.org/10.1016/j.ijforecast.2021.03.012)：分组消融与多面板时间图的组织方式。

采用白底、蓝橙绿灰配色和线型区分模型。以上为排版参考，曲线与数值均来自本研究。
