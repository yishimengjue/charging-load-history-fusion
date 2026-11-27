export function paperFigure(name, title, description) {
  const path = `figures/paper/${name}`;
  return `<figure class="paper-figure">
    <div class="paper-figure-heading"><h3>${title}</h3><span class="paper-scroll-hint">横向滑动查看完整图表</span></div>
    <div class="paper-plot-window" tabindex="0" role="region" aria-label="${title}">
      <img src="${path}.png" alt="${description}" loading="lazy">
    </div>
    <figcaption><p>${description}</p><nav aria-label="图件下载">
      <a href="${path}.png" target="_blank" rel="noopener">查看大图</a>
      <a href="${path}.pdf" download>下载 PDF</a>
      <a href="${path}.svg" download>下载 SVG</a>
    </nav></figcaption>
  </figure>`;
}

export function paperResultSections() {
  return new Map([
    ['input-effects', `<section id="input-effects" class="chapter current-chapter">
      <header class="section-heading"><h2>近期信息重组与远期历史分别带来多少收益</h2>
        <p>蓝色圆点对应近期信息重组，橙色方点对应增加远期记录，绿色菱形对应两步总改动。</p></header>
      ${paperFigure('Fig02_history_input_ablation', '图2　历史输入的消融结果',
        '横轴为平均绝对误差降低值。点在零线右侧表示改善，左侧表示退步。四个数据范围分别保留各自的单位和刻度。')}
      <p class="current-note">OP一至三月，重组近7天信息使MAE下降<b>6.17%</b>；加入更早历史后，总降幅为<b>7.55%</b>。</p>
      <details class="reading-details"><summary>查看比较对象与区间定义</summary><div class="detail-body">
        <p>近期信息重组比较“仅近期序列”和“近期序列＋近7天同小时记录”。增加远期历史比较“近7天”与“近远期18条记录”。两步总改动比较“仅近期”与“近远期”。</p>
        <p>各组保持神经结构、训练样本和预算一致。点为三种子的平均损失差，横线为2000次非循环7日配对日期块重采样得到的95%区间，条件于已拟合预测。SOFIE近期重组的区间窄于图中符号。</p>
      </div></details>
    </section>`],
    ['daily-curves', `<section id="daily-curves" class="chapter current-chapter">
      <header class="section-heading"><h2>真实日负荷与近期重组前后的预测</h2>
        <p>上排展示负荷曲线，下排展示每个小时的绝对误差变化。</p></header>
      ${paperFigure('Fig03_observed_and_predicted_daily_load', '图3　三个日期的预测曲线',
        '黑线为真实负荷，灰色虚线为仅近期预测，蓝线为加入近7天同小时记录的预测。下排正值表示误差下降，负值表示误差增加。')}
      <details class="reading-details"><summary>查看日期选择与计量说明</summary><div class="detail-body">
        <p>编号7的观测平均负荷位于OP九条序列的中位位置。在其35个完整日中，依次选取日均负荷接近中位数、日峰值接近第90百分位、充电小时占比接近第25百分位的三个不同日期。第三个日期有19/24小时为正负荷，百分位是在该编号可选日期内计算。</p>
        <p>曲线固定使用随机种子42，日期由真实负荷特征决定。每天协调世界时零点起报，预测步24对应次日零点。OP真实值为每个整点前300秒内最后一条有效功率记录。</p>
      </div></details>
    </section>`],
    ['temporal-results', `<section id="temporal-results" class="chapter current-chapter">
      <header class="section-heading"><h2>连续日期收益与各预测步误差</h2>
        <p>逐日比较近期信息重组的收益，再查看第1至24小时的平均预测误差。</p></header>
      ${paperFigure('Fig04_daily_and_horizon_errors', '图4　时间维度上的预测效果',
        '上排每个点代表一天，正值表示近期重组降低了日均误差。下排曲线比较三种输入在各预测步上的平均绝对误差。')}
      <p class="current-note">OP一至三月有<b>67/90天</b>的日均误差下降；七月有<b>12/25天</b>下降。总体MAE按有效评分目标数加权。</p>
      <details class="reading-details"><summary>查看统计口径</summary><div class="detail-body">
        <p>每日统计汇总当日全部可评分序列与预测步。逐步统计汇总评价期内同一预测步的全部可评分位置。两者均先计算三个种子的损失，再取平均。</p>
        <p>近远期18条记录由前1至7天的同小时值，以及前14、21、…、84天的同小时值组成。灰、蓝、橙三条曲线分别对应仅近期、近7天、近远期输入。</p>
      </div></details>
    </section>`]
  ]);
}
