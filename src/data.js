window.PAPER_DATA = {
  updated: '2026-11-13',
  glossary: [
    {term:'TimeXer',kind:'借鉴模型 · 时间序列预测',definition:'一种Transformer预测架构：从连续历史中学习变化模式，并结合目标序列之外的辅助信息预测未来。',use:'本文借用其编码结构，将近期168小时负荷编码，再与另行构建的历史摘要拼接。这里的历史摘要与融合设计属于本文的研究内容。',source:'Wang等，2024 · TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables',url:'https://arxiv.org/abs/2402.19072v4'},
    {term:'OP',kind:'数据来源 · Optimise Prime',definition:'英国Optimise Prime项目的公开充电记录。OP是本页对这一数据来源的简称。',use:'构造小时边界前5分钟内最后一条功率记录，预测未来24步，误差单位为kW。一个发布方编号对应一条计量序列，物理站点映射待确认。',source:'UK Power Networks · UKPN Optimise Prime',url:'https://ukpowernetworks.opendatasoft.com/explore/dataset/ukpn-optimise-prime/information/'},
    {term:'AMS',kind:'数据简称 · 挪威住宅车库',definition:'本研究对挪威住宅车库小时充电电量任务的简称，不是预测模型。数据来自公寓建筑电动汽车充电数据集。',use:'使用单车库计量序列，检验方法在另一来源的小时电量任务中的表现，误差单位为kWh。',source:'Sorensen，2021 · Residential electric vehicle charging datasets from apartment buildings，V1',url:'https://doi.org/10.17632/jbks2rcwyj.1'},
    {term:'LCL',kind:'数据来源 · Low Carbon London',definition:'伦敦Low Carbon London项目的居民电动汽车充电试验数据。本页使用充电记录，不是家庭总用电记录。',use:'固定48个对象，评价2013年11月至2014年1月的预测误差。目标为来源整点平均功率记录；物理单位换算待确认，保留原始单位。',source:'本页数据说明 · Low Carbon London居民充电试验',url:'#task'},
    {term:'LightGBM',kind:'借鉴模型 · 决策树预测器',definition:'通过逐轮增加决策树来修正预测误差。输入是一行特征，例如历史负荷、星期和统计量。',use:'作为普通树基线和增加历史统计的树模型；其预测也用于与神经网络做等权组合。',source:'Ke等，2017 · LightGBM: A Highly Efficient Gradient Boosting Decision Tree；链接为官方项目及论文入口',url:'https://github.com/microsoft/LightGBM'},
    {term:'MAE',kind:'评价指标 · 平均绝对误差',definition:'逐点计算预测值与真实值之差的绝对值，再取平均。数值越小，预测的平均偏差幅度越小。',use:'例如预测8 kW、实际10 kW，该点绝对误差为2 kW。不同数据的单位分别保留，不能直接比较其MAE大小。',source:'本页评价指标与计算示例',url:'#evidence'},
    {term:'历史摘要',kind:'本文输入 · 历史统计特征',definition:'从过去同类时段提取均值、分位数、零值比例及可用记录数量等数值，压缩表达历史负荷分布；不是文字摘要。',use:'每个未来小时从7／28／84天三个参照池分别计算统计，并加入可用性标记，共54维，与近期序列表示融合。',source:'本页历史取数与摘要计算示例',url:'#history'},
    {term:'特征拼接',kind:'融合操作 · 并排输入',definition:'把两组数值特征连接成一个更长的向量，再交给网络学习它们与预测目标的关系，不是把数值相加。',use:'将64维近期表示与54维历史摘要拼成118维输入，映射后用于预测未来负荷。',source:'本页输入特征拼接与预测流程',url:'#model'},
    {term:'消融实验',kind:'实验方法 · 去掉组件后比较',definition:'保持主要训练设置一致，去掉一个输入或组件，重新训练并比较误差，用于判断该部分带来的增量。',use:'本文从训练开始移除历史摘要数值及可用性标记，保留近期输入、样本和训练预算，比较有摘要与无摘要网络。',source:'本页历史摘要的匹配消融实验',url:'#evidence'}
  ],
  windowExample: {
    origin: '2022-03-29T00:00:00.000Z', targetHour: 18,
    pools: [
      {days:7, ages:[1,2,3,4,5,6,7], rule:'取过去7个起报日对应的18点', capacity:7},
      {days:28, ages:[7,14,21,28], rule:'28天内，只取同星期的18点', capacity:4},
      {days:84, ages:[7,14,21,28,35,42,49,56,63,70,77,84], rule:'84天内，只取同星期的18点', capacity:12}
    ],
    records:[{age:7,value:0},{age:14,value:6},{age:21,value:null},{age:28,value:12}]
  },
  scopes: [
    { id: 'op_Jan_Mar', short: 'OP · 一至三月', label: 'Optimise Prime · 2022年1—3月', unit: 'kW', points: 18469,
      plain: 4.988239, tree: 4.776484, neural: 4.660014, without: 4.938161, average: 4.528269,
      withoutAverage: 4.629994, conditional: 4.535370, twoTrees: 4.600276, supported: true,
      note: '合并MAE下降；三月单月加入摘要后增加0.69%。' },
    { id: 'op_July', short: 'OP · 七月', label: 'Optimise Prime · 2022年7月1—25日', unit: 'kW', points: 4775,
      plain: 3.552177, tree: 3.480602, neural: 3.555579, without: 3.597243, average: 3.443720,
      withoutAverage: 3.456581, conditional: 3.446100, twoTrees: 3.505494, supported: false,
      note: '摘要带来1.16%的点估计下降，四种差值区间均跨零。' },
    { id: 'ams', short: 'AMS · 三个月', label: '挪威住宅车库 · 2019年11月—2020年1月', unit: 'kWh', points: 2136,
      plain: 3.163978, tree: 3.161769, neural: 3.183752, without: 3.239926, average: 3.117280,
      withoutAverage: 3.142198, conditional: 3.148614, twoTrees: null, supported: true,
      note: '合并MAE下降；十二月单月加入摘要后增加1.45%。' },
    { id: 'lcl', short: 'LCL · 三个月', label: '伦敦居民充电 · 2013年11月—2014年1月', unit: '来源原始单位', points: 88889,
      plain: 0.158841, tree: 0.160477, neural: 0.152106, without: 0.154418, average: 0.155519,
      withoutAverage: 0.156824, conditional: null, twoTrees: 0.158924, supported: true,
      distribution: 0.155410, note: '三个单月的神经MAE均下降；神经单分支低于外部平均。' }
  ],
  lclGroups: [
    {label: '零负荷', count: 71459, neural: 0.005197, average: 0.010082, contribution: 0.003927},
    {label: '普通正负荷', count: 9479, neural: 1.038300, average: 1.027705, contribution: -0.001130},
    {label: '高负荷', count: 7951, neural: 0.415943, average: 0.422828, contribution: 0.000616}
  ]
};
