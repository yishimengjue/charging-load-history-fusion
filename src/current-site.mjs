import {readFileSync} from 'node:fs';
import {buildFigures} from './current-figures.mjs';
const d=JSON.parse(readFileSync(new URL('../research/current-evidence.json',import.meta.url),'utf8'));
const fmt=n=>n.toFixed(6);
const pct=(a,b)=>((a-b)/a*100).toFixed(2);
const unit=s=>s.unit==='source_raw'?'来源原始单位':s.unit;
const head=(n,title,desc='')=>`<header class="section-heading"><span class="section-no">${n}</span><h2>${title}</h2>${desc?`<p>${desc}</p>`:''}</header>`;
const section=(id,body)=>`<section id="${id}" class="chapter current-chapter">${body}</section>`;
const detail=(label,body)=>`<details class="reading-details"><summary>${label}</summary><div class="detail-body">${body}</div></details>`;
const table=(heads,rows,caption)=>`<div class="table-wrap"><table class="current-table"><caption>${caption}</caption><thead><tr>${heads.map(h=>`<th scope="col">${h}</th>`).join('')}</tr></thead><tbody>${rows.map(r=>`<tr>${r.map((v,i)=>i===0?`<th scope="row">${v}</th>`:`<td>${v}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
const figure=(name,alt,caption)=>`<figure class="current-figure"><img src="figures/${name}.svg" alt="${alt}" loading="lazy"><figcaption>${caption} <a href="figures/${name}.svg" download>下载 SVG</a></figcaption></figure>`;
const cards=items=>`<div class="current-cards">${items.map(([title,body])=>`<article><h3>${title}</h3><p>${body}</p></article>`).join('')}</div>`;
const action=(href,label)=>`<a class="current-link" href="${href}">${label} →</a>`;
const overviewRows=d.scopes.map(s=>[s.label,fmt(s.mae.recent),fmt(s.mae.aligned),`<b class="${s.mae.aligned<s.mae.recent?'text-green':'text-red'}">${s.mae.aligned<s.mae.recent?'下降':'增加'} ${Math.abs(Number(pct(s.mae.recent,s.mae.aligned))).toFixed(2)}%</b>`,unit(s)]);
const sample=(label,heading,body)=>`<div class="history-example"><span>${label}</span><h3>${heading}</h3><p>${body}</p></div>`;
export function currentSections(){
  buildFigures(d);
  const m=new Map();
  m.set('top',section('top',`<div class="current-hero"><div><span class="eyebrow">研究图解 / 2026.11.26</span><h1>把历史同小时记录<br>接入充电负荷预测</h1><p class="hero-lead">连续序列描述<b>最近一周如何变化</b>，同小时记录提供<b>过去对应时段的负荷参照</b>。两路信息在同一个神经网络中融合，预测未来24个小时位置。</p><div class="current-actions">${action('method.html','方法与示例')}${action('results.html','数据与结果')}</div></div><aside class="current-highlight"><span>OP · 2022年1—3月</span><strong>7.55<span>%</span></strong><b>相对仅近期输入，MAE下降</b><p>4.938161 → 4.565178 kW</p><small>同结构、同训练预算的三种子平均误差</small></aside></div>`));
  m.set('contributions',section('contributions',head('01 / 研究贡献','从历史记录的组织方式入手降低误差')+cards([
    ['方法：按预测时刻组织历史','每个未来小时接收<b>18个同小时历史槽位</b>，与近期序列表示拼接。历史值、是否启用、是否观测分别表达。'],
    ['证据：拆开两种信息作用','近7天组重组已有168小时信息；近远期组再加入第2—12周记录。用同结构对照区分<b>信息重组</b>与<b>更早历史</b>的作用。'],
    ['结果：定位误差来自哪里','按零负荷、普通正负荷和高负荷拆分收益。四个范围均出现<b>正负荷误差减少、零负荷误差增加</b>，净效果因数据而异。']
  ])+`<p class="current-note">方法贡献落在<b>目标时段历史的输入组织与验证</b>。TimeXer是借用的编码架构；预测组合另作完整方案比较。</p>`+table(['评价范围','仅近期 MAE','近远期 MAE','误差变化','单位'],overviewRows,'当前输入改动的结果；每行在同一数据范围内比较。')+`<p class="current-note">OP七月点估计下降2.00%，差值区间跨零；LCL仅增加近7天记录的误差更低。详见实验结果。</p>`));
  m.set('route',section('route',head('02 / 路线演进','从直接预测走向目标时段历史融合')+`<ol class="current-route"><li><span>01</span><div><b>直接预测与会话分解</b><p>建立点预测；尝试分解会话后重建负荷。</p></div></li><li><span>02</span><div><b>历史统计与序列模型</b><p>加入同类时段摘要，比较树、TimeXer及输出平均。</p></div></li><li class="selected"><span>03</span><div><b>逐条历史与目标时刻对齐</b><p>当前主线：保留历史值，区分近7天与更早记录。</p></div></li><li><span>04</span><div><b>跨数据验证与误差分解</b><p>用LCL、SOFIE和朴素预测检验输入收益。</p></div></li></ol>`+detail('展开完整路线图',figure('route','五阶段路线：直接预测、历史统计、序列组合、逐条历史、跨数据比较。','不同阶段不是可叠加的收益。'))+action('exploration.html','近期尝试与取舍')+action('methods.html','历年方法与结果表')));
  m.set('conclusion',section('conclusion',head('03 / 当前结论','保留历史融合，按数据选择输入与输出')+`<div class="current-summary"><p><b>OP：</b>近远期网络优于同候选中位数；已有树与摘要神经组合仍有更低MAE。</p><p><b>LCL：</b>近7天重组已带来收益，继续增加远期记录没有降低误差。</p><p><b>SOFIE：</b>零负荷占87.31%，空闲时段增加的误差超过充电时段收益。</p></div>`));
  m.set('task',section('task',head('01 / 预测任务','利用已知历史预测未来24个小时位置')+cards([
    ['输入','同一计量序列的过去168小时、同小时历史记录、观测标记与已知日历。'],
    ['输出','未来24个小时位置各一个负荷值。模型在<b>月初训练</b>，随后<b>逐日起报</b>。'],
    ['评价','用MAE衡量预测值与观测值的平均绝对差。越低越好；跨月按观测目标数汇总。']
  ])+table(['数据','计量对象与目标','评价时间','评分目标 / 已起报目标'],d.scopes.map(s=>[s.label.split(' · ')[0],`${s.description}<br><b>${unit(s)}</b>`,s.label.split(' · ')[1],`${s.observed.toLocaleString('en-US')} / ${s.issued.toLocaleString('en-US')}`]),'评价的是充电负荷；LCL不是家庭总用电。')+detail('时间戳与连续评价方式','<p>OP、LCL的近期序列结束于起报时刻t，目标标签为t＋1至t＋24小时。SOFIE的输入为[t−168小时,t)，第h个目标为[t＋h−1,t＋h)的平均交付功率。</p><p>每次起报固定预测24步，预测窗口内不读取真实负荷更新本次预测。训练只使用月初已经结束的目标窗口。未知目标不补零，也不参与评分。</p><p>当前评价覆盖所列连续月份，而非全年。SOFIE交付功率包含光伏、储能与控制条件下的实际运行结果。</p>')));
  m.set('history',section('history',head('02 / 历史输入','为每个未来小时提供同小时负荷参照','当前保留逐条历史记录，不是统计摘要。')+`<div class="current-duo">${sample('连续路径','过去168小时','按照时间顺序输入，保留启停、持续充电与相邻时段变化。')}${sample('参照路径','每个目标步18条候选','取前1—7天同小时，再取前14、21、…、84天同小时；后者与目标星期相同。')}</div>`+cards([
    ['仅近期','候选槽位全部关闭，仅使用连续168小时。'],
    ['近期＋近7天','启用前1—7天同小时值；这些值<b>已在168小时输入内</b>。比较信息重组的作用。'],
    ['近期＋近远期','再启用前2—12周同小时值，提供<b>168小时之外</b>的历史。']
  ])+table(['历史槽位状态','负荷输入','启用标记','观测标记'],[['未启用','0','0','0'],['启用但缺记录','0','1','0'],['有记录且负荷为零','0','1','1'],['有记录且负荷为正','缩放负荷','1','1']],'18槽 × 3项 = 54维；零负荷与缺失记录采用不同编码。')+detail('历史候选与原摘要的区别','<p>原摘要在7／28／84天历史池计算27项统计及27项可用性标记，用少量数值概括负荷分布。当前候选保留18条历史值及各自标记，交给预测器学习。</p><p>对齐实验与错开12小时实验采用共同观测规则：两个对应槽均有记录时才保留。后续近7天和近远期复验沿用该规则。原摘要采用原历史资格筛选，因此摘要与逐条历史比较的是两种输入流程。</p>')));
  m.set('model',section('model',head('03 / 融合方法','近期表示与历史记录进入同一个预测器')+`<div class="current-method-layout">${figure('method','近期168小时经TimeXer得到64维表示；与54维历史向量拼接，映射回64维，经计量序列输出头产生24步预测。','两种历史输入在网络内部融合。')}<div>`+cards([
    ['序列编码','TimeXer把168小时划为14个12小时片段，学习连续变化模式，再形成每个预测步的64维表示。'],
    ['特征拼接','将64维近期表示与54维历史值和标记连接为118维，再映射为64维。<b>不是历史功率直接相加。</b>'],
    ['输出适配','编码器共享训练，每条计量序列适配自己的输出头。<b>近期基础预测与历史融合修正相加</b>，再恢复量纲并归零负值。']
  ])+detail('训练与实现参数','<p>编码宽度64、2层、4个注意力头；融合层118→64，解码器64→32→1。查询包含可学习步长、已知日历、序列与尺度信息，并做层归一化。</p><p>实际候选与全空候选两种输入条件共用模型，损失权重95%与5%。共享训练80轮，输出头适配20轮；学习率0.0003／0.001，AdamW，批大小64，种子42／43／44。</p><p>各序列按训练标签P90与1的较大值缩放。训练优化恢复量纲的绝对误差，训练时不截断输出；推理时把负值归零。</p><p>融合层近期子矩阵单位初始化；步长表示标准差0.02；解码器末层零初始化。统计树不进入该神经模型的输入。</p>')+`</div></div>`));
  m.set('workflow',section('workflow',head('04 / 验证链路','分开检验输入收益、学习器作用与完整方案')+`<div class="current-lanes"><article><span>输入对照</span><b>同一个神经结构</b><p>仅近期 → 近7天 → 近远期</p><strong>分开信息重组与远期记录的作用</strong></article><article><span>学习器对照</span><b>相同历史候选</b><p>候选中位数 / 共享树 / 神经网络</p><strong>比较历史值如何转为预测</strong></article><article><span>完整方案</span><b>已有强基线</b><p>历史统计树 / 原摘要网络 / 等权组合</p><strong>保留实际误差更低的方案</strong></article><article><span>跨数据检验</span><b>按来源分别评价</b><p>OP / LCL / SOFIE</p><strong>再按负荷状态分解误差</strong></article></div>`+detail('可复验的比较规则','<p>同来源各神经输入组保持训练样本、尺度、骨干与预算一致。三种子先分别计算误差再平均，不先平均预测。</p><p>配对日期块采用7／14日与循环／非循环四种设置，每种2000次；区间基于已拟合预测。OP两段均已用于研究，LCL与SOFIE为固定候选后的连续时期评价。</p><p>新增共享树为350轮固定LightGBM，共3种输入×4个月=12次拟合。它与神经拥有相同可用信息与标签，但训练程序不同；已有强统计树另列比较。</p>')));
  m.set('worked-prediction',section('worked-prediction',head('05 / 时间示例','周二零点预测当天18点的负荷','以下日期说明取数方式，不是实测预测曲线。')+`<div class="current-timeline">${sample('起报','2022年3月22日 · 周二00:00','此刻冻结本次输入，预测未来24步。18点是其中一个目标。')}${sample('连续输入','3月15日01:00 → 3月22日00:00','读取168小时连续记录，保留时间顺序。')}${sample('近7天参照','3月15日至21日 · 每天18:00','得到7个槽位：重组近期输入中已有的记录。')}${sample('远期参照','3月8日、3月1日、2月22日…','依次向前取第2—12周周二18:00，共11个槽位。')}${sample('融合与输出','7＋11 = 18个历史槽位','历史值及标记与18点的近期表示拼接，输出该时刻的负荷预测。')}</div>`+`<p class="current-note"><b>为何按目标时刻取数：</b>预测18点时提供过去18点记录，让预测器直接看到对应时段的负荷，而不是只从连续曲线中自行寻找。</p>`));
  const scopeButtons=`<div class="current-scope-controls" hidden aria-label="选择评价数据">${d.scopes.map((s,i)=>`<button type="button" data-current-scope="${s.id}" aria-pressed="${i===0}">${s.label}</button>`).join('')}</div>`;
  const comparisonPanels=d.scopes.map(s=>`<article class="current-scope-panel" data-current-panel="${s.id}"><div class="current-result-layout">${figure('mae-'+s.id,`${s.label}；${Object.entries(s.mae).map(([k,v])=>k+' '+fmt(v)).join('；')}`,'各图从零开始；紫色为近远期输入，绿色为近7天输入。')}<div class="current-result-copy"><h3>${s.label}</h3><p>${s.description}</p><p><b>${s.observed.toLocaleString('en-US')}个评分目标</b> · 零负荷占${s.zero_pct.toFixed(2)}%</p><p class="current-result-number ${s.mae.aligned<s.mae.recent?'text-green':'text-red'}">MAE${s.mae.aligned<s.mae.recent?'下降':'增加'} ${Math.abs(Number(pct(s.mae.recent,s.mae.aligned))).toFixed(2)}%</p><p>近远期输入，相对仅近期输入</p><p>${{ 'op-winter':'近7天重组已带来6.17%的下降；再加入远期记录，误差从4.633271降至4.565178 kW。','op-july':'点估计下降，四种日期块差值区间均跨零。','lcl':'近7天组MAE为0.144633，低于近远期的0.144900；本期更早历史未继续降低误差。','sofie':'仅近期与近7天误差接近；加入远期后，零负荷位置的误差增加。'}[s.id]}</p></div></div>${detail('展开差值区间与完整分数',table(['输入','MAE'],Object.entries(s.mae).map(([k,v])=>[{recent:'仅近期',near:'近期＋近7天',aligned:'近期＋近远期',summary:'原统计摘要',zero:'全部预测为零'}[k],fmt(v)]),`单位：${unit(s)}`)+table(['日期块设置','MAE下降量的95%区间'],Object.entries(s.contrasts.find(c=>c.reference==='recent'&&c.candidate==='aligned').intervals).map(([k,v])=>[{noncircular_7:'非循环7日',noncircular_14:'非循环14日',circular_7:'循环7日',circular_14:'循环14日'}[k],v.map(fmt).join(' 至 ')]),'正值表示近远期输入降低误差。'))}</article>`).join('');
  m.set('evidence',section('evidence',head('01 / 预测结果','同一预测器，比较三种历史输入','先选择数据查看输入效果，再比较其他学习器与已有完整方案。')+scopeButtons+comparisonPanels+`<h3 class="current-subhead">同候选预测与已有方案</h3>`+table(['方法','OP一至三月 MAE','OP七月 MAE','比较角色'],[
    ['同候选中位数',...d.controls.map(c=>fmt(c.mae.median_aligned)),'不训练模型，历史值直接给出预测'],
    ['同信息共享树',...d.controls.map(c=>fmt(c.mae.tree_aligned)),'固定配置的学习器对照'],
    ['近远期神经模型',...d.scopes.slice(0,2).map(s=>`<b>${fmt(s.mae.aligned)}</b>`),'当前输入研究主线'],
    ['历史统计树 · 旧流程',...d.oldSystems.map(s=>fmt(s.tree)),'保留已有强基线'],
    ['树与原摘要神经各50% · 旧流程',...d.oldSystems.map(s=>`<b class="text-green">${fmt(s.average)}</b>`),'本表两范围内MAE最低']
  ],'同一OP评分范围，单位kW；旧流程使用原历史统计特征。')+`<p class="current-note">近远期神经相对同候选中位数，两个范围分别下降<b>5.54%</b>与<b>1.23%</b>。三月中位数3.804072低于神经3.865710 kW；七月额外收益的区间跨零。</p>`));
  m.set('tradeoff',section('tradeoff',head('02 / 误差来源','充电时段的收益与空闲时段的代价','与仅近期输入相比：右侧为减少误差，左侧为增加误差。')+d.scopes.map(s=>`<article data-current-panel="${s.id}" class="current-scope-panel"><div class="current-result-layout">${figure('errors-'+s.id,`${s.label}的加权误差贡献：零负荷${fmt(s.contribution.zero)}，普通正负荷${fmt(s.contribution.ordinary)}，高负荷${fmt(s.contribution.high)}。`,'分组贡献之和等于总体MAE下降量。')}<div class="current-result-copy"><h3>${s.label}</h3><p><b>零负荷：</b>历史输入使平均预测误差增加。</p><p><b>普通正负荷与高负荷：</b>历史输入使平均预测误差减少。</p><p class="current-note">${s.id==='sofie'?'本期87.31%目标为零负荷，空闲时段代价超过正负荷收益，总体MAE增加。':'本期正负荷收益超过零负荷代价，总体MAE下降。'}</p></div></div></article>`).join('')+detail('贡献计算方式','<p><b>组贡献 = 该组目标数占比 ×（仅近期组MAE − 近远期组MAE）。</b>零负荷、普通正负荷和高负荷互斥，三组贡献相加得到净变化。</p><p>高负荷采用每条序列当月训练正值的P90阈值，并包含与阈值相等的值。这是误差分配统计，不是负荷成因的因果估计。</p>')));
  m.set('latest-research',section('latest-research',head('01 / 近期尝试','围绕历史输入组织推进验证','绿色为当前主线；灰色为参照；红色为未进入主线的尝试。')+table(['研究环节','方法与结果','数据','当前选择'],[
    ['历史重组','近7天输入相对仅近期：MAE下降6.17%／0.35%／1.21%／0.0022%','OP一至三月／七月；LCL；SOFIE','<b class="text-green">保留，区分已有信息重组</b>'],
    ['更早历史','近远期输入相对仅近期：下降7.55%／2.00%／1.03%；增加2.09%','同上','<b class="text-green">当前输入研究主线</b>'],
    ['同候选中位数','MAE 4.833063／3.569309 kW','OP一至三月／七月','保留朴素对照'],
    ['同信息共享树','MAE 4.959536／3.751773 kW；12次拟合完成','OP一至三月／七月','保留学习器对照'],
    ['充电状态探针','状态树功率MAE 0.241377，零预测0.239608 kW；增加0.74%','SOFIE二至三月','<b class="text-red">当前版本停止</b>'],
    ['日周期架构 SparseTSF','尚未运行，无结果','拟在OP固定范围执行','待开展']
  ],'2026年11月26日汇总；各行百分比均按所列比较对象计算。')+`<p class="current-note">会话分解、共享训练、动态权重、输出修正、价格支线等早期尝试保留在完整方法表。它们不是当前方法的累计组件。</p>`+action('methods.html','按方向检索全部历史实验')));
  m.set('downloads',section('downloads',head('01 / 当前资料','图表与汇总数据')+cards([
    ['汇总结果',`${action('research/current-evidence.json','下载 JSON')}四个评价范围、简单对照、分组贡献、区间与来源哈希。`],
    ['方法与路线',`${action('figures/method.svg','方法图 SVG')}${action('figures/route.svg','路线图 SVG')}矢量格式，文字与数值可检查。`],
    ['研究与文献',`${action('research/CURRENT_RESULTS.md','当前研究说明')}${action('literature.html','文献综述')}当前网页解释逐条历史输入；文献正文保留综述版本。`]
  ])+table(['数据范围','预测比较图','误差贡献图'],d.scopes.map(s=>[s.label,`<a href="figures/mae-${s.id}.svg" download>下载 SVG</a>`,`<a href="figures/errors-${s.id}.svg" download>下载 SVG</a>`]),'图表由同一份公开汇总数据生成。')));
  m.set('reproduction',section('reproduction',head('02 / 复现入口','区分当前汇总与既有数据发布包')+`<p>本次更新发布<b>网页、图表与汇总结果</b>。既有 <b>repro-v0.1</b> 发布包对应原摘要融合流程，不等同于本次近远期输入的完整复现包。</p>`+`<div class="current-actions">${action('REPRODUCE.md','既有复现说明')}${action('DATA_LICENSES.md','数据许可')}${action('https://github.com/yishimengjue/charging-load-history-fusion/releases/tag/repro-v0.1','既有发布包')}</div>`+detail('原论文与实验档案','<p><a href="research/METHODS_AND_RESULTS_v0.5.md">原方法与结果文档 v0.5</a> · <a href="publication.json">历史档案索引</a> · <a href="reproduction-manifest.json">既有数据包清单</a>。</p><p>旧稿、旧表和当前图表各自保留日期与比较范围，避免跨版本混用结论。</p>')));
  return m;
}
