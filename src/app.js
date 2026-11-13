(() => {
  'use strict';
  const data = window.PAPER_DATA;
  const colors = { plain: '#687e9c', tree: '#f4ad69', without: '#687e9c', neural: '#79c5f2', average: '#80d7bd' };
  const fmt = n => n.toFixed(6);
  const gain = (base, candidate) => (base - candidate) / base * 100;
  const termCard = document.querySelector('#term-card');
  const closeTermButton = termCard.querySelector('.term-close');
  let activeTerm = null;
  const terms = new Map(data.glossary.map(entry => [entry.term, entry]));
  const termPattern = new RegExp(data.glossary.map(entry => /[A-Za-z]/.test(entry.term) ? `\\b${entry.term}\\b` : entry.term).join('|'), 'g');
  function termButton(term) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'term-trigger';
    button.dataset.term = term;
    button.textContent = term;
    button.setAttribute('aria-label', `${term}：查看简介`);
    button.setAttribute('aria-haspopup', 'dialog');
    button.setAttribute('aria-expanded', 'false');
    button.setAttribute('aria-controls', 'term-card');
    return button;
  }
  function explainTerms(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) {
      const node = walker.currentNode;
      if (!node.parentElement.closest('a,button,svg,script,style,h1,noscript') && node.textContent.match(termPattern)) nodes.push(node);
    }
    for (const node of nodes) {
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      for (const match of node.textContent.matchAll(termPattern)) {
        fragment.append(node.textContent.slice(cursor, match.index), termButton(match[0]));
        cursor = match.index + match[0].length;
      }
      fragment.append(node.textContent.slice(cursor));
      node.replaceWith(fragment);
    }
  }
  function positionTerm() {
    if (!activeTerm) return;
    const margin = 12;
    termCard.style.maxHeight = `${window.innerHeight - 2 * margin}px`;
    const anchor = activeTerm.getBoundingClientRect();
    const card = termCard.getBoundingClientRect();
    const left = Math.max(margin, Math.min(anchor.left, window.innerWidth - card.width - margin));
    let top = anchor.bottom + 10;
    if (top + card.height > window.innerHeight - margin) top = anchor.top - card.height - 10;
    termCard.style.left = `${left}px`;
    termCard.style.top = `${Math.max(margin, Math.min(top, window.innerHeight - card.height - margin))}px`;
  }
  function closeTerm(restoreFocus = false) {
    const previous = activeTerm;
    if (previous) previous.setAttribute('aria-expanded', 'false');
    activeTerm = null;
    termCard.hidden = true;
    if (restoreFocus && previous?.isConnected) previous.focus({preventScroll:true});
  }
  document.addEventListener('click', event => {
    const trigger = event.target.closest('[data-term]');
    if (trigger) {
      if (activeTerm === trigger) { closeTerm(true); return; }
      closeTerm();
      const entry = terms.get(trigger.dataset.term);
      if (!entry) return;
      activeTerm = trigger;
      for (const [id, value] of [['term-title',entry.term],['term-kind',entry.kind],['term-definition',entry.definition],['term-use',entry.use]]) document.getElementById(id).textContent = value;
      const sourceLink = document.querySelector('#term-source-link');
      sourceLink.textContent = entry.source;
      sourceLink.href = entry.url;
      if (entry.url.startsWith('https://')) { sourceLink.target = '_blank'; sourceLink.rel = 'noopener noreferrer'; }
      else { sourceLink.removeAttribute('target'); sourceLink.removeAttribute('rel'); }
      termCard.hidden = false;
      trigger.setAttribute('aria-expanded','true');
      positionTerm();
      closeTermButton.focus({preventScroll:true});
    } else if (!termCard.contains(event.target)) closeTerm();
  });
  closeTermButton.addEventListener('click', () => closeTerm(true));
  document.addEventListener('keydown', event => { if (event.key === 'Escape' && activeTerm) { event.preventDefault(); closeTerm(true); } });
  document.addEventListener('focusin', event => { if (activeTerm && event.target !== activeTerm && !termCard.contains(event.target)) closeTerm(); });
  document.querySelector('#term-source-link').addEventListener('click', () => closeTerm(true));
  window.addEventListener('resize', positionTerm);
  window.addEventListener('scroll', positionTerm, {passive:true});
  document.querySelector('#term-directory').append(...data.glossary.map(entry => termButton(entry.term)));
  document.querySelector('#contribution-results').innerHTML = data.scopes.map(scope => `<tr><td>${scope.short}</td><td>${fmt(scope.without)}</td><td>${fmt(scope.neural)}</td><td class="${scope.supported?'text-green':'text-orange'}"><b>${gain(scope.without,scope.neural).toFixed(2)}%</b></td><td>${scope.unit}</td></tr>`).join('');
  const example=data.windowExample;
  const exampleDate=age=>{
    const date=new Date(Date.parse(example.origin)-age*86400000);
    return `${date.getUTCMonth()+1}/${date.getUTCDate()}`;
  };
  document.querySelector('#calendar-pools').innerHTML=example.pools.map(pool=>`<article class="calendar-pool"><div class="pool-heading"><b>${pool.days}<small>天范围</small></b><span>最多${pool.capacity}个参照日</span></div><p>${pool.rule}</p><div class="date-chips">${pool.ages.map(age=>`<span class="${age===7?'overlap':''}">${exampleDate(age)}</span>`).join('')}</div><small>每个日期均取18点；筛选后可能少于${pool.capacity}条。</small></article>`).join('');
  document.querySelector('#sample-records').innerHTML=example.records.map(record=>`<div class="sample-record ${record.value===null?'missing':''}"><span>${exampleDate(record.age)} · 周二18点</span><strong>${record.value===null?'未知':`${record.value}<small> kW</small>`}</strong><small>${record.value===null?'不当作零，不进入均值':'观测已知，参与统计'}</small></div>`).join('');
  const known=example.records.filter(r=>r.value!==null).map(r=>r.value);
  const mean=known.reduce((sum,n)=>sum+n,0)/known.length;
  const zeroCount=known.filter(n=>n===0).length;
  const sorted=known.toSorted((a,b)=>a-b);
  const median=sorted[Math.ceil(known.length*.5)-1];
  document.querySelector('#sample-metrics').innerHTML=[
    ['历史均值',`${mean} kW`,'(0＋6＋12) ÷ 3'],
    ['中位数',`${median} kW`,'排序后的中间值'],
    ['零负荷比例',`${(zeroCount/known.length*100).toFixed(1)}%`,'1个零值 ÷ 3条已知记录'],
    ['已观测比例',`${known.length/example.records.length*100}%`,'3条已知 ÷ 4个候选日']
  ].map(([label,value,calculation])=>`<div><span>${label}</span><strong>${value}</strong><small>${calculation}</small></div>`).join('');
  function barGroup(scope, arms) {
    const max = Math.max(scope.plain,scope.tree,scope.without,scope.neural,scope.average) * 1.1;
    return `<div class="bars" role="img" aria-label="MAE对比，单位${scope.unit}，各图横轴从零开始。${arms.map(([key, label]) => `${label}${fmt(scope[key])}`).join('；')}">${arms.map(([key,label]) => `<div class="metric-row"><div class="metric-label"><span>${label}</span><b>${fmt(scope[key])}</b></div><div class="metric-track"><span class="metric-fill" style="--width:${scope[key]/max*100}%;--bar-color:${colors[key]}"></span></div></div>`).join('')}<div class="bar-axis"><span>0</span><span>${max.toFixed(scope.id === 'lcl' ? 3 : 2)} ${scope.unit}</span></div></div>`;
  }
  function card(scope, options) {
    return `<article class="comparison-card"><span class="micro-label">${options.kicker}</span><h3>${options.title}</h3><p class="card-caption">${options.caption}</p>${barGroup(scope,options.arms)}<div class="card-result"><strong class="${options.color}">${options.result}</strong><p>${options.description}</p><span class="outcome ${options.badgeClass || ''}">${options.badge}</span></div></article>`;
  }
  function renderScope(id) {
    closeTerm();
    const scope=data.scopes.find(s => s.id===id);
    if(!scope) return;
    for(const button of document.querySelectorAll('[data-scope]')) button.setAttribute('aria-pressed',String(button.dataset.scope===id));
    document.querySelector('#scope-heading').innerHTML=`<strong>${scope.label}</strong><span>${scope.points.toLocaleString('en-US')}个观测评分点 · MAE越低越好 · 各图从0起</span>`;
    const treeGain=gain(scope.plain,scope.tree),summaryGain=gain(scope.without,scope.neural);
    const averageWins=scope.average<Math.min(scope.neural,scope.tree);
    document.querySelector('#comparison-panels').innerHTML=[
      card(scope,{kicker:'对照 A / 树中的特征',title:'树模型的历史统计特征对照',caption:'同配置树模型，增加27项历史统计。',arms:[['plain','普通树'],['tree','增加历史统计的树']],color:treeGain>=0?'text-green':'text-red',result:`MAE${treeGain>=0?'下降':'增加'} ${Math.abs(treeGain).toFixed(2)}%`,description:scope.id==='lcl'?'本期直接加统计特征增加误差。摘要的作用还取决于学习方式。':scope.id==='ams'?'两种树的误差接近。单独增加统计特征，点估计变化小。':'对同一树模型，历史统计提供了额外预测信息。',badge:'相对普通树的点估计',badgeClass:treeGain<0?'negative':''}),
      card(scope,{kicker:'对照 B / 网络中的信息',title:'序列网络的历史摘要消融',caption:'相同训练预算，从训练开始使用或移除摘要。',arms:[['without','无历史摘要的网络'],['neural','有历史摘要的网络']],color:scope.supported?'text-green':'text-orange',result:`MAE下降 ${summaryGain.toFixed(2)}%`,description:scope.note,badge:scope.supported?'四种差值区间均支持下降':'四种差值区间均跨零',badgeClass:scope.supported?'':'caution'}),
      card(scope,{kicker:'对照 C / 网络外的组合',title:'树与神经网络的等权组合',caption:'神经与统计树的非负预测各占50%。',arms:[['neural','有摘要的神经预测'],['tree','历史统计树预测'],['average','两个预测各占50%']],color:averageWins?'text-green':'text-orange',result:averageWins?'平均MAE低于两个组成分支':`神经MAE比平均低 ${gain(scope.average,scope.neural).toFixed(2)}%`,description:averageWins?`平均预测相对普通树下降${gain(scope.plain,scope.average).toFixed(2)}%。这是完整流程收益，不是摘要单独的贡献。`:'取平均没有继续降低误差。神经单分支保留了更多零负荷位置的低误差。',badge:averageWins?'当前合并范围的点MAE排序':'本期不支持平均优于神经',badgeClass:averageWins?'':'caution'})
    ].join('');
  }
  for(const button of document.querySelectorAll('[data-scope]')) button.addEventListener('click',()=>{renderScope(button.dataset.scope);explainTerms(document.querySelector('#evidence'));});
  document.querySelector('#gain-summary').innerHTML=data.scopes.map(s=>`<article class="gain-tile ${s.supported?'':'caution'}"><span>${s.short}</span><strong>${gain(s.without,s.neural).toFixed(2)}%</strong><p>神经MAE下降${s.supported?' · 四种区间支持':' · 区间跨零'}</p></article>`).join('');
  document.querySelector('#architecture-rows').innerHTML=data.scopes.filter(s=>s.conditional!==null).map(s=>`<tr><td>${s.short} / ${s.unit}</td><td>${fmt(s.conditional)}</td><td>${fmt(s.average)}</td></tr>`).join('');
  const groupColors=['#f4ad69','#80d7bd','#79c5f2'];
  const total=data.lclGroups.reduce((sum,g)=>sum+g.count,0);
  document.querySelector('#mix-bar').innerHTML=data.lclGroups.map((g,i)=>`<span style="width:${g.count/total*100}%;background:${groupColors[i]}"></span>`).join('');
  document.querySelector('#mix-legend').innerHTML=data.lclGroups.map((g,i)=>`<div><i class="dot" style="background:${groupColors[i]}"></i><span>${g.label}</span><b>${(g.count/total*100).toFixed(2)}%</b></div>`).join('');
  const limit=.0045;
  document.querySelector('#contribution-chart').innerHTML=data.lclGroups.map(g=>`<div class="contribution-row"><div class="contribution-label"><span>${g.label}</span><b class="${g.contribution>0?'text-red':'text-green'}">${g.contribution>0?'+':''}${fmt(g.contribution)}</b></div><div class="diverging-track" role="img" aria-label="${g.label}使总体MAE${g.contribution>0?'增加':'减少'}${fmt(Math.abs(g.contribution))}"><span class="diverging-fill ${g.contribution<0?'negative':''}" style="--width:${Math.abs(g.contribution)/limit*50}%"></span></div></div>`).join('');
  renderScope(data.scopes[0].id);
  explainTerms(document.querySelector('main'));
  if('IntersectionObserver' in window){
    const sections=[...document.querySelectorAll('.chapter')];
    const observer=new IntersectionObserver(entries=>{
      const active=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top)[0];
      if(!active)return;
      const id=active.target.id==='tradeoff'?'evidence':active.target.id;
      for(const link of document.querySelectorAll('.topbar nav a')){
        const selected=link.getAttribute('href')===`#${id}`;
        link.classList.toggle('active',selected);
        if(selected)link.setAttribute('aria-current','location');else link.removeAttribute('aria-current');
      }
    },{rootMargin:'-15% 0px -60% 0px'});
    sections.forEach(s=>observer.observe(s));
  }
})();
