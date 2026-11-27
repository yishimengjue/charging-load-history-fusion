import {readFileSync,writeFileSync} from 'node:fs';
import {currentSections} from './src/current-site.mjs';

const read=name=>readFileSync(new URL('src/'+name,import.meta.url),'utf8');
const source=read('index.html');
const main=source.match(/<main id="main">([\s\S]*?)<\/main>/)[1];
const sections=new Map();
let depth=0,start=0,id='';
// The authored body uses balanced section tags; retain nested method tables intact.
for(const match of main.matchAll(/<\/?section\b[^>]*>/g)){
  if(match[0].startsWith('</')){
    if(--depth===0)sections.set(id,main.slice(start,match.index+match[0].length));
  }else{
    if(depth++===0){start=match.index;id=match[0].match(/\bid="([^"]+)"/)?.[1];}
  }
}
if(depth!==0||sections.size!==15)throw new Error('Unexpected research section structure');
const archivedExtensions=sections.get('latest-research').replace('id="latest-research"','id="extensions-2026-11-19"').replaceAll('当前','当时');
const historicalMethods=sections.get('methods').replaceAll('当前','当时');
const archiveAt=historicalMethods.lastIndexOf('</section>');
sections.set('methods',historicalMethods.slice(0,archiveAt)+`<details class="reading-details"><summary>2026年11月19日扩展实验归档</summary><div class="detail-body">${archivedExtensions}</div></details>`+historicalMethods.slice(archiveAt));
// Keep the historical experiment inventory; render current explanations from the checked snapshot.
for(const [id,html] of currentSections())sections.set(id,html);
const pages=[
  {file:'index.html',label:'研究总览',ids:['top','contributions','route','conclusion']},
  {file:'abstract.html',label:'研究摘要',ids:['abstract']},
  {file:'method.html',label:'方法与示例',ids:['task','history','model','workflow','worked-prediction'],
    intro:'从输入到预测，按五个小节理解完整方法。',tabsLabel:'方法小节',tabs:['预测任务','历史记录','融合模型','训练验证','时间示例']},
  {file:'results.html',label:'实验结果',ids:['evidence','input-effects','daily-curves','temporal-results','tradeoff'],
    intro:'从总体误差到真实曲线，按小节查看改动与收益。',tabsLabel:'结果小节',tabs:['总体结果','输入消融','日负荷曲线','连续评价','误差来源']},
  {file:'methods.html',label:'完整方法表',ids:['methods'],intro:'按研究方向检索已经尝试的方法、数据与结果。'},
  {file:'exploration.html',label:'近期尝试',ids:['latest-research'],intro:'围绕历史输入的近期实验、结果和选择。'},
  {file:'resources.html',label:'资料与复现',ids:['downloads','reproduction'],intro:'论文文件、参考文献、数据包与复现实验入口。'}
];
const navigationPages=[...pages.slice(0,2),{file:'literature.html',label:'文献综述'},...pages.slice(2)];
const navigationFor=file=>navigationPages.map(p=>`<a href="${p.file}"${p.file===file?' class="active" aria-current="page"':''}>${p.label}</a>`).join('');
const pagerFor=file=>{
  const index=navigationPages.findIndex(p=>p.file===file);
  const previous=navigationPages[index-1],next=navigationPages[index+1];
  return `<nav class="page-turns" aria-label="页面顺序">${previous?`<a href="${previous.file}"><small>上一页</small><b>← ${previous.label}</b></a>`:'<span></span>'}${next?`<a href="${next.file}"><small>下一页</small><b>${next.label} →</b></a>`:'<a href="index.html"><small>返回</small><b>研究总览 →</b></a>'}</nav>`;
};
const anchors={};
for(const page of pages)for(const section of page.ids){
  if(!sections.has(section))throw new Error(`Missing section ${section}`);
  for(const match of sections.get(section).matchAll(/\bid="([^"]+)"/g))anchors[match[1]]=page.file;
}
const termCard=source.match(/<aside id="term-card"[\s\S]*?<\/aside>/)[0];
const scripts=['data.js','app.js','methods.js','reading.js','current.js'];
const script=code=>`<script>${code.replaceAll('</script','<\\/script')}</script>`;
const css=read('style.css')+'\n'+read('theme.css')+'\n'+read('current.css');
for(const page of pages){
  const index=navigationPages.findIndex(p=>p.file===page.file);
  const navigation=navigationFor(page.file);
  const intro=page.intro?`<header class="page-heading"><span class="section-no">充电负荷研究 / ${String(index+1).padStart(2,'0')}</span><h1>${page.label}</h1><p>${page.intro}</p></header>`:'';
  const tabs=page.tabs?`<nav class="local-tabs" aria-label="${page.tabsLabel}" data-section-switcher>${page.ids.map((id,i)=>`<a href="#${id}">${String(i+1).padStart(2,'0')} ${page.tabs[i]}</a>`).join('')}</nav>`:'';
  const pager=pagerFor(page.file);
  let body=page.ids.map(id=>sections.get(id)).join('\n');
  if(page.file==='methods.html'){
    const groups=[['direct','点预测与标签'],['sessions','会话分解'],['sharing','共享训练'],['sequence','序列模型'],['fusion','历史融合'],['weights','组合权重'],['outputs','输出与损失'],['validation','验证与消融'],['price','价格支线']];
    body=body.replace(/(<section class="method-controls"[^>]*>)/,`$1<label class="group-select">研究方向<select id="group-filter"><option value="all">全部方向</option>${groups.map(([id,label],i)=>`<option value="${id}"${i===0?' selected':''}>${label}</option>`).join('')}</select></label>`);
  }
  let html=`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="充电负荷历史融合研究：方法、结果与复现"><title>${page.label} · 充电负荷预测</title><style>${css}</style></head>
<body data-page="${page.file}"><a class="skip-link" href="#main">跳到正文</a>
<header class="topbar"><a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true">↗</span>充电负荷预测<span class="brand-sub"> / 研究图解</span></a><nav aria-label="阅读导航">${navigation}</nav></header>
<main id="main">${intro}${tabs}${body}${pager}<footer><span>目标时段历史与近期序列融合的充电负荷预测</span><a href="index.html">研究总览</a><small>页面更新 2026.11.27</small></footer></main>${termCard}
${script(`window.PAPER_SITE=${JSON.stringify({page:page.file,anchors,sectionIds:page.tabs?page.ids:[]})};`)}
${scripts.map(name=>script(read(name))).join('\n')}
</body></html>`;
  html=html.replace(/href="#([^"]+)"/g,(original,anchor)=>
    anchors[anchor]&&anchors[anchor]!==page.file?`href="${anchors[anchor]}#${anchor}"`:original);
  writeFileSync(new URL(page.file,import.meta.url),html);
}
writeFileSync(new URL('literature.html',import.meta.url),read('literature.html')
  .replace('<!--SITE_NAV-->',navigationFor('literature.html'))
  .replace('<!--PAGE_TURNS-->',pagerFor('literature.html'))
  .replace('/*LITERATURE_STYLE*/',read('literature.css')));
console.log(`Built ${pages.length} research pages and the literature onepage`);
