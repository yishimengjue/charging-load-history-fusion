import {readFileSync,writeFileSync} from 'node:fs';

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
const pages=[
  {file:'index.html',label:'研究总览',ids:['top','contributions','route','conclusion']},
  {file:'method.html',label:'方法与示例',ids:['task','history','model','workflow','worked-prediction'],
    intro:'从输入到预测，按五个小节理解完整方法。',tabs:['预测任务','历史摘要','融合模型','训练验证','计算示例']},
  {file:'results.html',label:'实验结果',ids:['evidence','tradeoff'],intro:'先比较平均误差，再解释不同负荷状态下的收益。'},
  {file:'exploration.html',label:'新增尝试',ids:['latest-research'],intro:'近期结构优化、状态输入与输出修正的结果和取舍。'},
  {file:'methods.html',label:'完整方法表',ids:['methods'],intro:'按研究方向检索已经尝试的方法、数据与结果。'},
  {file:'resources.html',label:'资料与复现',ids:['downloads','reproduction'],intro:'论文文件、参考文献、数据包与复现实验入口。'}
];
const anchors={};
for(const page of pages)for(const section of page.ids){
  if(!sections.has(section))throw new Error(`Missing section ${section}`);
  for(const match of sections.get(section).matchAll(/\bid="([^"]+)"/g))anchors[match[1]]=page.file;
}
const termCard=source.match(/<aside id="term-card"[\s\S]*?<\/aside>/)[0];
const scripts=['data.js','app.js','methods.js','reading.js'];
const script=code=>`<script>${code.replaceAll('</script','<\\/script')}</script>`;
const css=read('style.css')+'\n'+read('theme.css');
for(const [index,page] of pages.entries()){
  const navigation=pages.map(p=>`<a href="${p.file}"${p.file===page.file?' class="active" aria-current="page"':''}>${p.label}</a>`).join('');
  const intro=page.intro?`<header class="page-heading"><span class="section-no">CHARGING LOAD / ${String(index+1).padStart(2,'0')}</span><h1>${page.label}</h1><p>${page.intro}</p></header>`:'';
  const tabs=page.tabs?`<nav class="local-tabs" aria-label="方法小节" data-section-switcher>${page.ids.map((id,i)=>`<a href="#${id}">${String(i+1).padStart(2,'0')} ${page.tabs[i]}</a>`).join('')}</nav>`:'';
  const previous=pages[index-1],next=pages[index+1];
  const pager=`<nav class="page-turns" aria-label="页面顺序">${previous?`<a href="${previous.file}"><small>上一页</small><b>← ${previous.label}</b></a>`:'<span></span>'}${next?`<a href="${next.file}"><small>下一页</small><b>${next.label} →</b></a>`:'<a href="index.html"><small>返回</small><b>研究总览 →</b></a>'}</nav>`;
  let body=page.ids.map(id=>sections.get(id)).join('\n');
  if(page.file==='methods.html'){
    const groups=[['direct','点预测与标签'],['sessions','会话分解'],['sharing','共享训练'],['sequence','序列模型'],['fusion','历史融合'],['weights','组合权重'],['outputs','输出与损失'],['validation','验证与消融'],['price','价格支线']];
    body=body.replace(/(<section class="method-controls"[^>]*>)/,`$1<label class="group-select">研究方向<select id="group-filter"><option value="all">全部方向</option>${groups.map(([id,label],i)=>`<option value="${id}"${i===0?' selected':''}>${label}</option>`).join('')}</select></label>`);
  }
  let html=`<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light"><meta name="description" content="充电负荷历史融合研究：方法、结果与复现"><title>${page.label} · 充电负荷预测</title><style>${css}</style></head>
<body data-page="${page.file}"><a class="skip-link" href="#main">跳到正文</a>
<header class="topbar"><a class="brand" href="index.html"><span class="brand-mark" aria-hidden="true">↗</span>充电负荷预测<span class="brand-sub"> / 研究图解</span></a><nav aria-label="阅读导航">${navigation}</nav></header>
<main id="main">${intro}${tabs}${body}${pager}<footer><span>历史摘要与近期序列融合的充电负荷预测</span><a href="index.html">研究总览</a><small>研究更新 2026.11.19</small></footer></main>${termCard}
${script(`window.PAPER_SITE=${JSON.stringify({page:page.file,anchors,sectionIds:page.tabs?page.ids:[]})};`)}
${scripts.map(name=>script(read(name))).join('\n')}
</body></html>`;
  html=html.replace(/href="#([^"]+)"/g,(original,anchor)=>
    anchors[anchor]&&anchors[anchor]!==page.file?`href="${anchors[anchor]}#${anchor}"`:original);
  writeFileSync(new URL(page.file,import.meta.url),html);
}
console.log(`Built ${pages.length} linked research pages`);
