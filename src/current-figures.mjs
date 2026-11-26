import {mkdirSync,writeFileSync} from 'node:fs';
const C={ink:'#292832',muted:'#706b79',line:'#e5e1e8',violet:'#6c4ed9',green:'#26705c',red:'#aa5368',gray:'#a5a1af'};
const esc=s=>String(s).replaceAll('&','&amp;').replaceAll('<','&lt;');
const text=(x,y,s,size=18,color=C.ink,weight=400,anchor='start')=>`<text x="${x}" y="${y}" font-size="${size}" fill="${color}" font-weight="${weight}" text-anchor="${anchor}">${esc(s)}</text>`;
const rect=(x,y,w,h,fill)=>`<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="5" fill="${fill}"/>`;
function frame(title,desc,height,body){return `<svg xmlns="http://www.w3.org/2000/svg" width="440" height="${height}" viewBox="0 0 440 ${height}" role="img" aria-labelledby="title desc"><title id="title">${esc(title)}</title><desc id="desc">${esc(desc)}</desc><style>text{font-family:"PingFang SC","Microsoft YaHei",sans-serif}</style>${rect(0,0,440,height,'#fff')}${body}</svg>`;}
function flow(title,subtitle,steps){
  let body=text(22,34,title,22,C.ink,700)+text(22,64,subtitle,15,C.muted);
  steps.forEach(([heading,line1,line2],i)=>{
    const y=90+i*125;
    body+=rect(22,y,396,98,i===steps.length-1?'#edf7f1':'#f3f0fc')+text(38,y+28,`${i+1}. ${heading}`,20,C.violet,700)+text(38,y+56,line1,17)+text(38,y+80,line2,16,C.muted);
    if(i<steps.length-1)body+=text(220,y+119,'↓',22,C.muted,400,'middle');
  });
  return frame(title,steps.map(s=>s.join('：')).join('。'),90+steps.length*125,body);
}
function mae(s){
  const labels={recent:'仅近期168小时',near:'近期＋近7天同小时',aligned:'近期＋近远期同小时',summary:'近期＋原统计摘要',zero:'全部预测为零'};
  const entries=Object.entries(s.mae),max=Math.max(...entries.map(([,v])=>v))*1.05;
  let body=text(22,32,s.label,19,C.ink,700)+text(22,60,`MAE越低越好 · ${s.unit==='source_raw'?'来源原始单位':s.unit}`,16,C.muted);
  entries.forEach(([key,value],i)=>{
    const y=96+i*72;
    body+=text(22,y,labels[key],17)+text(418,y,value.toFixed(6),17,C.ink,700,'end')+rect(22,y+12,396,18,'#f1eef5')+rect(22,y+12,value/max*396,18,key==='aligned'?C.violet:key==='near'?C.green:C.gray);
  });
  const y=96+entries.length*72;
  body+=text(22,y,'0',15,C.muted)+text(418,y,`${max.toFixed(3)}`,15,C.muted,400,'end');
  return frame(`${s.label}的MAE比较`,entries.map(([k,v])=>`${labels[k]}：${v.toFixed(6)}`).join('；'),y+20,body);
}
function errors(s){
  const entries=[['零负荷',s.contribution.zero],['普通正负荷',s.contribution.ordinary],['高负荷',s.contribution.high],['净变化',s.mae.recent-s.mae.aligned]];
  const max=Math.max(...entries.map(([,v])=>Math.abs(v)))*1.1;
  let body=text(22,32,s.label,19,C.ink,700)+text(22,60,'← 增加误差            减少误差 →',17,C.muted);
  entries.forEach(([label,v],i)=>{
    const y=96+i*74,w=Math.abs(v)/max*190;
    body+=text(22,y,label,18)+text(418,y,`${v>0?'+':''}${v.toFixed(6)}`,18,v>=0?C.green:C.red,700,'end')+rect(22,y+12,396,20,'#f5f3f7')+rect(v<0?220-w:220,y+12,w,20,v>=0?C.green:C.red)+`<path d="M220 ${y+8}v28" stroke="${C.muted}"/>`;
  });
  body+=text(22,412,`单位：${s.unit==='source_raw'?'来源原始单位':s.unit} · 正值表示总体MAE下降`,15,C.muted);
  return frame(`${s.label}的误差贡献`,entries.map(([l,v])=>`${l}：${v.toFixed(6)}`).join('；'),438,body);
}
export function buildFigures(data){
  const dir=new URL('../figures/',import.meta.url);mkdirSync(dir,{recursive:true});
  const save=(name,svg)=>writeFileSync(new URL(name+'.svg',dir),svg);
  save('method',flow('从两种历史表示到24步预测','同一计量序列 · 所有输入均在起报前取得',[
    ['连续历史保留变化顺序','过去168小时负荷＋观测标记','TimeXer编码 → 每个目标步的64维表示'],
    ['同小时记录提供对应参照','前1—7天＋前2—12周，共18个槽位','每槽：负荷值、启用标记、观测标记'],
    ['拼接两路信息','64维近期表示＋54维历史向量','118维 → 64维融合表示'],
    ['按计量序列适配输出','近期基础预测＋历史融合修正','恢复量纲，负值在推理时归零'],
    ['预测未来24个小时位置','每个小时产生一个负荷预测值','月初训练，随后逐日起报与评分']
  ]));
  save('route',flow('研究路线与当前选择','箭头表示研究推进，不表示收益逐项叠加',[
    ['直接预测与会话分解','直接预测作为起点；尝试先分解再重建','分解误差会传递，未成为当前输出路径'],
    ['保留数据与历史统计','保留已观测标签；构建同类时段统计','成为后续模型的数据准备与参照'],
    ['序列网络与预测组合','TimeXer读取近期序列并融合历史摘要','旧树与神经组合：OP一至三月4.528269 kW'],
    ['逐条历史输入与归因','区分近7天重组、增加远期记录','当前近远期网络：OP一至三月4.565178 kW'],
    ['跨数据与朴素方法比较','LCL误差下降，SOFIE误差上升','保留差异；下一项周期模型尚未运行']
  ]));
  for(const s of data.scopes){save('mae-'+s.id,mae(s));save('errors-'+s.id,errors(s));}
}
