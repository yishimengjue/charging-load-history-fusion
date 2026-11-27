import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, existsSync} from 'node:fs';
const root=new URL('../',import.meta.url);
const files=['index.html','method.html','results.html','exploration.html','methods.html','resources.html'];
const read=file=>readFileSync(new URL(file,root),'utf8');
test('six independently addressable pages contain only their assigned research sections',()=>{
  const sections={
    'index.html':['top','contributions','route','conclusion'],
    'method.html':['task','history','model','workflow','worked-prediction'],
    'results.html':['evidence','input-effects','daily-curves','temporal-results','tradeoff'],
    'exploration.html':['latest-research'],
    'methods.html':['methods'],
    'resources.html':['downloads','reproduction']
  };
  for(const file of files){
    assert(existsSync(new URL(file,root)),`${file} must be built`);
    const html=read(file);
    for(const [owner,ids] of Object.entries(sections))for(const id of ids)
      assert.equal(html.includes(`id="${id}"`),owner===file,`${file}: ${id}`);
    assert(html.includes('aria-current="page"'));
    assert(html.includes('name="color-scheme" content="light"'));
  }
});
test('Chinese publication figures appear in their corresponding reading modules',()=>{
  const placements=[
    ['method.html','history','Fig01_recent_history_organization'],
    ['method.html','history','Fig05_history_before_after'],
    ['results.html','input-effects','Fig02_history_input_ablation'],
    ['results.html','daily-curves','Fig03_observed_and_predicted_daily_load'],
    ['results.html','temporal-results','Fig04_daily_and_horizon_errors']
  ];
  for(const [page,id,name] of placements){
    const section=read(page).split(`id="${id}"`)[1]?.split('</section>')[0];
    assert(section?.includes(`figures/paper/${name}.png`),`${page} / ${id}`);
    for(const extension of ['png','pdf','svg'])assert(existsSync(new URL(`figures/paper/${name}.${extension}`,root)));
    const svg=read(`figures/paper/${name}.svg`);
    assert(/[\u4e00-\u9fff]/.test(svg));
    assert(!/Recent sequence|Forecast horizon|Observed|MAE reduction/.test(svg));
    assert(svg.includes('Heiti SC') && !svg.includes('Heiti TC'), 'use Simplified Chinese glyphs');
    assert(section.includes(`data-paper-figure="${name}"`), 'render selectable inline SVG');
    assert(section.includes('<text'), 'keep figure labels as actual text');
    assert(!section.includes(`<img src="figures/paper/${name}.png"`), 'PNG is download only');
  }
  assert(read('results.html').includes('aria-label="结果小节"'));
});
test('figure explanations expose the selection example and all three comparisons',()=>{
  const history=read('method.html').split('id="history"')[1]?.split('</section>')[0];
  assert(history.includes('没有增加新的观测数据'));
  assert(history.includes('3月23日18点'));
  for(const value of ['使用前','使用后','9.92','4.60','5.32'])assert(history.includes(value),value);
  const input=read('results.html').split('id="input-effects"')[1]?.split('</section>')[0];
  const visible=input.split('<details')[0];
  for(const label of ['仅近期序列','近7天同小时记录','近远期18条记录','改动前MAE','改动后MAE','1.47%'])assert(visible.includes(label),label);
});
test('cross-page links resolve to actual files and section anchors',()=>{
  for(const file of files){
    assert(existsSync(new URL(file,root)),file);
    const html=read(file).split('<script>')[0];
    const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
    assert.equal(ids.length,new Set(ids).size,`${file} duplicate IDs`);
    for(const match of html.matchAll(/(?:href|src)="([^"]+)"/g)){
      const href=match[1];
      if(/^https:/.test(href))continue;
      const [path,anchor]=href.split('#');
      const target=path||file;
      assert(existsSync(new URL(decodeURIComponent(target),root)),`${file}: ${href}`);
      if(anchor)assert(read(target).includes(`id="${anchor}"`),`${file}: ${href}`);
    }
  }
});
test('method page exposes short independently switchable sections',()=>{
  assert(existsSync(new URL('method.html',root)));
  const html=read('method.html');
  assert(html.includes('data-section-switcher'));
  assert(html.includes('aria-label="方法小节"'));
});
