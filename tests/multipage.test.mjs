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
  }
  assert(read('results.html').includes('aria-label="结果小节"'));
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
