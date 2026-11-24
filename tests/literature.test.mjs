import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import vm from 'node:vm';
const root=new URL('../',import.meta.url);
const read=name=>readFileSync(new URL(name,root),'utf8');
const page=()=>{assert(existsSync(new URL('literature.html',root)),'literature page must be built');return read('literature.html');};

test('literature onepage is reachable from all research pages',()=>{
  const html=page();
  assert(html.includes('<html lang="zh-CN">'));
  assert(html.includes('href="literature.html" class="active" aria-current="page"'));
  for(const name of ['index','method','results','exploration','methods','resources'])
    assert(read(name+'.html').includes('href="literature.html"'),name);
});
test('page exposes the three journal sections and references',()=>{
  const html=page();
  for(const id of ['flow','studies','position','references'])assert(html.includes(`id="${id}"`),id);
  assert.equal((html.match(/class="review-prose"/g)||[]).length,3);
  assert.equal((html.match(/<p class="review-paragraph">/g)||[]).length,9);
  for(const heading of ['预测对象与历史信息组织','历史选择与信息融合','预测组合与研究定位'])assert(html.includes(heading));
  assert(!html.includes('文本消融'));
  assert(!html.includes('GRU-D'));
  assert(html.includes('<strong>近期序列</strong>'));
  assert(html.includes('<strong>目标时段历史</strong>'));
});
test('twelve core citations retain evidence status',()=>{
  const html=page();
  assert.equal((html.match(/<details class="reference"/g)||[]).length,12);
  for(let i=1;i<=12;i++)assert(html.includes(`id="ref-${i}"`));
  assert.equal((html.match(/data-publication="preprint"/g)||[]).length,2);
  assert.equal((html.match(/data-publication="metadata-only"/g)||[]).length,0);
  assert(html.includes('占用率'));
  assert(html.includes('外部特征评估'));
  assert(html.includes('https://doi.org/10.1038/s41597-025-04874-4'));
});
test('all local links and citations resolve; private files are not published',()=>{
  const html=page().split('<script>')[0];
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
  assert.equal(ids.length,new Set(ids).size);
  for(const[,href]of html.matchAll(/href="([^"]+)"/g)){
    if(href.startsWith('https:'))continue;
    const[file,id]=href.split('#');
    const target=file||'literature.html';
    assert(existsSync(new URL(target,root)),href);
    if(id)assert(read(target).includes(`id="${id}"`),href);
  }
  assert(!/\/Users\/|file:\/\/|localhost|填补空白|首次提出|赋能|极大/.test(html));
});
test('references work without JavaScript and focused citation can expand details',()=>{
  const html=page();
  assert.equal((html.match(/<summary>/g)||[]).length,12);
  assert(html.includes('prefers-reduced-motion'));
  assert(html.includes(':focus-visible'));
  assert(html.includes('hashchange'));
  assert(!html.includes('<!--SITE_NAV-->'));
  assert(!html.includes('/*LITERATURE_STYLE*/'));
});
test('all old pages measure the taller navigation even without method tabs',()=>{
  const properties={};
  const header={getBoundingClientRect:()=>({height:181})};
  const document={
    querySelector:selector=>selector==='.topbar'?header:null,
    documentElement:{style:{setProperty:(key,value)=>{properties[key]=value;}}},
    getElementById:()=>null,addEventListener:()=>{}
  };
  vm.runInNewContext(read('src/reading.js'),{document,window:{PAPER_SITE:{anchors:{}},addEventListener:()=>{}},location:{hash:''}});
  assert.equal(properties['--header-height'],'181px');
  assert(read('src/theme.css').includes('scroll-padding-top:calc(var(--header-height'));
});
