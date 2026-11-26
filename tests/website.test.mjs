import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const pages=['index.html','method.html','results.html','exploration.html','methods.html','resources.html'];
const pageHtml=pages.map(file=>readFileSync(resolve(root,file),'utf8'));
const html=pageHtml.map(page=>page.split('<main id="main">')[1].split('<nav class="page-turns"')[0]).join('\n');
test('separated pages preserve unique research anchors and bundled scripts',()=>{
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
  assert.equal(ids.length,new Set(ids).size);
  for(const page of pageHtml)assert(page.indexOf('<script>window.PAPER_DATA')>page.indexOf('</main>'));
});
test('timeline example explains which records are available before prediction',()=>{
  const example=html.split('id="worked-prediction"')[1].split('</section>')[0];
  for(const text of ['2022年3月22日','3月15日至21日','3月8日','7＋11 = 18','不是实测预测曲线'])assert(example.includes(text),text);
  const origin=Date.parse('2022-03-22T00:00:00Z');
  assert.equal(new Date(origin).getUTCDay(),2);
  for(const age of [1,2,3,4,5,6,7,14,21,28,35,42,49,56,63,70,77,84]){
    const candidate=origin+18*3600000-age*86400000;
    assert(candidate<=origin);
    if(age>=14)assert.equal(new Date(candidate).getUTCDay(),2);
  }
});
test('matched summary results reproduce the headline percentage gains',()=>{
  const context={window:{}};vm.runInNewContext(readFileSync(resolve(root,'src/data.js'),'utf8'),context);
  const data=context.window.PAPER_DATA;
  assert.deepEqual(Array.from(data.scopes,s=>((s.without-s.neural)/s.without*100).toFixed(2)),['5.63','1.16','1.73','1.50']);
  assert.equal(data.scopes.find(s=>s.id==='op_July').supported,false);
  const lcl=data.scopes.find(s=>s.id==='lcl');assert(lcl.neural<lcl.average);
});
test('method inventory and release data scope remain explicit',()=>{
  const publication=JSON.parse(readFileSync(resolve(root,'publication.json'),'utf8'));
  assert.equal(publication.sources.length,56);
  assert.equal(publication.overviewRows,74);
  assert.equal(publication.archivedComparisons,117);
  assert.equal(publication.rawDataIncluded,true);
  const manifest=JSON.parse(readFileSync(resolve(root,'reproduction-manifest.json'),'utf8'));
  assert.equal(manifest.packages.length,3);
  assert(manifest.files.length>=813);
  assert(html.includes('id="method-search"'));
  assert(html.includes('id="status-filter"'));
});
test('latest experiments keep scope, comparator and decision beside their results',()=>{
  assert(html.includes('id="latest-research"'));
  assert(html.includes('2026.11.26'));
  for(const text of ['同候选中位数','同信息共享树','充电状态探针','SparseTSF','当前版本停止','尚未运行'])assert(html.includes(text),text);
  assert(html.includes('0.241377'));
  assert(readFileSync(resolve(root,'src/reading.js'),'utf8').includes('revealDestination'));
});
test('method filtering distinguishes nested archive rows from the outer reading panel',()=>{
  const outerPanel={};
  const innerArchive={open:false,querySelector:()=>({})};
  const input={value:'',addEventListener:(_,fn)=>{input.change=fn;}};
  const select={value:'all',addEventListener:(_,fn)=>{select.change=fn;}};
  const counter={},empty={};
  const rows=[
    {dataset:{status:'adopted',search:'TimeXer'},closest:()=>outerPanel},
    {dataset:{status:'stopped',search:'old method'},closest:()=>innerArchive}
  ];
  const section={contains:node=>node===innerArchive,querySelector:()=>innerArchive,
    querySelectorAll:selector=>selector==='tbody tr'?rows:[]};
  const nodes={'#method-search':input,'#status-filter':select,'#filter-count':counter,'#method-empty':empty};
  const document={querySelector:selector=>nodes[selector],querySelectorAll:()=>[section]};
  vm.runInNewContext(readFileSync(resolve(root,'src/methods.js'),'utf8'),{document});
  assert.equal(counter.textContent,'总览 1 条 · 早期细项 1 条');
  input.value='TimeXer';input.change();
  assert.equal(counter.textContent,'总览 1 条 · 早期细项 0 条（已展开匹配细项）');
  input.value='';input.change();
  assert.equal(counter.textContent,'总览 1 条 · 早期细项 1 条');
  select.value='adopted';select.change();
  assert.equal(counter.textContent,'总览 1 条 · 早期细项 0 条（已展开匹配细项）');
});
