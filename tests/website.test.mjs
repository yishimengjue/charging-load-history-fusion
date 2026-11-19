import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync,existsSync} from 'node:fs';
import {resolve,dirname} from 'node:path';
import {fileURLToPath} from 'node:url';
import vm from 'node:vm';
const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const html=readFileSync(resolve(root,'index.html'),'utf8');
test('single page has unique anchors and bundled internal links',()=>{
  const ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
  assert.equal(ids.length,new Set(ids).size);
  assert.equal((html.match(/<main\b/g)||[]).length,1);
  for(const match of html.matchAll(/(?:href|src)="([^"]+)"/g)){
    const href=match[1];
    if(href.startsWith('#'))assert(ids.includes(href.slice(1)),href);
    else if(!/^https:/.test(href))assert(existsSync(resolve(root,decodeURIComponent(href))),href);
  }
  assert(html.indexOf('<script>window.PAPER_DATA')>html.indexOf('</main>'));
});
test('example arithmetic is consistent with its table',()=>{
  const example=html.split('id="worked-prediction"')[1].split('<section id="task"')[0];
  const rows=[...example.matchAll(/<tr><th>(18|19)点<\/th>((?:<td[^>]*>.*?<\/td>)+)<\/tr>/g)]
    .map(m=>[...m[2].matchAll(/<td[^>]*>([\d.]+)<\/td>/g)].map(c=>Number(c[1])));
  assert.equal(rows.length,2);
  for(const[n,s,f,y,en,ef]of rows){assert.equal(f,(n+s)/2);assert.equal(en,Math.abs(n-y));assert.equal(ef,Math.abs(f-y));}
  assert.equal(rows.reduce((a,r)=>a+r[4],0)/rows.length,2);
  assert.equal(rows.reduce((a,r)=>a+r[5],0)/rows.length,3);
  assert(example.includes('虚构讲解数据'));
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
  assert(html.includes('2026.11.19'));
  for(const id of ['pooling-extension','state-extension','duration-extension','residual-extension']){
    const row=html.match(new RegExp(`<tr id="${id}"[\\s\\S]*?</tr>`))?.[0];
    assert(row,id);
    assert(row.includes('OP'),id);
    assert(row.includes('不采用')||row.includes('扩展比较'),id);
  }
  assert(html.includes('4.464749 → 4.647791'));
  assert(html.includes('4.660014 → 4.528610'));
  assert(html.includes('展开流程图与训练细节'));
  assert(html.includes('revealDestination'));
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
