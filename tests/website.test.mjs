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
test('method inventory and data exclusion remain explicit',()=>{
  const publication=JSON.parse(readFileSync(resolve(root,'publication.json'),'utf8'));
  assert.equal(publication.sources.length,56);
  assert.equal(publication.overviewRows,74);
  assert.equal(publication.archivedComparisons,117);
  assert.equal(publication.rawDataIncluded,false);
  assert(html.includes('id="method-search"'));
  assert(html.includes('id="status-filter"'));
});
