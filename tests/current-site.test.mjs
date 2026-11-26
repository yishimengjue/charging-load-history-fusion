import {test} from 'node:test';
import assert from 'node:assert/strict';
import {existsSync, readFileSync} from 'node:fs';
const read=p=>readFileSync(new URL('../'+p,import.meta.url),'utf8');
test('current public evidence separates sources and balances error contributions',()=>{
  assert(existsSync(new URL('../research/current-evidence.json',import.meta.url)),'public aggregate evidence is generated');
  const d=JSON.parse(read('research/current-evidence.json'));
  assert.equal(d.date,'2026-11-26');
  assert.deepEqual(d.scopes.map(s=>((s.mae.recent-s.mae.aligned)/s.mae.recent*100).toFixed(2)),['7.55','2.00','1.03','-2.09']);
  for(const s of d.scopes)assert(Math.abs(Object.values(s.contribution).reduce((a,b)=>a+b,0)-(s.mae.recent-s.mae.aligned))<1e-10);
  assert.equal(d.scopes[2].unit,'source_raw');
  assert(d.oldSystems[0].average<d.scopes[0].mae.aligned);
  assert(!read('research/current-evidence.json').includes('/Users/'));
});
test('four figure groups are downloadable and have text alternatives',()=>{
  for(const file of ['method','route',...['op-winter','op-july','lcl','sofie'].flatMap(s=>['mae-'+s,'errors-'+s])]){
    const path='figures/'+file+'.svg';
    assert(existsSync(new URL('../'+path,import.meta.url)),path);
    const svg=read(path);assert(svg.includes('<title'));assert(svg.includes('<desc'));assert(!svg.includes('NaN'));
  }
});
test('current pages explain raw inputs and retain stronger prior comparisons',()=>{
  const method=read('method.html').split('<script>')[0];
  assert(method.includes('18条'));assert(method.includes('14、21'));assert(method.includes('不是统计摘要'));
  const results=read('results.html').split('<script>')[0];
  for(const value of ['4.528269','3.443720','3.525426','0.318600'])assert(results.includes(value),value);
  assert(results.includes('来源原始单位'));assert(results.includes('旧流程'));
  const explore=read('exploration.html').split('<script>')[0];
  assert(explore.includes('SparseTSF'));assert(explore.includes('尚未运行'));
});
test('superseded extension results remain in dated archive rather than disappear',()=>{
  const archive=read('methods.html').split('<script>')[0];
  for(const value of ['4.528610','4.647791','pooling-extension','state-extension','duration-extension','residual-extension'])assert(archive.includes(value),value);
  assert(archive.includes('2026年11月19日扩展实验归档'));
  assert(!archive.includes('当前神经结构'));
});
