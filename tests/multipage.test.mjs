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
    'results.html':['evidence','tradeoff'],
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
