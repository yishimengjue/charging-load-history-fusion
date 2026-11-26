// Run only when refreshing the checked aggregate snapshot, not during site builds.
import {readFileSync, writeFileSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {basename, dirname} from 'node:path';
const read=p=>JSON.parse(readFileSync(new URL('../'+p,import.meta.url),'utf8'));
const evidence=read('review_response_2026-11-26/PAPER_EVIDENCE_CHECK.json');
const controls=read('reproduced-history-simple-controls-v1/results.json');
if(!evidence.passed||!controls.passed)throw Error('Evidence audit must pass');
const names=[
  ['op-winter','OP · 2022年1—3月','9条计量序列；小时边界前300秒内最后一条功率记录。'],
  ['op-july','OP · 2022年7月1—25日','同一批9条计量序列；另一季节的连续25天。'],
  ['lcl','LCL · 2014年2—4月','居民电动汽车充电试验；原48个对象，各月38／37／37个对象进入评分。'],
  ['sofie','SOFIE · 2026年4—6月','8台充电设备；整点累计电量差换算为小时平均交付功率。']
];
const result={
  date:'2026-11-26',metric:'MAE',aggregation:'Mean of three seed losses, weighted by observed targets across months; not prediction ensembling.',
  scopes:evidence.rows.map((r,i)=>({id:names[i][0],label:names[i][1],description:names[i][2],unit:r.unit,observed:r.observed,issued:r.issued,zero_pct:r.zero_pct,mae:r.mae,contrasts:r.contrasts,contribution:r.decomposition_gain.recent})),
  controls:Object.entries(controls.scopes).map(([scope,r])=>({scope,points:r.points,mae:r.groups.all.mae,pairs:r.pairs})),
  controlMonthly:Object.entries(controls.monthly).map(([month,r])=>({month,points:r.points,mae:r.groups.all.mae})),
  oldSystems:[{scope:'jan_mar',plain:4.988239,tree:4.776484,summary:4.660014,average:evidence.old_op_equal_average_mae.jan_mar},{scope:'july',plain:3.552177,tree:3.480602,summary:3.555579,average:evidence.old_op_equal_average_mae.july}],
  activity:evidence.activity,
  sources:Object.entries(evidence.source_sha256).map(([p,sha256])=>({id:basename(dirname(p))+'/'+basename(p),sha256})),
};
const controlPath=new URL('../reproduced-history-simple-controls-v1/results.json',import.meta.url);
result.sources.push({id:'reproduced-history-simple-controls-v1/results.json',sha256:createHash('sha256').update(readFileSync(controlPath)).digest('hex')});
writeFileSync(new URL('../research/current-evidence.json',import.meta.url),JSON.stringify(result,null,2)+'\n');
console.log('Exported aggregate scores only; no raw observations or local paths.');
