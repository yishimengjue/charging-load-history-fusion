(() => {
  'use strict';
  const input=document.querySelector('#method-search');
  if(!input) return;
  const select=document.querySelector('#status-filter');
  const groupFilter=document.querySelector('#group-filter');
  const groupSections=[...document.querySelectorAll('.method-group')];
  const archiveState=new Map();
  let filtering=false;
  function filter(){
    const query=input.value.trim().toLocaleLowerCase();
    const status=select.value;
    const active=Boolean(query)||status!=='all';
    if(active&&!filtering)for(const section of groupSections){const detail=section.querySelector('details');if(detail)archiveState.set(detail,detail.open);}
    let overview=0,archived=0;
    for(const section of groupSections){
      let count=0;
      for(const row of section.querySelectorAll('tbody tr')){
        const groupMatches=!groupFilter||groupFilter.value==='all'||section.id===groupFilter.value;
        const visible=groupMatches&&(status==='all'||row.dataset.status===status)&&row.dataset.search.toLocaleLowerCase().includes(query);
        row.hidden=!visible;
        if(visible){count++;if(section.contains(row.closest('details')))archived++;else overview++;}
      }
      for(const table of section.querySelectorAll('.method-table-scroll'))table.hidden=![...table.querySelectorAll('tbody tr')].some(row=>!row.hidden);
      const detail=section.querySelector('details');
      if(detail){detail.hidden=!detail.querySelector('tbody tr:not([hidden])');if(active)detail.open=!detail.hidden;else if(filtering)detail.open=archiveState.get(detail)||false;}
      section.hidden=count===0;
    }
    filtering=active;
    document.querySelector('#method-empty').hidden=overview+archived>0;
    document.querySelector('#filter-count').textContent=`总览 ${overview} 条 · 早期细项 ${archived} 条${active?'（已展开匹配细项）':''}`;
  }
  input.addEventListener('input',filter);
  select.addEventListener('change',filter);
  groupFilter?.addEventListener('change',filter);
  filter();
})();
