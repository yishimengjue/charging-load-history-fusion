(() => {
  'use strict';
  const site = window.PAPER_SITE;
  const methods = document.querySelector('#methods');
  if(methods){
  const heading = methods.querySelector('.methods-hero');
  heading.querySelector('h2').textContent = '完整方法与早期实验记录';
  heading.querySelector('.hero-lead').textContent = '从点预测、会话分解到历史融合，按研究方向查看方法、数据、效果与取舍。';
  heading.querySelector('.result-note').textContent = '此表为历次实验档案，绿色表示当时保留的组件。当前选择与新增验证见“近期尝试”页。';
  }

  const sectionIds = site?.sectionIds || [];
  let activeSection=sectionIds[0];
  const tabNavigation=document.querySelector('[data-section-switcher]');
  const header=document.querySelector('.topbar');
  if(header){
    const updateOffsets=()=>{
      document.documentElement.style.setProperty('--header-height',`${header.getBoundingClientRect().height}px`);
      if(tabNavigation)document.documentElement.style.setProperty('--tabs-height',`${tabNavigation.getBoundingClientRect().height}px`);
    };
    updateOffsets();
    if('ResizeObserver' in window){
      const observer=new ResizeObserver(updateOffsets);
      observer.observe(header);
      if(tabNavigation)observer.observe(tabNavigation);
    }
  }
  function selectSection(id){
    activeSection=id;
    for(const sectionId of sectionIds)document.getElementById(sectionId).hidden=sectionId!==id;
    for(const link of document.querySelectorAll('[data-section-switcher] a')){
      const selected=link.hash===`#${id}`;
      link.classList.toggle('active',selected);
      if(selected)link.setAttribute('aria-current','true');else link.removeAttribute('aria-current');
    }
  }

  function revealDestination(hash = location.hash) {
    let id;
    try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
    const destination=site?.anchors[id];
    if(destination && destination!==site.page){
      location.replace(destination+hash);
      return;
    }
    const target = document.getElementById(id);
    const section=target?.closest('main > section');
    if(sectionIds.length)selectSection(sectionIds.includes(section?.id)?section.id:activeSection);
    if (!target) return;
    const group=target.closest('.method-group');
    const groupFilter=document.querySelector('#group-filter');
    if(group && groupFilter){
      groupFilter.value=group.id;
      groupFilter.dispatchEvent(new Event('change'));
    }
    let opened = false;
    // Direct links to a section also open its first-level reading panel.
    const panel = ['methods', 'workflow'].includes(id)
      ? target.querySelector(':scope > details.reading-details') : null;
    if (panel && !panel.open) { panel.open = true; opened = true; }
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS' && !parent.open) { parent.open = true; opened = true; }
    }
    if (opened || sectionIds.length) requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  }
  window.addEventListener('hashchange', () => revealDestination());
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href^="#"]');
    if (link) revealDestination(link.hash);
  });
  revealDestination();
})();
