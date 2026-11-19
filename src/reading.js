(() => {
  'use strict';
  // Keep the archived ledger searchable without making it the default reading path.
  const methods = document.querySelector('#methods');
  const heading = methods.querySelector('.methods-hero');
  const archive = document.createElement('details');
  archive.className = 'reading-details';
  const summary = document.createElement('summary');
  summary.textContent = '展开完整方法表、搜索与早期实验';
  archive.append(summary);
  for (const node of [...methods.children]) if (node !== heading) archive.append(node);
  methods.append(archive);
  heading.querySelector('h2').textContent = '完整方法与早期实验记录';
  heading.querySelector('.hero-lead').textContent = '从点预测、会话分解到历史融合，按研究方向查看方法、数据、效果与取舍。';
  heading.querySelector('.result-note').textContent = '绿色表示保留在研究路线中的组件；新增扩展实验见上方“新增尝试”。';
  const counts = heading.querySelector('.ledger-counts');
  if (counts) archive.insertBefore(counts, summary.nextSibling);

  function revealDestination(hash = location.hash) {
    let id;
    try { id = decodeURIComponent(hash.slice(1)); } catch { return; }
    const target = document.getElementById(id);
    if (!target) return;
    let opened = false;
    // Direct links to a section also open its first-level reading panel.
    const panel = ['methods', 'workflow'].includes(id)
      ? target.querySelector(':scope > details.reading-details') : null;
    if (panel && !panel.open) { panel.open = true; opened = true; }
    for (let parent = target.parentElement; parent; parent = parent.parentElement) {
      if (parent.tagName === 'DETAILS' && !parent.open) { parent.open = true; opened = true; }
    }
    if (opened) requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  }
  window.addEventListener('hashchange', () => revealDestination());
  document.addEventListener('click', event => {
    const link = event.target.closest('a[href^="#"]');
    if (link) revealDestination(link.hash);
  });
  revealDestination();
})();
