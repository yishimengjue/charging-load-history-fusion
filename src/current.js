(() => {
  const controls=document.querySelector('.current-scope-controls');
  if(!controls)return;
  const buttons=[...controls.querySelectorAll('[data-current-scope]')];
  const panels=[...document.querySelectorAll('[data-current-panel]')];
  function select(id){
    for(const button of buttons)button.setAttribute('aria-pressed',String(button.dataset.currentScope===id));
    for(const panel of panels)panel.hidden=panel.dataset.currentPanel!==id;
  }
  for(const button of buttons)button.addEventListener('click',()=>select(button.dataset.currentScope));
  controls.hidden=false;
  select(buttons[0].dataset.currentScope);
})();
