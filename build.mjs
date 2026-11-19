import {readFileSync,writeFileSync} from 'node:fs';
const read=n=>readFileSync(new URL('src/'+n,import.meta.url),'utf8');
let html=read('index.html').replace('<link rel="stylesheet" href="style.css">',()=>'<style>'+read('style.css')+'</style>');
const names=['data.js','app.js','methods.js','reading.js'];
for(const name of names)html=html.replace('<script src="'+name+'" defer></script>','');
html=html.replace('</body>',()=>names.map(name=>'<script>'+read(name).replaceAll('</script','<\\/script')+'</script>').join('\n')+'\n</body>');
writeFileSync(new URL('index.html',import.meta.url),html);
console.log('Built self-contained single page');
