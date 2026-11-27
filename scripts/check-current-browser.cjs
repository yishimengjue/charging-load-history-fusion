const assert=require('node:assert/strict');
const {chromium}=require('playwright');
const fs=require('node:fs');
(async()=>{
  const browser=await chromium.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true});
  const errors=[],report=[];
  const base=process.env.SITE_URL||'http://127.0.0.1:55986/';
  fs.mkdirSync('/private/tmp/current-site-qa',{recursive:true});
  for(const width of [1440,390]){
    const page=await browser.newPage({viewport:{width,height:1000}});
    page.on('pageerror',e=>errors.push(e.message));
    for(const file of ['index.html','abstract.html','method.html','results.html','exploration.html','methods.html','resources.html','literature.html']){
      await page.goto(base+file);await page.waitForLoadState('networkidle');
      assert(await page.locator('h1,h2').count()>0,file);
      const size=await page.evaluate(()=>({scroll:document.documentElement.scrollWidth,client:document.documentElement.clientWidth}));
      assert(size.scroll<=size.client+1,`${file} at ${width} overflows: ${JSON.stringify(size)}`);
      if(file==='abstract.html'){
        assert.equal(await page.locator('#abstract>article>p').count(),1);
        assert(await page.locator('.topbar a[aria-current="page"]').evaluate(el=>el.getAttribute('href')==='abstract.html'));
        assert.deepEqual(await page.locator('.page-turns a').evaluateAll(links=>links.map(a=>a.getAttribute('href'))),['index.html','method.html']);
      }
      // Hidden details contain lazy images; request decoding before checking their assets.
      for(const img of await page.locator('img').all())assert(await img.evaluate(async el=>{el.loading='eager';await el.decode();return el.complete&&el.naturalWidth>0;}),`${file} image`);
      for(const svg of await page.locator('[data-paper-figure]').all()){
        assert(await svg.locator('text').count()>0,'figure keeps real text');
        assert(await svg.locator('text').first().evaluate(el=>getComputedStyle(el).fontFamily.includes('Heiti SC')),'Simplified Chinese font');
        assert.equal(await svg.locator('button').count(),0,'glossary must not alter SVG text');
      }
      if(file==='method.html'){
        for(const id of ['task','history','model','workflow','worked-prediction']){
          await page.locator(`[data-section-switcher] a[href="#${id}"]`).click();
          assert(await page.locator('#'+id).isVisible(),id);
          assert.equal(await page.locator('main>section:visible').count(),1);
          if(id==='history'){
            const text=page.locator('[data-paper-figure] text').filter({hasText:'功率'}).first();
            assert(await text.evaluate(el=>{const range=document.createRange();range.selectNodeContents(el);const s=window.getSelection();s.removeAllRanges();s.addRange(range);const value=s.toString();s.removeAllRanges();return value.includes('功率');}),'Chinese figure text is selectable');
            await page.locator('.paper-explainer').scrollIntoViewIfNeeded();
            await page.screenshot({path:`/private/tmp/current-site-qa/${width}-history.png`,fullPage:true,animations:'disabled'});
          }
        }
        await page.locator('[data-section-switcher] a[href="#model"]').click();
        await page.locator('[data-term="TimeXer"]').first().click();
        assert(await page.locator('#term-card').isVisible());
        const box=await page.locator('#term-card').boundingBox();assert(box.x>=0&&box.x+box.width<=width+1);
        await page.keyboard.press('Escape');assert(!(await page.locator('#term-card').isVisible()));
        assert.equal(await page.evaluate(()=>document.activeElement.dataset.term),'TimeXer');
      }
      if(file==='results.html'){
        for(const section of ['evidence','input-effects','daily-curves','temporal-results','tradeoff']){
          await page.locator(`[data-section-switcher] a[href="#${section}"]`).click();
          assert(await page.locator('#'+section).isVisible(),section);
          assert.equal(await page.locator('main>section:visible').count(),1);
          await page.evaluate(()=>new Promise(resolve=>requestAnimationFrame(()=>{window.scrollTo({top:0,behavior:'instant'});requestAnimationFrame(resolve);})));
          await page.screenshot({path:`/private/tmp/current-site-qa/${width}-${section}.png`,fullPage:true,animations:'disabled'});
        }
        for(const id of ['op-winter','op-july','lcl','sofie']){
          await page.locator(`[data-current-scope="${id}"]:visible`).click();
          assert.equal(await page.locator('[data-current-panel]:visible').count(),1);
          assert.equal(await page.locator(`[data-current-panel="${id}"]:visible`).count(),1);
          assert.equal(await page.locator(`[data-current-scope="${id}"][aria-pressed="true"]`).count(),2);
        }
        await page.locator('[data-section-switcher] a[href="#evidence"]').click();
        assert.equal(await page.locator('[data-current-panel="sofie"]:visible').count(),1);
      }
      if(file==='methods.html'){
        await page.locator('#group-filter').selectOption('all');
        await page.locator('#method-search').fill('TimeXer');
        assert(!/总览 0 条 · 早期细项 0 条/.test(await page.locator('#filter-count').innerText()));
      }
      if(file==='literature.html'){
        await page.locator('a.citation[href="#ref-4"]').click();
        assert(await page.locator('#ref-4').evaluate(el=>el.open));
        await page.locator('#ref-4 summary').click();
        assert(!(await page.locator('#ref-4').evaluate(el=>el.open)));
      }
      await page.evaluate(()=>window.scrollTo({top:0,behavior:'instant'}));
      await page.screenshot({path:`/private/tmp/current-site-qa/${width}-${file}.png`,fullPage:true,animations:'disabled'});
      report.push({width,file,overflow:false});
    }
    await page.close();
  }
  const nojs=await browser.newPage({javaScriptEnabled:false,viewport:{width:390,height:900}});
  await nojs.goto(base+'results.html');assert.equal(await nojs.locator('[data-current-panel]:visible').count(),8);
  await nojs.close();await browser.close();
  assert.deepEqual(errors,[]);
  console.log(JSON.stringify({passed:true,pages:report,checks:'images, overflow, method tabs, glossary focus, scope controls, method search, no-JS fallback'},null,2));
})().catch(e=>{console.error(e);process.exit(1)});
