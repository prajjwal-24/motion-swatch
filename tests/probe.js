const puppeteer=require('puppeteer-core');
(async()=>{const b=await puppeteer.launch({executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args:['--no-sandbox']});
const p=await b.newPage(); p.on('pageerror',e=>console.log('ERR',String(e)));
await p.goto('http://localhost:8000/index.html',{waitUntil:'networkidle0'});
const t=require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg','utf8');
console.log(await p.evaluate(async(t)=>{
 const {loadUploadedSVG,library}=window.__ms;
 await loadUploadedSVG(t,'Scene3.svg'); await new Promise(r=>setTimeout(r,600));
 const svgs=[...document.querySelectorAll('svg')].map(s=>({id:s.id,cls:s.className.baseVal,parent:s.parentElement&&s.parentElement.id,paths:s.querySelectorAll('path').length}));
 const ids=[...document.querySelectorAll('[id]')].map(e=>e.id).filter(i=>/scarf/i.test(i));
 return {svgs, scarfIds:ids, lib:library.getAll().map(m=>m.id).slice(-4)};
},t));
await b.close();})();
