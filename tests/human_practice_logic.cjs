/* Software-only checks, using a DOM stub: no browser, network or human answers. */
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const path=require('node:path'),root=path.resolve(__dirname,'..');
const html=fs.readFileSync(path.join(root,'runs/human-practice-v1/QA.html'),'utf8');
const script=html.match(/<script>([\s\S]*)<\/script>/)[1];
class Element{
 constructor(id=''){this.id=id;this.value='';this.textContent='';this.checked=false;this.hidden=false;this.disabled=false;this.style={};this.listeners={};this.dataset={};this.clientWidth=1000;this.clientHeight=id==='spectrum'?220:165;this.width=0;this.height=0;this.currentTime=0;this.playbackRate=1;this.paused=true;this.files=[];this.classList={toggle(){}};}
 addEventListener(n,fn){this.listeners[n]=fn;} appendChild(){} click(){if(this.onclick)this.onclick();} pause(){this.paused=true;} play(){this.paused=false;return Promise.resolve();}
 getBoundingClientRect(){return{left:0,top:0,width:this.clientWidth,height:this.clientHeight};}
 getContext(){return new Proxy({},{get:(t,k)=>t[k]??(()=>{}),set:(t,k,v)=>(t[k]=v,true)});}
}
const elements=new Map();const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
for(const m of html.matchAll(/id="([^"\s]+)"/g))get(m[1]);
const modes=[];for(const s of ['setup','ending']){get(s+'_status');for(const f of ['low','best','high']){get(s+'_'+f);const b=new Element();b.dataset.mode=s+'_'+f;modes.push(b);}}
const sounds=['none','breath','other','speech','unclear'].map(v=>{const e=new Element();e.name='sound';e.value=v;return e;});
const doc={getElementById:get,createElement:()=>new Element(),querySelector:s=>s==='[name=sound][value=none]'?sounds[0]:null,
 querySelectorAll:s=>s==='[data-mode]'?modes:s==='[name=sound]'?sounds:s==='[name=sound]:checked'?sounds.filter(x=>x.checked):[...elements.values(),...sounds]};
const storage=new Map();let exported=null;const context=vm.createContext({document:doc,window:{devicePixelRatio:1,addEventListener(){}},
 localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
 Image:class{constructor(){this.complete=false;this.naturalWidth=0;}},Blob,URL:{createObjectURL:b=>(exported=b,'blob:software-qa'),revokeObjectURL(){}},
 setTimeout:fn=>fn(),confirm:()=>true,console,Date,Math,JSON,Number,String,Object,Array,Promise});
const run=s=>vm.runInContext(s,context);run(script);let checks=0;function check(name,fn){fn();checks++;console.log('PASS',name);}
check('starts with six anonymous trials, blank annotations and hidden plots',()=>{assert.equal(run('PACK.trials.length'),6);assert.equal(run('answer().setup.best'),null);assert.equal(get('annotation').hidden,true);assert.equal(get('qa').hidden,false);});
check('initial listening answers gate plot reveal',()=>{get('reveal').onclick();assert.equal(run('answer().revealed'),false);get('transcript').value='matches';get('perceived').value='gap';get('reveal').onclick();assert.equal(run('answer().revealed'),true);assert.equal(run('answer().firstListening.transcript'),'matches');});
check('first listening answers stay preserved after revision',()=>{get('transcript').value='unsure';run('readFields()');assert.equal(run('answer().firstListening.transcript'),'matches');get('transcript').value='matches';run('readFields()');});
for(const k of ['reading','experience','device','prior'])get(k).value=k==='device'?'SOFTWARE QA DEVICE':k==='prior'?'unsure':k==='reading'?'comfortable':'none';
get('setup_status').value=get('ending_status').value='located';
for(const [k,v] of Object.entries({setup_low:1,setup_best:1.01,setup_high:1.02,ending_low:1.2,ending_best:1.21,ending_high:1.22}))get(k).value=String(v);
sounds[0].checked=true;get('confidence').value='medium';get('disruption').value='no';run('readFields()');
check('valid ranges save and derived gap is not rounded to a fake timestamp',()=>{assert.equal(run('issues(answer()).length'),0);assert.equal(run('save()'),true);assert.equal(run('answer().reviewComplete'),true);assert.match(get('gapLabel').textContent,/200\.0 ms/);});
check('saved completion survives next/back navigation',()=>{run('load(1);load(0)');assert.equal(run('answer().reviewComplete'),true);assert.equal(get('setup_best').value,1.01);});
check('reversed uncertainty and out-of-audio positions cannot be completed',()=>{get('setup_low').value='1.5';run('readFields()');assert.ok(run('issues(answer()).some(s=>s.includes("earliest"))'));assert.equal(run('save()'),false);get('setup_low').value='-0.2';run('readFields()');assert.ok(run('issues(answer()).some(s=>s.includes("valid setup"))'));get('setup_low').value='1';});
check('plot position maps to original timeline after zoom',()=>{run('view=[2,3];mode="setup_best";plotClick({currentTarget:$("wave"),clientX:518})');assert.equal(run('answer().setup.best'),2.5);});
check('unlocatable endpoints remain null rather than zero',()=>{for(const s of ['setup','ending']){get(s+'_status').value='unlocatable';for(const f of ['low','best','high'])get(s+'_'+f).value='';}get('closure').checked=true;get('confidence').value='low';run('readFields()');assert.equal(run('save()'),true);assert.equal(run('answer().setup.best'),null);assert.match(get('gapLabel').textContent,/No gap/);});
(async()=>{
 run('exportData()');const p=JSON.parse(await exported.text());check('export is QA-tagged, portable and retains blank uncertainty',()=>{assert.equal(p.purpose,'software_qa');assert.equal(p.completedCount,1);assert.equal(p.trials.P01.setup.best,null);assert.equal(p.trialOrder.length,6);assert.equal(p.languageAbility,'fluent_or_native');});
 get('import').files=[{text:async()=>JSON.stringify({...p,packId:'wrong-pack'})}];await get('import').onchange({target:get('import')});check('wrong-pack import is rejected without replacing answers',()=>{assert.match(get('message').textContent,/Import failed/);assert.equal(run('session.packId'),p.packId);});
 get('import').files=[{text:async()=>JSON.stringify(p)}];await get('import').onchange({target:get('import')});check('same-pack backup restores answers',()=>{assert.match(get('message').textContent,/Backup restored/);assert.equal(run('answer().closure'),true);});
 const out={kind:'software_logic_tests_not_human_annotation',checks_passed:checks,real_browser_test:false,
  browser_preview:'blocked by browser local-file URL policy; no workaround attempted',
  qa_html_sha256:require('node:crypto').createHash('sha256').update(html).digest('hex')};
 fs.writeFileSync(path.join(root,'runs/human-practice-v1/logic-checks.json'),JSON.stringify(out,null,2)+'\n');
})().catch(e=>{console.error(e);process.exitCode=1});
