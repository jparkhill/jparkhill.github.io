# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "beautifulsoup4", "pypdf", "pyyaml", "fonttools", "pymupdf"]
# ///
"""Download advertised public PDFs, or print sanitized Terray article HTML.
Run: uv run scripts/published_work/archive.py [--blogs]
All attempts, including HTTP errors, are retained. Re-running skips successes.
"""
import concurrent.futures, hashlib, html, io, json, re, subprocess, sys, tempfile, unicodedata
from pathlib import Path
from urllib.parse import urljoin
import requests,yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2]; S=ROOT/'artifacts/agt_41423e075e48/sources'; A=ROOT/'assets/published-work'
CHROME='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
def norm(s):return re.sub('[^a-z0-9]','',unicodedata.normalize('NFKD',s).lower())
def validate(data,e):
 if not data.startswith(b'%PDF-') or len(data)<=5000:raise ValueError('Not a nontrivial PDF')
 r=PdfReader(io.BytesIO(data));text='\n'.join(p.extract_text() or '' for p in r.pages);n=norm(text)
 titles=[e['title']]+e.get('alternate_titles',[])
 exact=next((t for t in titles if norm(t) in n),None)
 if not exact:
  import pymupdf
  text='\n'.join(p.get_text() for p in pymupdf.open(stream=data,filetype='pdf'));n=norm(text)
  exact=next((t for t in titles if norm(t) in n),None)
 if len(text)<300 or not exact:raise ValueError(f'Article title not found in extracted content ({len(text)} characters)')
 if any(t in text[:1500].lower() for t in ['just a moment','checking your browser','verify you are human','access denied']):raise ValueError('Challenge/error page')
 return {'bytes':len(data),'pages':len(r.pages),'text_characters':len(text),'matched_title':exact,'sha256':hashlib.sha256(data).hexdigest(),'text_excerpt':text[:300],'validation':'passed'}
def download(e):
 attempts=[]
 for u in e.get('pdf_candidates',[]):
  try:
   r=requests.get(u,timeout=45,headers={'User-Agent':'PublishedWorkArchive/1.0 (personal academic bibliography)'});a={'url':u,'status':r.status_code,'final_url':r.url,'bytes':len(r.content)};attempts.append(a)
   if r.status_code!=200:continue
   try:check=validate(r.content,e)
   except Exception as ex:a['validation_error']=str(ex);continue
   path=A/('materials' if e['kind'] in ['dissertation','conference-abstract','dataset'] else 'papers')/(e['id'].lower()+'.pdf');path.write_bytes(r.content)
   return e['id'],{'pdf':'/'+str(path.relative_to(ROOT)),'pdf_status':'archived','pdf_note':'Public repository or publisher copy; version/title verified from extracted text.','download_url':u,'check':check,'attempts':attempts}
  except Exception as ex:attempts.append({'url':u,'error':str(ex)})
 status='blocked' if e.get('pdf_candidates') else ('unavailable' if e['kind'] in ['conference-abstract','dataset'] else 'paywalled')
 return e['id'],{'pdf_status':status,'pdf_note':'Advertised public PDF could not be retrieved or failed content validation; see manifest attempts.' if status=='blocked' else ('No open PDF location found in the reconciled scholarly API records. Publisher link retained; access may vary.' if status=='paywalled' else 'No public PDF located for this abstract or dataset; canonical record retained.'),'attempts':attempts}
def blog(e):
 slug=e['url'].split('/')[-1];raw=S/'terray-pages'/(slug+'.html');s=BeautifulSoup(raw.read_text(),'html.parser');main=s.find('main')
 for t in main.select('style,script,nav,button,iframe,svg,a[aria-label^="Back to reference"]'):t.decompose()
 for t in main.select('li[id^="fn-"] > span:first-child'):
  t.decompose()
 for t in main.select('a[href="/news-and-research"]'):
  t.decompose()
 for t in main.find_all(True):
  for k in ['class','style','width','height','srcset','sizes','loading']:
   t.attrs.pop(k,None)
  for attr in ['src','href']:
   if t.get(attr):t[attr]=urljoin(e['url'],t[attr])
  if t.name=='img':t['loading']='eager'
 css='''@page { size: A4; margin: 17mm 16mm; } body { font: 11pt/1.5 Georgia,serif; color:#20252c; } h1 {font: bold 25pt/1.15 Arial,sans-serif;} h2 {font: bold 18pt/1.25 Arial,sans-serif;} h3,h4 {font: bold 14pt/1.3 Arial,sans-serif;} h1,h2,h3,h4 {break-after:avoid;} p,li {orphans:3;widows:3;} img,svg,video {max-width:100%;max-height:220mm;height:auto;break-inside:avoid;} a {color:#235b7a;overflow-wrap:anywhere;} table {border-collapse:collapse;font-size:9pt;width:100%;} td,th {padding:5px;border:1px solid #ccc;} pre {white-space:pre-wrap;font-size:9pt;} figure {margin:1em 0;break-inside:avoid;} .archive-source {font:9pt/1.5 Arial,sans-serif;color:#46505c;border-bottom:1px solid #ddd;padding-bottom:10px;}'''
 body='<!doctype html><html><head><meta charset="utf-8"><title>'+html.escape(e['title'])+'</title><style>'+css+'</style></head><body><p class="archive-source">Terray Therapeutics • '+str(e['year'])+'<br>Source: '+html.escape(e['url'])+'</p>'+str(main)+'</body></html>'
 path=A/'terray'/(slug+'.pdf');scratch=ROOT/'tmp/pdfs';scratch.mkdir(parents=True,exist_ok=True);f=scratch/(slug+'.html');f.write_text(body)
 cmd=[CHROME,'--headless','--disable-gpu','--no-pdf-header-footer','--run-all-compositor-stages-before-draw','--virtual-time-budget=15000','--print-to-pdf='+str(path),f.as_uri()]
 with tempfile.TemporaryDirectory(prefix='pw-chrome-') as profile:
  cmd.insert(1,'--user-data-dir='+profile);proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=90)
 if proc.returncode or not path.exists():raise RuntimeError(proc.stderr.decode()[-1000:])
 check=validate(path.read_bytes(),e)
 return e['id'],{'pdf':'/'+str(path.relative_to(ROOT)),'pdf_status':'archived','pdf_note':'Chrome print of complete article body, with print styling and original figures; byline and source URL preserved.','check':check,'print_command':[c for c in cmd if not c.startswith('--user-data-dir')],'attempts':[{'url':e['url'],'status':200,'method':'Chrome headless print'}]}
if __name__=='__main__':
 import fcntl
 lock=open(S/'archive.lock','w');fcntl.flock(lock,fcntl.LOCK_EX)
 groups=yaml.safe_load((ROOT/'_data/published_work.yml').read_text());out=S/'archive-results.json';results=json.loads(out.read_text()) if out.exists() else {};blogs='--blogs' in sys.argv
 entries=[e for g in groups for e in g['entries'] if (g['id']=='terray')==blogs and (e.get('pdf_status')!='archived' or '--reprint' in sys.argv) and (blogs or 'pdf_candidates' in e)]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2 if blogs else 5) as pool:
  jobs={pool.submit(blog if blogs else download,e):e for e in entries}
  for f in concurrent.futures.as_completed(jobs):
   e=jobs[f]
   try:k,v=f.result();results[k]=v;print(k,v['pdf_status'],v.get('check',{}).get('bytes',''),flush=True)
   except Exception as ex:results[e['id']]={'pdf_status':'blocked','pdf_note':str(ex),'attempts':[]};print(e['id'],'ERROR',ex,flush=True)
   out.write_text(json.dumps(results,indent=2))
