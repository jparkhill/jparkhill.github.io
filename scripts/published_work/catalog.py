# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "beautifulsoup4", "pypdf", "pyyaml"]
# ///
"""Reconcile preserved public-source responses and read-only resume inputs.
Run: uv run scripts/published_work/catalog.py
Network collection snapshots are retained in artifacts/agt_41423e075e48/sources.
"""
import hashlib, json, re, shutil, unicodedata, xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader
ROOT=Path(__file__).resolve().parents[2]
S=ROOT/'artifacts/agt_41423e075e48/sources'
A=ROOT/'assets/published-work'
def read(name): return json.loads((S/name).read_text())
def norm(s): return re.sub(r'[^a-z0-9]', '',unicodedata.normalize('NFKD',s).lower())
def doi(s): return re.sub(r'^https?://(dx\.)?doi.org/','',s or '').lower()
def slug(s): return re.sub(r'[^a-z0-9]+','-',unicodedata.normalize('NFKD',s).encode('ascii','ignore').decode().lower()).strip('-')[:100]
def ident(s): return 'work-'+hashlib.sha256(s.encode()).hexdigest()[:12]
works=read('openalex-works.txt')['results']+[read(n+'.txt') for n in ['openalex-missing-polaron','openalex-missing-exciton','openalex-orbital-correct']]
# A title-only repository record and a retitled preprint are the same research works.
merges={'W3105032437':'W2150504126','W2731573197':'W2620906374'}
records={}; source_matches=[]
for w in works:
 key=w['id'].split('/')[-1];target=merges.get(key,key)
 e=records.setdefault(target,{'id':target,'source':[],'alternate_urls':[],'alternate_titles':[],'pdf_candidates':[],'openalex_ids':[]})
 e['source'].append('openalex:'+key);e['openalex_ids'].append(key)
 if key==target:
  loc=w.get('primary_location') or {};src=loc.get('source') or {}
  e.update(title=w['display_name'],authors=[a.get('raw_author_name') or a['author']['display_name'] for a in w['authorships']],year=w['publication_year'],venue=src.get('display_name') or 'Scholarly repository',url=w.get('doi') or loc.get('landing_page_url') or w['id'],doi=doi(w.get('doi')),kind=w['type'],oa_reported=w.get('open_access',{}).get('is_oa',False))
 else:e['alternate_titles'].append(w['display_name'])
 if w.get('doi'):e['alternate_urls'].append(w['doi'])
 for l in w['locations']:
  for u in [l.get('pdf_url'),l.get('landing_page_url')]:
   if not u:continue
   e['alternate_urls'].append(u)
   if l.get('is_oa') and (u==l.get('pdf_url') or '.pdf' in u or '/purl/' in u):e['pdf_candidates'].append(u)
 source_matches.append({'source':'openalex','source_id':key,'title':w['display_name'],'catalog_id':target,'match':'merged version' if key!=target else 'included'})
# ORCID/Crossref identify additional, independently verified versions and authors.
cr=read('crossref-works.txt')['message']['items']+read('crossref-author-matched.json')
conference=read('crossref-conference.txt')['message']; cr.append(conference);cr.append(read('crossref-qchem4.txt')['message'])
for x in cr:
 d=doi(x['DOI'])
 if d=='10.1016/s0020-7837(73)80017-2':continue # Different John Parkhill, urban libraries in 1973.
 if d=='10.26434/chemrxiv-2023-bdkgm':key='W4391568364'
 else:key=next((k for k,e in records.items() if e.get('doi')==d),None)
 if key is None and d==conference['DOI']:
  key=ident(d);records[key]={'id':key,'source':[],'alternate_urls':[],'alternate_titles':[],'pdf_candidates':[],'openalex_ids':[], 'title':x['title'][0].title(),'authors':[a.get('given','')+' '+a.get('family','') for a in x['author']],'year':2017,'venue':x['container-title'][0],'url':x['URL'],'doi':d,'kind':'conference-abstract','oa_reported':False}
 if key is None:continue
 e=records[key];e['source'].append('crossref:'+d)
 if d==e.get('doi') and len(x.get('author',[]))>len(e['authors']):
  e['authors']=[' '.join([a.get('given',''),a.get('family','')]).strip() for a in x['author']]
  e['author_list_source']='Crossref complete author list (OpenAlex truncates at 100 authors)'
 if d==e.get('doi') and e.get('kind')=='erratum' and x.get('type')=='journal-article':e['kind']='article'
 if x.get('title') and norm(x['title'][0])!=norm(e['title']):e['alternate_titles'].append(BeautifulSoup(x['title'][0],'html.parser').get_text())
 if d not in [doi(e['url'])]:e['alternate_urls'].append('https://doi.org/'+d)
 source_matches.append({'source':'crossref','source_id':d,'title':x['title'][0],'catalog_id':key,'match':'included / DOI or verified version'})
# arXiv has machine-readable DOI relationships as well as alternate preprint titles.
ns={'a':'http://www.w3.org/2005/Atom','x':'http://arxiv.org/schemas/atom'}
for x in ET.fromstring((S/'arxiv-author.txt').read_text()).findall('a:entry',ns):
 u=x.findtext('a:id',namespaces=ns).replace('http:','https:');aid=re.search(r'(\d{4}\.\d{4,5})',u)[1];title=' '.join(x.findtext('a:title',namespaces=ns).split());d=doi(x.findtext('x:doi',namespaces=ns))
 key=next((k for k,e in records.items() if any(aid in a for a in e['alternate_urls']) or (d and d==e.get('doi'))),None)
 if not key:raise ValueError(('Unreconciled arXiv entry',aid,title))
 e=records[key];e['source'].append('arxiv:'+aid);e['alternate_urls'].append(u);e['alternate_titles'].append(title);e['pdf_candidates'].insert(0,'https://arxiv.org/pdf/'+aid)
 source_matches.append({'source':'arxiv','source_id':aid,'title':title,'catalog_id':key,'match':'verified version'})
for group in read('orcid-works.txt')['group']:
 x=group['work-summary'][0];title=x['title']['title']['value'];ds=[doi(a.get('external-id-normalized',{}).get('value') or a['external-id-value']) for a in x.get('external-ids',{}).get('external-id',[]) if a['external-id-type']=='doi']
 key=next((k for k,e in records.items() if e.get('doi') in ds or norm(title)==norm(e['title']) or (title.startswith('Measurement of the third-order') and e.get('doi')=='10.1002/jrs.3102') or (title.startswith('Near-Infrared Spectroscopy') and e.get('doi')=='10.1086/529484')),None)
 if not key:raise ValueError(('Unreconciled ORCID entry',title))
 records[key]['source'].append('orcid:'+str(x['put-code']));source_matches.append({'source':'orcid','source_id':str(x['put-code']),'title':title,'catalog_id':key,'match':'DOI / normalized title'})
# Prefer known public repository copies. Never guess a paywalled PDF endpoint.
records['W2245173478']['pdf_candidates'].insert(0,'https://escholarship.org/content/qt7nk0q94d/qt7nk0q94d.pdf')
records['W2026172453']['pdf_candidates'].append('https://web.phys.ntu.edu.tw/jdchai/Papers/p27.pdf')
records['W2151703051']['pdf_candidates'].append('https://articles.adsabs.harvard.edu/pdf/2008ApJ...680.1243M')
for e in records.values():
 for u in e['alternate_urls']:
  m=re.search(r'(?:PMC|articles/)(\d{6,})',u)
  if m and ('ncbi' in u or 'europepmc' in u):e['pdf_candidates'].append('https://europepmc.org/articles/PMC'+m[1]+'?pdf=render')
# Terray full news index plus old insights URLs. Keep one entry per article title.
terray=[];inventory=read('terray-inventory.json')
for x in inventory:
 p=x.get('post',{});authors=[a['name'] for a in (p.get('authors') or [])];related=any('Parkhill' in a for a in authors)
 resume_link='enhanced-parsing' in x['url']
 # Include the platform overview because it introduces the ML work, and the
 # follow-up TerraBind post; clearly preserve their actual authorship.
 context=any(t in x['url'] for t in ['introducing-emmi-where','terrabind-generating-synthetic'])
 if p.get('sourceType')=='external':
  e=next((e for e in records.values() if norm(e['title'])==norm(x['title'])),None)
  x['disposition']='scholarly link already cataloged' if e else 'external news / no Parkhill byline'
  if e:e['source'].append('terray-research-index');e['alternate_urls'].append(x['url']);x['catalog_id']=e['id']
 elif related or resume_link or context:
  old=next((t for t in terray if norm(t['title'])==norm(x['title'])),None)
  if old:old['alternate_urls'].append(x['url']);x['disposition']='duplicate insights URL';x['catalog_id']=old['id'];continue
  e={'id':'terray-'+slug(p['title']),'title':p['title'],'authors':authors or ['Terray Therapeutics'],'year':int(p['date'][:4]),'venue':'Terray Therapeutics','url':x['url'],'kind':'blog','source':['terray-news-index','terray-sitemap'] if '/insights/' in x['url'] else ['terray-news-index'],'alternate_urls':[],'pdf_status':'pending','inclusion_reason':'John Parkhill byline' if related else ('Resume-linked collaboration' if resume_link else 'Related Terray ML platform context; byline preserved')}
  terray.append(e);x['disposition']=e['inclusion_reason'];x['catalog_id']=e['id']
 else:x['disposition']='No Parkhill byline or mention; unrelated company news'
# Resume annotation URLs and TeX links are all reconciled individually.
r=PdfReader(ROOT/'assets/johnparkhill-resume.pdf');pdf_urls=list(dict.fromkeys(a.get_object().get('/A',{}).get('/URI') for p in r.pages for a in p.get('/Annots',[]) if a.get_object().get('/A',{}).get('/URI')))
tex_path=Path('/Users/johnparkhill/job_hunt_2026/master/resume/johnparkhill.tex');tex=tex_path.read_text();tex_urls=list(dict.fromkeys(re.findall(r'\\(?:link|href)\{(https?://[^}]+)\}',tex)))
material_specs={
 'https://scholar.google.com/citations?user=mkEouY4AAAAJ':('Google Scholar profile','Google Scholar',None,['John Parkhill'],'profile'),
 'https://github.com/jparkhill/TensorMol':('TensorMol: open-source neural network model chemistry','GitHub',2018,['John Parkhill and contributors'],'software'),
 'https://www.chemistryworld.com/news/neural-network-predicts-bond-energies-like-a-pro/3007598.article':('Neural network predicts bond energies like a pro','Chemistry World',2017,['Chemistry World'],'press'),
 'https://web.archive.org/web/20150928033434/http://aspuru.chem.harvard.edu/john-parkhill/':('Harvard research profile and publications','Aspuru-Guzik Group / Internet Archive',2015,['John Parkhill'],'research-profile'),
 'http://www.q-chem.com':('Q-Chem electronic structure package','Q-Chem',None,['Q-Chem developers; contributor John Parkhill'],'software'),
}
materials=[];resume_matches=[]
for source,urls in [('resume_pdf',pdf_urls),('resume_tex',tex_urls)]:
 for u in urls:
  key=next((e['id'] for e in records.values() if u in e['alternate_urls'] or u==e['url'] or (('2511.10590' in u or '2602.07735' in u or '1509.00062' in u) and any(re.search(r'\d{4}\.\d{4,5}',u)[0] in a for a in e['alternate_urls'])) or ('64e8137fdd1a73847f73f7aa' in u and e['id']=='W4391568364') or ('2024.08.22.609169' in u and e['id']=='W4401828945')),None)
  if not key:key=next((e['id'] for e in terray if e['url']==u),None)
  if not key:
   spec=material_specs.get(u)
   if not spec:raise ValueError(('Unmapped resume link',u))
   old=next((e for e in materials if e['url']==u),None)
   if old:e=old
   else:
    title,venue,year,authors,kind=spec;e={'id':ident(u),'title':title,'authors':authors,'venue':venue,'year':year,'kind':kind,'url':u,'source':[],'pdf_status':'not_applicable'};materials.append(e)
   key=e['id']
  e=next(e for e in list(records.values())+terray+materials if e['id']==key);e['source'].append(source)
  resume_matches.append({'source':source,'url':u,'catalog_id':key,'match':'included'})
resume={'id':'resume','title':'John Parkhill — Resume','authors':['John Parkhill'],'venue':'Personal resume','year':2026,'kind':'resume','url':'/assets/johnparkhill-resume.pdf','pdf':'/assets/published-work/materials/johnparkhill-resume.pdf','pdf_status':'archived','source':['resume_pdf','resume_tex']};materials.append(resume);shutil.copyfile(ROOT/'assets/johnparkhill-resume.pdf',ROOT/resume['pdf'].lstrip('/'))
for name,u,title in [('github-profile','https://github.com/jparkhill','GitHub profile'),('linkedin-profile','https://www.linkedin.com/in/johnaparkhill/','LinkedIn profile')]:
 materials.append({'id':name,'title':title,'authors':['John Parkhill'],'venue':title.split()[0],'year':None,'kind':'profile','url':u,'source':['resume_pdf visible header','resume_tex visible header'],'pdf_status':'not_applicable'})
materials.append({'id':'gtc-2024','title':'Live from GTC: A Conversation with Terray Therapeutics','authors':['John Parkhill','Tony Paikeday'],'venue':'NVIDIA GTC','year':2024,'kind':'talk','url':'https://www.nvidia.com/en-us/on-demand/session/gtc24-ep64036/','source':['NVIDIA session page / public web search'],'pdf_status':'not_applicable'})
# Preserve completed archive results across regeneration.
archive=read('archive-results.json') if (S/'archive-results.json').exists() else {}
for e in list(records.values())+terray:
 if e['id'] in archive:e.update({k:v for k,v in archive[e['id']].items() if k in ['pdf','pdf_status','pdf_note']})
 else:e['pdf_status']='pending'
 for k in ['source','alternate_urls','alternate_titles','pdf_candidates','openalex_ids']:
  if k in e:e[k]=list(dict.fromkeys(e[k]))
 if 'alternate_urls' in e:e['alternate_urls']=[u for u in e['alternate_urls'] if u!=e['url']]
# API calls supply article provenance, not evidence of direct Scholar coverage.
papers=[]
for e in records.values():
 if e['kind'] in ['dissertation','conference-abstract','dataset']:materials.append(e)
 else:papers.append(e)
groups=[{'title':'Papers & Preprints','id':'papers','entries':papers},{'title':'Terray Blog Posts','id':'terray','entries':terray},{'title':'Talks, Patents & Resume Materials','id':'materials','entries':materials}]
for g in groups:g['entries'].sort(key=lambda e:(-(e.get('year') or 0),e['title']))
(ROOT/'_data/published_work.yml').write_text(yaml.safe_dump(json.loads(json.dumps(groups)),allow_unicode=True,sort_keys=False,width=110))
(S/'reconciliation.json').write_text(json.dumps({'source_matches':source_matches,'resume_matches':resume_matches,'resume_pdf_links':pdf_urls,'resume_tex_links':tex_urls,'terray_inventory':[{k:v for k,v in x.items() if k not in ['post','keys']} for x in inventory],'excluded':[{'source':'crossref-author','doi':'10.1016/s0020-7837(73)80017-2','reason':'Different author / urban library publication in 1973'}, {'source':'openalex-orbital.txt','doi':'10.1080/00268976.2018.1448947','reason':'Exploratory DOI lookup resolved to unrelated authors; not used'}, {'source':'pubmed-search.txt','reason':'Initial-based query matched Julian Parkhill; all ten results excluded'}, {'source':'arxiv-all','reason':'Two other Parkhill authors, speech synthesis and visual-experience datasets; excluded'}]},indent=2))
print({g['id']:len(g['entries']) for g in groups});print('resume links',len(pdf_urls),len(tex_urls),'source records',len(source_matches))
