# /// script
# requires-python = ">=3.11"
# dependencies = ["requests", "beautifulsoup4"]
# ///
"""Refresh the successful public metadata inputs used by catalog.py.
Run only when intentionally updating the catalog. This preserves full responses,
including all 38 news-index records; it does not query blocked Google Scholar.
"""
import concurrent.futures,json
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
S=Path(__file__).resolve().parents[2]/'artifacts/agt_41423e075e48/sources'
URLS={
 'openalex-authors.txt':'https://api.openalex.org/authors?search=John%20Parkhill&per-page=20',
 'openalex-works.txt':'https://api.openalex.org/works?filter=author.id:A5026991176&per-page=200',
 'orcid-works.txt':'https://pub.orcid.org/v3.0/0000-0001-5739-9620/works',
 'crossref-works.txt':'https://api.crossref.org/works?filter=orcid:0000-0001-5739-9620&rows=200',
 'crossref-author.txt':'https://api.crossref.org/works?query.author=John%20Parkhill&rows=100',
 'crossref-qchem4.txt':'https://api.crossref.org/works/10.1080/00268976.2014.952696',
 'crossref-conference.txt':'https://api.crossref.org/works/10.15278/isms.2017.wb02',
 'openalex-missing-polaron.txt':'https://api.openalex.org/works/https://doi.org/10.1063/1.4762441',
 'openalex-missing-exciton.txt':'https://api.openalex.org/works/https://doi.org/10.1063/1.3689858',
 'openalex-orbital-correct.txt':'https://api.openalex.org/works/https://doi.org/10.1080/00268976.2017.1342009',
 'arxiv-author.txt':'https://export.arxiv.org/api/query?search_query=au:Parkhill_John&max_results=100',
 'terray-news.txt':'https://www.terraytx.ai/news-and-research',
 'terray-sitemap.txt':'https://www.terraytx.ai/sitemap.xml',
 'terray-old-sitemap.txt':'https://terraytx.com/sitemap.xml',
}
def fetch(item):
 name,url=item;r=requests.get(url,headers={'Accept':'application/json'} if name=='orcid-works.txt' else {},timeout=50)
 r.raise_for_status();(S/name).write_text(r.text);return {'name':name,'url':url,'status':r.status_code,'final_url':r.url,'bytes':len(r.content)}
if __name__=='__main__':
 S.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:log=list(ex.map(fetch,URLS.items()))
 soup=BeautifulSoup((S/'terray-news.txt').read_text(),'html.parser');j=json.loads(soup.find('script',id='__NEXT_DATA__').string);(S/'terray-next.json').write_text(json.dumps(j,indent=2))
 items=j['props']['pageProps']['page']['content'][1]['allItems'];urls=['https://www.terraytx.ai/'+x['slug']['current'] for x in items]
 urls += [e.text for e in ET.fromstring((S/'terray-sitemap.txt').read_text()).iter() if e.tag.endswith('loc') and '/insights/' in e.text]
 (S/'terray-pages').mkdir(exist_ok=True)
 def page(url):
  r=requests.get(url,timeout=45);r.raise_for_status();(S/'terray-pages'/(url.split('/')[-1]+'.html')).write_text(r.text);s=BeautifulSoup(r.text,'html.parser');d=json.loads(s.find('script',id='__NEXT_DATA__').string)['props']['pageProps'];p=d.get('post',d.get('page',{}))
  return {'url':url,'final_url':r.url,'status':r.status_code,'title':s.h1.get_text(' ',strip=True),'parkhill':'parkhill' in s.get_text(' ',strip=True).lower(),'keys':list(d),'post':p}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:inventory=list(ex.map(page,urls))
 (S/'terray-inventory.json').write_text(json.dumps(inventory,indent=2))
 cr=json.loads((S/'crossref-author.txt').read_text())['message']['items'];matched=[x for x in cr if any(a.get('family','').lower()=='parkhill' and a.get('given','').lower().startswith(('john','j.','j a')) for a in x.get('author',[]))];(S/'crossref-author-matched.json').write_text(json.dumps(matched,indent=2));(S/'collection-requests.json').write_text(json.dumps(log,indent=2))
 print(f'Refreshed {len(log)} metadata responses and {len(inventory)} Terray pages; review changes before regenerating.')
