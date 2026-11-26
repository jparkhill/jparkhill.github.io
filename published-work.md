---
layout: page
title: Published Work
permalink: /published-work/
---

Research in machine learning, molecular design, and quantum chemistry, alongside writing from Terray Therapeutics and other research materials.

[Google Scholar](https://scholar.google.com/citations?user=mkEouY4AAAAJ) · [ORCID](https://orcid.org/0000-0001-5739-9620) · [Resume]({{ '/assets/johnparkhill-resume.pdf' | relative_url }})

Local PDFs include open research copies and print editions of Terray articles. Preprints and published versions of the same work appear together. Author lists below follow the source records.

<nav aria-label="Published work sections">
{% for group in site.data.published_work %}
<a href="#{{ group.id }}">{{ group.title | escape }} ({{ group.entries | size }})</a>{% unless forloop.last %} · {% endunless %}
{% endfor %}
</nav>

{% for group in site.data.published_work %}
<section id="{{ group.id }}" class="published-work-section">
<h2>{{ group.title | escape }}</h2>
<ol class="published-work-list">
{% for work in group.entries %}
<li id="{{ work.id }}">
  <h3><a href="{{ work.url | relative_url }}">{{ work.title | escape }}</a></h3>
  <p class="work-meta">{% if work.year %}{{ work.year }} · {% endif %}{{ work.venue | escape }}</p>
  <details><summary>Authors</summary><p>{{ work.authors | join: ', ' | escape }}</p></details>
  <p class="work-links">
    <a href="{{ work.url | relative_url }}">{% if work.doi %}DOI / publication{% else %}Original source{% endif %}</a>
    {% if work.pdf %} · <a href="{{ work.pdf | relative_url }}">Download PDF</a>
    {% elsif work.pdf_status == 'paywalled' %} · <span>Publisher access; open PDF not located</span>
    {% elsif work.pdf_status == 'blocked' %} · <span>PDF retrieval unavailable</span>
    {% elsif work.pdf_status == 'unavailable' %} · <span>PDF not located</span>{% endif %}
  </p>
  {% if work.inclusion_reason %}<p class="work-meta">{{ work.inclusion_reason | escape }}</p>{% endif %}
  {% if work.alternate_urls.size > 0 %}
  <details><summary>Other versions &amp; sources</summary><ul>
  {% for url in work.alternate_urls %}<li><a href="{{ url }}">{{ url | escape }}</a></li>{% endfor %}
  </ul></details>
  {% endif %}
</li>
{% endfor %}
</ol>
</section>
{% endfor %}

<p class="work-meta">Catalog checked November 2026 against the resume, OpenAlex, ORCID, Crossref, arXiv, and Terray's news index and sitemap. Direct Google Scholar access was rate limited, so exact profile coverage could not be verified. <a href="{{ '/assets/published-work/MANIFEST.json' | relative_url }}">Archive provenance and retrieval details</a>.</p>

<style>
.published-work-section { margin-top: 2.5rem; scroll-margin-top: 1rem; }
.published-work-list { padding-left: 1.4rem; }
.published-work-list > li { margin-bottom: 1.7rem; padding-left: .25rem; }
.published-work-list h3 { font-size: 1.08rem; line-height: 1.45; margin: 0 0 .35rem; }
.published-work-list p { margin: .3rem 0; }
.work-meta, .work-links, .published-work-list details { font-size: .9rem; }
.published-work-list summary { cursor: pointer; }
.published-work-list details p, .published-work-list details li { overflow-wrap: anywhere; }
</style>
