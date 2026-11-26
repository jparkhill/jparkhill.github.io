Published Work catalog and public PDF archive

- Page: `published-work.md`, canonical `/published-work/`, alias `/published-work.html`.
- Catalog: `_data/published_work.yml`; full reconciliation and retrieval attempts: `assets/published-work/MANIFEST.json`.
- Review report: `report/results.json`; figures and a rendered template preview are beside it.
- Regenerate from preserved metadata: `uv run scripts/published_work/catalog.py`.
- Retry downloads and prints sequentially: `uv run scripts/published_work/archive.py`, then `uv run scripts/published_work/archive.py --blogs`.
- Verify and regenerate manifest/report: `uv run scripts/published_work/verify.py`.
- `collect.py` intentionally refreshes successful public-source inputs; review differences before regeneration. Google Scholar was rate limited; it is not retried by that collector.

The operator's latest instruction says use the default branch without creating branches. This checkout's existing branch is `master`. No pushes or deployments were performed. Direct Scholar completeness and a full native Jekyll build remain unverified; see report caveats.
