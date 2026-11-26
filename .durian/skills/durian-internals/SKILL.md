---
name: durian-internals
description: Keep Durian's own operating knowledge accurate. Use whenever changing Durian code, schema, scripts, service units, transport, policy, secrets, workers, or deployment state.
tools: [read_file, write_file, list_dir, memory_search, wiki_search, shell, runtime_status, project_sync_wiki]
requires: []
---
# Durian internals

Durian must understand itself. The canonical internal reference lives in
`wiki/internal/`.

## When to use

Use this skill when you change any of:

- `src/durian/` code or CLI commands
- SurrealDB schema or migrations
- service units or setup/bootstrap scripts
- worker protocol, policy engine, secrets, or update flow
- machine topology, ports, service state, or known caveats
- what a child or hand-started harness is told (`DURIAN.md`, `AGENTS.md`, the
  skills under `integrations/`)

## Required update

Before declaring the change complete:

1. Update the matching `wiki/internal/*.md` page, and set its `updated:` date.
2. Update local gitignored `wiki/ops/current-state.md` if installation-specific deployment facts changed (never commit hostnames, IPs, or personal data to `wiki/internal/`).
3. **A new page goes into `wiki/internal/index.md` in the same commit**, with one
   line saying what it is for. A page that is not in an index is not found.
4. Every page needs `title`, `type` and `updated` in its frontmatter.
5. Do not put secret values in the internal KB; use secret names/refs only.
6. Run `uv run python scripts/wiki-lint.py` and read its output for the pages
   you touched. An error on one of your pages (missing frontmatter, not linked
   from an index, broken link) blocks the change. `scripts/test.sh` prints only
   the linter's last line, so a green suite does not mean the wiki is clean.
7. Run `make hygiene` (`python -m durian.security.hygiene`) to verify zero
   credentials or private terms appear in tracked files.
8. Rebuild the index with `durian reload` or the service `/reload` endpoint.
9. Verify with `durian status`, `durian machines`, and a relevant agtask/report.

## Say which claim you are making

A fact about Durian is one of three things; write the one that is true.

- **written**: the change exists in a checkout or worktree.
- **committed**: it is on `main`.
- **deployed**: the running service booted from a commit that contains it
  (`durian deployed <sha>`).

A capability that was built but has not run for real is written as that:
"not live", "tested headless only", "not yet tried on the Mac". If a fact is
uncertain, write `status: pending/unknown` instead of inventing it.

## What belongs where

- How something works and why (generic product documentation): `wiki/internal/`.
- What is true of this specific deployment today: local gitignored `wiki/ops/current-state.md`, dated.
- What went wrong and what was learned on this installation: local gitignored `wiki/ops/incidents.md`.
- Rules every agent in this checkout must follow: `AGENTS.md`. Keep it short
  and link to the page.

Generated blocks (between `durian:...:auto` markers) and generated machine pages
(`wiki/ops/machines/`) are local installation state; do not commit them to core git.

The internal KB is operational documentation, not marketing prose.

## Why the list is this long

On 2026-11-15 a session that added five features updated the right pages and
still left two of them out of the index, one without a `type`, and
`current-state.md` dated twelve days stale, because it never ran the linter.
