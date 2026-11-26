---
name: hierarchical-index
description: Build or improve a hierarchical index (a topic tree with a summary at each node and every item on exactly one leaf) over a raw, unorganised collection - a long markdown list such as the reading list, a directory of files, or exported connector items such as a scraped Google Drive - and optionally re-organise the collection from that index into directories or a table, losslessly, with a manifest and a check. Use when the operator asks to organise, index, sort, categorise, file or restructure a pile of items, or asks "what is in X" about a large unsorted collection.
tools: [shell, read_file, memory_search]
requires: [bin:uv]
---
# Hierarchical index

One published workflow, **Hierarchical index** (category Knowledge / Indexing), does every
model call: summarise each item, grow a topic tree a few items at a time, place each item on
one leaf (the PaperQA pattern: per-item summary first, then a tree built from the summaries).
Plain code around it reads the source, keeps the index and writes the pages. The service
publishes the workflow itself when it starts; there is nothing to publish by hand.
Internals: `wiki/internal/hierarchical-index.md`.

## When to use it

- A collection has outgrown its shape: one long list, a directory nobody sorted, a Drive export.
- Someone wants to browse it by topic, or wants it re-filed into folders or a table.

Not for: answering a question about one document (read it), or a collection of under about
fifteen items (sort it by hand).

## What you get

| Thing | Where | Backed up by |
|---|---|---|
| The index (data) | `<wiki>/knowledge/indexes/<name>/index.json` | nightly state archive (`wiki` tree) |
| Browsable pages | `index.md` beside it, `topics/<topic>.md` per top-level topic | same; also searchable in the KB |
| List of all indexes | `<wiki>/knowledge/indexes/index.md`, linked once from the wiki's root `index.md` | same |
| Re-organised copy (optional) | `<data>/organised/<name>-<format>/` with `manifest.json` | restic raw item `data/organised` |
| The collection as knowledge-base pages (optional) | `<wiki>/knowledge/<name>/`: index page, topic pages, one page per item, `index.json`, `manifest.json` | nightly state archive (`wiki` tree) |
| Workflow definition and run records | `workflow*` database tables | nightly database dump |

Every item has a stable id, a one-sentence summary, exactly one leaf and a pointer back to
its source (file and lines, relative path, or connector source/account/external id).

## Build the index

```bash
uv run python -m durian.indexing.cli build --service \
    --kind markdown_list --source notes/bookmarks.md --sections "Links" \
    --name "Bookmarks" --hint "links worth keeping"
```

(The reading list was built this way from `wiki/projects/reading-list.md` and then filed as pages:
it now lives in `wiki/knowledge/reading/`, see below. Do not build from the old page; it is a pointer.)

- `--kind markdown_list`: top-level bullets (with their indented continuation lines);
  `--sections` limits it to bullets under those headings, so logs and todos stay out.
- `--kind directory --source DIR`: every file under DIR. Add `--metadata-only` when the model
  may see only paths and sizes.
- `--kind connector --source FILE`: rows written by
  `durian connectors export ACCOUNT --source drive --out FILE`. Folders, shared drives and
  deleted rows are not items.
- `--service` runs on the running Durian through the admin workflow API with your CLI login,
  on the version the service published at its last start.
- No service to use (a release not deployed yet, a machine without one): replace `--service`
  with `--local-db DIR --cap FILE`. The same workflow then runs in the command's own process
  on a database of its own, calling the installation's default model with the key read from
  the secret store there; `FILE` is a committed JSON spend cap (`max_calls`,
  `max_input_tokens`, `max_output_tokens`, `max_usd`, `notional_price_usd_per_mtok`) and the
  build stops when it is reached. Commit the cap before the first call.
- Each run is one model call of at most 2048 output tokens and 30 seconds, and the default
  model spends part of that on reasoning, so items go 8 to a call (`--batch`); a failed call
  is sent again in halves. Expect a little under one call per item on a first build (measured:
  126 items, 107 calls of which 30 were cut off and re-sent, 14 minutes). `--max-top` (8), `--max-sub` (6) and `--split-over` (10) shape
  the tree: a top-level topic with more than `--split-over` items gets subtopics.

The command prints the item count, leaves, depth, how many model calls it made, how many runs
failed and were re-sent, and `unsummarized_items` / `unplaced_items` (both should be empty).
Compare the item count with what the source says it holds before you report anything.

## Connector items: metadata only

For connector rows the model sees exactly five fields: name, folder path, type, modified and
created dates. Bodies, descriptions, owners, links and file contents are never put in a model
input; the adapter has no switch for it. Sending document contents to a model needs the
operator's say-so and is a different task (the Drive wiki builder, `durian connectors
drive-wiki`). To confirm after a run, read the prompt inputs of its run records:
`GET /workflows/runs/<run_id>` (`attempts[].inputs`); the run ids are in `index.json`
under `history[].runs`.

## Improve it: re-run, and move things by hand

The index is meant to be corrected and re-run.

- Move an item: `uv run python -m durian.indexing.cli move --name "Bookmarks" --item <id> --leaf <leaf id>`
  (or edit that item's `leaf` in `index.json`). It is marked `placed_by: manual`.
- Re-run the same `build` command after the source gained items. Items already in the index
  stay where they are, whether the model or a person put them there. Only new items are
  summarised and placed, into the existing leaves. An item whose summary fell back to its
  title in an earlier build (`unsummarized_items`) is summarised again. Items the source no longer has are dropped
  and listed in the run record (`removed_items`).
- The tree is grown once, on the first build. To rename or add a topic, edit `nodes` in `index.json`
  (`id`, `parent`, `title`, `summary`), then `python -m durian.indexing.cli render --name ...`.
- Never edit the `.md` pages: they are rewritten from `index.json`.
- Items the model could not place land in a leaf called **Unsorted**; move them by hand.

## File a collection as knowledge-base pages (the reading list's layout)

```bash
uv run python -m durian.indexing.cli reorganise --name "Bookmarks" --format pages --out wiki/knowledge/bookmarks
uv run python -m durian.indexing.cli check --out wiki/knowledge/bookmarks                    # exit 0 = lossless
```

One page per item (its bytes unchanged between `<!-- item:begin <id> -->` and `<!-- item:end -->`),
a page per top-level topic, an index page, `about.md` with a markdown list's text outside its items,
and the index moved to `<out>/index.json`: the pages are the collection from then on. Use
`--index-dir <out>` for `move`, `render` and re-runs (`build --kind kb_pages --source <out>`).
Edit an entry only between its markers; everything else on the pages is regenerated. Documents
under `data/papers/` are linked from the entry pages, never copied into the wiki.

**The reading list** lives this way in `wiki/knowledge/reading/` (since 2026-11-21). Add to it only
with `reading_list_add` (Telegram `/reading-list`): it writes the entry's page and the workflow files
it under a topic (Unsorted when no model can run). Move an entry:
`uv run python -m durian.indexing.cli move --index-dir wiki/knowledge/reading --item <id> --leaf <leaf id>`.
Prove the move lost nothing:
`uv run python -m durian.indexing.cli check --out wiki/knowledge/reading --source artifacts/agt_77aeb04151ac/inventory/reading-list.pre-change.md`.

## Re-organise (optional) and prove nothing was lost

```bash
uv run python -m durian.indexing.cli reorganise --name "Bookmarks" --format table        # or directories
uv run python -m durian.indexing.cli check --out data/organised/bookmarks-table            # exit 0 = lossless
```

- It writes to a **new, empty** location and never touches the source. `directories` gives
  `<topic>/<subtopic>/<file>` with each item's bytes unchanged; `table` gives `items.csv`,
  one row per item (text as UTF-8, anything else base64).
- `manifest.json` lists, per item: item id, source sha256, output path and row, output sha256,
  transform.
- `check` reads the **source** again, recomputes every source hash and rebuilds each item's
  bytes from the output. It exits 1 when an item is missing, duplicated or altered, when the
  output holds something the manifest does not name, or when the source has changed since
  (re-run `build` and re-organise into a fresh location). `--source PATH` checks against a
  pinned copy of the source.
- Replacing the original with the re-organised copy is a separate decision for the operator.
  Do not delete or overwrite a source because the check passed.

## Report

Say how many items the source holds and how many the index holds (they must be equal), the
number of leaves, anything in Unsorted, the model and number of calls, and link the page
`wiki/knowledge/indexes/<name>/index.md`. If you re-organised, give the check's counts and
exit code.

## Worked example in this repository

`artifacts/agt_eb80fd87543d/reading-list/build.sh` is the exact command the real reading-list
index was built with (real provider, committed cap in `artifacts/agt_eb80fd87543d/spend_cap.json`);
`artifacts/agt_eb80fd87543d/run/real_model_runs.py fixtures|improve` does the same for a
directory fixture, a Drive-shaped connector export, and a re-run on a copy of the reading list.
