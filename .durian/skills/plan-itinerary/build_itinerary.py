#!/usr/bin/env python3
"""Build a decision-first itinerary page from one JSON file.  Standard library only.

    python3 build_itinerary.py itinerary.json --out artifacts/<id>/report [--results] [--no-online-map]

Writes into --out:

    itinerary.html     one self-contained phone page: decisions, starred map, days, rules
    map.svg        the same map as a figure for the standard Durian report
    results.json   with --results: headline, the map figure and one table per decision
                   are written into it (other keys already there are kept)

It refuses to build (exit 1, reasons on stderr) when the brief breaks the rules
in SKILL.md: too long, a decision without a pick, a place without coordinates,
several sources with no account of where they disagreed.  Fix itinerary.json and
run it again; do not work around the check.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from urllib.parse import quote

LIMITS = {"now": 40, "why": 45, "option_side": 22, "block": 30, "rule": 30, "options": 4, "decisions": 5,
          "top_words": 450, "total_words": 1800}
esc = html.escape


def words(value: object) -> int:
    return len(str(value or "").split())


def check(brief: dict) -> list[str]:
    """Every way this brief breaks the rules; empty means it may be built."""
    bad: list[str] = []
    places = {p.get("id"): p for p in brief.get("places") or []}
    if not str(brief.get("title") or "").strip():
        bad.append("title is missing")
    if not str(brief.get("now") or "").strip():
        bad.append("`now` is missing: one sentence saying the next move")
    elif words(brief["now"]) > LIMITS["now"]:
        bad.append(f"`now` is {words(brief['now'])} words; the limit is {LIMITS['now']}")
    decisions = brief.get("decisions") or []
    if not decisions:
        bad.append("no decisions: a brief exists to settle at least one")
    if len(decisions) > LIMITS["decisions"]:
        bad.append(f"{len(decisions)} decisions; keep the {LIMITS['decisions']} that matter and move the rest to days or rules")
    for n, d in enumerate(decisions, 1):
        tag = f"decision {n} ({str(d.get('question') or '?')[:40]})"
        if not d.get("question"):
            bad.append(f"{tag}: no question")
        if not d.get("deadline"):
            bad.append(f"{tag}: no deadline (when must this be settled?)")
        options = d.get("options") or []
        if not 2 <= len(options) <= LIMITS["options"]:
            bad.append(f"{tag}: {len(options)} options; give 2 to {LIMITS['options']}, ranked")
        ids = [o.get("place") or o.get("label") for o in options]
        if d.get("pick") is None:
            if not d.get("blocked_on"):
                bad.append(f"{tag}: no pick. Recommend one, or say in `blocked_on` what single fact is missing")
        elif d["pick"] not in ids:
            bad.append(f"{tag}: pick {d['pick']!r} is not one of its options")
        if words(d.get("why")) > LIMITS["why"]:
            bad.append(f"{tag}: `why` is {words(d.get('why'))} words; the limit is {LIMITS['why']}")
        if d.get("pick") is not None and not d.get("why"):
            bad.append(f"{tag}: a pick needs a `why`")
        for o in options:
            if o.get("place") and o["place"] not in places:
                bad.append(f"{tag}: option place {o['place']!r} is not in places")
            for side in ("for", "against"):
                if words(o.get(side)) > LIMITS["option_side"]:
                    bad.append(f"{tag}: option {o.get('place') or o.get('label')}: `{side}` is over {LIMITS['option_side']} words")
            if "verified" not in o:
                bad.append(f"{tag}: option {o.get('place') or o.get('label')}: say `verified` true or false "
                           "(was the price or availability checked for these dates?)")
        if d.get("pick") in places and not places[d["pick"]].get("star"):
            bad.append(f"{tag}: the picked place {d['pick']!r} must have star: true so it stands out on the map")
    for p in brief.get("places") or []:
        name = p.get("name") or p.get("id")
        if not p.get("id") or not p.get("name"):
            bad.append(f"place {name!r}: needs id and name")
        lat, lon = p.get("lat"), p.get("lon")
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            bad.append(f"place {name!r}: needs numeric lat and lon (look them up; do not guess)")
    real = [p for p in brief.get("places") or [] if p.get("kind") != "anchor"]
    if len(real) < 2:
        bad.append("fewer than two places: the map needs the options on it")
    if not any(p.get("star") for p in real):
        bad.append("no starred place: star what you recommend")
    for day in brief.get("days") or []:
        for b in day.get("blocks") or []:
            if b.get("place") and b["place"] not in places:
                bad.append(f"day {day.get('date')}: block place {b['place']!r} is not in places")
            if words(b.get("what")) > LIMITS["block"]:
                bad.append(f"day {day.get('date')} {b.get('when')}: over {LIMITS['block']} words; link the detail instead")
    for r in brief.get("rules") or []:
        if words(r.get("then")) > LIMITS["rule"]:
            bad.append(f"rule {r.get('if')!r}: over {LIMITS['rule']} words")
    sources = brief.get("sources") or []
    if not sources:
        bad.append("no sources: list every pack, task or page this brief was built from")
    if len(sources) > 1 and "disagreements" not in brief:
        bad.append(f"{len(sources)} sources but no `disagreements`: say where they differed and how you ruled "
                   "(an empty list means you checked and they agree)")
    for item in brief.get("disagreements") or []:
        if len(item.get("positions") or []) < 2 or not item.get("ruling"):
            bad.append(f"disagreement {item.get('topic')!r}: needs both positions and a ruling")
    top = words(brief.get("now")) + sum(
        words(d.get("question")) + words(d.get("why")) + sum(words(o.get("for")) + words(o.get("against")) for o in d.get("options") or [])
        for d in decisions)
    if top > LIMITS["top_words"]:
        bad.append(f"the decision section is {top} words; the limit is {LIMITS['top_words']}")
    total = words(json.dumps({k: v for k, v in brief.items() if k not in {"places", "lines", "sources"}}, ensure_ascii=False))
    if total > LIMITS["total_words"]:
        bad.append(f"the brief is about {total} words; the limit is {LIMITS['total_words']}. Link the long material, do not inline it")
    return bad


# ---- the map ----
def maps_url(place: dict) -> str:
    if place.get("url"):
        return str(place["url"])
    query = f"{place['name']} {place.get('address') or ''}".strip() if place.get("address") else f"{place['lat']},{place['lon']}"
    return "https://www.google.com/maps/search/?api=1&query=" + quote(query)


def numbered(brief: dict) -> list[dict]:
    """Real places in map order: starred first, then as written.  Anchors get no number."""
    real = [p for p in brief.get("places") or [] if p.get("kind") != "anchor"]
    real.sort(key=lambda p: not p.get("star"))
    return [{**p, "n": i} for i, p in enumerate(real, 1)]


def star_path(cx: float, cy: float, r: float) -> str:
    pts = []
    for i in range(10):
        radius = r if i % 2 == 0 else r * 0.45
        angle = -math.pi / 2 + i * math.pi / 5
        pts.append(f"{cx + radius * math.cos(angle):.1f},{cy + radius * math.sin(angle):.1f}")
    return "M" + " L".join(pts) + " Z"


def render_map(brief: dict, width: int = 640) -> str:
    """Drawn for a phone: it is shown about 400 px wide, so marks and type are large."""
    places = numbered(brief)
    anchors = [p for p in brief.get("places") or [] if p.get("kind") == "anchor"]
    lines = brief.get("lines") or []
    pts = [(p["lat"], p["lon"]) for p in places + anchors] + [tuple(pt) for line in lines for pt in line.get("path") or []]
    lat0, lat1 = min(p[0] for p in pts), max(p[0] for p in pts)
    lon0, lon1 = min(p[1] for p in pts), max(p[1] for p in pts)
    k = math.cos(math.radians((lat0 + lat1) / 2))  # a degree of longitude is shorter than one of latitude
    span_x, span_y = max((lon1 - lon0) * k, 1e-4), max(lat1 - lat0, 1e-4)
    pad_l, pad_r, pad_t, pad_b = 50, 110, 56, 56
    scale = (width - pad_l - pad_r) / span_x
    height = int(min(max(span_y * scale + pad_t + pad_b, 360), width))
    scale = min(scale, (height - pad_t - pad_b) / span_y)
    off_x = pad_l + ((width - pad_l - pad_r) - span_x * scale) / 2
    off_y = pad_t + ((height - pad_t - pad_b) - span_y * scale) / 2

    def xy(lat: float, lon: float) -> tuple[float, float]:
        return off_x + (lon - lon0) * k * scale, off_y + (lat1 - lat) * scale

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
           f'font-family="system-ui,-apple-system,Segoe UI,sans-serif" role="img" aria-label="Map of the options">',
           f'<rect width="{width}" height="{height}" fill="#f4f1ea"/>']
    for line in lines:
        path = " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(xy(*pt) for pt in line["path"]))
        out.append(f'<path d="{path}" fill="none" stroke="{esc(line.get("color") or "#9aa3ad")}" stroke-width="3" '
                   f'stroke-linecap="round" stroke-linejoin="round" opacity="0.7"/>')
        x, y = xy(*line["path"][0])
        out.append(f'<text x="{x + 6:.1f}" y="{y - 6:.1f}" font-size="13" fill="#6b7480">{esc(str(line.get("name") or ""))}</text>')
    for a in anchors:
        x, y = xy(a["lat"], a["lon"])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="#8a8f98"/>'
                   f'<text x="{x + 8:.1f}" y="{y + 5:.1f}" font-size="14" font-style="italic" fill="#6b7480">{esc(a["name"])}</text>')
    taken: list[tuple[float, float, float, float]] = []  # label boxes already placed

    def place_label(x: float, y: float, text: str, r: float) -> tuple[float, float, str]:
        w, h = len(text) * 8.4 + 4, 19
        spots = [(x + r + 4, y + 4, "start"), (x - r - 4, y + 4, "end"), (x, y - r - 5, "middle"), (x, y + r + 14, "middle"),
                 (x + r + 4, y - 10, "start"), (x + r + 4, y + 18, "start"), (x - r - 4, y - 10, "end"), (x - r - 4, y + 18, "end")]
        for lx, ly, anchor in spots:
            left = lx if anchor == "start" else lx - w if anchor == "end" else lx - w / 2
            box = (left, ly - h + 3, left + w, ly + 3)
            inside = box[0] >= 2 and box[2] <= width - 2 and box[1] >= 2 and box[3] <= height - 2
            if inside and not any(box[0] < t[2] and t[0] < box[2] and box[1] < t[3] and t[1] < box[3] for t in taken):
                taken.append(box)
                return lx, ly, anchor
        return spots[0]

    marks = []
    for p in places:
        x, y = xy(p["lat"], p["lon"])
        r = 17 if p.get("star") else 13
        taken.append((x - r, y - r, x + r, y + r))
        marks.append((p, x, y, r))
    for p, x, y, r in reversed(marks):  # starred drawn last, on top
        if p.get("star"):
            out.append(f'<path d="{star_path(x, y, r + 3)}" fill="#f5b301" stroke="#7a5600" stroke-width="1.5"/>')
            fill = "#3b2a00"
        else:
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="#2f6fde" stroke="#fff" stroke-width="2"/>')
            fill = "#fff"
        out.append(f'<text x="{x:.1f}" y="{y + 5:.1f}" font-size="14" font-weight="700" text-anchor="middle" fill="{fill}">{p["n"]}</text>')
    for p, x, y, r in marks:  # starred labels claim their spots first
        lx, ly, anchor = place_label(x, y, p["name"], r + (3 if p.get("star") else 0))
        weight = "700" if p.get("star") else "500"
        out.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="15.5" font-weight="{weight}" text-anchor="{anchor}" fill="#1c1f24" '
                   f'stroke="#f4f1ea" stroke-width="3" paint-order="stroke">{esc(p["name"])}</text>')
    km_per_px = 111.32 / scale
    nice = next((v for v in (0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500) if v / km_per_px >= 70), 1000)
    bar = nice / km_per_px
    out.append(f'<g transform="translate(24,{height - 26})"><path d="M0,0 h{bar:.0f} M0,-5 v10 M{bar:.0f},-5 v10" stroke="#555" stroke-width="1.5" fill="none"/>'
               f'<text x="{bar / 2:.0f}" y="-8" font-size="13" text-anchor="middle" fill="#555">{nice:g} km</text></g>')
    out.append(f'<g transform="translate({width - 30},34)"><path d="M0,-16 L6,6 L0,2 L-6,6 Z" fill="#555"/>'
               f'<text y="21" font-size="13" text-anchor="middle" fill="#555">N</text></g>')
    out.append(f'<g transform="translate(24,22)"><path d="{star_path(0, -4, 8)}" fill="#f5b301" stroke="#7a5600"/>'
               f'<text x="14" font-size="14" fill="#333">recommended</text>'
               f'<circle cx="126" cy="-4" r="7" fill="#2f6fde"/><text x="139" font-size="14" fill="#333">other option</text></g>')
    out.append("</svg>")
    return "".join(out)


# ---- the page ----
CSS = """
:root{--bg:#fbfaf7;--card:#fff;--ink:#1c1f24;--muted:#667080;--line:#e3dfd6;--accent:#1f5fd0;--gold:#f5b301;--warn:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#14161a;--card:#1d2026;--ink:#eceef2;--muted:#9aa3b2;--line:#2c313a;--accent:#7fb0ff;--warn:#f0a94a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:760px;margin:0 auto;padding:16px;overflow-wrap:break-word}h1{font-size:1.35rem;margin:.2em 0}h2{font-size:1.05rem;margin:1.6em 0 .5em;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}
.now{font-size:1.15rem;font-weight:650;border-left:5px solid var(--gold);padding:.5em .8em;background:var(--card);border-radius:6px;margin:.8em 0}
.muted{color:var(--muted);font-size:.88rem}.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
.q{font-weight:700;font-size:1.08rem}.chip{display:inline-block;font-size:.78rem;padding:.1em .6em;border-radius:99px;border:1px solid var(--line);color:var(--muted);margin-left:.4em;white-space:nowrap}
.pick{font-size:1.25rem;font-weight:750;margin:.45em 0 .15em}.pick .s{color:var(--gold)}.blocked{color:var(--warn);font-weight:650}
.btns a{display:inline-block;margin:.5em .5em 0 0;padding:.5em .9em;border-radius:8px;background:var(--accent);color:#fff;text-decoration:none;font-weight:600}
.btns a.alt{background:transparent;color:var(--accent);border:1px solid var(--accent)}
table{border-collapse:collapse;width:100%;margin-top:.8em;font-size:.9rem}th,td{text-align:left;padding:.4em .5em;border-top:1px solid var(--line);vertical-align:top}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase}td:first-child,th:first-child{min-width:6.5em}tr.picked td{font-weight:650}.scroll{overflow-x:auto}
#map svg{width:100%;height:auto;border-radius:10px;border:1px solid var(--line)}#live{height:420px;border-radius:10px;border:1px solid var(--line);display:none}
ol.legend{columns:2;padding-left:1.4em;font-size:.92rem}ol.legend li{break-inside:avoid;margin:.15em 0}@media(max-width:520px){ol.legend{columns:1}}
details{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:.6em .9em;margin:.5em 0}summary{font-weight:650;cursor:pointer}
dl{display:grid;grid-template-columns:max-content 1fr;gap:.3em .9em;margin:.6em 0}dt{color:var(--muted);white-space:nowrap}dd{margin:0}a{color:var(--accent)}
"""


def render_html(brief: dict, svg: str, *, online_map: bool = True) -> str:
    places = {p["id"]: p for p in numbered(brief)}

    def name(pid: str | None, fallback: str = "") -> str:
        p = places.get(pid or "")
        return f'<a href="{esc(maps_url(p))}">{esc(p["name"])}</a> <span class="muted">#{p["n"]}</span>' if p else esc(fallback or pid or "")

    out = [f"<h1>{esc(brief['title'])}</h1>",
           f'<p class="muted">{esc(str(brief.get("updated") or ""))}</p>' if brief.get("updated") else "",
           f'<div class="now">{esc(brief["now"])}</div>', "<h2>Decide</h2>"]
    for d in brief["decisions"]:
        options = d["options"]
        picked = next((o for o in options if (o.get("place") or o.get("label")) == d.get("pick")), None)
        card = [f'<div class="card"><div class="q">{esc(d["question"])}<span class="chip">by {esc(str(d["deadline"]))}</span></div>']
        if picked:
            card.append(f'<div class="pick"><span class="s">★</span> {name(picked.get("place"), picked.get("label"))}'
                        + (f' <span class="muted">{esc(str(picked["price"]))}</span>' if picked.get("price") else "") + "</div>")
            card.append(f"<div>{esc(str(d.get('why') or ''))}</div>")
            buttons = []
            for i, act in enumerate(picked.get("actions") or []):
                href = act.get("url") or ("tel:" + "".join(c for c in str(act.get("phone") or "") if c.isdigit() or c == "+"))
                buttons.append(f'<a class="{"alt" if i else ""}" href="{esc(href)}">{esc(str(act.get("label") or "Open"))}</a>')
            if picked.get("place") in places:
                buttons.append(f'<a class="alt" href="{esc(maps_url(places[picked["place"]]))}">Map</a>')
            card.append(f'<div class="btns">{"".join(buttons)}</div>')
        else:
            card.append(f'<div class="pick blocked">Not settled: {esc(str(d.get("blocked_on")))}</div>')
        rows = "".join(
            f'<tr class="{"picked" if o is picked else ""}"><td>{name(o.get("place"), o.get("label"))}</td>'
            f'<td>{esc(str(o.get("price") or ""))}<br><span class="muted">{"checked" if o.get("verified") else "not checked"}</span></td>'
            f'<td>{esc(str(o.get("for") or ""))}</td><td>{esc(str(o.get("against") or ""))}</td></tr>' for o in options)
        card.append(f'<div class="scroll"><table><tr><th>Option</th><th>Cost</th><th>For</th><th>Against</th></tr>{rows}</table></div>')
        if d.get("unknowns"):
            card.append('<p class="muted">Not known: ' + esc("; ".join(map(str, d["unknowns"]))) + "</p>")
        out.append("".join(card) + "</div>")
    out.append('<h2>Map</h2><div id="map">' + svg + '</div><div id="live"></div>')
    out.append('<ol class="legend">' + "".join(
        f'<li>{"★ " if p.get("star") else ""}<a href="{esc(maps_url(p))}">{esc(p["name"])}</a>'
        + (f' <span class="muted">{esc(str(p["note"]))}</span>' if p.get("note") else "") + "</li>" for p in places.values()) + "</ol>")
    if brief.get("days"):
        out.append("<h2>Days</h2>")
        for i, day in enumerate(brief["days"]):
            blocks = "".join(
                f'<dt>{esc(str(b.get("when") or ""))}</dt><dd>{esc(str(b.get("what") or ""))}'
                + (f' — {name(b["place"])}' if b.get("place") else "")
                + (f'<br><span class="muted">else: {esc(str(b["fallback"]))}</span>' if b.get("fallback") else "") + "</dd>"
                for b in day.get("blocks") or [])
            out.append(f'<details{" open" if i == 0 else ""}><summary>{esc(str(day.get("date")))}'
                       + (f' — {esc(str(day["headline"]))}' if day.get("headline") else "") + f"</summary><dl>{blocks}</dl></details>")
    if brief.get("rules"):
        out.append("<h2>If things change</h2><dl>" + "".join(
            f'<dt>{esc(str(r.get("if")))}</dt><dd>{esc(str(r.get("then")))}</dd>' for r in brief["rules"]) + "</dl>")
    if brief.get("disagreements"):
        rows = "".join(
            f'<tr><td>{esc(str(x.get("topic")))}</td><td>'
            + "<br>".join(f'<b>{esc(str(p.get("source")))}:</b> {esc(str(p.get("says")))}' for p in x["positions"])
            + f'</td><td>{esc(str(x["ruling"]))}</td></tr>' for x in brief["disagreements"])
        out.append(f'<h2>Where the sources disagreed</h2><div class="scroll"><table><tr><th>Topic</th><th>They said</th><th>Ruling</th></tr>{rows}</table></div>')
    links = "".join(f'<li><a href="{esc(str(link.get("url")))}">{esc(str(link.get("label")))}</a></li>' for link in brief.get("detail_links") or [])
    sources = "".join(f'<li><b>{esc(str(s.get("name")))}</b>' + (f' — {esc(str(s["used_for"]))}' if s.get("used_for") else "") + "</li>"
                      for s in brief.get("sources") or [])
    out.append(f"<details><summary>Sources and the long version</summary><ul>{sources}</ul><ul>{links}</ul></details>")
    script = ""
    if online_map:  # a real street map when the phone is online; the drawing above stays when it is not
        pins = [{"n": p["n"], "name": p["name"], "lat": p["lat"], "lon": p["lon"], "star": bool(p.get("star")), "url": maps_url(p)}
                for p in places.values()]
        script = ('<script>(function(){var P=' + json.dumps(pins).replace("</", "<\\/") + ';var c=document.createElement("link");c.rel="stylesheet";'
                  'c.href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css";document.head.appendChild(c);var s=document.createElement("script");'
                  's.src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js";s.onload=function(){try{var el=document.getElementById("live");'
                  'el.style.display="block";var m=L.map(el);L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",'
                  '{maxZoom:19,attribution:"© OpenStreetMap"}).addTo(m);var b=[];P.forEach(function(p){var i=L.divIcon({className:"",'
                  'html:"<div style=\\"font:700 12px system-ui;text-align:center;line-height:26px;width:26px;height:26px;border-radius:50%;'
                  'border:2px solid "+(p.star?"#7a5600":"#fff")+";background:"+(p.star?"#f5b301":"#2f6fde")+";color:"+(p.star?"#3b2a00":"#fff")+'
                  '"\\">"+(p.star?"★":p.n)+"</div>",iconSize:[26,26],iconAnchor:[13,13]});var a=document.createElement("a");a.href=p.url;'
                  'a.textContent=(p.star?"★ ":"")+p.n+". "+p.name;L.marker([p.lat,p.lon],{icon:i,zIndexOffset:p.star?1000:0}).addTo(m).bindPopup(a);'
                  'b.push([p.lat,p.lon])});m.fitBounds(b,{padding:[30,30]});document.getElementById("map").style.display="none"}'
                  'catch(e){document.getElementById("live").style.display="none";document.getElementById("map").style.display=""}};'
                  'document.head.appendChild(s)})();</script>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{esc(brief['title'])}</title><style>{CSS}</style></head><body><main>{''.join(out)}</main>{script}</body></html>")


def results_fragment(brief: dict) -> dict:
    """What the standard Durian report shows first: the picks, the map, one table per decision."""
    names = {p["id"]: p for p in numbered(brief)}
    lines = [f"**Now:** {brief['now']}", ""]
    tables = []
    for d in brief["decisions"]:
        picked = next((o for o in d["options"] if (o.get("place") or o.get("label")) == d.get("pick")), None)
        label = (names.get(picked.get("place"), {}).get("name") or picked.get("label")) if picked else None
        lines.append(f"- **{d['question']}** (by {d['deadline']}): "
                     + (f"**{label}**" + (f", {picked['price']}" if picked.get("price") else "") + f". {d.get('why') or ''}" if picked
                        else f"not settled; missing: {d.get('blocked_on')}"))
        tables.append({
            "title": d["question"],
            "columns": [{"key": "rank", "label": "Rank", "definition": "1 is the recommendation"},
                        {"key": "option", "label": "Option"}, {"key": "cost", "label": "Cost"},
                        {"key": "checked", "label": "Checked", "definition": "price or availability confirmed for these dates"},
                        {"key": "for", "label": "For"}, {"key": "against", "label": "Against"}],
            "rows": [{"rank": i, "option": names.get(o.get("place"), {}).get("name") or o.get("label"), "cost": o.get("price") or "",
                      "checked": "yes" if o.get("verified") else "no", "for": o.get("for") or "", "against": o.get("against") or ""}
                     for i, o in enumerate(sorted(d["options"], key=lambda o: o is not picked), 1)]})
    lines += ["", "Open **itinerary.html** for the page with buttons, the live map and the day plan."]
    return {"headline": "\n".join(lines), "tables": tables,
            "figures": [{"path": "map.svg", "title": "Map of the options",
                         "caption": "Gold stars are the recommendations; numbers match the list in itinerary.html."}]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a decision-first itinerary page (see SKILL.md).")
    ap.add_argument("brief", help="itinerary.json (the plan, in the shape SKILL.md describes)")
    ap.add_argument("--out", required=True, help="output directory, normally artifacts/<agtask>/report")
    ap.add_argument("--results", action="store_true", help="also write headline, map figure and decision tables into results.json")
    ap.add_argument("--no-online-map", action="store_true", help="leave out the street-map upgrade (page then loads nothing from the network)")
    ap.add_argument("--check", action="store_true", help="only check the brief; write nothing")
    args = ap.parse_args()
    try:
        brief = json.loads(Path(args.brief).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {args.brief}: {exc}", file=sys.stderr)
        return 2
    bad = check(brief)
    if bad:
        print(f"itinerary.json breaks {len(bad)} rule(s); nothing was written:", file=sys.stderr)
        for line in bad:
            print("  - " + line, file=sys.stderr)
        return 1
    if args.check:
        print("itinerary.json is within the rules")
        return 0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    svg = render_map(brief)
    (out / "map.svg").write_text(svg)
    (out / "itinerary.html").write_text(render_html(brief, svg, online_map=not args.no_online_map))
    wrote = ["itinerary.html", "map.svg"]
    if args.results:
        path = out / "results.json"
        current = json.loads(path.read_text()) if path.is_file() else {}
        fragment = results_fragment(brief)
        current["headline"] = fragment["headline"]
        current["figures"] = fragment["figures"] + [f for f in current.get("figures") or [] if f.get("path") != "map.svg"]
        titles = {t["title"] for t in fragment["tables"]}
        current["tables"] = fragment["tables"] + [t for t in current.get("tables") or [] if t.get("title") not in titles]
        path.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n")
        wrote.append("results.json")
    print(f"wrote {', '.join(wrote)} to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
