---
name: plan-itinerary
description: Plan an itinerary of any kind (a trip, a day out, a conference, a house-hunting tour, a run of errands or site visits) and deliver it as a short decision-first page with a starred map. Use whenever the operator asks for a plan, schedule or itinerary that involves places and choices such as where to stay, where to eat or which route to take.
tools: [fetch_url, read_file, write_file, shell, agtask_list, agtask_show, memory_search]
requires: [bin:python3]
---
# Plan an itinerary

The reader is standing somewhere with a phone and has to choose. The page you
deliver must let them make each choice in under a minute. Research can be as
deep as the task needs; the page cannot be long.

Written 2026-11-15 after a trip pack came back at 30,000 words with 62
headings, no map, tonight's hotel buried among 45 activity blocks, and a
second model's pack on the same trip left unread.

## What you deliver

1. `itinerary.json`, the whole plan in the shape below.
2. `itinerary.html` and `map.svg`, built from it by the script in this folder.
   Never write the page by hand.
3. The long material (every option you researched, sources, block libraries)
   as separate files, linked from `detail_links`. It is never pasted into the page.

```bash
python3 .durian/skills/plan-itinerary/build_itinerary.py itinerary.json \
    --out artifacts/<agtask_id>/report --results
```

`--results` writes the picks, the map and one table per decision into
`results.json`, so the standard Durian report opens with them. The script
refuses to build when a rule below is broken and says which. Fix the JSON.

## The order of the page

| Level | What it holds | Limit |
|---|---|---|
| 1. Now | The next move, one sentence | 40 words |
| 2. Decide | Each choice the reader must make: your pick, the reason, the deadline, the alternatives | 5 decisions, 2 to 4 options each, 45 words of reason |
| 3. Map | Every option as a numbered pin; what you recommend is a gold star | every place needs real coordinates |
| 4. Days | Time blocks, one line each, with a fallback | 30 words a block |
| 5. If things change | Rain, a tired child, a cancelled train, two spare hours | 30 words a rule |
| 6. Sources | What this was built from and where the sources disagreed | |

Whole page: 1,800 words. Everything else is a link.

## Rules

**Recommend.** Every decision has a `pick`. If one missing fact stops you
(tonight's room rate), set `pick` to null and name that fact in `blocked_on`.
Four equally weighted plans is not an answer. If you built alternatives, the
page carries one and links the others.

**Put the urgent decision first.** Order decisions by deadline. "Where do we
sleep tonight" comes before anything about Wednesday.

**Say what you checked.** Each option has `verified`: true only if you
confirmed the price or availability for these dates. An estimate is written as
an estimate ("$140 to 220, estimate").

**Every decision needs an action.** The pick carries `actions`: a booking
link, a phone number. The reader should be able to act from the card.

**Coordinates are looked up.** Take latitude and longitude from a geocoder or
the venue's map listing. Do not estimate them: a pin in the wrong place is
worse than no pin. If the places are spread out, add two or three `anchor`
places (the airport, the city centre) so the map reads at a glance.

**Use everything already written.** Before planning, look for other work on
the same request: sibling tasks with the same title (`agtask_list`), messages
from other agents, earlier packs in the workspace, KB pages
(`memory_search`). List every one in `sources`. With more than one source you
must fill `disagreements`: the topic, what each said, your ruling. Do not
silently pick one. An empty list means you compared them and they agree.

**Constraints come from the operator.** Carry the fixed points into the plan
and do not relax them: flight times, who is travelling, mobility limits,
budget, things already booked. If one is unknown and changes the plan, ask.

## itinerary.json

```json
{
  "title": "Lisbon, 3 days, two adults and a toddler",
  "updated": "2026-11-15 13:40, before booking",
  "now": "Book the Alfama flat before 18:00; the rate is held until then.",
  "decisions": [
    {
      "question": "Where do we stay?",
      "deadline": "today 18:00",
      "pick": "alfama",
      "why": "Ten minutes' flat walk to the tram and the river, a lift, and a kitchen for early dinners.",
      "options": [
        {"place": "alfama", "price": "EUR 140 a night", "verified": true,
         "for": "Lift, kitchen, flat walk to the tram", "against": "Street noise until 23:00",
         "actions": [{"label": "Book", "url": "https://example.com/book"}, {"label": "Call", "phone": "+351 21 000 0000"}]},
        {"place": "chiado", "price": "EUR 165 to 190, estimate", "verified": false,
         "for": "Central, quiet", "against": "Steep hill with a stroller"}
      ],
      "unknowns": ["Whether the Alfama flat has a cot"]
    }
  ],
  "places": [
    {"id": "alfama", "name": "Alfama flat", "kind": "stay", "lat": 38.7119, "lon": -9.1296, "star": true, "note": "lift, kitchen"},
    {"id": "chiado", "name": "Chiado hotel", "kind": "stay", "lat": 38.7106, "lon": -9.1421},
    {"id": "airport", "name": "Airport", "kind": "anchor", "lat": 38.7742, "lon": -9.1342}
  ],
  "days": [
    {"date": "Mon 16", "headline": "River and old town", "blocks": [
      {"when": "09:00 to 12:00", "what": "Tram 28 loop, then the castle garden", "place": "alfama", "fallback": "Oceanarium if it rains"}
    ]}
  ],
  "rules": [{"if": "Raining", "then": "Oceanarium, then lunch at the market hall"}],
  "sources": [{"name": "Task agt_aaa (Claude)", "used_for": "stays, transit"}, {"name": "Task agt_bbb (Codex)", "used_for": "food, day plan"}],
  "disagreements": [
    {"topic": "Where to stay", "positions": [{"source": "Claude", "says": "Alfama, for the lift"}, {"source": "Codex", "says": "Chiado, for quiet"}],
     "ruling": "Alfama: the hill rules Chiado out with a stroller"}
  ],
  "detail_links": [{"label": "All 14 stays compared", "url": "STAYS.md"}]
}
```

Optional: `lines` draws routes on the map
(`[{"name": "Tram 28", "color": "#c9a227", "path": [[38.71, -9.13], [38.72, -9.14]]}]`),
a place may carry `address` (used for its map link) or its own `url`.

## Before you report

- `build_itinerary.py --check itinerary.json` passes.
- Open `itinerary.html` and read only the first screen. Could the operator act
  on it without scrolling? If not, the pick or the action is missing.
- Each starred pin is where the place really is.
- The handoff says which options were checked for the dates and which were not.

The page shows a street map with the same pins when the phone is online and
keeps the drawn map when it is not. Pass `--no-online-map` if the page must
load nothing from the network.
