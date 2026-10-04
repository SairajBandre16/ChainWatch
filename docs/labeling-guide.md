# Extraction labeling guide (owner task)

Goal: 100+ hand-labeled news items so extraction quality can be measured honestly (step 1.4).
Time needed: roughly 1 to 2 hours for 118 items.

Two ways to label: assisted (an LLM drafts, you confirm) or fully by hand in a spreadsheet. Both
write the same CSV and can be mixed. The `label_source` column records which way each item was done.

## Option A: assisted labeling (faster)

```bash
uv run python -m chainwatch.extraction.label_assist --dry-run --limit 5   # optional preview first
uv run python -m chainwatch.extraction.label_assist                       # label; Ctrl+C or q to stop
uv run python -m chainwatch.extraction.label_assist stats                 # counts per label_source
```

For each unlabeled item the tool shows the article, a draft label from qwen2.5:3b (same prompt as the
extraction pipeline), and asks:

- `a` accept the draft as is
- `e` edit: every field is asked again with the draft as default (Enter keeps, `-` clears)
- `m` manual: every field is asked with no defaults
- `s` skip, `q` quit

Nothing is written without your choice. Rows you already labeled are never touched. Progress is saved
after every item, so re-running resumes where you stopped. Close the CSV in Excel while the tool runs.

The dry run writes `data/eval/label_assist_preview.csv` (gitignored) and never changes the template;
its rows are marked `draft_unconfirmed` and are never imported as labels.

**Bias warning.** Seeing a draft pulls your answer toward it, and qwen2.5:3b is one of the models being
scored. Read the article *before* the draft, and use `e` freely. Results will be reported separately
for `accepted` vs `edited`/`manual` items (D16).

## Option B: label by hand in a spreadsheet

### Steps

1. Open `data/eval/labeling_template.csv` in Excel or LibreOffice (it is UTF-8 with BOM).
   If you want a fresh one: `uv run python -m chainwatch.extraction.eval template --n 120`.
2. For each row, read `title` and `text` (open `url` if unsure) and fill in the label columns.
3. Save as CSV (keep the same file name and columns).
4. Run `uv run python -m chainwatch.extraction.eval import-csv`. This writes `data/eval/extraction_labels.jsonl`.
5. Commit both files. Tell Claude "labels are done" and it will run the scorer for every model.

Do **not** look at model outputs in `data/processed/extractions/` while labeling by hand. That biases
the labels. Leave `label_source` and `draft_model` blank (they mean "labeled by hand").

## Columns to fill

| Column | Value |
|---|---|
| `is_disruption` | `yes` or `no`. Leave blank only for rows you skip (blank rows are ignored). |
| `event_type` | one of: `port_closure`, `port_congestion`, `canal_or_strait_blockage`, `labor_strike`, `severe_weather`, `conflict_or_attack`, `accident`, `sanctions_or_regulation`, `infrastructure_failure`, `other` |
| `location` | most specific place named: "Port of Vancouver", "Strait of Hormuz", "Red Sea" |
| `country_code` | 2-letter ISO code of the place (`IR`, `IN`, `IE`); blank for open sea |
| `port_code` | UN/LOCODE only if the place is a port (`CAVAN`, `INNSA`, `NLRTM`); else blank |
| `severity` | 1 = minor, local, hours; 2 = local, a day or two; 3 = regional, multi-day; 4 = major lane affected, weeks; 5 = global impact |
| `notes` | anything ambiguous (optional) |

Leave the event columns blank when `is_disruption` is `no`.

## What counts as a disruption

Yes: a physical event happening now or imminent that stops or slows goods movement: port closures or
congestion, strikes, attacks on ships, canal/strait blockages, storms hitting ports, accidents, sanctions
that block a trade route, cyber outages at terminals.

No: market reports, freight-rate commentary, company deals, opinion pieces, forecasts without a concrete
event, events fully resolved long ago.

Edge cases: an article about the *effects* of a known ongoing disruption (e.g. rerouting because of Red
Sea attacks) is `yes`, labeled with the underlying event and location.

## Several events in one item

Copy the row (same `news_id`) once per extra event and fill the event columns differently.
`is_disruption` must be `yes` on every copy.

## Lookup help

- UN/LOCODE search: https://unece.org/trade/cefact/unlocode-code-list-country-and-territory
- ISO country codes: https://www.iso.org/obp/ui/#search
