# Backtest specification

Written and committed **before** the backtest engine and before any backtest data was extracted.
Nothing in this file may be changed after seeing backtest results. Any later change must be logged in
`docs/results.md` as a new, separately labeled run, with the original run kept.

## Question

If ChainWatch had been running on historical news, how early (or how late) would it have flagged the
India to Europe trade lanes for two real disruptions, and how often would it raise false alarms in quiet
periods?

## Disruptions (ground truth from the public record)

| id | Disruption | Onset (UTC date) | Target nodes |
|---|---|---|---|
| `suez_2021` | Ever Given grounds in the Suez Canal and blocks it (refloated 2021-03-29) | 2021-03-23 | SUEZ_CANAL |
| `red_sea_2023` | Houthi attacks on merchant ships; major carriers (Maersk, Hapag-Lloyd) suspend Red Sea transits | 2023-12-15 | RED_SEA, BAB_EL_MANDEB, GULF_OF_ADEN |

**Disruption onset** is the first UTC date on which normal India to Europe container traffic through the
target nodes was physically stopped or diverted by carriers: the grounding day for Suez, the first major
carrier suspension for the Red Sea. For `red_sea_2023` we also report lead time against a secondary
reference date, the first attack on merchant shipping (Galaxy Leader seized, **2023-11-19**), labeled as
such; the primary metric uses 2023-12-15.

## Watched lanes

Primary lane: `INNSA-NLRTM` (Nhava Sheva to Rotterdam). Secondary: every lane with focus `india_europe` or
`india_ireland` in `data/processed/lanes.csv` (reported as a group).

## System configuration (frozen)

- News: GDELT 1.0 daily event files, logistics URL filter from `ingest.sources` as of this commit.
  Published date = the file's date.
- Extractors: (a) keyword baseline `extraction.baseline` as of this commit; (b) optionally `qwen2.5:3b` with
  prompt `extract_v1`, run on the same items. No prompt or rule changes for the backtest.
- Linking and exposure: `graph.linking` and `TradeGraph.lane_exposure_score` as of this commit.
- **Lookback:** a day's signal uses events published in the 7 days ending that day (inclusive).
- **Flag threshold:** τ = 0.5 on the lane exposure score.

## Definitions

- **Daily signal** for lane L on day d: the exposure score of L from events visible at the end of d.
- **Flag:** lane L is flagged on day d if its daily signal is ≥ τ.
- **On-target flag** for disruption D: a flag where at least one contributing event is linked to one of D's
  target nodes.
- **Evaluation window** for D: from onset − 30 days to onset + 7 days (inclusive).
- **First warning date:** the first day in D's window with an on-target flag on the primary lane.
- **Lead time** = onset − first warning date, in days. Positive = warned before onset. Zero = flagged on the
  onset day. Negative = flagged after onset (a detection delay). No on-target flag in the window = **missed**.
- **Control windows** (no disruption closing or diverting India to Europe traffic for a day or more is known):
  2019-09-01 to 2019-09-30 and 2022-09-01 to 2022-09-30.
- **False-alarm rate** = flagged days / total days, over the control windows, on the primary lane.
  Any flag in a control window is a false alarm, even if triggered by a real but minor incident.
  We also report the number of false-alarm episodes (runs of consecutive flagged days).
- **Pre-onset off-target flags:** flags in an evaluation window before onset whose contributing events are
  not on target nodes; reported, not counted in lead time.

## Rules

- τ, lookback, windows, onsets, target nodes and lanes are fixed above. No tuning on backtest events.
- Every number reported must come from `python -m chainwatch.backtest.run` and be logged in `docs/results.md`.
- GDELT days that fail to download are listed in the results, not silently skipped.

## Known limitations (stated in advance)

- GDELT event files carry URLs, not headlines; the URL slug is used as the headline text.
- The keyword baseline only knows a small gazetteer; it can miss events phrased without a known place.
- Two disruptions are a tiny sample. Results illustrate behavior; they are not statistically strong.
