# Backtest v2 scoring (post-hoc, pre-registered)

Written and committed **before** running v2. This is a post-hoc change: it is informed by the v1
diagnosis in `docs/results.md` (R6), so its results are reported as a separate, clearly labeled run
(R6b). The v1 spec (`docs/backtest-spec.md`) and the R6 numbers are kept unchanged.

## What changes

Only how the daily lane signal combines events. Everything else is identical to v1: GDELT data,
extractor, linking, onsets, target nodes, windows, lanes, τ = 0.5, 7-day lookback, metric definitions.

1. **Drop country-level matches from the flag signal.** A hit whose node is a country (`CTRY:XX`) is
   ignored. Reason: a country name plus an alarm keyword says nothing about a specific port or route
   (R6: all 45 false-alarm days had a `CTRY:IN` contribution). Country hits still appear in briefs.
2. **Count each story once.** Hits that share the same publication date, event type and graph node are
   treated as one story and contribute only their highest weight. Reason: one incident reported under
   dozens of URLs is one piece of evidence, not dozens (R6 example: 2022-09-02 sea-drone story).

Then score = 1 - prod(1 - w) over the remaining story-level weights, as in v1.

## Rules

- These two rules are fixed here, with no parameters to tune. v2 is run **once**; whatever it gives is
  reported, including if lead times get worse or a disruption is missed.
- No further changes after seeing v2 results without a new pre-registered file and a new run label.
- Caveat stated in advance: because the rules were chosen after looking at v1 control-window flags, the
  v2 false-alarm rate is optimistic. A fair test needs new control windows not inspected before.
