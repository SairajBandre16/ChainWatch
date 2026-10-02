# Resume bullet drafts

Fill `{placeholders}` only from `docs/results.md`. Filled values below are already recorded there;
`{pending: ...}` items need a run that has not happened yet.

1. Built **ChainWatch**, an end-to-end supply-chain early-warning system (Python, Ollama, NetworkX,
   LightGBM, FastAPI, Streamlit) that turns news from RSS and GDELT into lane-level disruption risk for
   India to Europe shipping, fully offline and free; {119} automated tests run offline in under a minute ({23.3} s).
2. Designed an LLM extraction pipeline with Pydantic schemas, self-repair retries and disk caching;
   compared two local models (schema failure rate {1.5%} vs {10.6%}) and built a per-field P/R/F1 harness
   ({pending: extraction F1 on 100+ hand-labeled items}).
3. Trained a late-delivery model on {180,519} orders with leak-free point-in-time features and purged
   time splits; LightGBM reached test ROC AUC {0.776} vs {0.724} for a one-rule baseline, with SHAP-based
   plain-language explanations.
4. Built a tool-calling agent with code-level guard rails (citation checks, recomputed scores, route
   feasibility) and a spec-first backtest on {164} days of GDELT news: flagged the 2023 Red Sea crisis
   {≥30} days before carrier suspensions and the 2021 Suez blockage on day {0}, and diagnosed a {0.75}
   false-alarm rate to syndicated duplicate stories ({pending: false-alarm rate after dedup run}).
