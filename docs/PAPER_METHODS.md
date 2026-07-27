# Paper methods: implementations and Polymarket adaptations

This file records exactly what was implemented from each paper and
where this pipeline deliberately adapts, so results are never
misattributed to the original methods.

## Towards Systematic Intraday News Screening (arXiv 2304.05115)

**Implemented faithfully**

* Five-minute liquidity vectors x_t = (spread-in-ticks, turnover,
  volatility, best-level book size) — `liquidity_bars`
  (`build-liquidity-bars`).  Best-level size is the paper's book-size
  variable; total depth is stored separately and never substituted.
* The K=2 jump model, eq. 3.1: `min Σ‖x_t − θ_{m_t}‖² + λ Σ 1{m_t ≠
  m_{t+1}}`, solved by alternating centroid updates and
  dynamic-programming mode assignment (`fit-liquidity-modes`).  DP
  optimality is tested against brute force.  The lower-volatility mode
  is labelled calm, the other event, matching the paper's Mode 1 /
  Mode 2 convention.
* The impact screen, eq. 4.1: news arriving in bin t is impactful iff
  a calm→event transition occurs at the boundary immediately before or
  after the arrival bin (`screen-news-impact`).

**Adaptations (ours, versioned in `liquidity_mode_runs`)**

* *Stationarization.*  The paper uses per-time-of-day location/IQR
  over ~750 equity trading days.  Prediction markets trade
  continuously with far shorter histories, so reference cells are
  (condition, UTC hour), median/IQR of log(x+eps) fitted on TRAINING
  bars only, with per-condition global fallback for thin cells and an
  IQR floor — both recorded in the persisted reference stats.
* *Volatility.*  The paper uses the uncertainty-zones estimator; we
  use realized log-odds variance from book-mid observations (prices
  live on (0,1); returns are computed on the logit).
* *Lambda selection.*  Chosen on training data only as the smallest
  candidate reaching a minimum mean mode duration (default 3 bins),
  mirroring the paper's persistence trade-off; a fixed λ can be
  supplied.  Deterministic volatility-split initialization replaces
  random restarts so refits are reproducible.
* *Prices vs sentiment.*  The paper retrains a Naive Bayes sentiment
  classifier on the screened sample.  Here the screen decides IMPACT
  (market reaction) while LLM relevance scoring decides SEMANTICS
  (relevance/direction); the two are never conflated.  A faithful
  headline-classifier replication benchmark remains future work.

**Strict availability and the two decoders.**  The full-sequence DP
backtracks from the end of the segment, so a smoothed mode at t can
depend on later bars — those assignments (`mode_label`, basis
`retrospective_smoothed`) are for paper replication and offline
analysis only.  The ONLINE decoder (`mode_label_online`, basis
`online_filtered`) is the forward pass: the mode at t uses
observations only through t, verified by a prefix-invariance test, so
`screen_available_at = arrival_bin_end + bin_seconds` is a true claim
only for online-basis screens, and only those may feed live DRC
(`impactful_news_asof` filters to them by default).  Both bases are
persisted side by side.

**Screening unit.**  The claim (article-level release) x market, as in
the paper's per-release screening; later confirmations and corrections
in the same event family are screened independently, with the family
id carried for downstream aggregation.

**Sparse turnover.**  Five-minute prediction-market bars legitimately
have zero executions, which collapses turnover IQRs and would let a
lone trade dominate every distance.  Turnover therefore decomposes
into PRESENCE (a bounded binary dimension) and LEVEL GIVEN PRESENCE
(log1p, reference stats fitted on positive-turnover bars only; absent
turnover contributes level z = 0), with per-variable minimum scale
floors and +/-10 winsorization — all recorded in the persisted
reference stats.  Reference-cell fallback is cell -> per-condition ->
POOLED, so markets absent from training still standardize and receive
assignments (transfer is tested).  Mode-run ids are SHA-256 over the
full reference statistics, configuration, model version and ordered
training-bar identity.

Bars form a regular grid with explicit incomplete bins; incomplete
bars break DP chains instead of being interpolated.

## Pervasive Underreaction

**Implemented** (`underreaction-analysis`):

* 15-minute log-odds interval returns from complete book-mid bars
  (contiguous pairs only — gaps break, never bridge);
* the news / non-news decomposition under two specifications: any
  relevant claim in the interval (``all_relevant``) and only
  liquidity-screen-impactful claims (``screened_impactful``,
  basis-aware), plus fixed-UTC-day aggregation;
* interval-level local projections of future log-odds changes on
  r_news and r_nonnews with market fixed effects, controls
  (probability level, spread, log1p turnover, trailing volatility) and
  CR1 cluster-robust standard errors by market and by UTC day — the
  paper's central finding corresponds to beta_news > 0 with the
  non-news loading far smaller;
* a permutation placebo (news labels shuffled within market) that must
  collapse the news loading;
* event-level initial response vs later drift, same-direction
  continuation, and the absorption fraction A_e = |initial| /
  (|initial| + |drift|) — explicitly OUR adaptation, labelled on every
  record; missing closes yield missing outcomes, never zeros (drift
  endpoints require a fresh close near the target time);
* distraction proxies computable from Polymarket data alone
  (cross-market claim volume, unrelated active families, weekend
  timing, optional event-mode prevalence) and the interaction
  regression testing whether drift after news strengthens under high
  distraction.

**Adaptations, marked as such:** interval-level local projections
replace the paper's daily aggregation (the daily view is also
reported); A_e is ours.  **Untested, not replicated:** the
analyst-revision mechanism — it requires an external expectations
series the pipeline does not have; the report says so explicitly.

**Model-availability discipline (screens):** an online-basis screen
additionally requires the mode model to have EXISTED before the news
(``news_time >= fit_cutoff``; violations are ``model_unavailable``),
and single-boundary non-transitions are ``partial_coverage`` rather
than reliable negatives — only two clear boundaries certify
non-impact.  Mode-run ids hash the full training data (per-bar values
and feature versions) and the fitted centroids.  Pooled-fallback
transfer to unseen markets is implemented but scientifically
unvalidated across categories; stratified reference cells are future
work before transfer results are used in claims.

## Relevance availability policies

`relevance_snapshot_asof(availability_policy=...)`:

* `online_scored` (default): text available AND scorer actually run
  before the cutoff — the only defensible policy for live DRC claims.
* `retrospective_source`: text availability alone — for clearly
  labelled backtests/replications with frozen scorers; `run-analysis`
  replays use this and record it in the run config and context
  coverage.
