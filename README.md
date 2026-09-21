# Ontario Grid Brain: Demand & Price Forecasting, Risk Classification, and Battery Arbitrage Optimization

## Summary

An end-to-end data science pipeline built on 23 years (2002–2025) of real Ontario electricity grid data, covering the full range of classical data science techniques: time-series forecasting, classification, optimization, and Monte Carlo simulation. The project forecasts electricity demand and price (comparing XGBoost and LSTM model performance), classifies hours by demand risk, optimizes a grid-battery arbitrage strategy against those forecasts, and rigorously stress-tests the resulting financial outcome against real-world uncertainty, including validating (and disproving at times) its own risk model.

**Headline results:**
- Demand forecasting: **3.0–3.3% MAPE** (XGBoost), beating an LSTM baseline (4.4–5.6% MAPE)
- Price forecasting: **19–26% MAE improvement** over a naive persistence baseline
- Critical-demand-hour classification: recall improved from **60–76% to 75–88%** via validated threshold tuning, while avoiding a simpler alternative's precision collapse to 32%
- Battery arbitrage optimization: real backtested dollar figures across three years, plus a rigorously diagnosed and honestly-reported failure mode
- Monte Carlo risk simulation: validated against real outcomes, revealing (and correctly diagnosing) a genuine, well-evidenced limitation of resampling-based risk methods for this domain

---

## Motivation

While visiting rural southern Ontario, I was struck by how much infrastructure was quietly doing its job in the background. There were wind turbines dotting fields, transmission lines stretching across long stretches of nothing, all built to serve places with a handful of homes just as reliably as it serves a city of millions. Canada is enormous, and the idea that one interconnected system has to serve both a farmhouse and downtown Toronto, at the same standard of reliability, all the time, stuck with me.

One of the nights during that trip, the power went out without a storm or obvious cause. It got me thinking about something I'd genuinely never considered before: electricity is a business, with real costs, real supply constraints, and real decisions being made every hour about how much to generate and at what price — but it's also invisible infrastructure in the truest sense. Nobody thinks about the grid when it works. The moment it doesn't, it's the only thing anyone can think about.

That curiosity turned into this project: an attempt to actually engage with the forecasting and decision-making problems a real grid participant faces; predicting demand, predicting price, and making a real economic decision (battery arbitrage) under real uncertainty. The simulation piece in particular was intentional as I specifically wanted hands-on experience building something that tests its own assumptions statistically against real-world variance, rather than trusting a single backtested result at face value.

---

## Tech Stack

pandas, NumPy, scikit-learn (RandomizedSearchCV, TimeSeriesSplit, permutation importance, StandardScaler), XGBoost, TensorFlow/Keras (LSTM), PuLP (linear programming), joblib.

---

## Data Sources

| Source | What | Range |
|---|---|---|
| IESO (Independent Electricity System Operator) | Hourly Ontario Demand | 2002–2025 |
| IESO | Hourly Ontario Energy Price (HOEP) | 2002–Apr 30, 2025 (retired; replaced by a two-part Day-Ahead + Load Forecast Deviation Adjustment structure under IESO's Market Renewal Program) |
| Environment Canada | Hourly temperature, Toronto Pearson Airport | 2002–2025 (two stations: #5097 through June 13, 2013; #51459 thereafter, following a real station handover) |

### Real data engineering challenges solved
- **A mid-project market structure change**: HOEP was retired May 1, 2025, replaced by a two-part pricing mechanism. Resolved by scoping the analysis to the well-documented legacy HOEP era (2002–Apr 2025) rather than stitching together a short-lived, retention-limited new data source (IESO's public reports enforce a 90-day rolling retention window on the newer report type, discovered empirically).
- **A silent weather station hardware replacement** (station 5097 → 51459, June 2013) — identified during initial data cleaning, then confirmed and resolved by querying both stations across the transition month.
- **Missing values** (53 temperature, 8 price hours) handled via a two-tier strategy: short gaps (≤3 hours) linearly interpolated; longer gaps filled with a climatological normal (same month/day/hour, averaged across all other years), a documented trade-off that smooths over any coincidental extreme-weather hours in exchange for a complete dataset.
- **Rate limiting / connection instability** from a government data portal, resolved with exponential backoff, jitter, and an automatic in-run retry pass.

---

## Exploratory Findings

- **Surplus Baseload Generation (SBG)**: HOEP legitimately sits at $0 for a meaningful share of hours (mostly overnight and in shoulder seasons), when abundant nuclear/hydro/wind output exceeds demand. Confirmed directly against real temperature and time-of-day data rather than assumed.
- **2002 price volatility**: The four most extreme historical price spikes in the dataset all fall in Aug–Sep 2002, consistent with Ontario's well-documented market-opening volatility that led to a government-imposed price cap by that November.

---

## Phase-by-Phase Results

### 1. Demand Forecasting
Two architectures were deliberately given *different* input representations to make the comparison meaningful: XGBoost received hand-engineered lag/rolling/calendar features (validated via autocorrelation analysis and permutation importance, not chosen by feel), while an LSTM received raw 168-hour sequences of demand and temperature, left to discover temporal structure by itself.

| Year | XGBoost MAPE | LSTM MAPE |
|---|---|---|
| 2023 | 3.09% | 4.40% |
| 2024 | 3.30% | 4.56% |
| 2025 (partial) | 3.05% | 4.05% |

**Finding**: hand-engineered features captured most of the exploitable temporal structure. XGBoost consistently outperformed the LSTM, though the comparison is disclosed as imperfect (XGBoost received a full hyperparameter search while the LSTM used reasonable defaults).

### 2. Price Forecasting
Reused the same feature scaffold, with the demand model's own forecast added as a new input feature (demand is one of the two real forces setting marginal price, alongside supply-side conditions).

| Year | Model MAE | Naive (t-24h) MAE | Improvement |
|---|---|---|---|
| 2023 | $8.39/MWh | $10.35/MWh | 19% |
| 2024 | $9.88/MWh | $12.75/MWh | 22% |
| 2025 (partial) | $20.61/MWh | $27.86/MWh | 26% |

**A key diagnostic finding**: raw MAPE looked alarming (66–90%), but investigation revealed this was concentrated almost entirely in the lowest price quartile (106–205% MAPE there vs. 18–34% everywhere else), a metric artifact caused by near-zero SBG-era prices, not genuine model failure. Confirmed by the naive-baseline comparison above and an explicit price-bucket breakdown.

### 3. Classification — Demand Risk Regimes
Hours classified into Low/Normal/Peak/Critical using training-data-only percentile thresholds (25th/75th/95th). Three approaches compared on Critical-class recall specifically, since a missed real Critical hour is operationally more costly than a false alarm:

| Approach | Critical Recall (2023/24/25) | Critical Precision (2023/24/25) |
|---|---|---|
| XGBoost, default threshold | 0.76 / 0.63 / 0.60 | 0.59 / 0.72 / 0.71 |
| Logistic Regression | 0.90 / 0.78 / 0.95 | 0.57 / 0.60 / **0.32** |
| XGBoost, tuned threshold (0.301, chosen on validation data only) | 0.88 / 0.76 / 0.75 | 0.48 / 0.65 / 0.58 |

**Finding**: the tuned-threshold model delivers most of Logistic Regression's recall improvement without its precision collapse in 2025 (a real "alarm fatigue" risk in a deployed system) — the best-balanced option for the problem at hand, not just defaulting to whichever model had the highest raw accuracy.

### 4. Battery Dispatch Optimization
A linear program (PuLP) decides hourly charge/discharge for a 100 MWh / 25 MW / 90%-efficiency battery, verified against synthetic test cases before running on real data.

| Year | Forecast-driven | Naive fixed schedule | Perfect hindsight | % of max captured |
|---|---|---|---|---|
| 2023 | $370,114 | $527,527 | $982,021 | 37.7% |
| 2024 | $481,496 | $644,142 | $1,262,865 | 38.1% |
| 2025 (partial) | $325,017 | $261,760 | $878,284 | 37.0% |

**Finding**: the forecast-driven policy underperformed a naive fixed-time-of-day schedule on 63–73% of days, and lost money outright on 18–30% of days. Diagnosis: price forecast error doesn't just cost missed upside, it can actively cause wrong-direction trades (discharging when actually cheap, charging when actually expensive), a materially different and more costly failure mode than plain forecast inaccuracy. A proposed fix (a "confidence deadband," only trading when forecasted price spread is large enough to trust) was tested rigorously on validation data and did not hold up.

### 5. Monte Carlo Risk Simulation
A block-bootstrap simulation (resampling whole historical days of price forecast error, preserving within-day correlation) quantified how much the battery's profit could plausibly have varied given forecast uncertainty.

**Finding — validated, not just trusted**: the actual realized outcome fell at the 0th–8th percentile of the simulated distribution across all three years — meaning real outcomes were worse than nearly every simulated scenario. Two independent explanations were tested and ruled out systematically:
1. *Validation-year volatility mismatch* — ruled out; 2023's actual forecast-error volatility was smaller than 2022's, not larger.
2. *Insufficient historical sample size* — ruled out; expanding to a 20-year, model-independent outlook made calibration *worse*, not better, because the rolling-baseline method used to build it inadvertently preserved normal daily price-cycle shape, which reinforced (rather than randomized) the battery's existing schedule.

**Conclusion**: real damaging days are driven by idiosyncratic, correlated supply-side shocks (unplanned outages, extreme weather) that generic historical resampling in any form tested cannot reproduce. This is a known, documented limitation of block-bootstrap methods for risks driven by rare structural events rather than generic statistical noise, not a fixable implementation bug.

---
