# Financial NLP for Market Signal Modeling and Risk-Aware Portfolio Allocation

An end-to-end machine learning project that explores how **financial news text can be converted into market signals and downstream portfolio decisions**.

The project combines financial NLP, equity return data, signal evaluation, portfolio construction, transaction-cost modeling, and constraint-aware risk controls. The goal is not simply to ask whether a language model classifies news sentiment well, but whether its outputs remain useful once translated into realistic investment decisions.

---

## Project Overview

The pipeline follows five stages:

1. **Financial data preparation**
   - Clean and align financial news with ticker-level market data.
   - Construct forward equity returns for multiple prediction horizons.

2. **Financial NLP signal generation**
   - Apply **FinBERT** to financial news.
   - Extract positive, negative, and neutral sentiment probabilities.
   - Convert model outputs into a continuous ticker-level sentiment signal.

3. **Signal evaluation**
   - Directional accuracy
   - Cross-sectional rank information coefficient (Rank IC)
   - Signal-bucket return analysis
   - Top-minus-bottom return spreads
   - 1-day and 5-day forward-return horizons

4. **Portfolio construction**
   - Build daily long-short portfolios from the strongest positive and negative signals.
   - Compare a direct signal-weighted allocator with a **constraint-aware repaired allocator**.

5. **Realistic portfolio evaluation**
   - Position-size constraints
   - Gross-exposure normalization
   - Turnover controls
   - Transaction costs
   - Return, volatility, Sharpe ratio, drawdown, turnover, concentration, and constraint-violation analysis

---

## Why This Project?

A predictive NLP signal is not automatically a useful financial strategy.

A model can appear attractive in isolation but become fragile once realistic portfolio considerations are introduced. This project therefore evaluates the **full ML-to-decision pipeline**:

> Financial News → NLP Signal → Predictive Evaluation → Portfolio Allocation → Risk Constraints → Transaction Costs → Performance

This makes the project an applied study of how machine-learning outputs interact with real-world financial decision constraints.

---

## Data

The FinBERT experiment uses a processed panel containing:

- **1,487 ticker-date observations**
- Financial news text
- Daily stock prices and returns
- 1-day forward returns
- 5-day forward returns
- Rolling volatility features

The FinBERT signal is computed from model-implied sentiment probabilities for each news observation.

---

## NLP Model

The project uses **FinBERT**, a transformer model designed for financial-language sentiment analysis.

For each observation, the model produces probabilities for:

- Positive
- Negative
- Neutral

These probabilities are transformed into a continuous financial-news sentiment signal used for downstream analysis.

---

## Signal Evaluation

The NLP signal is evaluated independently of portfolio performance.

| Horizon | Directional Accuracy | Mean Rank IC | Top − Bottom Return |
|---|---:|---:|---:|
| 1 Day | 46.9% | 0.017 | -0.71% |
| 5 Day | 53.1% | 0.024 | 2.51% |

The raw predictive relationship is modest, which motivates evaluating whether portfolio construction and risk controls materially affect downstream outcomes.

---

## Portfolio Construction

A daily long-short portfolio is formed using the strongest FinBERT signals.

The primary experiment uses:

- Top 5 positive-signal assets
- Bottom 5 negative-signal assets
- Gross exposure = 1.0

Two allocation approaches are compared.

### Naive Signal-Weighted Allocator

Directly converts signal strength into portfolio weights.

### Constraint-Aware Allocator

Repairs the initial allocation by applying:

- Maximum absolute position size: **10%**
- Gross-exposure renormalization
- Maximum portfolio turnover: **0.50**

The repaired portfolio is then checked again for constraint violations before backtesting.

---

## Main Results

### 1-Day Horizon, 5 bps Transaction Cost

| Metric | Naive | Constraint-Aware |
|---|---:|---:|
| Annualized Return | 6.23% | **8.90%** |
| Annualized Volatility | 19.14% | **7.95%** |
| Sharpe Ratio | 0.41 | **1.11** |
| Maximum Drawdown | -17.40% | **-3.31%** |
| Average Turnover | 1.53 | **0.50** |
| Constraint Violation Rate | 100% | **0%** |

The constraint-aware portfolio materially reduced volatility, drawdown, turnover, and constraint violations while improving net risk-adjusted performance.

### 1-Day Horizon, Zero Transaction Cost

| Metric | Naive | Constraint-Aware |
|---|---:|---:|
| Annualized Return | **28.81%** | 16.05% |
| Annualized Volatility | 19.15% | **7.94%** |
| Sharpe Ratio | 1.42 | **1.91** |
| Maximum Drawdown | -14.55% | **-2.77%** |

Without transaction costs, the naive strategy earns a higher raw return because it trades much more aggressively. The repaired portfolio nevertheless achieves substantially better risk-adjusted performance.

---

## Robustness: 5-Day Horizon

The 5-day portfolio experiment is substantially weaker.

At 5 bps transaction costs, both naive and repaired strategies produced negative annualized performance. This result is intentionally retained because it highlights an important research lesson:

> A financial ML signal should not be judged from a single favorable horizon or backtest.

The project therefore reports both positive and negative findings rather than selecting only the strongest result.

---

## Key Takeaways

- Financial NLP signals should be evaluated **downstream**, not only through text-classification metrics.
- Weak-to-moderate predictive signals can behave very differently once converted into portfolios.
- Turnover and position concentration can make frictionless backtests misleading.
- Risk constraints can materially improve portfolio stability.
- Transaction costs can reverse conclusions drawn from zero-cost simulations.
- Negative results across alternative horizons are an important part of robust financial ML evaluation.

---

## Tech Stack

- **Python**
- **PyTorch / Hugging Face Transformers**
- **FinBERT**
- **pandas / NumPy**
- **Matplotlib**
- Modular backtesting and risk-control utilities
- Jupyter notebooks for research and analysis

---

## Example Workflow

```python
# 1. Load aligned news and market data
news_panel = pd.read_csv("data/processed/news_price_panel.csv")

# 2. Generate FinBERT sentiment signals
news_finbert = add_finbert_signal(
    news_panel,
    text_col="text"
)

# 3. Construct signal-weighted long-short portfolio
weights = make_signal_weighted_long_short_weights(
    signal_df=news_finbert,
    signal_col="signal_finbert",
    top_k=5,
    bottom_k=5
)

# 4. Apply portfolio constraints
weights_repaired = clip_and_renormalize_weights(
    weights,
    max_abs_weight=0.10
)

weights_repaired = apply_turnover_limit(
    weights_repaired,
    max_turnover=0.50
)

# 5. Backtest and apply transaction costs
returns = run_backtest(
    weights=weights_repaired,
    panel=news_finbert,
    forward_return_col="fwd_ret_1d"
)
```

---

## Evaluation Metrics

The project reports metrics across both the ML signal and the resulting portfolio.

**Signal-level**
- Directional accuracy
- Mean and median Rank IC
- IC positive rate
- Bucket returns
- Top-minus-bottom spread

**Portfolio-level**
- Annualized return
- Annualized volatility
- Sharpe ratio
- Maximum drawdown
- Average turnover
- Maximum position concentration
- Constraint-violation rate

---

## Limitations

This is a research project rather than a production trading system.

Important limitations include:

- Relatively small financial-news sample
- Limited historical period
- Dependence on a pretrained sentiment model
- Modest raw signal strength
- Weak 5-day portfolio results
- Simplified transaction-cost assumptions
- Simplified portfolio and risk constraints
- No claim that historical backtest performance will persist out of sample

These limitations are useful extensions for future work rather than hidden assumptions.

---

## Possible Extensions

- Walk-forward / rolling out-of-sample evaluation
- Larger and more diverse financial-news datasets
- Alternative financial language models
- Fine-tuning versus frozen pretrained embeddings
- Sector and market-neutral portfolio construction
- Volatility-scaled position sizing
- More realistic nonlinear transaction-cost models
- Temporal leakage tests
- Regime-specific performance analysis
- Ablation studies across signal and portfolio components

---

## Disclaimer

This repository is for **educational and research purposes only**. The results are historical simulations and should not be interpreted as investment advice or evidence of future trading performance.
