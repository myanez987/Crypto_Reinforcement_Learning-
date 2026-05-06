[README_Crypto_RL.md](https://github.com/user-attachments/files/27426252/README_Crypto_RL.md)
# 📈 Crypto Reinforcement Learning Trading Agent

A reinforcement learning agent that learns to trade across 25 cryptocurrencies simultaneously using a custom [Gymnasium](https://gymnasium.farama.org/) environment and the **Proximal Policy Optimization (PPO)** algorithm via `stable-baselines3`.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/myanez987/Crypto_Reinforcement_Learning-/blob/main/Crypto_Reinforcement_Trading.ipynb)

---

## Overview

This project builds a self-learning trading agent that interacts with a simulated cryptocurrency market. At each timestep the agent observes prices, its current cash balance, and its holdings across all assets, then outputs a continuous buy/sell/hold decision for each coin. The agent is rewarded based on the **change in total portfolio value**, incentivizing it to grow wealth over time.

Key design choices:
- **Custom trading environment** built on Gymnasium's `gym.Env` interface
- **Continuous action space** — the agent controls the *fraction* of balance to buy or fraction of holdings to sell, not just discrete actions
- **Transaction costs** modeled as a 0.1% fee per trade
- **PPO algorithm** for stable, sample-efficient policy gradient training

---

## Environment: `CryptoTradingEnv`

| Parameter | Value | Description |
|---|---|---|
| `initial_balance` | `$100,000` | Starting cash |
| `transaction_fee` | `0.001` (0.1%) | Applied to every buy and sell |
| `n_cryptos` | `25` | Number of assets traded simultaneously |

### Observation Space

A flat vector of shape `(2 × n_cryptos + 1,)` containing:

```
[ price_0, ..., price_24,  balance,  holdings_0, ..., holdings_24 ]
```

All values are continuous, lower-bounded at 0.

### Action Space

A continuous vector of shape `(n_cryptos,)` with values in `[-1, 1]`:

| Value | Behavior |
|---|---|
| `> 0` | **Buy** — spend that fraction of current cash balance on asset `i` |
| `< 0` | **Sell** — sell that fraction of current holdings of asset `i` |
| `= 0` | **Hold** — no trade |

### Reward

```
reward = portfolio_value_after_step - portfolio_value_before_step
```

Portfolio value = cash balance + Σ(holdings × current prices). The agent is directly incentivized to maximize dollar returns.

### Episode Termination

The episode ends when the agent has stepped through all rows of the historical price data.

---

## Agent: PPO

The agent uses **Proximal Policy Optimization** with an MLP (multi-layer perceptron) policy via `stable-baselines3`.

```python
from stable_baselines3 import PPO

env = CryptoTradingEnv(historical_data=historical_data)
model = PPO("MlpPolicy", env, verbose=0)
model.learn(total_timesteps=10000)
```

PPO is well-suited here because:
- It handles continuous action spaces natively
- Its clipped surrogate objective prevents destructively large policy updates
- It performs well on custom environments without extensive hyperparameter tuning

---

## Results

After training, the agent is evaluated on the full historical dataset in deterministic mode (`model.predict(..., deterministic=True)`). Returns are tracked at each timestep and plotted as **USD profit/loss relative to the $100,000 starting balance**.

```
Returns (USD) vs. Time Steps
```

The chart shows how the learned policy performs against the simulated market over the evaluation period.

---

## Getting Started

### Run in Google Colab (Recommended)

Click the badge at the top of this README — no local setup required.

### Run Locally

**1. Clone the repo**
```bash
git clone https://github.com/myanez987/Crypto_Reinforcement_Learning-.git
cd Crypto_Reinforcement_Learning-
```

**2. Install dependencies**
```bash
pip install stable-baselines3 gymnasium pandas matplotlib numpy
```

**3. Open the notebook**
```bash
jupyter notebook Crypto_Reinforcement_Trading.ipynb
```

---

## Using Your Own Data

Replace the `historical_data` DataFrame with real OHLCV or close-price data. The only requirement is that it be a `pandas.DataFrame` where:
- Each **row** is a timestep (e.g., daily or hourly)
- Each **column** is a separate cryptocurrency's price series

```python
import pandas as pd

historical_data = pd.read_csv("your_crypto_prices.csv", index_col=0, parse_dates=True)
env = CryptoTradingEnv(historical_data=historical_data)
```

Suggested free data sources: [CoinGecko API](https://www.coingecko.com/en/api), [Binance API](https://binance-docs.github.io/apidocs/), [Yahoo Finance via `yfinance`](https://github.com/ranaroussi/yfinance).

---

## Project Structure

```
Crypto_Reinforcement_Learning-/
├── Crypto_Reinforcement_Trading.ipynb   # Main notebook
└── README.md
```

---

## Potential Extensions

- **Longer training** — increase `total_timesteps` significantly (100k–1M) for better convergence
- **Real market data** — plug in live OHLCV feeds from CoinGecko, Binance, or CCXT
- **Richer observation space** — add technical indicators (RSI, MACD, Bollinger Bands)
- **Alternative algorithms** — try A2C, SAC, or TD3 from `stable-baselines3`
- **Hyperparameter tuning** — use Optuna with `sb3-contrib` for automated search
- **Walk-forward validation** — train on one time window, evaluate on a future unseen window to avoid lookahead bias

---

## Dependencies

| Package | Purpose |
|---|---|
| `stable-baselines3` | PPO algorithm and training loop |
| `gymnasium` | Custom environment base class |
| `pandas` | Historical price data management |
| `numpy` | Observation/action array operations |
| `matplotlib` | Returns visualization |

---

## License

MIT License. See `LICENSE` for details.
