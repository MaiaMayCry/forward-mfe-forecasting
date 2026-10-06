# RSI Permutation Test

A time-series machine-learning pipeline for predicting the maximum short-term price return of stocks using RSI technical indicators, correlation filtering, permutation importance, and several regression models.

The default configuration downloads 15 years of `SPY` market data and predicts the best achievable forward five-period return based on the subsequent closing prices.

## Installation and Usage

Clone the repository and install dependencies:

```bash
git clone https://github.com/MaiaMayCry/rsi\_permutation\_test.git
cd rsi\_permutation\_test
pip install -r requirements.txt 
```

Run the default experiment:

```bash
python run.py
```

Run a customized experiment:

```bash
python run.py \
  --ticker AAPL \
  --period 10y \
  --corr-threshold 0.90 \
  --label-window 5 \
  --n-iter 25 \
  --random-state 123
```

## Features

- **Data Pipeline**: Automated data processing, cleaning, and splitting with temporal gap to reduce leakage
- **Feature Creation**: Engineering indicators such as RSI, MACD and EMA as features
- **Correlation Filtering**: Removing highly correlated features with configurable coefficients (default `spearman`)
- **Randomized Search CV**: Hyperparameter selection and testing of machine learning models
- **Permutation Importance**: Testing features to assess statistical significance
- **Model Configuration**: Modular configuration system for easy experimentation
- **Baseline Comparisons**: Model's performance is compared against zero-return, mean and median baselines

## Evaluation Design

The data is divided chronologically into:

- Training data: used for hyperparameter search and permutation
- Validation data: used to select the final model
- Test data: used once for final out-of-sample evaluation

The default configuration is:

| Parameter | Default |
|---|---:|
| Ticker | `SPY` |
| Data period | `15y` |
| Correlation threshold | `0.95` |
| Label window | `5` trading days |
| Randomized-search iterations | `15` |
| Random state | `42` |

## Project structure

```text
.
├── data\_pipeline.py      # Data download, feature engineering, labels, and splits
├── feature\_filtering.py  # Correlation filtering and permutation importance
├── model\_configs.py      # Models and hyperparameter distributions
├── modeling.py           # Model search and evaluation utilities
└── run.py                # Main pipeline entry point
```

## Limitations

- The target is the maximum future return available during a window, best used as an indicator not a complete trading strategy
- TimeSeriesSplit is used on RandomizedSearch but walk forward validation is missing on model evaluation
- A relatively small amount of data for some models like XGBoost
- Correlation filtering is arbitrary, meaning they are kept or removed by order of appearance

## Future Improvements

- Increase the amount of data by performing the test with multiple stocks
- Implement walk forward validation on the model evaluation steps
- Save experiment metrics and selected features to `results/` folder
- Forecast potential loss as well as best return to create a proper trading strategy
