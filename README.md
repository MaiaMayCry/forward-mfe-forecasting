# RSI Permutation Test
The model predicts the maximum price movement within a future window, enabling dynamic profit targets and directional signals similar to Donchian Channels but with forward-looking predictions rather than historical lookback.

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
- **Randomized Search CV**: Hyperparameter selection for a collection of machine learning models
- **Permutation Importance**: Testing features to assess statistical significance for each model
- **Baseline Comparisons**: Model's performance is evaluated on hold-out data and compared against zero-return, mean and median baselines
- **Model Configuration**: Modular configuration system for easy experimentation
- **Logging**: File and console logging with configurable levels to track pipeline execution

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

## Results

### Model Selection & Performance

Evaluating the 4 best models on the validation data, Ridge has the best results and it's then used for the final run with the test-data

| Model | Validation MAE | # Features | Test MAE |
|-------|----------------|-----------|----------|
| **Ridge** | 0.009930 | 15 | **0.007845** |
| HistGradientBoosting | 0.009931 | 10 | — |
| GradientBoosting | 0.010109 | 12 | — |
| ElasticNet | 0.010105 | 5 | — |

### Baseline Comparison

The model significantly outperforms common naive prediction strategies by 10-79%:

| Baseline Strategy | Test MAE | Error Ratio vs. Ridge |
|-------------------|----------|----------------------|
| **Ridge Model** | **0.007845** | — |
| Median target value | 0.008645 | 1.10x |
| Mean target value | 0.009109 | 1.16x |
| Zero return (predict 0) | 0.014010 | 1.79x |

## Project structure

```text
.
├── data\_pipeline.py      # Data download, feature engineering, labels, and splits
├── feature\_filtering.py  # Correlation filtering and permutation importance
├── model\_configs.py      # Models and hyperparameters to be used
├── modeling.py           # Model search and evaluation utilities
├── logging_setup.py      # Logging configuration (file + console output)
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
- Forecast potential loss as well as best return to create a proper trading strategy
