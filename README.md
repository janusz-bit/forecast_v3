# 4-week sales forecast

I forecast the daily number of units sold across the retail network using the
[Demand forecasting dataset](https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset).
I show a historical forecast for a held-out period of 28 days: **January 3–30, 2024**.
In a single [notebook](notebooks/prognoza_sprzedazy.ipynb), I present the data,
analysis, features, training, and results. I compare four model families:
a baseline, Lasso, LightGBM, and Prophet.

## Two scenarios

I train the models on data before **January 3, 2024**. I use the final 28 days,
**January 3–30**, for evaluation. I have 732 days of history before this period;
lags and rolling means reduce the training set to 677 observations.

| Scenario | What does the model know when making a forecast? |
|---|---|
| **History only** | Data through January 2 and the dates of future days |
| **Known prices and promotions** | The same information, plus prices, promotions, and discounts for the next 28 days |

In the second scenario, the actual validation values of `price`, `promotion`,
and `discount` stand in for a perfect company plan. The dataset does not contain
a separate file with such a plan. Future sales, `Demand`, epidemic status, and
weather are excluded from model inputs in both scenarios. I do not assume a
future epidemic timeline.

**Note: possible data leakage.** If the validation prices, promotions, and
discounts were not actually known at forecast time, using them constitutes
data leakage and may produce an overly optimistic evaluation. I therefore
treat the second scenario as a conditional forecast, valid if the plan is
known in advance.

## Data and features

I sum `Units Sold` across stores and products, obtaining 760 daily totals
from 76,000 records. I explore epidemic status, prices, promotions, discounts,
demand, weather, and the calendar. The descriptive analysis covers the entire
dataset; I calculate historical training features using only data before
validation.

In this dataset, sales are strongly associated with epidemic status (r=−0.90),
price (r=0.92), and demand (r=0.98). I do not interpret this as evidence of
causation. Future `Demand` and epidemic status are unknown at forecast time,
so I do not use them as features for future days. Using these unavailable
values would be leakage; a strong correlation alone is not leakage. Weather
has a weaker association with sales, as do promotions and discounts.

The shared historical features include sales, epidemic status, price, demand,
promotion, and discount from 28 days earlier, plus average sales and epidemic
status over a 28-day window ending 28 days before the date being forecast.
From the calendar, I derive the day of the week, season, annual cycles,
and sine/cosine cycles of 7, 14, and 28 days.

| Model | History only | Known prices and promotions |
|---|---|---|
| 28-day mean | Constant mean of the last 28 days | The same mean |
| Lasso | History, calendar with cycles, and AR(1) residual correction | History, calendar, and plan |
| LightGBM | History and calendar without short cycles | History, calendar with cycles, and plan |
| Prophet | History, built-in weekly seasonality, and 14/28-day cycles | The same features plus the plan |

In each scenario, I select one LightGBM model after comparing parameters and
predict sales for all 28 days in a single call. Without the plan, I use 21
features: calendar features, values from 28 days earlier, and lagged rolling
means. For the plan, I add 7-day lags and 7-day rolling means of prices,
promotions, and discounts; future values of these three variables are allowed
in this scenario.

For Lasso, I standardize features using the training set and use `alpha=25`.
The displayed coefficients refer to standardized features, not their original
units. I add an AR(1) correction only in the "History only" scenario. A residual
is sales minus the fitted Lasso value on the training set. AR predicts it as
`intercept + coefficient × previous residual`. I fit it on 676 pairs of
consecutive residuals. I start the forecast with the last residual from
January 2, then use my own predicted residuals, without validation sales.

For LightGBM, I loop over 1,024 combinations in each scenario: four values
each for `n_estimators`, `learning_rate`, `num_leaves`, `min_child_samples`,
and `reg_lambda`. I edit the grid in `LIGHTGBM_GRID` in the notebook.
I show the 10 best results, all four metrics, and the selected parameters.
I keep the model with the lowest MAE separately for each scenario.
Before training, I show the names of the features passed to the model.
Immediately after each forecast plot, I show the error metrics.
I finish the analysis with the "Model comparison" section, containing a table
and a short conclusion; in "Results for the app", I only save the output files.

## Results

MAE is the mean absolute error in units per day:

| Model | History only | Known prices and promotions |
|---|---:|---:|
| 28-day mean | 1053.50 | 1053.50 |
| **Lasso** | **742.99** | **229.30** |
| LightGBM | 1074.16 | 251.85 |
| Prophet | 985.41 | 244.51 |

Lasso with AR(1) has the lowest MAE in the "History only" scenario, while Lasso
without correction performs best with the known plan. Without the plan, the
correction reduces MAE from 1022.68 to 742.99. The AR coefficient is about
0.798, so the last error affects several subsequent days, with its influence
gradually fading. This does not mean the model knows when the sales jump
will occur.

LightGBM tuning reduced MAE with the known plan from 262.29 to 251.85.
Without the plan, the previous parameters remained best, with MAE of 1074.16.

In the notebook, I show forecast curves and all metrics before and after
correction for the "History only" scenario. With the known plan, I use Lasso
without AR. The better result in the second scenario depends on knowing future
prices and promotions; an actual company plan might be less accurate.

I also save RMSE, MAPE, and bias in the [full metrics table](outputs/metrics.csv).
RMSE gives more weight to large errors, MAPE expresses error as a percentage,
and bias is `100 × sum(forecast − sales) / sum(sales)`.
Positive bias means total sales are overestimated.

I select variants and parameters by MAE on **the same validation window**.
I fit model weights only on earlier data, but the result is a retrospective
evaluation rather than an independent performance test.

## Getting started

```bash
nix develop
uv sync --frozen
uv run jupyter lab
```

In Jupyter, I select the **Python 3 (ipykernel)** kernel from the project
environment (`.venv`) and run the notebook from the first cell. The
`notebooks/prognoza_sprzedazy.py` file is its Jupytext representation,
not a second model module. The notebook saves only three output files:

- [predictions.csv](outputs/predictions.csv) — forecasts for both scenarios;
- [metrics.csv](outputs/metrics.csv) — eight rows of metrics;
- [protocol.json](outputs/protocol.json) — scenario names and selected models.

The app reads these files and the daily data aggregation:

```bash
uv run streamlit run src/app.py
```

I select a scenario and a model, then view its forecast alongside the baseline
and the metrics table. The date filter changes the plot range; metrics cover
the full 28 days. `src/forecast_v3` contains only data and app result loading.

`flake.nix` provides Pandoc and XeLaTeX for exporting the executed notebook:

```bash
uv run jupyter nbconvert --to pdf notebooks/prognoza_sprzedazy.ipynb
```
