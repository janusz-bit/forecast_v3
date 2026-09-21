As a Data Science and Python expert, separate readable analysis from reusable code:
- Treat the only notebook, `notebooks/prognoza_sprzedazy.ipynb`, as a self-contained, linear narrative: a question or goal, a short code cell, a result, and a conclusion. Show aggregation, analysis, feature preparation, training, forecasting, and metrics there, without requiring the reader to look in `src`. Do not put classes or long blocks of definitions in it.
- Keep only technical data loading and app support in `src/forecast_v3`. Do not store a second implementation of the notebook's analysis or models there. Avoid extra abstraction layers and helper functions for a single simple operation.
- Train each model directly in its own notebook section, without defining a model function. In that section, produce forecasts for earlier windows and the final training run. Show and explain genuinely shared feature and metric calculations in the notebook.
- Prioritize readability: show successive calculations using clear intermediate variables. Do not add separate data validation functions or manually raised exceptions in simple model and metric functions.
- Use regular imports from `forecast_v3`. Do not modify `sys.path` or dynamically load notebook code. Importing the package must not load data, train models, display results, or save results.
- The notebook must work when its cells are run from the beginning. The `.py` file paired through Jupytext is only a text representation of the notebook, not a module to import.
- Do not add a test directory. After changes, running the notebook from the beginning and checking that the app opens the saved results is sufficient.
- Write section titles, explanations, and conclusions in English, using the first person. Preserve the reasoning behind decisions, method limitations, and the distinction between future prices/promotions and variants without future observations.
- Include `discount` in every model that uses `promotion`. Variants without future knowledge may use only history and explicit assumptions; allow future prices, promotions, and discounts only in separately labeled conditional variants.
- In every notebook code proposal, add the following at the very beginning of the first cell:
  %load_ext autoreload
  %autoreload 2
- Follow clean programming practices: type hints, docstrings, and returning new DataFrames instead of modifying them in place (`inplace=True` is prohibited).
