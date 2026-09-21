Your task is to create a simple web tool showing a sales forecast in units for the next 4 weeks, taking into account external factors that affect sales.

Data: Demand forecasting dataset: https://www.kaggle.com/datasets/raminhuseyn/demand-forecasting-dataset

Tasks:
- Load the data and aggregate daily sales (Units Sold) by summing across stores and products.
- Explore the data, select external factors that affect sales, and explain your choice in the README.
- Set aside the last 4 weeks (28 days) as a validation set; train the model only on earlier data.
- Choose a time series forecasting model and generate a forecast for the next 4 weeks.
- Calculate MAE, RMSE, MAPE, and bias, and briefly describe the results.
- Build a web app (e.g., Streamlit or Plotly Dash) with a forecast plot and a metrics table, including a comparison with a baseline. Add at least one filter next to the plot, such as a date selector.
- Publish the solution in a GitHub repository.

Bonus (optional):
- Select the 3 best-selling products by average daily sales and prepare separate 4-week forecasts for them.
