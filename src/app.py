"""I compare four models in two scenarios over a 28-day holdout."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from forecast_v3.dashboard import load_dashboard_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Sales forecast", page_icon=":material/monitoring:", layout="wide")
st.title("Retail sales forecast")
st.caption(
    "I sum sales across all stores and products. "
    "I compare four models in two scenarios over the same 28 days."
)


daily, predictions, metrics, protocol = load_dashboard_data(PROJECT_ROOT)

st.sidebar.header("Model comparison")
scenario = st.sidebar.selectbox(
    "Available data",
    options=list(protocol["scenarios"]),
    format_func=protocol["scenarios"].get,
    key="scenario",
)
predictions = predictions.loc[predictions["scenario"].eq(scenario)].drop(columns="scenario")
metrics = metrics.loc[metrics["scenario"].eq(scenario)].drop(columns="scenario")
labels = protocol["models"][scenario]
selected_model = protocol["selected_models"][scenario]
validation_start, validation_end = predictions.index.min(), predictions.index.max()

if scenario == "history":
    st.caption(
        "I use only history before validation and calendar features. "
        "I use validation sales to evaluate and select models and parameters. "
        "I do not use them for training or as features."
    )
else:
    st.caption(
        "I know prices, promotions and discounts for the 28-day forecast. "
        "In this simulation, I treat their validation values as a perfect company plan. "
        "Future sales, demand and epidemic status remain unavailable to the models."
    )

alternatives = [name for name in labels if name != "mean28"]
model_name = st.sidebar.selectbox(
    "Model to compare with baseline",
    options=alternatives,
    index=alternatives.index(selected_model),
    format_func=labels.get,
    key=f"model_{scenario}",
)
st.sidebar.caption(
    f"I selected {labels[selected_model]} based on MAE."
)
st.sidebar.subheader("Date filter")
st.sidebar.caption(
    "I change the chart range with the slider below the axis or the 28 days, 90 days and All buttons."
)

st.subheader("Sales and forecast")
st.caption(
    f"History through {protocol['training_end']} ({protocol['training_days']} days). "
    f"Forecast: {validation_start:%Y-%m-%d}–{validation_end:%Y-%m-%d}. "
    "This is a historical evaluation, not a forecast beyond the dataset."
)
with st.container(horizontal=True):
    st.metric("MAE (units/day)", f"{metrics.loc[model_name, 'MAE']:.2f}", border=True)
    st.metric("RMSE (units/day)", f"{metrics.loc[model_name, 'RMSE']:.2f}", border=True)
    st.metric("MAPE (%)", f"{metrics.loc[model_name, 'MAPE (%)']:.2f}%", border=True)
    st.metric("Bias (%)", f"{metrics.loc[model_name, 'Bias (%)']:+.2f}%", border=True)

improvement = 100 * (1 - metrics.loc[model_name, "MAE"] / metrics.loc["mean28", "MAE"])
st.caption(
    f"Improvement over baseline: {improvement:+.2f}%. "
    "A positive value means lower MAE; a negative value means higher MAE."
)

figure = go.Figure()
figure.add_trace(go.Scatter(
    x=daily.index, y=daily["units"], name="Actual sales",
    line={"color": "#38BDF8", "width": 1.5},
))
for name, color, style in (
    ("mean28", "#FBBF24", "dot"),
    (model_name, "#F472B6", "solid"),
):
    figure.add_trace(go.Scatter(
        x=predictions.index, y=predictions[name], name=labels[name],
        line={"color": color, "width": 2, "dash": style},
    ))
figure.add_vrect(
    x0=validation_start, x1=validation_end,
    fillcolor="rgba(241,245,249,0.10)", line_width=0,
    annotation_text="validation: 28 days", annotation_position="top left",
)
figure.update_layout(
    template="plotly_dark", hovermode="x unified", height=540,
    margin={"l": 10, "r": 10, "t": 120, "b": 120},
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    yaxis={"title": "Sales (units/day)", "gridcolor": "#334155"},
    xaxis={
        "gridcolor": "#334155",
        "range": [validation_end - pd.Timedelta(days=89), validation_end],
        "rangeslider": {"visible": True, "thickness": 0.10},
        "rangeselector": {
            "yanchor": "top", "y": -0.38,
            "buttons": [
                {"count": 28, "label": "28 days", "step": "day", "stepmode": "backward"},
                {"count": 90, "label": "90 days", "step": "day", "stepmode": "backward"},
                {"step": "all", "label": "All"},
            ],
        },
    },
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.04, "x": 0},
    uirevision="validation-chart",
)
st.plotly_chart(
    figure, width="stretch", theme=None, key="validation_chart",
    config={"displaylogo": False, "scrollZoom": True, "responsive": True},
)
st.caption("The date filter changes the chart range; metrics always cover the full 28 days.")

st.subheader("Model errors")
metric_columns = ["MAE", "RMSE", "MAPE (%)", "Bias (%)"]
metrics_view = metrics.loc[list(labels), metric_columns].rename(columns={
    "MAE": "MAE (units/day)", "RMSE": "RMSE (units/day)",
})
metrics_view = metrics_view.assign(Model=[labels[name] for name in metrics_view.index])
metrics_view = metrics_view[["Model", "MAE (units/day)", "RMSE (units/day)", "MAPE (%)", "Bias (%)"]]
st.dataframe(
    metrics_view, hide_index=True, width="stretch", key="validation_metrics",
    column_config={
        "MAE (units/day)": st.column_config.NumberColumn(format="%.2f"),
        "RMSE (units/day)": st.column_config.NumberColumn(format="%.2f"),
        "MAPE (%)": st.column_config.NumberColumn(format="%.2f %%"),
        "Bias (%)": st.column_config.NumberColumn(format="%+.2f %%"),
    },
)

with st.expander("How do I evaluate the forecast?"):
    st.markdown(
        "The baseline is the fixed mean of the last 28 training days. "
        "No model updates its predictions using validation sales.\n\n"
        "- **MAE**: mean absolute error in units.\n"
        "- **RMSE**: an error metric that gives more weight to large errors.\n"
        "- **MAPE (%)**: mean absolute percentage error.\n"
        "- **Bias (%)**: 100 × sum(forecast − sales) / sum(sales); "
        "a positive value means overestimation.\n\n"
        "I selected the default model by MAE over the displayed 28 days. "
        "I use the same window for evaluation, so I do not treat this result as an independent test."
    )

with st.expander("Daily forecasts"):
    daily_view = predictions[["actual", *labels]].rename(columns={
        "actual": "Actual sales", **labels,
    }).reset_index()
    st.dataframe(
        daily_view, hide_index=True, width="stretch", key="daily_predictions",
        column_config={
            "Date": st.column_config.DateColumn(format="YYYY-MM-DD"),
            **{
                name: st.column_config.NumberColumn(format="%.2f")
                for name in daily_view.columns if name != "Date"
            },
        },
    )
    st.download_button(
        "Download forecasts (CSV)", data=daily_view.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"forecasts_{scenario}_28_days.csv", mime="text/csv",
        on_click="ignore", key="download_predictions",
    )
