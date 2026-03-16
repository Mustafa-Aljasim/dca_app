import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots


YEAR_SECONDS = 365.25 * 24 * 3600


# =========================================================
# Core Arps and Cumulative Functions
# =========================================================
def arps_rate(t_years, qi, Di, b):
    """Arps decline: q(t) = qi / (1 + b*Di*t)^(1/b), exponential if b -> 0."""
    t_years = np.asarray(t_years, dtype=float)
    qi = float(qi)
    Di = float(Di)
    b = float(b)

    if np.isclose(b, 0.0):
        return qi * np.exp(-Di * t_years)

    denom = np.clip(1.0 + b * Di * t_years, 1e-12, np.inf)
    return qi / (denom ** (1.0 / b))


def running_cumulative(dates, rates):
    """Running cumulative aligned with dates using trapezoids."""
    if len(dates) < 2:
        return np.zeros(len(dates), dtype=float)

    d = pd.to_datetime(dates).to_numpy()
    r = np.asarray(rates, dtype=float)
    dt_days = np.diff(d).astype("timedelta64[s]").astype(float) / 86400.0
    avg_rates = 0.5 * (r[1:] + r[:-1])
    segment_areas = avg_rates * dt_days
    return np.concatenate(([0.0], np.cumsum(segment_areas)))


# =========================================================
# Data Loading
# =========================================================
@st.cache_data
def parse_oil_upload(uploaded_file):
    """Parses uploaded oil-rate CSV (expects date + oil/rate columns)."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f"There was an error processing the oil-rate file: {exc}"

    cols = {str(c).lower().strip(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if "date" in k), None)

    oil_col = None
    oil_aliases = {"oil", "oil_rate", "rate", "qo", "q", "oilrate"}
    for key, original in cols.items():
        if key in oil_aliases:
            oil_col = original
            break
    if oil_col is None:
        oil_col = next((cols[k] for k in cols if "oil" in k and "rate" in k), None)

    if date_col is None or oil_col is None:
        return None, "Oil CSV must include a date column and an oil rate column."

    out = (
        pd.DataFrame(
            {
                "Date": pd.to_datetime(df[date_col], errors="coerce"),
                "OIL": pd.to_numeric(df[oil_col], errors="coerce"),
            }
        )
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if out.empty:
        return None, "No valid oil-rate data found after cleaning."

    return out, None


@st.cache_data
def parse_well_upload(uploaded_file):
    """Parses uploaded well-count CSV (expects date + well count columns)."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f"There was an error processing the well-count file: {exc}"

    cols = {str(c).lower().strip(): c for c in df.columns}
    date_col = next((cols[k] for k in cols if "date" in k), None)

    well_col = None
    preferred = {"well_count", "wellcount", "wells", "well", "count"}
    for key, original in cols.items():
        if key in preferred:
            well_col = original
            break
    if well_col is None:
        well_col = next((cols[k] for k in cols if "well" in k and "count" in k), None)

    if date_col is None or well_col is None:
        return None, "Well-count CSV must include a date column and a well count column."

    out = (
        pd.DataFrame(
            {
                "Date": pd.to_datetime(df[date_col], errors="coerce"),
                "Well_Count": pd.to_numeric(df[well_col], errors="coerce"),
            }
        )
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    if out.empty:
        return None, "No valid well-count data found after cleaning."

    return out, None


@st.cache_data
def make_dummy_oil_data():
    """Creates deterministic dummy oil-rate data for app testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2017-01-01", periods=120, freq="MS")
    t_years = (dates - dates[0]).days / 365.25
    base_rate = arps_rate(t_years, qi=3200, Di=0.62, b=0.85)
    noise = rng.normal(0.0, 90.0, size=len(base_rate))
    oil = np.clip(base_rate + noise, 30.0, None)
    return pd.DataFrame({"Date": dates, "OIL": oil})


@st.cache_data
def make_dummy_well_data():
    """Creates deterministic dummy well-count data for upload testing."""
    dates = pd.date_range(start="2017-01-01", periods=120, freq="MS")
    wells = np.ones(len(dates), dtype=float) * 4
    wells[18:] = 6
    wells[42:] = 8
    wells[72:] = 9
    return pd.DataFrame({"Date": dates, "Well_Count": wells})


@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode("utf-8")


# =========================================================
# Fit Engine
# =========================================================
def fit_decline(df, fit_mask, forecast_years=10, forecast_start_rate=None):
    """Fit Arps on selected history and forecast from end of full history."""
    fit_df = df.loc[fit_mask].copy().sort_values("Date")
    if fit_df.empty or len(fit_df) < 5:
        raise ValueError("Select at least 5 points for fitting.")

    t0 = fit_df["Date"].iloc[0]
    t_fit = (fit_df["Date"] - t0).dt.total_seconds().to_numpy() / YEAR_SECONDS
    y_fit = fit_df["OIL"].astype(float).to_numpy()

    valid = np.isfinite(y_fit) & (y_fit > 0)
    if valid.sum() < 5:
        raise ValueError("Not enough positive data points in selected fit window.")

    qi0 = float(np.nanmax(y_fit[valid]))
    Di0 = 0.5
    b0 = 0.7
    bounds = ([1e-8, 1e-6, 0.0], [1e9, 5.0, 2.0])

    popt, _ = curve_fit(
        arps_rate,
        t_fit[valid],
        y_fit[valid],
        p0=[qi0, Di0, b0],
        bounds=bounds,
        maxfev=30000,
    )

    fitted_params = {"qi": float(popt[0]), "Di": float(popt[1]), "b": float(popt[2])}
    used_params = dict(fitted_params)

    hist_actual_cum = running_cumulative(df["Date"], df["OIL"])
    forecast_start_date = df["Date"].iloc[-1]
    t_forecast_start = float((forecast_start_date - t0).total_seconds() / YEAR_SECONDS)
    q_start = float(df["OIL"].iloc[-1]) if forecast_start_rate is None else float(forecast_start_rate)
    q_start = max(q_start, 1e-8)

    Di = used_params["Di"]
    b = used_params["b"]
    if np.isclose(b, 0.0):
        used_params["qi"] = q_start * np.exp(Di * t_forecast_start)
    else:
        used_params["qi"] = q_start * ((1.0 + b * Di * t_forecast_start) ** (1.0 / b))

    t_hist = (df["Date"] - t0).dt.total_seconds().to_numpy() / YEAR_SECONDS
    hist_model_rate = arps_rate(t_hist, **used_params)

    forecast_dates = pd.date_range(
        start=forecast_start_date,
        periods=int(forecast_years * 12) + 1,
        freq="MS",
    )[1:]
    t_fore = (forecast_dates - t0).total_seconds() / YEAR_SECONDS
    forecast_rate = arps_rate(t_fore, **used_params)

    forecast_plot_dates = pd.DatetimeIndex([forecast_start_date]).append(pd.DatetimeIndex(forecast_dates))
    forecast_plot_rate = np.concatenate(([q_start], forecast_rate))

    combined_dates = pd.concat([df["Date"], pd.Series(forecast_dates)], ignore_index=True)
    combined_rates = pd.concat([df["OIL"], pd.Series(forecast_rate)], ignore_index=True)
    combined_cum = running_cumulative(combined_dates, combined_rates)

    forecast_cum = combined_cum[len(df) :]
    forecast_plot_cum = np.concatenate(([hist_actual_cum[-1]], forecast_cum))
    hist_model_cum = running_cumulative(df["Date"], hist_model_rate)

    return {
        "fit_df": fit_df,
        "fit_start": fit_df["Date"].min(),
        "fit_end": fit_df["Date"].max(),
        "fitted_params": fitted_params,
        "used_params": used_params,
        "hist_model_rate": hist_model_rate,
        "hist_model_cum": hist_model_cum,
        "hist_actual_cum": hist_actual_cum,
        "forecast_start_rate": q_start,
        "forecast_dates": forecast_dates,
        "forecast_rate": forecast_rate,
        "forecast_cum": forecast_cum,
        "forecast_plot_dates": forecast_plot_dates,
        "forecast_plot_rate": forecast_plot_rate,
        "forecast_plot_cum": forecast_plot_cum,
    }


# =========================================================
# Selection helpers
# =========================================================
def _get_item_or_attr(obj, key, default=None):
    """Read a field from either dict-like or attribute-style objects."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def extract_selection_indices(plot_state):
    """Extracts selected point indices from Streamlit plot selection state."""
    if plot_state is None:
        return None

    selection = _get_item_or_attr(plot_state, "selection")
    if selection is None:
        return None

    # Preferred path: Streamlit provides selected point indices directly.
    point_indices = _get_item_or_attr(selection, "point_indices")
    if point_indices is not None:
        try:
            return sorted({int(i) for i in point_indices})
        except Exception:
            pass

    points = _get_item_or_attr(selection, "points", [])
    if points is None:
        points = []

    indices = []
    for point in points:
        curve_number = _get_item_or_attr(point, "curve_number")
        if curve_number not in (None, 0):
            # We only fit on the first (historical) trace.
            continue

        val = _get_item_or_attr(point, "customdata")
        if isinstance(val, (list, tuple, np.ndarray)):
            val = val[0] if len(val) > 0 else None

        if val is None:
            for key in ("point_index", "pointNumber", "point_number", "pointIndex"):
                candidate = _get_item_or_attr(point, key)
                if candidate is not None:
                    val = candidate
                    break

        try:
            if val is not None:
                indices.append(int(val))
        except Exception:
            continue

    return sorted(set(indices))


def mask_from_selection(df, selected_indices):
    """Converts selected points into a contiguous date-window mask."""
    if not selected_indices:
        return pd.Series(True, index=df.index), None, None

    valid_indices = [i for i in selected_indices if i in df.index]
    if not valid_indices:
        return pd.Series(True, index=df.index), None, None

    sel_dates = df.loc[valid_indices, "Date"]
    fit_start = sel_dates.min()
    fit_end = sel_dates.max()
    fit_mask = (df["Date"] >= fit_start) & (df["Date"] <= fit_end)
    return fit_mask, fit_start, fit_end


def mask_from_date_range(df, start_date, end_date):
    """Build fit mask from manual date-range input."""
    fit_start = pd.to_datetime(start_date)
    fit_end = pd.to_datetime(end_date)
    if fit_start > fit_end:
        fit_start, fit_end = fit_end, fit_start
    fit_mask = (df["Date"] >= fit_start) & (df["Date"] <= fit_end)
    return fit_mask, fit_start, fit_end


def store_selection_from_widget(plot_widget_key, state_key):
    """
    Callback-safe selection sync.
    Reads chart selection from session_state widget key and stores persistent indices.
    """
    plot_state = st.session_state.get(plot_widget_key)
    selected_indices = extract_selection_indices(plot_state)
    st.session_state[state_key] = [] if selected_indices is None else selected_indices


def forecast_download_frame(result):
    out = pd.DataFrame(
        {
            "Date": result["forecast_dates"],
            "Forecast_Rate": result["forecast_rate"],
            "Forecast_Cumulative": result["forecast_cum"],
        }
    )
    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")
    return out


def cumulative_time_figure(df, result, title):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["CumOil"],
            mode="lines",
            name="Historical Cumulative",
            line={"color": "#2CA02C", "width": 3},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result["forecast_plot_dates"],
            y=result["forecast_plot_cum"],
            mode="lines",
            name="Forecast Cumulative",
            line={"color": "#D62728", "dash": "dash", "width": 3},
        )
    )
    fig.update_layout(
        title=title,
        height=520,
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Cumulative Oil")
    return fig


# =========================================================
# Streamlit App
# =========================================================
st.set_page_config(layout="wide")
st.title("Interactive Decline Curve Analysis")
st.caption(
    "Use box/lasso selection on each plot to choose the fit window. "
    "If no points are selected, the full history is used."
)

st.sidebar.header("Data")
oil_upload = st.sidebar.file_uploader(
    "Upload Oil-Rate CSV (optional)", type="csv", key="oil_upload"
)
well_upload = st.sidebar.file_uploader(
    "Optional: Upload Well-Count CSV", type="csv", key="well_upload"
)

if oil_upload is not None:
    oil_df, oil_err = parse_oil_upload(oil_upload)
    if oil_err:
        st.error(oil_err)
        st.stop()
    st.sidebar.success(f"Loaded {len(oil_df)} oil-rate rows.")
else:
    oil_df = make_dummy_oil_data()
    st.sidebar.info("No oil file uploaded. Using built-in dummy oil-rate data.")

well_df = None
if well_upload is not None:
    well_df, well_err = parse_well_upload(well_upload)
    if well_err:
        st.sidebar.error(well_err)
    else:
        st.sidebar.success(f"Loaded {len(well_df)} well-count rows.")

with st.sidebar.expander("Dummy test files"):
    st.download_button(
        "Download dummy oil CSV",
        data=convert_df_to_csv(make_dummy_oil_data()),
        file_name="dummy_oil_rate.csv",
        mime="text/csv",
        key="dl_dummy_oil",
    )
    st.download_button(
        "Download dummy well-count CSV",
        data=convert_df_to_csv(make_dummy_well_data()),
        file_name="dummy_well_count.csv",
        mime="text/csv",
        key="dl_dummy_well",
    )

st.sidebar.header("Analysis Controls")
forecast_years = st.sidebar.number_input(
    "Forecast Years", min_value=1, max_value=50, value=10, step=1
)
use_log_scale = st.sidebar.checkbox("Use log scale for oil rate", value=False)

df = oil_df.copy().sort_values("Date").reset_index(drop=True)
df["CumOil"] = running_cumulative(df["Date"], df["OIL"])
df["PointID"] = df.index.astype(int)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()

fit_window_mode = st.sidebar.radio(
    "Fit window source",
    options=["Plot selection", "Manual date range"],
    index=0,
    help="Use plot selection or force a manual date range for fitting.",
)

manual_fit_range = (min_date, max_date)
if fit_window_mode == "Manual date range":
    manual_fit_range = st.sidebar.date_input(
        "Manual fit date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

if isinstance(manual_fit_range, (tuple, list)) and len(manual_fit_range) >= 2:
    manual_fit_start = manual_fit_range[0]
    manual_fit_end = manual_fit_range[1]
else:
    manual_fit_start = min_date
    manual_fit_end = max_date

for key in ("sel_time", "sel_q_np"):
    if key not in st.session_state:
        st.session_state[key] = []

tab1, tab2 = st.tabs(
    [
        "1) Oil Rate vs Time",
        "2) Cum Oil vs Oil Rate",
    ]
)

# =========================================================
# Tab 1: Oil Rate vs Time + Optional Well Count
# =========================================================
with tab1:
    left, right = st.columns([1, 5])
    with left:
        if st.button("Clear selection", key="clear_tab1"):
            st.session_state["sel_time"] = []
            st.rerun()
    with right:
        tab1_start_rate = st.number_input(
            "First forecast point rate (at last historical date)",
            min_value=0.0,
            value=float(df["OIL"].iloc[-1]),
            step=25.0,
            format="%.2f",
            key="tab1_start_rate",
            help="This anchors the first red forecast point to the last historical date.",
        )

    sel_mask_1, sel_start_1, sel_end_1 = mask_from_selection(df, st.session_state["sel_time"])
    if fit_window_mode == "Manual date range":
        fit_mask_1, fit_start_1, fit_end_1 = mask_from_date_range(df, manual_fit_start, manual_fit_end)
    else:
        fit_mask_1, fit_start_1, fit_end_1 = sel_mask_1, sel_start_1, sel_end_1

    try:
        result_1 = fit_decline(
            df,
            fit_mask_1,
            forecast_years=forecast_years,
            forecast_start_rate=tab1_start_rate,
        )
        fit_error_1 = None
    except Exception as exc:
        result_1 = None
        fit_error_1 = str(exc)

    rows = 2 if well_df is not None else 1
    titles = ["Oil Rate vs Time (select points for fit)"]
    if well_df is not None:
        titles.append("Well Count vs Time")

    fig1 = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=(rows == 2),
        vertical_spacing=0.08,
        subplot_titles=tuple(titles),
    )

    marker_colors = np.where(fit_mask_1, "#1f77b4", "#B0B0B0")

    fig1.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["OIL"],
            mode="markers",
            name="Historical Oil Rate",
            customdata=df["PointID"],
            marker={"size": 8, "color": marker_colors},
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Rate=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if result_1 is not None:
        fig1.add_trace(
            go.Scatter(
                x=df["Date"],
                y=result_1["hist_model_rate"],
                mode="lines",
                name="Model Fit",
                line={"color": "#0057B8", "width": 3},
            ),
            row=1,
            col=1,
        )
        fig1.add_trace(
            go.Scatter(
                x=result_1["forecast_plot_dates"],
                y=result_1["forecast_plot_rate"],
                mode="lines",
                name="Forecast",
                line={"color": "#D62728", "dash": "dash", "width": 3},
            ),
            row=1,
            col=1,
        )

    if fit_start_1 is not None and fit_end_1 is not None:
        fig1.add_vrect(
            x0=fit_start_1,
            x1=fit_end_1,
            fillcolor="rgba(31, 119, 180, 0.08)",
            line_width=0,
            row=1,
            col=1,
        )

    if well_df is not None:
        fig1.add_trace(
            go.Scatter(
                x=well_df["Date"],
                y=well_df["Well_Count"],
                mode="lines+markers",
                name="Well Count",
                marker={"size": 5, "color": "#2CA02C"},
                line={"color": "#2CA02C", "width": 2},
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Wells=%{y:.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig1.update_yaxes(title_text="Well Count", row=2, col=1)
        fig1.update_xaxes(title_text="Date", row=2, col=1)
    else:
        fig1.update_xaxes(title_text="Date", row=1, col=1)

    fig1.update_yaxes(
        title_text="Oil Rate",
        type="log" if use_log_scale else "linear",
        row=1,
        col=1,
    )
    fig1.update_layout(
        height=740 if rows == 2 else 520,
        template="plotly_white",
        dragmode="select",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
        key="plot_tab1",
        on_select=lambda: store_selection_from_widget("plot_tab1", "sel_time"),
        selection_mode=("points", "box", "lasso"),
    )

    if fit_window_mode == "Manual date range":
        st.caption(
            f"Manual fit range active: {pd.to_datetime(fit_start_1).date()} to {pd.to_datetime(fit_end_1).date()}"
        )

    if fit_error_1:
        st.error(f"Fit error: {fit_error_1}")
    else:
        st.write(
            f"Fit window: **{result_1['fit_start'].date()}** to **{result_1['fit_end'].date()}** "
            f"({len(result_1['fit_df'])} points)"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("qi", f"{result_1['fitted_params']['qi']:.2f}")
        c2.metric("Di", f"{result_1['fitted_params']['Di']:.4f}")
        c3.metric("b", f"{result_1['fitted_params']['b']:.3f}")

        st.plotly_chart(
            cumulative_time_figure(
                df,
                result_1,
                "Cumulative Oil vs Time (Historical + Forecast from Tab 1)",
            ),
            use_container_width=True,
            key="cum_plot_tab1",
        )

        st.download_button(
            "Download forecast CSV (Tab 1)",
            data=convert_df_to_csv(forecast_download_frame(result_1)),
            file_name="forecast_tab1_rate_vs_time.csv",
            mime="text/csv",
            key="dl_tab1_forecast",
        )


# =========================================================
# Tab 2: Cum Oil vs Oil Rate
# =========================================================
with tab2:
    left, right = st.columns([1, 5])
    with left:
        if st.button("Clear selection", key="clear_tab2"):
            st.session_state["sel_q_np"] = []
            st.rerun()
    with right:
        tab2_start_rate = st.number_input(
            "First forecast point rate (at last historical date)",
            min_value=0.0,
            value=float(df["OIL"].iloc[-1]),
            step=25.0,
            format="%.2f",
            key="tab2_start_rate",
            help="This anchors the first red forecast point to the last historical date.",
        )

    sel_mask_2, sel_start_2, sel_end_2 = mask_from_selection(df, st.session_state["sel_q_np"])
    if fit_window_mode == "Manual date range":
        fit_mask_2, fit_start_2, fit_end_2 = mask_from_date_range(df, manual_fit_start, manual_fit_end)
    else:
        fit_mask_2, fit_start_2, fit_end_2 = sel_mask_2, sel_start_2, sel_end_2

    try:
        result_2 = fit_decline(
            df,
            fit_mask_2,
            forecast_years=forecast_years,
            forecast_start_rate=tab2_start_rate,
        )
        fit_error_2 = None
    except Exception as exc:
        result_2 = None
        fit_error_2 = str(exc)

    marker_colors = np.where(fit_mask_2, "#1f77b4", "#B0B0B0")

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=df["CumOil"],
            y=df["OIL"],
            mode="markers",
            name="Historical (select points)",
            customdata=df["PointID"],
            marker={"size": 8, "color": marker_colors},
            hovertemplate="Cum=%{x:.2f}<br>Rate=%{y:.2f}<extra></extra>",
        )
    )

    if result_2 is not None:
        fig2.add_trace(
            go.Scatter(
                x=df["CumOil"],
                y=result_2["hist_model_rate"],
                mode="lines",
                name="Model Fit",
                line={"color": "#0057B8", "width": 3},
            )
        )
        fig2.add_trace(
            go.Scatter(
                x=result_2["forecast_plot_cum"],
                y=result_2["forecast_plot_rate"],
                mode="lines",
                name="Forecast",
                line={"color": "#D62728", "dash": "dash", "width": 3},
            )
        )

    fig2.update_layout(
        title="Cumulative Oil vs Oil Rate (select points for fit window)",
        height=600,
        template="plotly_white",
        dragmode="select",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )
    fig2.update_xaxes(title_text="Cumulative Oil")
    fig2.update_yaxes(title_text="Oil Rate", type="log" if use_log_scale else "linear")

    st.plotly_chart(
        fig2,
        use_container_width=True,
        key="plot_tab2",
        on_select=lambda: store_selection_from_widget("plot_tab2", "sel_q_np"),
        selection_mode=("points", "box", "lasso"),
    )

    if fit_window_mode == "Manual date range":
        st.caption(
            f"Manual fit range active: {pd.to_datetime(fit_start_2).date()} to {pd.to_datetime(fit_end_2).date()}"
        )

    if fit_error_2:
        st.error(f"Fit error: {fit_error_2}")
    else:
        st.write(
            f"Fit window: **{result_2['fit_start'].date()}** to **{result_2['fit_end'].date()}** "
            f"({len(result_2['fit_df'])} points)"
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("qi", f"{result_2['fitted_params']['qi']:.2f}")
        c2.metric("Di", f"{result_2['fitted_params']['Di']:.4f}")
        c3.metric("b", f"{result_2['fitted_params']['b']:.3f}")

        st.plotly_chart(
            cumulative_time_figure(
                df,
                result_2,
                "Cumulative Oil vs Time (Historical + Forecast from Tab 2)",
            ),
            use_container_width=True,
            key="cum_plot_tab2",
        )

        st.download_button(
            "Download forecast CSV (Tab 2)",
            data=convert_df_to_csv(forecast_download_frame(result_2)),
            file_name="forecast_tab2_cum_vs_rate.csv",
            mime="text/csv",
            key="dl_tab2_forecast",
        )
