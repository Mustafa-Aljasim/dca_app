# e:\Code_Folder\Code_DCA\dca_streamlit_example.py
import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io

# =========================================================
# Core Arps and Cumulative Functions (from dca_cli.py)
# =========================================================
def arps_rate(t_years, qi, Di, b):
    """
    Arps decline: q(t) = qi / (1 + b*Di*t)^(1/b), exponential if b -> 0.
    t_years in years, Di in 1/yr.
    """
    t_years = np.asarray(t_years, dtype=float)
    qi = float(qi); Di = float(Di); b = float(b)
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
    dt_days = np.diff(d).astype('timedelta64[s]').astype(float) / 86400.0
    avg_rates = 0.5 * (r[1:] + r[:-1])
    segment_areas = avg_rates * dt_days
    return np.concatenate(([0.0], np.cumsum(segment_areas)))

# =========================================================
# Data Loading and Analysis Functions (adapted for Streamlit)
# =========================================================
@st.cache_data
def parse_upload(uploaded_file):
    """Parses a file uploaded via Streamlit."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        return None, f"There was an error processing this file: {e}"

    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[c] for c in cols if "date" in c), None)
    oil_col = next((cols[c] for c in cols if c in ("oil", "oil_rate", "rate", "qo", "q")), None)

    if date_col is None or oil_col is None:
        return None, "CSV must include 'Date' and 'OIL' columns (case-insensitive)."

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce"),
        "OIL": pd.to_numeric(df[oil_col], errors="coerce")
    }).dropna().sort_values("Date").reset_index(drop=True)

    if out.empty:
        return None, "No valid data found in the CSV after cleaning."

    return out, None

def perform_decline_analysis(df, forecast_years=10, forecast_start_date=None, override_params=None):
    """
    Fits an Arps model to the data and generates a forecast.
    Returns fitted parameters, historical data with model, and forecast data.
    """
    if df.empty or len(df) < 5:
        raise ValueError("Not enough data points for fitting after cleaning.")
    df = df.copy() # Avoid SettingWithCopyWarning

    t_years = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds() / (365.25 * 24 * 3600)
    y = df["OIL"].astype(float).values
    mask = np.isfinite(y) & (y > 0)

    if mask.sum() < 5:
        raise ValueError("Not enough positive data points for fitting.")

    qi0 = float(np.nanmax(y[mask]))
    Di0 = 0.5
    b0 = 0.7
    bounds = ([1e-8, 1e-6, 0.0], [1e9, 5.0, 2.0])

    popt, _ = curve_fit(arps_rate, t_years[mask], y[mask], p0=[qi0, Di0, b0], bounds=bounds, maxfev=20000)
    params = {"qi": popt[0], "Di": popt[1], "b": popt[2]}
    
    # Use override parameters if provided
    final_params = params.copy()
    if override_params:
        final_params.update(override_params)

    df["Model_Rate"] = arps_rate(t_years, **final_params)

    last_date = forecast_start_date if forecast_start_date is not None else df["Date"].iloc[-1]

    forecast_dates = pd.date_range(start=last_date, periods=int(forecast_years * 12) + 1, freq="MS")[1:]
    t_fore_years = (forecast_dates - df["Date"].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
    q_fore = arps_rate(t_fore_years, **final_params)

    forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast_Rate": q_fore})

    return params, df, forecast_df

# =========================================================
# Streamlit App
# =========================================================
st.set_page_config(layout="wide")
st.title("Interactive Decline Curve Analysis (Streamlit Version)")

# --- 1. File Uploader ---
uploaded_file = st.file_uploader(
    "Upload a CSV file with 'Date' and 'OIL' columns", type="csv"
)

if uploaded_file is None:
    st.info("Please upload a data file to begin.")
    st.stop()

# --- 2. Data Loading and Caching ---
df, error_msg = parse_upload(uploaded_file)
if error_msg:
    st.error(f"Error: {error_msg}")
    st.stop()

# --- 3. Sidebar Controls ---
st.sidebar.header("Analysis Controls")

min_date, max_date = df['Date'].min().date(), df['Date'].max().date()

start_date, end_date = st.sidebar.date_input(
    "Historical Data Range for Fit:",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
    help="Select the portion of the historical data to use for the curve fit."
)

forecast_years = st.sidebar.number_input(
    "Forecast Years:", min_value=1, max_value=50, value=10, step=1
)

st.sidebar.header("Plotting Options")
use_log_scale = st.sidebar.checkbox("Use Log Scale for Rate Plot", value=False,
                                    help="Toggles the Y-axis of the top chart between linear and logarithmic scales.")

# --- 4. Perform Analysis ---
try:
    # Filter the dataframe for the decline fit based on the date picker
    fit_df = df[(df['Date'] >= pd.to_datetime(start_date)) & (df['Date'] <= pd.to_datetime(end_date))].copy()
    # The forecast should always start from the end of the *entire* history
    full_history_last_date = df['Date'].iloc[-1]
    # This call gets the TRUE fitted parameters and the historical model fit line
    params, hist_df_fit, _ = perform_decline_analysis(fit_df, forecast_years, forecast_start_date=full_history_last_date)
except (ValueError, RuntimeError) as e:
    st.error(f"Analysis Error: {e}")
    st.stop()

# --- 5. Display Results ---

# Display parameters in the sidebar
st.sidebar.subheader("Fitted Arps Parameters")
st.sidebar.metric("qi (Initial Rate)", f"{params['qi']:.2f}")
st.sidebar.metric("Di (Initial Decline)", f"{params['Di']:.4f}")
st.sidebar.metric("b (Hyperbolic Factor)", f"{params['b']:.3f}")

st.sidebar.subheader("Manual Forecast Override")

# The rate at the start of the forecast is the last point of the historical model fit
t_at_forecast_start = (full_history_last_date - fit_df['Date'].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
forecast_start_rate = arps_rate(t_at_forecast_start, **params)

q_forecast_start_override = st.sidebar.number_input(
    "Forecast Start Rate:",
    min_value=0.0,
    value=float(forecast_start_rate),
    step=100.0,
    format="%.2f",
    help="Manually set the rate at the beginning of the forecast period."
)

# --- 4b. Generate forecast using potentially overridden parameters ---
# The historical fit (hist_df_fit) is NOT recalculated.

# To make the override intuitive, we calculate a new 'qi' that makes the curve
# pass through the user's desired 'q_forecast_start_override' at the forecast start time.
b = params['b']
Di = params['Di']
# This is the inverse of the Arps equation to solve for qi
adjusted_qi = q_forecast_start_override * ((1.0 + b * Di * t_at_forecast_start) ** (1.0 / b))

forecast_params = {"qi": adjusted_qi, "Di": Di, "b": b}

forecast_dates = pd.date_range(start=full_history_last_date, periods=int(forecast_years * 12) + 1, freq="MS")[1:]
t_fore_years = (forecast_dates - fit_df["Date"].iloc[0]).total_seconds() / (365.25 * 24 * 3600)
q_fore = arps_rate(t_fore_years, **forecast_params)
forecast_df = pd.DataFrame({"Date": forecast_dates, "Forecast_Rate": q_fore})

# --- 6. Create Plots ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                    subplot_titles=("Decline Curve Analysis: Rate vs. Time", "Cumulative Production vs. Time"))

# Rate Plot
fig.add_trace(go.Scatter(x=df["Date"], y=df["OIL"], mode='markers', name='All Historical Data', marker=dict(color='gray', opacity=0.7)), row=1, col=1)
fig.add_trace(go.Scatter(x=hist_df_fit["Date"], y=hist_df_fit["Model_Rate"], mode='lines', name='Model Fit', line=dict(color='blue', width=3)), row=1, col=1)
fig.add_trace(go.Scatter(x=forecast_df["Date"], y=forecast_df["Forecast_Rate"], mode='lines', name='Forecast', line=dict(color='red', dash='dash')), row=1, col=1)

# Cumulative Plot
full_hist_cum_df = df[df['Date'] <= hist_df_fit['Date'].iloc[-1]]
comb_dates = pd.concat([full_hist_cum_df["Date"], forecast_df["Date"]], ignore_index=True)
comb_rates = pd.concat([full_hist_cum_df["OIL"], forecast_df["Forecast_Rate"]], ignore_index=True)
comb_cum = running_cumulative(comb_dates, comb_rates)

fig.add_trace(go.Scatter(x=comb_dates, y=comb_cum, mode='lines', name='Cumulative (Hist+Forecast)', line=dict(color='green')), row=2, col=1)

fig.update_layout(height=700, template='plotly_white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
fig.update_yaxes(title_text="Oil Rate (per day)", type='log' if use_log_scale else 'linear', row=1, col=1)
fig.update_yaxes(title_text="Cumulative Oil", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# --- 7. Prepare and Offer Download ---

# Calculate cumulative for the forecast period
hist_cum_end = comb_cum[len(full_hist_cum_df)-1]
forecast_inc_cum = running_cumulative(forecast_df['Date'], forecast_df['Forecast_Rate'])
forecast_df['Forecast_Cumulative'] = hist_cum_end + forecast_inc_cum

download_df = forecast_df[['Date', 'Forecast_Rate', 'Forecast_Cumulative']].copy()
download_df['Date'] = download_df['Date'].dt.strftime('%Y-%m-%d')

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv_data = convert_df_to_csv(download_df)

st.sidebar.download_button(
   label="Download Forecast CSV",
   data=csv_data,
   file_name="forecast_data.csv",
   mime="text/csv",
)
