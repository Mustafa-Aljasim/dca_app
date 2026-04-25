import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots


YEAR_SECONDS = 365.25 * 24 * 3600
YEAR_DAYS = 365.25


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


def arps_forecast_from_anchor(t_years, q_start, Di, b):
    """Forecast from a new origin using starting rate q_start at t=0."""
    t_years = np.asarray(t_years, dtype=float)
    q_start = max(float(q_start), 1e-8)
    Di = max(float(Di), 1e-12)
    b = float(b)

    if np.isclose(b, 0.0):
        return q_start * np.exp(-Di * t_years)

    denom = np.clip(1.0 + b * Di * t_years, 1e-12, np.inf)
    return q_start / (denom ** (1.0 / b))


def arps_to_effective_annual_decline(Di, b):
    """Convert Arps decline parameters to model-consistent effective annual percentage drop."""
    Di = np.asarray(Di, dtype=float)
    b = np.asarray(b, dtype=float)

    exp_case = 1.0 - np.exp(-Di)
    safe_b = np.where(np.isclose(b, 0.0), 1.0, b)
    denom = np.clip(1.0 + b * Di, 1e-12, np.inf)
    hyperbolic_case = 1.0 - np.power(denom, -1.0 / safe_b)
    return np.where(np.isclose(b, 0.0), exp_case, hyperbolic_case)


def build_prediction_cases(result, p10_multiplier=0.85, p90_multiplier=1.15):
    """Build simple P10/P50/P90 cases by varying forecast decline around the base case."""
    dates = pd.DatetimeIndex(result['forecast_plot_dates'])
    q0 = float(result['forecast_start_rate'])
    Di0 = max(float(result['forecast_start_D']), 1e-12)
    b0 = float(result['forecast_start_b'])
    t_fore = np.arange(0, len(dates), dtype=float) / 12.0

    def case(mult):
        Di_case = max(Di0 * float(mult), 1e-12)
        rate = arps_forecast_from_anchor(t_fore, q0, Di_case, b0)
        uplift = np.asarray(result.get('forecast_plot_rate_uplift', np.zeros_like(rate)), dtype=float)
        if len(uplift) == len(rate):
            rate = rate + uplift
        return {'multiplier': float(mult), 'Di': float(Di_case), 'rate': rate}

    cases = {
        'P10': case(p10_multiplier),
        'P50': case(1.0),
        'P90': case(p90_multiplier),
    }
    return {'dates': dates, 'cases': cases}


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


def watercut_from_wor(wor):
    wor = np.asarray(wor, dtype=float)
    return wor / (1.0 + wor)


def wor_from_watercut(watercut):
    watercut = np.asarray(watercut, dtype=float)
    denom = np.clip(1.0 - watercut, 1e-12, np.inf)
    return watercut / denom


# =========================================================
# Data Loading
# =========================================================
@st.cache_data
def parse_oil_upload(uploaded_file):
    """Parses uploaded oil/water-rate CSV (expects date + qo + qw columns)."""
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return None, f"There was an error processing the rate file: {exc}"

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

    water_col = None
    water_aliases = {"water", "water_rate", "qw", "waterrate", "qwater"}
    for key, original in cols.items():
        if key in water_aliases:
            water_col = original
            break
    if water_col is None:
        water_col = next((cols[k] for k in cols if "water" in k and "rate" in k), None)

    if date_col is None or oil_col is None or water_col is None:
        return None, "Rate CSV must include date, oil rate (qo), and water rate (qw) columns."

    out = (
        pd.DataFrame(
            {
                "Date": pd.to_datetime(df[date_col], errors="coerce"),
                "OIL": pd.to_numeric(df[oil_col], errors="coerce"),
                "WATER": pd.to_numeric(df[water_col], errors="coerce"),
            }
        )
        .dropna()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    out = out[(out["OIL"] > 0) & (out["WATER"] >= 0)].copy()

    if out.empty:
        return None, "No valid oil/water-rate data found after cleaning."

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
    """Creates deterministic dummy oil/water-rate data for app testing."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(start="2017-01-01", periods=120, freq="MS")
    t_years = (dates - dates[0]).days / YEAR_DAYS
    base_rate = arps_rate(t_years, qi=3200, Di=0.62, b=0.85)
    noise = rng.normal(0.0, 90.0, size=len(base_rate))
    oil = np.clip(base_rate + noise, 30.0, None)
    watercut = np.clip(0.08 + 0.72 * (1.0 - np.exp(-0.22 * t_years)), 0.02, 0.95)
    wor = wor_from_watercut(watercut)
    water = oil * wor
    return pd.DataFrame({"Date": dates, "OIL": oil, "WATER": water})


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

def fit_decline(
    df,
    fit_mask,
    forecast_years=10,
    forecast_start_rate=None,
    forecast_uplifts=None,
    decline_model="Hyperbolic",
    forecast_anchor_mode="Practical re-anchored",
):
    """Fit selected Arps model on chosen history and forecast from end of full history."""
    fit_df = df.loc[fit_mask].copy().sort_values("Date")
    if fit_df.empty or len(fit_df) < 5:
        raise ValueError("Select at least 5 points for fitting.")

    t0 = fit_df["Date"].iloc[0]
    t_fit = (fit_df["Date"] - t0).dt.total_seconds().to_numpy() / YEAR_SECONDS
    y_fit = fit_df["OIL"].astype(float).to_numpy()

    valid = np.isfinite(y_fit) & (y_fit > 0)
    if valid.sum() < 5:
        raise ValueError("Not enough positive data points in selected fit window.")

    model_key = str(decline_model).strip().lower()

    def arps_model_selected(t_years, qi, Di, b_free):
        if model_key == "exponential":
            return arps_rate(t_years, qi, Di, 0.0)
        if model_key == "harmonic":
            return arps_rate(t_years, qi, Di, 1.0)
        return arps_rate(t_years, qi, Di, b_free)

    qi0 = float(np.nanmax(y_fit[valid]))
    Di0 = 0.5

    if model_key == "exponential":
        popt, _ = curve_fit(
            lambda t, qi, Di: arps_rate(t, qi, Di, 0.0),
            t_fit[valid],
            y_fit[valid],
            p0=[qi0, Di0],
            bounds=([1e-8, 1e-6], [1e9, 5.0]),
            maxfev=30000,
        )
        fitted_params = {"qi": float(popt[0]), "Di": float(popt[1]), "b": 0.0}
    elif model_key == "harmonic":
        popt, _ = curve_fit(
            lambda t, qi, Di: arps_rate(t, qi, Di, 1.0),
            t_fit[valid],
            y_fit[valid],
            p0=[qi0, Di0],
            bounds=([1e-8, 1e-6], [1e9, 5.0]),
            maxfev=30000,
        )
        fitted_params = {"qi": float(popt[0]), "Di": float(popt[1]), "b": 1.0}
    else:
        b0 = 0.7
        popt, _ = curve_fit(
            arps_rate,
            t_fit[valid],
            y_fit[valid],
            p0=[qi0, Di0, b0],
            bounds=([1e-8, 1e-6, 0.0], [1e9, 5.0, 1.0]),
            maxfev=30000,
        )
        fitted_params = {"qi": float(popt[0]), "Di": float(popt[1]), "b": float(popt[2])}

    hist_actual_cum = running_cumulative(df["Date"], df["OIL"])
    forecast_start_date = df["Date"].iloc[-1]
    q_start_user = float(df["OIL"].iloc[-1]) if forecast_start_rate is None else float(forecast_start_rate)
    q_start_user = max(q_start_user, 1e-8)

    # Plot fitted model only from fit-window start onward.
    hist_model_df = df.loc[df["Date"] >= t0, ["Date", "CumOil"]].copy()
    t_hist_model = (hist_model_df["Date"] - t0).dt.total_seconds().to_numpy() / YEAR_SECONDS
    hist_model_rate = arps_rate(t_hist_model, **fitted_params)
    hist_model_df["Model_Rate"] = hist_model_rate

    # Forecast anchor options
    # 1) Practical re-anchored: use last/manual rate with fitted Di and b
    # 2) Rigorous parameter preservation: preserve fitted model, move to forecast origin
    anchor_mode_key = str(forecast_anchor_mode).strip().lower()
    t_anchor = float((forecast_start_date - t0).total_seconds() / YEAR_SECONDS)
    q_anchor_model = float(arps_rate(np.array([t_anchor]), **fitted_params)[0])
    q_anchor_model = max(q_anchor_model, 1e-8)

    if np.isclose(fitted_params["b"], 0.0):
        D_anchor_model = fitted_params["Di"]
    else:
        D_anchor_model = fitted_params["Di"] / max(1.0 + fitted_params["b"] * fitted_params["Di"] * t_anchor, 1e-12)

    if anchor_mode_key == "rigorous parameter preservation":
        q_forecast_start = q_anchor_model
        D_forecast_start = D_anchor_model
        b_forecast = fitted_params["b"]
        anchor_note = "Forecast preserves fitted Arps model at forecast start; manual start rate is ignored."
    else:
        q_forecast_start = q_start_user
        D_forecast_start = fitted_params["Di"]
        b_forecast = fitted_params["b"]
        anchor_note = "Forecast is re-anchored to the selected start rate using fitted Di and b from stable period."

    forecast_dates = pd.date_range(
        start=forecast_start_date,
        periods=int(forecast_years * 12) + 1,
        freq="MS",
    )[1:]
    t_fore = np.arange(1, len(forecast_dates) + 1, dtype=float) / 12.0
    forecast_rate_base = arps_forecast_from_anchor(
        t_fore, q_forecast_start, D_forecast_start, b_forecast
    )

    forecast_plot_dates = pd.DatetimeIndex([forecast_start_date]).append(pd.DatetimeIndex(forecast_dates))
    forecast_plot_rate_base = np.concatenate(([q_forecast_start], forecast_rate_base))

    uplift_plot_rate = np.zeros(len(forecast_plot_dates), dtype=float)
    if forecast_uplifts:
        plot_dates = pd.to_datetime(pd.Series(forecast_plot_dates)).to_numpy(dtype="datetime64[ns]")
        for ev in forecast_uplifts:
            ev_date = pd.to_datetime(ev.get("date"))
            ev_rate = float(ev.get("rate", 0.0))
            if not np.isfinite(ev_rate) or ev_rate <= 0:
                continue
            ev_dt = np.datetime64(ev_date)
            idx = int(np.searchsorted(plot_dates, ev_dt, side="left"))
            if 0 <= idx < len(uplift_plot_rate):
                base_at_event = max(float(forecast_plot_rate_base[idx]), 1e-12)
                uplift_plot_rate[idx:] += ev_rate * (forecast_plot_rate_base[idx:] / base_at_event)

    forecast_plot_rate = forecast_plot_rate_base + uplift_plot_rate
    forecast_rate = forecast_plot_rate[1:]

    combined_dates = pd.concat([df["Date"], pd.Series(forecast_dates)], ignore_index=True)
    combined_rates = pd.concat([df["OIL"], pd.Series(forecast_rate)], ignore_index=True)
    combined_cum = running_cumulative(combined_dates, combined_rates)

    forecast_cum = combined_cum[len(df):]
    forecast_plot_cum = np.concatenate(([hist_actual_cum[-1]], forecast_cum))
    hist_model_cum = running_cumulative(hist_model_df["Date"], hist_model_rate)
    hist_model_df["Model_Cum"] = hist_model_cum

    return {
        "fit_df": fit_df,
        "fit_start": fit_df["Date"].min(),
        "fit_end": fit_df["Date"].max(),
        "fitted_params": fitted_params,
        "fitted_effective_annual_decline": float(arps_to_effective_annual_decline(fitted_params["Di"], fitted_params["b"])),
        "decline_model": decline_model,
        "forecast_anchor_mode": forecast_anchor_mode,
        "hist_model_df": hist_model_df,
        "hist_model_rate": hist_model_rate,
        "hist_model_cum": hist_model_cum,
        "hist_actual_cum": hist_actual_cum,
        "forecast_start_rate_input": q_start_user,
        "forecast_start_rate": q_forecast_start,
        "forecast_anchor_rate_model": q_anchor_model,
        "forecast_anchor_D_model": D_anchor_model,
        "forecast_start_D": D_forecast_start,
        "forecast_start_b": b_forecast,
        "forecast_start_effective_annual_decline": float(arps_to_effective_annual_decline(D_forecast_start, b_forecast)),
        "forecast_anchor_note": anchor_note,
        "forecast_dates": forecast_dates,
        "forecast_rate_base": forecast_rate_base,
        "forecast_rate_uplift": uplift_plot_rate[1:],
        "forecast_rate": forecast_rate,
        "forecast_cum": forecast_cum,
        "forecast_plot_dates": forecast_plot_dates,
        "forecast_plot_rate_base": forecast_plot_rate_base,
        "forecast_plot_rate_uplift": uplift_plot_rate,
        "forecast_plot_rate": forecast_plot_rate,
        "forecast_plot_cum": forecast_plot_cum,
    }

def cumulative_from_wor_line(wor, m_ln, c_ln):
    """Invert ln(WOR)=m*Np+c for cumulative oil Np."""
    wor = np.asarray(wor, dtype=float)
    if np.any(wor <= 0):
        raise ValueError("WOR must stay positive for logarithmic inversion.")
    m_ln = float(m_ln)
    c_ln = float(c_ln)
    if np.isclose(m_ln, 0.0):
        raise ValueError("WOR-vs-cumulative fit slope is near zero; cannot invert to cumulative oil.")
    return (np.log(wor) - c_ln) / m_ln


def fit_wor_vs_cum_line(df, fit_mask):
    """Fit straight line on plotted space: ln(WOR)=m*Np+c."""
    valid_mask = fit_mask & np.isfinite(df["CumOil"]) & np.isfinite(df["lnWOR"])
    fit_df = df.loc[valid_mask, ["Date", "CumOil", "WOR", "lnWOR"]].copy().sort_values("Date")
    if fit_df.empty or len(fit_df) < 5:
        raise ValueError("Select at least 5 valid WOR points for fitting.")

    x = fit_df["CumOil"].astype(float).to_numpy()
    y = fit_df["lnWOR"].astype(float).to_numpy()
    if np.allclose(x, x[0]):
        raise ValueError("Selected cumulative-oil points are identical; cannot fit a straight line.")

    m, c = np.polyfit(x, y, 1)
    yhat = m * x + c
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - (ss_res / ss_tot)

    return {
        "fit_df": fit_df,
        "fit_start": fit_df["Date"].min(),
        "fit_end": fit_df["Date"].max(),
        "m_ln": float(m),
        "c_ln": float(c),
        "r2_log": r2,
    }


def fit_wor_vs_time_line(df, fit_mask):
    """Fit straight line for time trend: ln(WOR)=m*t_years+c."""
    valid_mask = fit_mask & np.isfinite(df["lnWOR"])
    fit_df = df.loc[valid_mask, ["Date", "WOR", "lnWOR"]].copy().sort_values("Date")
    if fit_df.empty or len(fit_df) < 5:
        raise ValueError("Select at least 5 valid WOR points for time forecasting.")

    t0 = fit_df["Date"].iloc[0]
    x = (fit_df["Date"] - t0).dt.total_seconds().to_numpy() / YEAR_SECONDS
    y = fit_df["lnWOR"].astype(float).to_numpy()
    if np.allclose(x, x[0]):
        raise ValueError("Selected dates are identical; cannot fit WOR time trend.")

    m, c = np.polyfit(x, y, 1)
    yhat = m * x + c
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if np.isclose(ss_tot, 0.0) else 1.0 - (ss_res / ss_tot)

    return {
        "fit_df": fit_df,
        "fit_start": fit_df["Date"].min(),
        "fit_end": fit_df["Date"].max(),
        "t0": t0,
        "m_ln_time": float(m),
        "c_ln_time": float(c),
        "r2_log": r2,
    }


def forecast_wor_at_date(target_date, fit_np, fit_time):
    """Forecast WOR/WC/CumOil at a specific date using straight-line equations."""
    target_date = pd.to_datetime(target_date)
    t_target = float((target_date - fit_time["t0"]).total_seconds() / YEAR_SECONDS)
    lnwor_target = fit_time["m_ln_time"] * t_target + fit_time["c_ln_time"]
    target_wor = float(np.clip(np.exp(lnwor_target), 1e-12, np.inf))
    target_wc = float(watercut_from_wor(target_wor))
    target_cum = float(cumulative_from_wor_line(target_wor, fit_np["m_ln"], fit_np["c_ln"]))
    return {
        "Date": target_date,
        "Forecast_WOR": target_wor,
        "Forecast_WaterCut": target_wc,
        "Forecast_Cumulative_Oil": target_cum,
    }


def build_wor_forecast(df, fit_mask, forecast_end_date):
    """Build WOR forecast up to user-selected date using straight-line fits."""
    fit_np = fit_wor_vs_cum_line(df, fit_mask)
    fit_time = fit_wor_vs_time_line(df, fit_mask)

    last_hist_date = pd.to_datetime(df["Date"].iloc[-1])
    forecast_end_date = pd.to_datetime(forecast_end_date)
    if forecast_end_date <= last_hist_date:
        raise ValueError(
            f"WOR forecast end date must be after last historical date ({last_hist_date.date()})."
        )

    monthly_dates = pd.date_range(start=last_hist_date, end=forecast_end_date, freq="MS")
    forecast_dates = monthly_dates[monthly_dates > last_hist_date]
    if len(forecast_dates) == 0 or pd.to_datetime(forecast_dates[-1]).date() != forecast_end_date.date():
        forecast_dates = forecast_dates.append(pd.DatetimeIndex([forecast_end_date]))
    forecast_dates = pd.DatetimeIndex(sorted(set(pd.to_datetime(forecast_dates))))

    t_fore = (forecast_dates - fit_time["t0"]).total_seconds() / YEAR_SECONDS
    lnwor_fore = fit_time["m_ln_time"] * t_fore + fit_time["c_ln_time"]
    forecast_wor = np.clip(np.exp(lnwor_fore), 1e-12, np.inf)
    forecast_cum = cumulative_from_wor_line(forecast_wor, fit_np["m_ln"], fit_np["c_ln"])
    forecast_wc = watercut_from_wor(forecast_wor)

    forecast_df = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecast_WOR": forecast_wor,
            "Forecast_WaterCut": forecast_wc,
            "Forecast_Cumulative_Oil": forecast_cum,
        }
    )
    forecast_df = forecast_df.replace([np.inf, -np.inf], np.nan).dropna().sort_values("Date").reset_index(drop=True)
    if forecast_df.empty:
        raise ValueError("Forecast produced no valid rows. Check fit window and data quality.")

    end_report = forecast_wor_at_date(forecast_end_date, fit_np, fit_time)
    return {
        "fit_np": fit_np,
        "fit_time": fit_time,
        "forecast_df": forecast_df,
        "end_report": end_report,
    }


def apply_liquid_rate_schedule(forecast_dates, schedule_starts, schedule_rates):
    """
    Apply piecewise-constant liquid rates to forecast dates.
    For each date, the active rate is from the latest segment start <= date.
    """
    if len(schedule_starts) == 0 or len(schedule_rates) == 0:
        raise ValueError("Provide at least one liquid-rate segment.")
    if len(schedule_starts) != len(schedule_rates):
        raise ValueError("Liquid-rate segment starts and rates count mismatch.")

    starts = pd.to_datetime(pd.Series(schedule_starts)).to_numpy(dtype="datetime64[ns]")
    rates = np.asarray(schedule_rates, dtype=float)
    if np.any(~np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Liquid rates must be non-negative finite numbers.")

    order = np.argsort(starts)
    starts = starts[order]
    rates = rates[order]

    dates = pd.to_datetime(pd.Series(forecast_dates)).to_numpy(dtype="datetime64[ns]")
    out = np.empty(len(dates), dtype=float)
    for i, dt in enumerate(dates):
        idx = int(np.searchsorted(starts, dt, side="right") - 1)
        if idx < 0:
            idx = 0
        out[i] = rates[idx]
    return out


def apply_step_rate_adjustments(dates, event_dates, event_deltas):
    """
    Build cumulative step-rate adjustment by date.
    Each delta is applied from its effective date onward.
    """
    if len(event_dates) == 0 or len(event_deltas) == 0:
        return np.zeros(len(dates), dtype=float)
    if len(event_dates) != len(event_deltas):
        raise ValueError("Step-adjustment dates and deltas count mismatch.")

    starts = pd.to_datetime(pd.Series(event_dates)).to_numpy(dtype="datetime64[ns]")
    deltas = np.asarray(event_deltas, dtype=float)
    if np.any(~np.isfinite(deltas)):
        raise ValueError("Step-adjustment deltas must be finite numbers.")

    order = np.argsort(starts)
    starts = starts[order]
    deltas = deltas[order]
    cum_deltas = np.cumsum(deltas)

    x_dates = pd.to_datetime(pd.Series(dates)).to_numpy(dtype="datetime64[ns]")
    idx = np.searchsorted(starts, x_dates, side="right") - 1
    out = np.zeros(len(x_dates), dtype=float)
    valid = idx >= 0
    if np.any(valid):
        out[valid] = cum_deltas[idx[valid]]
    return out


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
st.title("Interactive Decline Curve Analysis + WOR Forecast")
st.caption(
    "Use box/lasso selection on each plot to choose the fit window. "
    "Tab 3 uses straight-line ln(WOR) fitting and stops WOR forecast at your selected end date."
)

st.sidebar.header("Data")
oil_upload = st.sidebar.file_uploader(
    "Upload Oil/Water Rate CSV (optional)", type="csv", key="oil_upload"
)
well_upload = st.sidebar.file_uploader(
    "Optional: Upload Well-Count CSV", type="csv", key="well_upload"
)

if oil_upload is not None:
    oil_df, oil_err = parse_oil_upload(oil_upload)
    if oil_err:
        st.error(oil_err)
        st.stop()
    st.sidebar.success(f"Loaded {len(oil_df)} oil/water-rate rows.")
else:
    oil_df = make_dummy_oil_data()
    st.sidebar.info("No rate file uploaded. Using built-in dummy oil/water-rate data.")

well_df = None
if well_upload is not None:
    well_df, well_err = parse_well_upload(well_upload)
    if well_err:
        st.sidebar.error(well_err)
    else:
        st.sidebar.success(f"Loaded {len(well_df)} well-count rows.")

with st.sidebar.expander("Dummy test files"):
    st.download_button(
        "Download dummy oil/water CSV",
        data=convert_df_to_csv(make_dummy_oil_data()),
        file_name="dummy_oil_water_rate.csv",
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
if "WATER" not in df.columns:
    df["WATER"] = np.nan
df["OIL_RAW"] = df["OIL"].astype(float)

min_date = df["Date"].min().date()
max_date = df["Date"].max().date()
last_hist_date = pd.to_datetime(df["Date"].iloc[-1])

with st.sidebar.expander("Managed-rate adjustments"):
    use_hist_steps = st.checkbox(
        "Apply historical step adjustments",
        value=False,
        help="Each delta is cumulative from its date onward. Use negative values to subtract rate.",
    )
    n_hist_steps = int(
        st.number_input(
            "Historical step events",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="n_hist_steps",
        )
    )
    hist_step_dates = []
    hist_step_deltas = []
    if use_hist_steps and n_hist_steps > 0:
        for i in range(n_hist_steps):
            c1, c2 = st.columns(2)
            d = c1.date_input(
                f"Hist step {i + 1} date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                key=f"hist_step_date_{i}",
            )
            v = c2.number_input(
                f"Delta {i + 1}",
                value=0.0,
                step=10.0,
                format="%.3f",
                key=f"hist_step_delta_{i}",
            )
            hist_step_dates.append(pd.to_datetime(d))
            hist_step_deltas.append(float(v))

    use_forecast_uplifts = st.checkbox(
        "Apply forecast uplift events",
        value=False,
        help="Each event adds full rate at event date, then that added stream follows the same decline trend as main forecast.",
    )
    n_forecast_uplifts = int(
        st.number_input(
            "Forecast uplift events",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            key="n_forecast_uplifts",
        )
    )
    forecast_uplifts = []
    if use_forecast_uplifts and n_forecast_uplifts > 0:
        min_forecast_event_date = (last_hist_date + pd.DateOffset(days=1)).date()
        for i in range(n_forecast_uplifts):
            c1, c2 = st.columns(2)
            suggested = (last_hist_date + pd.DateOffset(years=i + 1)).date()
            d = c1.date_input(
                f"Uplift {i + 1} date",
                value=suggested,
                min_value=min_forecast_event_date,
                key=f"forecast_uplift_date_{i}",
            )
            v = c2.number_input(
                f"Rate {i + 1}",
                min_value=0.0,
                value=0.0,
                step=10.0,
                format="%.3f",
                key=f"forecast_uplift_rate_{i}",
            )
            forecast_uplifts.append({"date": pd.to_datetime(d), "rate": float(v)})

hist_adjust_active = use_hist_steps and len(hist_step_dates) > 0
if hist_adjust_active:
    df["Hist_Adjustment"] = apply_step_rate_adjustments(df["Date"], hist_step_dates, hist_step_deltas)
else:
    df["Hist_Adjustment"] = 0.0

df["OIL"] = np.clip(df["OIL_RAW"] + df["Hist_Adjustment"], 0.0, np.inf)
if hist_adjust_active and int((df["OIL"] <= 0).sum()) > 0:
    st.sidebar.warning("Some adjusted historical oil rates are <= 0 and will be ignored by decline fitting.")
active_forecast_uplifts = [ev for ev in forecast_uplifts if float(ev.get("rate", 0.0)) > 0]

df["CumOil"] = running_cumulative(df["Date"], df["OIL"])
df["WOR"] = np.where((df["OIL"] > 0) & (df["WATER"] >= 0), df["WATER"] / df["OIL"], np.nan)
df["WaterCut"] = np.where(np.isfinite(df["WOR"]) & (df["WOR"] >= 0), watercut_from_wor(df["WOR"]), np.nan)
df["lnWOR"] = np.where(df["WOR"] > 0, np.log(df["WOR"]), np.nan)
df["PointID"] = df.index.astype(int)

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

decline_model_choice = st.sidebar.selectbox(
    "Decline model",
    options=["Exponential", "Hyperbolic", "Harmonic"],
    index=1,
    help="Choose the Arps decline model to fit on the selected stable period.",
)

forecast_anchor_mode_choice = st.sidebar.selectbox(
    "Forecast anchoring mode",
    options=["Practical re-anchored", "Rigorous parameter preservation"],
    index=0,
    help=(
        "Practical re-anchored: start forecast from last/manual rate using fitted Di and b. "
        "Rigorous parameter preservation: preserve fitted Arps model at forecast start; for hyperbolic/harmonic this updates local D at the anchor date."
    ),
)

st.sidebar.caption("All calculations are internally converted to years.")

show_prediction_cases = st.sidebar.checkbox(
    "Show P10 / P50 / P90 prediction cases",
    value=True,
    help="Creates simple forecast sensitivity cases by varying decline rate around the base forecast.",
)
show_prediction_band = st.sidebar.checkbox(
    "Shade P10-P90 band",
    value=True,
    help="Displays the uncertainty envelope between P10 and P90 on forecast plots.",
)
p10_decline_multiplier = st.sidebar.number_input(
    "P10 decline multiplier",
    min_value=0.10,
    max_value=2.00,
    value=0.85,
    step=0.05,
    format="%.2f",
    help="P10 is optimistic by default, so it usually uses lower decline than the base case.",
)
p90_decline_multiplier = st.sidebar.number_input(
    "P90 decline multiplier",
    min_value=0.10,
    max_value=3.00,
    value=1.15,
    step=0.05,
    format="%.2f",
    help="P90 is conservative by default, so it usually uses higher decline than the base case.",
)

for key in ("sel_time", "sel_q_np", "sel_wor_np", "sel_wor_qo"):
    if key not in st.session_state:
        st.session_state[key] = []
for key in ("run_pcases_tab1", "run_pcases_tab2"):
    if key not in st.session_state:
        st.session_state[key] = False

st.caption("All calculations are internally converted to years.")

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "1) Oil Rate vs Time",
        "2) Cum Oil vs Oil Rate",
        "3) ln(WOR) vs Cum Oil",
        "4) Oil Rate from WOR + Liquid",
    ]
)

# =========================================================
# Tab 1: Oil Rate vs Time + Optional Well Count
# =========================================================
with tab1:
    left, center, right = st.columns([1, 1, 4])
    with left:
        if st.button("Clear selection", key="clear_tab1"):
            st.session_state["sel_time"] = []
            st.rerun()
    with center:
        if st.button("Run Simulation", key="run_sim_tab1"):
            st.session_state["run_pcases_tab1"] = True
    with right:
        tab1_start_rate = st.number_input(
            "First forecast point rate (at last historical date)",
            min_value=0.0,
            value=float(df["OIL"].iloc[-1]),
            step=25.0,
            format="%.2f",
            key="tab1_start_rate",
            help="Used in Practical re-anchored mode. In Rigorous parameter preservation mode, the fitted model determines the start rate at forecast origin.",
        )
        if forecast_anchor_mode_choice == "Rigorous parameter preservation":
            st.caption("Manual start rate is ignored in rigorous mode; forecast starts from model-preserved anchor at last history date.")

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
            forecast_uplifts=forecast_uplifts if use_forecast_uplifts else None,
            decline_model=decline_model_choice,
            forecast_anchor_mode=forecast_anchor_mode_choice,
        )
        prediction_cases_1 = None
        if show_prediction_cases and st.session_state.get("run_pcases_tab1", False):
            prediction_cases_1 = build_prediction_cases(
                result_1,
                p10_multiplier=p10_decline_multiplier,
                p90_multiplier=p90_decline_multiplier,
            )
        fit_error_1 = None
    except Exception as exc:
        result_1 = None
        prediction_cases_1 = None
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
            name="Historical Oil Rate (managed)",
            customdata=df["PointID"],
            marker={"size": 8, "color": marker_colors},
            hovertemplate="Date=%{x|%Y-%m-%d}<br>Rate=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    if hist_adjust_active:
        fig1.add_trace(
            go.Scatter(
                x=df["Date"],
                y=df["OIL_RAW"],
                mode="markers",
                name="Historical Oil Rate (raw)",
                marker={"size": 6, "color": "rgba(120,120,120,0.45)"},
                hovertemplate="Date=%{x|%Y-%m-%d}<br>Raw rate=%{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    if result_1 is not None:
        fig1.add_trace(
            go.Scatter(
                x=result_1["hist_model_df"]["Date"],
                y=result_1["hist_model_df"]["Model_Rate"],
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

        if show_prediction_cases and prediction_cases_1 is not None:
            if show_prediction_band:
                fig1.add_trace(
                    go.Scatter(
                        x=prediction_cases_1["dates"],
                        y=prediction_cases_1["cases"]["P90"]["rate"],
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                        name="P10-P90 band upper",
                        fill=None,
                    ),
                    row=1,
                    col=1,
                )
                fig1.add_trace(
                    go.Scatter(
                        x=prediction_cases_1["dates"],
                        y=prediction_cases_1["cases"]["P10"]["rate"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(214, 39, 40, 0.12)",
                        name="P10-P90 band",
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=1,
                )
            fig1.add_trace(
                go.Scatter(
                    x=prediction_cases_1["dates"],
                    y=prediction_cases_1["cases"]["P10"]["rate"],
                    mode="lines",
                    name="P10",
                    line={"color": "#2CA02C", "dash": "dot", "width": 2},
                ),
                row=1,
                col=1,
            )
            fig1.add_trace(
                go.Scatter(
                    x=prediction_cases_1["dates"],
                    y=prediction_cases_1["cases"]["P90"]["rate"],
                    mode="lines",
                    name="P90",
                    line={"color": "#9467BD", "dash": "dot", "width": 2},
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

    if hist_adjust_active:
        st.caption("Historical step adjustments are applied to oil rate before fitting.")
    if len(active_forecast_uplifts) > 0:
        st.caption(
            f"{len(active_forecast_uplifts)} forecast uplift event(s) are included; each adds full rate at event date and then declines with the same trend as main forecast."
        )

    if fit_window_mode == "Manual date range":
        st.caption(
            f"Manual fit range active: {pd.to_datetime(fit_start_1).date()} to {pd.to_datetime(fit_end_1).date()}"
        )

    if show_prediction_cases and not st.session_state.get("run_pcases_tab1", False):
        st.info("Click Run Simulation to generate P10 / P50 / P90 forecast cases.")

    if fit_error_1:
        st.error(f"Fit error: {fit_error_1}")
    else:
        st.write(
            f"Fit window: **{result_1['fit_start'].date()}** to **{result_1['fit_end'].date()}** "
            f"({len(result_1['fit_df'])} points)"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("qi", f"{result_1['fitted_params']['qi']:.2f}")
        c2.metric("Di (1/year)", f"{result_1['fitted_params']['Di']:.4f}")
        c3.metric("Effective annual decline", f"{100.0 * result_1['fitted_effective_annual_decline']:.2f}%")
        c4.metric("b", f"{result_1['fitted_params']['b']:.3f}")
        st.caption(f"Model: {result_1['decline_model']} | Forecast mode: {result_1['forecast_anchor_mode']}")
        if result_1['forecast_anchor_mode'] == 'Rigorous parameter preservation':
            st.caption(
                f"Forecast starts from model-preserved anchor at last history date: q={result_1['forecast_start_rate']:.2f}, D={result_1['forecast_start_D']:.4f}, effective={100.0 * result_1['forecast_start_effective_annual_decline']:.2f}%, b={result_1['forecast_start_b']:.3f}."
            )
        else:
            st.caption(
                f"Forecast starts from selected rate anchor: q={result_1['forecast_start_rate']:.2f}, D={result_1['forecast_start_D']:.4f}, effective={100.0 * result_1['forecast_start_effective_annual_decline']:.2f}%, b={result_1['forecast_start_b']:.3f}."
            )

        if show_prediction_cases and prediction_cases_1 is not None:
            case_rows_1 = []
            for case_name in ("P10", "P50", "P90"):
                rate_series = prediction_cases_1["cases"][case_name]["rate"]
                case_rows_1.append({
                    "Case": case_name,
                    "Decline multiplier": prediction_cases_1["cases"][case_name]["multiplier"],
                    "Start rate": float(rate_series[0]),
                    "End rate": float(rate_series[-1]),
                })
            st.dataframe(pd.DataFrame(case_rows_1), use_container_width=True, hide_index=True)

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
    left, center, right = st.columns([1, 1, 4])
    with left:
        if st.button("Clear selection", key="clear_tab2"):
            st.session_state["sel_q_np"] = []
            st.rerun()
    with center:
        if st.button("Run Simulation", key="run_sim_tab2"):
            st.session_state["run_pcases_tab2"] = True
    with right:
        tab2_start_rate = st.number_input(
            "First forecast point rate (at last historical date)",
            min_value=0.0,
            value=float(df["OIL"].iloc[-1]),
            step=25.0,
            format="%.2f",
            key="tab2_start_rate",
            help="Used in Practical re-anchored mode. In Rigorous parameter preservation mode, the fitted model determines the start rate at forecast origin.",
        )
        if forecast_anchor_mode_choice == "Rigorous parameter preservation":
            st.caption("Manual start rate is ignored in rigorous mode; forecast starts from model-preserved anchor at last history date.")

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
            forecast_uplifts=forecast_uplifts if use_forecast_uplifts else None,
            decline_model=decline_model_choice,
            forecast_anchor_mode=forecast_anchor_mode_choice,
        )
        prediction_cases_2 = None
        if show_prediction_cases and st.session_state.get("run_pcases_tab2", False):
            prediction_cases_2 = build_prediction_cases(
                result_2,
                p10_multiplier=p10_decline_multiplier,
                p90_multiplier=p90_decline_multiplier,
            )
        fit_error_2 = None
    except Exception as exc:
        result_2 = None
        prediction_cases_2 = None
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
                x=result_2["hist_model_df"]["CumOil"],
                y=result_2["hist_model_df"]["Model_Rate"],
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

        if show_prediction_cases and prediction_cases_2 is not None:
            case_cums_2 = {}
            for case_name in ("P10", "P50", "P90"):
                case_rates = prediction_cases_2["cases"][case_name]["rate"][1:]
                comb_dates = pd.concat([df["Date"], pd.Series(prediction_cases_2["dates"][1:])], ignore_index=True)
                comb_rates = pd.concat([df["OIL"], pd.Series(case_rates)], ignore_index=True)
                comb_cum = running_cumulative(comb_dates, comb_rates)
                case_cums_2[case_name] = np.concatenate(([result_2["hist_actual_cum"][-1]], comb_cum[len(df):]))
            if show_prediction_band:
                fig2.add_trace(
                    go.Scatter(
                        x=case_cums_2["P90"],
                        y=prediction_cases_2["cases"]["P90"]["rate"],
                        mode="lines",
                        line={"width": 0},
                        hoverinfo="skip",
                        showlegend=False,
                        name="P10-P90 band upper",
                        fill=None,
                    )
                )
                fig2.add_trace(
                    go.Scatter(
                        x=case_cums_2["P10"],
                        y=prediction_cases_2["cases"]["P10"]["rate"],
                        mode="lines",
                        line={"width": 0},
                        fill="tonexty",
                        fillcolor="rgba(214, 39, 40, 0.12)",
                        name="P10-P90 band",
                        hoverinfo="skip",
                    )
                )
            fig2.add_trace(
                go.Scatter(
                    x=case_cums_2["P10"],
                    y=prediction_cases_2["cases"]["P10"]["rate"],
                    mode="lines",
                    name="P10",
                    line={"color": "#2CA02C", "dash": "dot", "width": 2},
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=case_cums_2["P90"],
                    y=prediction_cases_2["cases"]["P90"]["rate"],
                    mode="lines",
                    name="P90",
                    line={"color": "#9467BD", "dash": "dot", "width": 2},
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

    if hist_adjust_active:
        st.caption("Historical step adjustments are applied to oil rate before fitting.")
    if len(active_forecast_uplifts) > 0:
        st.caption(
            f"{len(active_forecast_uplifts)} forecast uplift event(s) are included; each adds full rate at event date and then declines with the same trend as main forecast."
        )

    if fit_window_mode == "Manual date range":
        st.caption(
            f"Manual fit range active: {pd.to_datetime(fit_start_2).date()} to {pd.to_datetime(fit_end_2).date()}"
        )

    if show_prediction_cases and not st.session_state.get("run_pcases_tab2", False):
        st.info("Click Run Simulation to generate P10 / P50 / P90 forecast cases.")

    if fit_error_2:
        st.error(f"Fit error: {fit_error_2}")
    else:
        st.write(
            f"Fit window: **{result_2['fit_start'].date()}** to **{result_2['fit_end'].date()}** "
            f"({len(result_2['fit_df'])} points)"
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("qi", f"{result_2['fitted_params']['qi']:.2f}")
        c2.metric("Di (1/year)", f"{result_2['fitted_params']['Di']:.4f}")
        c3.metric("Effective annual decline", f"{100.0 * result_2['fitted_effective_annual_decline']:.2f}%")
        c4.metric("b", f"{result_2['fitted_params']['b']:.3f}")
        st.caption(f"Model: {result_2['decline_model']} | Forecast mode: {result_2['forecast_anchor_mode']}")
        if result_2['forecast_anchor_mode'] == 'Rigorous parameter preservation':
            st.caption(
                f"Forecast starts from model-preserved anchor at last history date: q={result_2['forecast_start_rate']:.2f}, D={result_2['forecast_start_D']:.4f}, effective={100.0 * result_2['forecast_start_effective_annual_decline']:.2f}%, b={result_2['forecast_start_b']:.3f}."
            )
        else:
            st.caption(
                f"Forecast starts from selected rate anchor: q={result_2['forecast_start_rate']:.2f}, D={result_2['forecast_start_D']:.4f}, effective={100.0 * result_2['forecast_start_effective_annual_decline']:.2f}%, b={result_2['forecast_start_b']:.3f}."
            )

        if show_prediction_cases and prediction_cases_2 is not None:
            case_rows_2 = []
            for case_name in ("P10", "P50", "P90"):
                rate_series = prediction_cases_2["cases"][case_name]["rate"]
                case_rows_2.append({
                    "Case": case_name,
                    "Decline multiplier": prediction_cases_2["cases"][case_name]["multiplier"],
                    "Start rate": float(rate_series[0]),
                    "End rate": float(rate_series[-1]),
                })
            st.dataframe(pd.DataFrame(case_rows_2), use_container_width=True, hide_index=True)

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


# =========================================================
# Tab 3: ln(WOR) vs Cum Oil (Straight-Line Fit)
# =========================================================
with tab3:
    st.latex(r"WOR = \frac{q_w}{q_o}")
    st.latex(r"WC = \frac{WOR}{1 + WOR}")

    wor_plot_df = df.loc[np.isfinite(df["lnWOR"])].copy()
    if len(wor_plot_df) < 5:
        st.error("Need at least 5 valid WOR points (qo > 0 and qw >= 0 with WOR > 0) to use this tab.")
    else:
        top_left, top_right = st.columns([1, 5])
        with top_left:
            if st.button("Clear selection", key="clear_tab3"):
                st.session_state["sel_wor_np"] = []
                st.rerun()
        with top_right:
            wor_forecast_end_date = st.date_input(
                "WOR forecast end date",
                value=(df["Date"].iloc[-1] + pd.DateOffset(years=5)).date(),
                min_value=(df["Date"].iloc[-1] + pd.DateOffset(days=1)).date(),
            )

        sel_mask_3, sel_start_3, sel_end_3 = mask_from_selection(df, st.session_state["sel_wor_np"])
        if fit_window_mode == "Manual date range":
            fit_mask_3, fit_start_3, fit_end_3 = mask_from_date_range(df, manual_fit_start, manual_fit_end)
        else:
            fit_mask_3, fit_start_3, fit_end_3 = sel_mask_3, sel_start_3, sel_end_3

        try:
            result_3 = build_wor_forecast(
                df=df,
                fit_mask=fit_mask_3,
                forecast_end_date=pd.to_datetime(wor_forecast_end_date),
            )
            fit_error_3 = None
        except Exception as exc:
            result_3 = None
            fit_error_3 = str(exc)

        marker_colors = np.where(
            fit_mask_3.loc[wor_plot_df.index].to_numpy(),
            "#1f77b4",
            "#B0B0B0",
        )

        fig3 = go.Figure()
        fig3.add_trace(
            go.Scatter(
                x=wor_plot_df["CumOil"],
                y=wor_plot_df["lnWOR"],
                mode="markers",
                name="Historical ln(WOR)",
                customdata=wor_plot_df["PointID"],
                marker={"size": 8, "color": marker_colors},
                hovertemplate="Point=%{customdata}<br>CumOil=%{x:.2f}<br>ln(WOR)=%{y:.4f}<extra></extra>",
            )
        )

        if result_3 is not None:
            fit_np = result_3["fit_np"]
            x_min = float(np.nanmin(wor_plot_df["CumOil"]))
            x_max_hist = float(np.nanmax(wor_plot_df["CumOil"]))
            x_max_fore = float(np.nanmax(result_3["forecast_df"]["Forecast_Cumulative_Oil"]))
            x_line = np.linspace(x_min, max(x_max_hist, x_max_fore), 250)
            y_line = fit_np["m_ln"] * x_line + fit_np["c_ln"]

            fig3.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    mode="lines",
                    name="Best fit straight line",
                    line={"color": "#0057B8", "width": 3},
                )
            )
            fig3.add_trace(
                go.Scatter(
                    x=result_3["forecast_df"]["Forecast_Cumulative_Oil"],
                    y=np.log(np.clip(result_3["forecast_df"]["Forecast_WOR"], 1e-12, np.inf)),
                    mode="lines",
                    name="Forecast path",
                    line={"color": "#D62728", "dash": "dash", "width": 3},
                )
            )

        fig3.update_layout(
            title="ln(WOR) vs Cumulative Oil (select points for fit window)",
            height=620,
            template="plotly_white",
            dragmode="select",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        fig3.update_xaxes(title_text="Cumulative Oil")
        fig3.update_yaxes(title_text="ln(WOR)")

        st.plotly_chart(
            fig3,
            use_container_width=True,
            key="plot_tab3",
            on_select=lambda: store_selection_from_widget("plot_tab3", "sel_wor_np"),
            selection_mode=("points", "box", "lasso"),
        )

        if fit_window_mode == "Manual date range":
            st.caption(
                f"Manual fit range active: {pd.to_datetime(fit_start_3).date()} to {pd.to_datetime(fit_end_3).date()}"
            )

        if fit_error_3:
            st.error(f"Fit error: {fit_error_3}")
        else:
            fit_np = result_3["fit_np"]
            fit_time = result_3["fit_time"]
            forecast_table = result_3["forecast_df"].copy()
            end_report = result_3["end_report"]

            st.write(
                f"Fit window: **{fit_np['fit_start'].date()}** to **{fit_np['fit_end'].date()}** "
                f"({len(fit_np['fit_df'])} valid WOR points)"
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("m (Np line)", f"{fit_np['m_ln']:.6g}")
            m2.metric("c (Np line)", f"{fit_np['c_ln']:.6g}")
            m3.metric("R2 Np-line", "N/A" if pd.isna(fit_np["r2_log"]) else f"{fit_np['r2_log']:.4f}")
            m4.metric("R2 time-line", "N/A" if pd.isna(fit_time["r2_log"]) else f"{fit_time['r2_log']:.4f}")

            st.code(
                f"ln(WOR) = {fit_np['m_ln']:.6g} * CumOil + {fit_np['c_ln']:.6g}\n"
                f"ln(WOR) = {fit_time['m_ln_time']:.6g} * t_years + {fit_time['c_ln_time']:.6g}\n"
                f"CumOil = (ln(WOR) - {fit_np['c_ln']:.6g}) / {fit_np['m_ln']:.6g}",
                language="text",
            )

            st.write("Reported values at the forecast stop date:")
            end_df = pd.DataFrame([end_report])
            st.dataframe(
                end_df.assign(Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%Y-%m-%d")),
                use_container_width=True,
                hide_index=True,
            )

            st.write("Forecast table (cumulative oil computed from WOR fit inversion):")
            st.dataframe(
                forecast_table.assign(Date=lambda x: x["Date"].dt.strftime("%Y-%m-%d")),
                use_container_width=True,
                hide_index=True,
            )

            wor_download = forecast_table.copy()
            wor_download["Date"] = wor_download["Date"].dt.strftime("%Y-%m-%d")
            st.download_button(
                "Download forecast CSV (Tab 3)",
                data=convert_df_to_csv(wor_download),
                file_name="forecast_tab3_wor_vs_cumoil.csv",
                mime="text/csv",
                key="dl_tab3_forecast",
            )


# =========================================================
# Tab 4: Oil-Rate Forecast from WOR + Liquid Rate
# =========================================================
with tab4:
    st.latex(r"q_o = \frac{q_l}{1 + WOR}")
    st.caption("Use forecasted WOR and a piecewise-constant liquid-rate schedule to forecast oil rate.")

    wor_plot_df_4 = df.loc[np.isfinite(df["lnWOR"])].copy()
    if len(wor_plot_df_4) < 5:
        st.error("Need at least 5 valid WOR points (qo > 0 and qw >= 0 with WOR > 0) to use this tab.")
    else:
        hist_last_date = pd.to_datetime(df["Date"].iloc[-1])
        default_end_date_4 = (hist_last_date + pd.DateOffset(years=5)).date()
        default_liq = float(df["OIL"].iloc[-1] + df["WATER"].iloc[-1]) if np.isfinite(df["WATER"].iloc[-1]) else float(df["OIL"].iloc[-1])

        top_left, top_right = st.columns([1, 5])
        with top_left:
            if st.button("Clear selection", key="clear_tab4"):
                st.session_state["sel_wor_qo"] = []
                st.rerun()
        with top_right:
            c1, c2 = st.columns(2)
            wor_forecast_end_date_4 = c1.date_input(
                "WOR forecast end date (Tab 4)",
                value=default_end_date_4,
                min_value=(hist_last_date + pd.DateOffset(days=1)).date(),
                key="tab4_end_date",
            )
            n_segments = int(
                c2.number_input(
                    "Liquid-rate segments",
                    min_value=1,
                    max_value=12,
                    value=1,
                    step=1,
                    key="tab4_n_segments",
                )
            )

            st.write("Liquid-rate schedule (piecewise constant):")
            seg_starts = []
            seg_rates = []
            for i in range(n_segments):
                s1, s2 = st.columns(2)
                suggested_start = (hist_last_date + pd.DateOffset(years=i)).date()
                seg_start = s1.date_input(
                    f"Segment {i + 1} start",
                    value=suggested_start,
                    key=f"tab4_seg_start_{i}",
                )
                seg_rate = s2.number_input(
                    f"Segment {i + 1} liquid rate",
                    min_value=0.0,
                    value=max(default_liq, 0.0),
                    step=10.0,
                    format="%.3f",
                    key=f"tab4_seg_rate_{i}",
                )
                seg_starts.append(pd.to_datetime(seg_start))
                seg_rates.append(float(seg_rate))

        sel_mask_4, sel_start_4, sel_end_4 = mask_from_selection(df, st.session_state["sel_wor_qo"])
        if fit_window_mode == "Manual date range":
            fit_mask_4, fit_start_4, fit_end_4 = mask_from_date_range(df, manual_fit_start, manual_fit_end)
        else:
            fit_mask_4, fit_start_4, fit_end_4 = sel_mask_4, sel_start_4, sel_end_4

        try:
            result_4 = build_wor_forecast(
                df=df,
                fit_mask=fit_mask_4,
                forecast_end_date=pd.to_datetime(wor_forecast_end_date_4),
            )

            forecast_table_4 = result_4["forecast_df"].copy()
            forecast_table_4["Liquid_Rate"] = apply_liquid_rate_schedule(
                forecast_table_4["Date"],
                seg_starts,
                seg_rates,
            )
            forecast_table_4["Forecast_OilRate"] = forecast_table_4["Liquid_Rate"] / (
                1.0 + forecast_table_4["Forecast_WOR"]
            )
            forecast_table_4["Forecast_WaterRate"] = forecast_table_4["Liquid_Rate"] - forecast_table_4["Forecast_OilRate"]

            end_report_4 = dict(result_4["end_report"])
            end_report_4["Liquid_Rate"] = float(
                apply_liquid_rate_schedule([end_report_4["Date"]], seg_starts, seg_rates)[0]
            )
            end_report_4["Forecast_OilRate"] = float(
                end_report_4["Liquid_Rate"] / (1.0 + end_report_4["Forecast_WOR"])
            )
            end_report_4["Forecast_WaterRate"] = float(end_report_4["Liquid_Rate"] - end_report_4["Forecast_OilRate"])
            fit_error_4 = None
        except Exception as exc:
            result_4 = None
            forecast_table_4 = None
            end_report_4 = None
            fit_error_4 = str(exc)

        marker_colors_4 = np.where(
            fit_mask_4.loc[wor_plot_df_4.index].to_numpy(),
            "#1f77b4",
            "#B0B0B0",
        )

        fig4_fit = go.Figure()
        fig4_fit.add_trace(
            go.Scatter(
                x=wor_plot_df_4["CumOil"],
                y=wor_plot_df_4["lnWOR"],
                mode="markers",
                name="Historical ln(WOR)",
                customdata=wor_plot_df_4["PointID"],
                marker={"size": 8, "color": marker_colors_4},
                hovertemplate="Point=%{customdata}<br>CumOil=%{x:.2f}<br>ln(WOR)=%{y:.4f}<extra></extra>",
            )
        )

        if result_4 is not None:
            fit_np_4 = result_4["fit_np"]
            x_min_4 = float(np.nanmin(wor_plot_df_4["CumOil"]))
            x_max_hist_4 = float(np.nanmax(wor_plot_df_4["CumOil"]))
            x_max_fore_4 = float(np.nanmax(result_4["forecast_df"]["Forecast_Cumulative_Oil"]))
            x_line_4 = np.linspace(x_min_4, max(x_max_hist_4, x_max_fore_4), 250)
            y_line_4 = fit_np_4["m_ln"] * x_line_4 + fit_np_4["c_ln"]

            fig4_fit.add_trace(
                go.Scatter(
                    x=x_line_4,
                    y=y_line_4,
                    mode="lines",
                    name="Best fit ln(WOR) line",
                    line={"color": "#0057B8", "width": 3},
                )
            )
            fig4_fit.add_trace(
                go.Scatter(
                    x=result_4["forecast_df"]["Forecast_Cumulative_Oil"],
                    y=np.log(np.clip(result_4["forecast_df"]["Forecast_WOR"], 1e-12, np.inf)),
                    mode="lines",
                    name="Forecast WOR path",
                    line={"color": "#D62728", "dash": "dash", "width": 3},
                )
            )

        fig4_fit.update_layout(
            title="ln(WOR) vs Cumulative Oil (fit used for oil-rate-from-liquid forecast)",
            height=560,
            template="plotly_white",
            dragmode="select",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )
        fig4_fit.update_xaxes(title_text="Cumulative Oil")
        fig4_fit.update_yaxes(title_text="ln(WOR)")

        st.plotly_chart(
            fig4_fit,
            use_container_width=True,
            key="plot_tab4_fit",
            on_select=lambda: store_selection_from_widget("plot_tab4_fit", "sel_wor_qo"),
            selection_mode=("points", "box", "lasso"),
        )

        if fit_window_mode == "Manual date range":
            st.caption(
                f"Manual fit range active: {pd.to_datetime(fit_start_4).date()} to {pd.to_datetime(fit_end_4).date()}"
            )

        if fit_error_4:
            st.error(f"Fit/forecast error: {fit_error_4}")
        else:
            st.write(
                f"Fit window: **{result_4['fit_np']['fit_start'].date()}** to **{result_4['fit_np']['fit_end'].date()}** "
                f"({len(result_4['fit_np']['fit_df'])} valid WOR points)"
            )

            fig4_rates = go.Figure()
            fig4_rates.add_trace(
                go.Scatter(
                    x=forecast_table_4["Date"],
                    y=forecast_table_4["Forecast_OilRate"],
                    mode="lines",
                    name="Forecast Oil Rate",
                    line={"color": "#2CA02C", "width": 3},
                )
            )
            fig4_rates.add_trace(
                go.Scatter(
                    x=forecast_table_4["Date"],
                    y=forecast_table_4["Liquid_Rate"],
                    mode="lines",
                    name="Applied Liquid Rate",
                    line={"color": "#D62728", "width": 2, "dash": "dot"},
                )
            )
            fig4_rates.add_trace(
                go.Scatter(
                    x=forecast_table_4["Date"],
                    y=forecast_table_4["Forecast_WaterRate"],
                    mode="lines",
                    name="Forecast Water Rate",
                    line={"color": "#0057B8", "width": 2, "dash": "dash"},
                )
            )
            fig4_rates.update_layout(
                title="Oil-Rate Forecast from Forecasted WOR and Liquid-Rate Schedule",
                height=520,
                template="plotly_white",
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            )
            fig4_rates.update_xaxes(title_text="Date")
            fig4_rates.update_yaxes(title_text="Rate")
            st.plotly_chart(fig4_rates, use_container_width=True, key="plot_tab4_rates")

            st.write("Reported values at forecast stop date:")
            st.dataframe(
                pd.DataFrame([end_report_4]).assign(Date=lambda x: pd.to_datetime(x["Date"]).dt.strftime("%Y-%m-%d")),
                use_container_width=True,
                hide_index=True,
            )

            st.write("Forecast table:")
            table_view_4 = forecast_table_4.copy()
            table_view_4["Date"] = table_view_4["Date"].dt.strftime("%Y-%m-%d")
            st.dataframe(table_view_4, use_container_width=True, hide_index=True)

            st.download_button(
                "Download forecast CSV (Tab 4)",
                data=convert_df_to_csv(table_view_4),
                file_name="forecast_tab4_oil_from_wor_liquid.csv",
                mime="text/csv",
                key="dl_tab4_forecast",
            )
