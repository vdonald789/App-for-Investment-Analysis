import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import requests, zipfile, io, warnings

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Factor Model Analyzer", page_icon="📈", layout="wide")

st.title("📈 Fama-French Factor Model Analyzer")
st.markdown(
    "Analyze stocks, passive ETFs, and active mutual funds using **CAPM** and the "
    "**Fama-French 3-Factor Model**. Adjust the inputs in the sidebar and click **Run Analysis**."
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")
ticker_input = st.sidebar.text_input("Tickers (comma-separated)", value="TSLA, SPY, FCNTX")
start_date   = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date     = st.sidebar.date_input("End Date",   value=pd.to_datetime("2025-12-31"))
run_btn      = st.sidebar.button("🚀 Run Analysis", type="primary")
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models included**\n- CAPM (1-factor)\n- Fama-French 3-Factor\n\n"
    "**Data sources**\n- Yahoo Finance (prices)\n- Ken French Data Library (direct download)"
)

# ── FF3 loader ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_ff3_factors():
    """Fetch FF3 monthly factors directly from Ken French's website."""
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_name = z.namelist()[0]
        raw = z.read(csv_name).decode("utf-8", errors="ignore")

    # ── Parse: collect only rows where first field is a 6-digit YYYYMM integer ──
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        # Need at least 5 fields; first must be a 6-digit number (YYYYMM)
        if len(parts) < 5:
            continue
        date_str = parts[0]
        if len(date_str) == 6 and date_str.isdigit():
            try:
                vals = [float(p) for p in parts[1:5]]
                rows.append([date_str] + vals)
            except ValueError:
                continue  # skip header-like rows that sneak through

    if not rows:
        raise ValueError("FF3 CSV parsing failed — no monthly rows found.")

    df = pd.DataFrame(rows, columns=["date", "Mkt-RF", "SMB", "HML", "RF"])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m") + pd.offsets.MonthEnd(0)
    df.set_index("date", inplace=True)
    df = df.div(100)   # convert from % to decimal
    return df


@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    prices = yf.download(tickers, start=str(start), end=str(end), progress=False)["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    return prices


def run_capm(data, ticker):
    Y = data[ticker + "_excess"].dropna()
    X = sm.add_constant(data.loc[Y.index, ["Mkt-RF"]])
    return sm.OLS(Y, X).fit()


def run_ff3_model(data, ticker):
    Y = data[ticker + "_excess"].dropna()
    X = sm.add_constant(data.loc[Y.index, ["Mkt-RF", "SMB", "HML"]])
    return sm.OLS(Y, X).fit()


def model_summary_table(model):
    label_map = {"const": "Alpha (α)", "Mkt-RF": "Market Beta", "SMB": "SMB", "HML": "HML"}
    rows = []
    for k in model.params.index:
        rows.append({
            "Factor":      label_map.get(k, k),
            "Coefficient": f"{model.params[k]:.4f}",
            "p-value":     f"{model.pvalues[k]:.4f}",
            "Significant": "✅" if model.pvalues[k] < 0.05 else "❌",
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Factor": "R²", "Coefficient": f"{model.rsquared:.4f}", "p-value": "", "Significant": ""}
    return df


# ── Guard ─────────────────────────────────────────────────────────────────────
if not run_btn:
    st.info("Configure your settings in the sidebar and click **Run Analysis** to get started.")
    st.stop()

tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]
if not tickers:
    st.error("Please enter at least one ticker symbol.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# ── Load ──────────────────────────────────────────────────────────────────────
with st.spinner("Downloading price data and Fama-French factors…"):
    try:
        prices = load_prices(tickers, start_date, end_date)
        ff3    = load_ff3_factors()
    except Exception as e:
        st.error(f"Data download failed: {e}")
        st.stop()

# Monthly returns, month-end aligned
monthly_prices = prices.resample("ME").last()
returns = monthly_prices.pct_change().dropna()
returns.index = returns.index + pd.offsets.MonthEnd(0)

# Slice FF3 to date range and merge
ff3_slice = ff3.loc[str(start_date): str(end_date)]
data = returns.join(ff3_slice, how="inner")

if "Mkt-RF" not in data.columns:
    st.error("Could not merge returns with FF3 factors. Try a wider date range.")
    st.stop()

for ticker in tickers:
    if ticker in data.columns:
        data[ticker + "_excess"] = data[ticker] - data["RF"]

valid_tickers = [t for t in tickers if t in data.columns]
if not valid_tickers:
    st.error("None of the tickers had usable data. Check your symbols and date range.")
    st.stop()

missing = set(tickers) - set(valid_tickers)
if missing:
    st.warning(f"No data found for: {', '.join(missing)}. Continuing with available tickers.")

# ── Section 1: Cumulative Returns ─────────────────────────────────────────────
st.subheader("📊 Cumulative Returns")
fig, ax = plt.subplots(figsize=(10, 4))
for ticker in valid_tickers:
    cum = (1 + data[ticker]).cumprod()
    ax.plot(cum.index, cum.values, label=ticker)
ax.set_title("Cumulative Returns (monthly resampled)")
ax.set_ylabel("Growth of $1")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout(); st.pyplot(fig); plt.close()

# ── Section 2: Descriptive Stats ──────────────────────────────────────────────
st.subheader("📋 Descriptive Statistics (Monthly Returns)")
st.dataframe(data[valid_tickers].describe().T.style.format("{:.4f}"), use_container_width=True)

# ── Section 3: Per-asset Factor Results ───────────────────────────────────────
st.subheader("🔬 Factor Model Results")
tabs = st.tabs(valid_tickers)

for tab, ticker in zip(tabs, valid_tickers):
    with tab:
        st.markdown(f"### {ticker}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**CAPM (1-Factor)**")
            try:
                st.dataframe(model_summary_table(run_capm(data, ticker)), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"CAPM failed: {e}")
        with col2:
            st.markdown("**Fama-French 3-Factor**")
            try:
                st.dataframe(model_summary_table(run_ff3_model(data, ticker)), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"FF3 failed: {e}")

        st.markdown("**Rolling 24-Month Factor Betas**")
        try:
            window, asset_col, factor_cols = 24, ticker + "_excess", ["Mkt-RF", "SMB", "HML"]
            sub = data[[asset_col] + factor_cols].dropna()
            if len(sub) >= window + 5:
                roll_results = []
                for i in range(len(sub) - window + 1):
                    chunk = sub.iloc[i: i + window]
                    m = sm.OLS(chunk[asset_col], sm.add_constant(chunk[factor_cols])).fit()
                    roll_results.append({"Date": chunk.index[-1], **m.params.to_dict()})
                roll_df = pd.DataFrame(roll_results).set_index("Date")
                fig2, ax2 = plt.subplots(figsize=(10, 3))
                for col, lbl in [("Mkt-RF", "Market β"), ("SMB", "SMB β"), ("HML", "HML β")]:
                    if col in roll_df.columns:
                        ax2.plot(roll_df.index, roll_df[col], label=lbl)
                ax2.axhline(0, color="black", lw=0.8, ls="--")
                ax2.set_title(f"{ticker} – Rolling 24-Month Factor Betas")
                ax2.legend(); ax2.grid(True, alpha=0.3)
                plt.tight_layout(); st.pyplot(fig2); plt.close()
            else:
                st.info("Not enough data for rolling betas (need > 29 months).")
        except Exception as e:
            st.warning(f"Rolling beta chart error: {e}")

# ── Section 4: Cross-asset comparison ────────────────────────────────────────
st.subheader("📐 Cross-Asset Factor Comparison (FF3)")
summary_rows = []
for ticker in valid_tickers:
    try:
        m = run_ff3_model(data, ticker)
        summary_rows.append({
            "Ticker":      ticker,
            "Alpha (α)":   round(m.params.get("const", np.nan), 4),
            "Market Beta": round(m.params.get("Mkt-RF", np.nan), 4),
            "SMB":         round(m.params.get("SMB", np.nan), 4),
            "HML":         round(m.params.get("HML", np.nan), 4),
            "R²":          round(m.rsquared, 4),
        })
    except Exception:
        pass

if summary_rows:
    summary_df = pd.DataFrame(summary_rows).set_index("Ticker")
    st.dataframe(summary_df.style.format("{:.4f}"), use_container_width=True)
    fig3, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["Market Beta", "SMB", "HML"]):
        summary_df[col].plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.axhline(0, color="black", lw=0.8); ax.set_title(col)
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
    plt.suptitle("FF3 Factor Exposures by Asset", fontsize=13)
    plt.tight_layout(); st.pyplot(fig3); plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: Yahoo Finance · FF3 factors from Ken French Data Library. "
    "All returns are monthly. Significance threshold: p < 0.05."
)
