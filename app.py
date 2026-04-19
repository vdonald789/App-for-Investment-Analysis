import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import getFamaFrenchFactors as gff
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Factor Model Analyzer", page_icon="📈", layout="wide")

st.title("📈 Fama-French Factor Model Analyzer")
st.markdown(
    "Analyze stocks, passive ETFs, and active mutual funds using **CAPM** and the "
    "**Fama-French 3-Factor Model**. Adjust the inputs in the sidebar and click **Run Analysis**."
)

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Settings")

default_tickers = "TSLA, SPY, FCNTX"
ticker_input = st.sidebar.text_input(
    "Tickers (comma-separated)",
    value=default_tickers,
    help="Enter up to 5 ticker symbols. Example: TSLA, SPY, FCNTX",
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2010-01-01"))
end_date   = st.sidebar.date_input("End Date",   value=pd.to_datetime("2025-12-31"))

run_btn = st.sidebar.button("🚀 Run Analysis", type="primary")

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Models included**\n"
    "- CAPM (1-factor)\n"
    "- Fama-French 3-Factor\n\n"
    "**Data sources**\n"
    "- Yahoo Finance (prices)\n"
    "- getFamaFrenchFactors (FF3 factors)"
)

# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_ff3_factors():
    ff3 = gff.famaFrench3Factor(frequency="m")
    ff3.rename(columns={"date_ff_factors": "Date"}, inplace=True)
    ff3.set_index("Date", inplace=True)
    return ff3


@st.cache_data(show_spinner=False)
def load_prices(tickers, start, end):
    prices = yf.download(tickers, start=str(start), end=str(end), progress=False)["Close"]
    if isinstance(prices, pd.Series):          # single ticker → make DataFrame
        prices = prices.to_frame(tickers[0])
    return prices


def run_capm(data, asset):
    Y = data[asset + "_excess"].dropna()
    X = sm.add_constant(data.loc[Y.index, ["Mkt-RF"]])
    model = sm.OLS(Y, X).fit()
    return model


def run_ff3(data, asset):
    Y = data[asset + "_excess"].dropna()
    X = sm.add_constant(data.loc[Y.index, ["Mkt-RF", "SMB", "HML"]])
    model = sm.OLS(Y, X).fit()
    return model


def model_summary_table(model, model_name):
    params = model.params
    pvals  = model.pvalues
    rows = []
    for k in params.index:
        label = {"const": "Alpha (α)", "Mkt-RF": "Market Beta", "SMB": "SMB", "HML": "HML"}.get(k, k)
        rows.append({
            "Factor": label,
            "Coefficient": f"{params[k]:.4f}",
            "p-value":     f"{pvals[k]:.4f}",
            "Significant": "✅" if pvals[k] < 0.05 else "❌",
        })
    df = pd.DataFrame(rows)
    df.loc[len(df)] = {"Factor": "R²", "Coefficient": f"{model.rsquared:.4f}", "p-value": "", "Significant": ""}
    return df


# ── Main logic ────────────────────────────────────────────────────────────────

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

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Downloading price data and Fama-French factors…"):
    try:
        prices = load_prices(tickers, start_date, end_date)
        ff3    = load_ff3_factors()
    except Exception as e:
        st.error(f"Data download failed: {e}")
        st.stop()

# Monthly returns
monthly_prices = prices.resample("ME").last()
returns = monthly_prices.pct_change().dropna()

# Merge with FF3
data = returns.join(ff3, how="inner")

# Check we have FF3 columns
if "Mkt-RF" not in data.columns:
    st.error("Could not merge returns with Fama-French factors. Check date range.")
    st.stop()

# Convert FF3 from % to decimal if needed
for col in ["Mkt-RF", "SMB", "HML", "RF"]:
    if col in data.columns and data[col].abs().mean() > 0.5:
        data[col] /= 100

# Excess returns
for ticker in tickers:
    if ticker in data.columns:
        data[ticker + "_excess"] = data[ticker] - data["RF"]

valid_tickers = [t for t in tickers if t in data.columns]
if not valid_tickers:
    st.error("None of the tickers had usable data. Please check your symbols and date range.")
    st.stop()

missing = set(tickers) - set(valid_tickers)
if missing:
    st.warning(f"No data found for: {', '.join(missing)}. Continuing with available tickers.")

# ── Section 1: Cumulative Returns Chart ───────────────────────────────────────
st.subheader("📊 Cumulative Returns")

fig, ax = plt.subplots(figsize=(10, 4))
for ticker in valid_tickers:
    cum = (1 + data[ticker]).cumprod()
    ax.plot(cum.index, cum.values, label=ticker)
ax.set_title("Cumulative Returns (monthly resampled)")
ax.set_ylabel("Growth of $1")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
st.pyplot(fig)
plt.close()

# ── Section 2: Descriptive Stats ─────────────────────────────────────────────
st.subheader("📋 Descriptive Statistics (Monthly Returns)")

desc = data[valid_tickers].describe().T
desc.index.name = "Ticker"
st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)

# ── Section 3: Factor Model Results ──────────────────────────────────────────
st.subheader("🔬 Factor Model Results")

tabs = st.tabs(valid_tickers)

for tab, ticker in zip(tabs, valid_tickers):
    with tab:
        st.markdown(f"### {ticker}")

        col1, col2 = st.columns(2)

        # CAPM
        with col1:
            st.markdown("**CAPM (1-Factor)**")
            try:
                capm = run_capm(data, ticker)
                st.dataframe(model_summary_table(capm, "CAPM"), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"CAPM failed: {e}")

        # FF3
        with col2:
            st.markdown("**Fama-French 3-Factor**")
            try:
                ff3_model = run_ff3(data, ticker)
                st.dataframe(model_summary_table(ff3_model, "FF3"), use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(f"FF3 failed: {e}")

        # Rolling betas chart
        st.markdown("**Rolling 24-Month FF3 Beta (Market)**")
        try:
            roll_results = []
            window = 24
            asset_col = ticker + "_excess"
            factor_cols = ["Mkt-RF", "SMB", "HML"]
            sub = data[[asset_col] + factor_cols].dropna()

            if len(sub) >= window + 5:
                for i in range(len(sub) - window + 1):
                    chunk = sub.iloc[i : i + window]
                    Y = chunk[asset_col]
                    X = sm.add_constant(chunk[factor_cols])
                    m = sm.OLS(Y, X).fit()
                    roll_results.append({"Date": chunk.index[-1], **m.params.to_dict()})

                roll_df = pd.DataFrame(roll_results).set_index("Date")

                fig2, ax2 = plt.subplots(figsize=(10, 3))
                for col, lbl in [("Mkt-RF", "Market β"), ("SMB", "SMB β"), ("HML", "HML β")]:
                    if col in roll_df.columns:
                        ax2.plot(roll_df.index, roll_df[col], label=lbl)
                ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
                ax2.set_title(f"{ticker} – Rolling 24-Month Factor Betas")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            else:
                st.info("Not enough data for rolling betas (need > 29 months).")
        except Exception as e:
            st.warning(f"Rolling beta chart error: {e}")

# ── Section 4: Side-by-side comparison ───────────────────────────────────────
st.subheader("📐 Cross-Asset Factor Comparison")

summary_rows = []
for ticker in valid_tickers:
    try:
        m = run_ff3(data, ticker)
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

    # Bar chart comparison
    fig3, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col in zip(axes, ["Market Beta", "SMB", "HML"]):
        summary_df[col].plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    plt.suptitle("FF3 Factor Exposures by Asset", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig3)
    plt.close()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Data: Yahoo Finance · Fama-French factors via getFamaFrenchFactors. "
    "All returns are monthly. Significance threshold: p < 0.05."
)
