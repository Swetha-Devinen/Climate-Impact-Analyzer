import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="Climate Impact Analyzer", layout="wide")
st.title("Climate & Environment Impact Analyzer") #main heading

# -----------------------------
# DB Derived Functions
# -----------------------------

@st.cache_data
def get_cities():
    conn = sqlite3.connect("climate.db")
    dfc = pd.read_sql_query(
        "SELECT DISTINCT city FROM weather_monthly ORDER BY city",
        conn
    )
    conn.close()
    return dfc["city"].tolist()


@st.cache_data
def query_filtered_data(selected_cities, start_date, end_date):
    placeholders = ",".join(["?"] * len(selected_cities))
    sql = f"""
        SELECT city, month, tavg, prcp, snow
        FROM weather_monthly
        WHERE city IN ({placeholders})
          AND month >= ?
          AND month <= ?
        ORDER BY city, month
    """

    params = list(selected_cities) + [
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    ]

    conn = sqlite3.connect("climate.db")
    dfq = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    dfq["month"] = pd.to_datetime(dfq["month"])
    return dfq


@st.cache_data
def get_min_max_month(selected_cities):
    placeholders = ",".join(["?"] * len(selected_cities))
    sql = f"""
        SELECT MIN(month) AS min_m, MAX(month) AS max_m
        FROM weather_monthly
        WHERE city IN ({placeholders})
    """
    conn = sqlite3.connect("climate.db")
    out = pd.read_sql_query(sql, conn, params=list(selected_cities))
    conn.close()
    min_m = pd.to_datetime(out.loc[0, "min_m"])
    max_m = pd.to_datetime(out.loc[0, "max_m"])
    return min_m, max_m


# -----------------------------
# Derived Functions
# -----------------------------

def zscore_flags(series: pd.Series, z: float = 2.0) -> pd.Series:
    s = series.dropna()
    if len(s) < 8:
        return pd.Series(False, index=series.index)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(False, index=series.index)
    zs = (series - mean) / std
    return zs.abs() >= z


def season_from_month(dt):
    m = dt.month
    if m in [12, 1, 2]:
        return "Winter"
    if m in [3, 4, 5]:
        return "Spring"
    if m in [6, 7, 8]:
        return "Summer"
    return "Fall"


# -----------------------------
# LOAD CITIES FROM DB
# -----------------------------
cities = get_cities()

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

view_mode = st.sidebar.radio("View mode", ["Single city", "Compare cities"])

if view_mode == "Single city":
    selected_cities = [st.sidebar.selectbox("City", cities)]
else:
    selected_cities = st.sidebar.multiselect(
        "Cities (choose up to 3)",
        options=cities,
        default=cities,
        max_selections=3
    )
    if not selected_cities:
        st.warning("Please select at least one city.")
        st.stop()

metric_label = st.sidebar.selectbox(
    "Metric",
    ["Temperature (tavg)", "Rainfall (prcp)", "Snowfall (snow)"]
)

metric_col = {
    "Temperature (tavg)": "tavg",
    "Rainfall (prcp)": "prcp",
    "Snowfall (snow)": "snow"
}[metric_label]

unit = {"tavg": "°C", "prcp": "mm", "snow": "mm"}[metric_col]

top_n = st.sidebar.slider("Top N extremes", 3, 20, 5, 1)
z = st.sidebar.slider("Anomaly sensitivity (Z-score)", 1.5, 4.0, 2.0, 0.1)

# -----------------------------
# DATE RANGE
# -----------------------------
min_m, max_m = get_min_max_month(selected_cities)

start_date, end_date = st.sidebar.date_input(
    "Month range",
    value=(min_m.date(), max_m.date()),
    min_value=min_m.date(),
    max_value=max_m.date()
)

start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

filtered = query_filtered_data(selected_cities, start_date, end_date)
filtered = filtered.sort_values(["city", "month"])


# =========================================================
# SECTION 1 — KPI SUMMARY
# =========================================================
st.subheader("KPI Summary")

if view_mode == "Single city":
    c = selected_cities[0]
    s = filtered[filtered["city"] == c]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Months", len(s))
    k2.metric("Avg Temp (°C)", f"{s['tavg'].mean():.2f}" if s["tavg"].notna().any() else "NA")
    k3.metric("Total Rain (mm)", f"{s['prcp'].sum():.1f}" if s["prcp"].notna().any() else "NA")
    k4.metric("Total Snow (mm)", f"{s['snow'].sum():.1f}" if s["snow"].notna().any() else "NA")
else:
    kpi = (filtered.groupby("city", as_index=False)
           .agg(
               months=("month", "count"),
               avg_temp=("tavg", "mean"),
               total_rain=("prcp", "sum"),
               total_snow=("snow", "sum")
           ))
    kpi["avg_temp"] = kpi["avg_temp"].round(2)
    kpi["total_rain"] = kpi["total_rain"].round(1)
    kpi["total_snow"] = kpi["total_snow"].round(1)
    st.dataframe(kpi, use_container_width=True)

st.divider()

# =========================================================
# SECTION 2 — TREND CHART
# =========================================================
st.subheader("Monthly Climate Trend Analysis")

plot_df = filtered[["city", "month", metric_col]].dropna()
if plot_df.empty or plot_df["month"].nunique() < 2:
    st.warning("Not enough data to plot trend for this range.")
else:
    fig = plt.figure()
    for city in selected_cities:
        sub = plot_df[plot_df["city"] == city]
        plt.plot(sub["month"], sub[metric_col], label=city)

    plt.title(f"{metric_label} Trend")
    plt.xlabel("Month")
    plt.ylabel(f"{metric_col} ({unit})")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# =========================================================
# SECTION 3 — EXTREMES (CHARTS)
# =========================================================
st.subheader("Extreme Climate Events")

ext_df = filtered[["city", "month", metric_col]].dropna().copy()
ext_df["month"] = pd.to_datetime(ext_df["month"])  #  ensure datetime

if ext_df.empty:
    st.warning("No data available for extremes.")
else:
    highs = ext_df.sort_values(metric_col, ascending=False).head(top_n).copy()
    lows = ext_df.sort_values(metric_col, ascending=True).head(top_n).copy()

    c1, c2 = st.columns(2)

    #  Highest 
    with c1:
        st.write(f"Top {top_n} highest {metric_label} months")

        fig, ax = plt.subplots()
        x = highs["month"].dt.strftime("%Y-%m")
        ax.bar(x, highs[metric_col])
        ax.set_title("Highest Months")
        ax.set_xlabel("Month")
        ax.set_ylabel(f"{metric_label} ({unit})")
        ax.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        st.pyplot(fig)

        with st.expander("Summary table"):
            st.dataframe(highs.rename(columns={metric_col: "value"}), use_container_width=True)

    #  Lowest 
    with c2:
        st.write(f"Top {top_n} lowest {metric_label} months")

        # Special handling: for snowfall, lowest values may be all 0 → chart looks empty
        if lows[metric_col].nunique() == 1 and lows[metric_col].iloc[0] == 0:
            st.info("Lowest values are all 0 for this metric in this date range (common for snowfall).")
        else:
            fig, ax = plt.subplots()
            x = lows["month"].dt.strftime("%Y-%m")
            ax.bar(x, lows[metric_col])
            ax.set_title("Lowest Months")
            ax.set_xlabel("Month")
            ax.set_ylabel(f"{metric_label} ({unit})")
            ax.tick_params(axis="x", rotation=45)
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("Summary table"):
            st.dataframe(lows.rename(columns={metric_col: "value"}), use_container_width=True)

st.divider()

# =========================================================
# SECTION 4 — ANOMALY ALERTS (VISUAL)
# =========================================================
st.subheader("Climate Anomaly Detection")

plot_df2 = filtered[["city", "month", metric_col]].dropna().copy()
plot_df2["month"] = pd.to_datetime(plot_df2["month"])  

if plot_df2.empty or plot_df2["month"].nunique() < 2:
    st.warning("Not enough data to visualize anomalies.")
else:
    alerts_list = []

    # Detect anomalies per city
    for city in selected_cities:
        s = plot_df2[plot_df2["city"] == city].sort_values("month").reset_index(drop=True)
        if s[metric_col].dropna().empty:
            continue
        flags = zscore_flags(s[metric_col], z=float(z))
        if flags.any():
            a = s.loc[flags, ["city", "month", metric_col]].copy()
            a = a.rename(columns={metric_col: "value"})
            alerts_list.append(a)

    #  Plot trend + anomaly points 
    fig, ax = plt.subplots()

    for city in selected_cities:
        sub = plot_df2[plot_df2["city"] == city].sort_values("month")
        ax.plot(sub["month"], sub[metric_col], label=f"{city} trend")

    if alerts_list:
        alerts = pd.concat(alerts_list, ignore_index=True).sort_values(["city", "month"])

        for city in selected_cities:
            a_city = alerts[alerts["city"] == city]
            if not a_city.empty:
                ax.scatter(a_city["month"], a_city["value"], s=90, label=f"{city} anomalies")

        st.warning(f"Detected {len(alerts)} anomalies (|z| ≥ {z}).")
    else:
        alerts = pd.DataFrame()
        st.success("No anomalies detected for selected range and threshold.")

    ax.set_title(f"Anomalies  — {metric_label} (Z ≥ {z})")
    ax.set_xlabel("Month")
    ax.set_ylabel(f"{metric_label} ({unit})")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig)

    # Optional table 
    if not alerts.empty:
        with st.expander("Summary Table"):
            st.dataframe(alerts, use_container_width=True)

st.divider()

# =========================================================
# SECTION 6 — Climate Shift Detector (Year-over-Year Delta)
# =========================================================
st.subheader("Climate Shift Detector (Year-over-Year Change)")

delta_df = filtered.copy()
delta_df["month"] = pd.to_datetime(delta_df["month"])  
delta_df["year"] = delta_df["month"].dt.year

year_sum = (delta_df.groupby(["city", "year"], as_index=False)
            .agg(
                avg_temp=("tavg", "mean"),
                total_rain=("prcp", "sum"),
                total_snow=("snow", "sum")
            ))

years = sorted(year_sum["year"].dropna().unique().tolist())

if len(years) < 2:
    st.info("Select a date range that includes at least 2 years to see climate shift detection.")
else:
    y1 = years[0]
    y2 = years[-1]

    base = year_sum[year_sum["year"] == y1].set_index("city")
    latest = year_sum[year_sum["year"] == y2].set_index("city")

    # Align cities and compute deltas safely
    combined = base[["avg_temp", "total_rain", "total_snow"]].join(
        latest[["avg_temp", "total_rain", "total_snow"]],
        how="inner",
        lsuffix=f"_{y1}",
        rsuffix=f"_{y2}"
    ).reset_index()

    combined["Δ Avg Temp (°C)"] = combined[f"avg_temp_{y2}"] - combined[f"avg_temp_{y1}"]
    combined["Δ Total Rain (mm)"] = combined[f"total_rain_{y2}"] - combined[f"total_rain_{y1}"]
    combined["Δ Total Snow (mm)"] = combined[f"total_snow_{y2}"] - combined[f"total_snow_{y1}"]

    delta = combined[["city", "Δ Avg Temp (°C)", "Δ Total Rain (mm)", "Δ Total Snow (mm)"]].copy()

    st.write(f"Change in **{y2} − {y1}** (per city)")
    st.dataframe(delta, use_container_width=True)

    # --- Delta charts ---
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots()
        ax.bar(delta["city"], delta["Δ Avg Temp (°C)"])
        ax.set_title("Δ Avg Temp")
        ax.set_xlabel("City")
        ax.set_ylabel("°C")
        plt.tight_layout()
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots()
        ax.bar(delta["city"], delta["Δ Total Rain (mm)"])
        ax.set_title("Δ Total Rain")
        ax.set_xlabel("City")
        ax.set_ylabel("mm")
        plt.tight_layout()
        st.pyplot(fig)

    with c3:
        # If all snow delta values are 0, it looks blank — show message
        if delta["Δ Total Snow (mm)"].nunique() == 1 and float(delta["Δ Total Snow (mm)"].iloc[0]) == 0.0:
            st.info("Snowfall change is 0 for all selected cities in this year range.")
        else:
            fig, ax = plt.subplots()
            ax.bar(delta["city"], delta["Δ Total Snow (mm)"])
            ax.set_title("Δ Total Snow")
            ax.set_xlabel("City")
            ax.set_ylabel("mm")
            plt.tight_layout()
            st.pyplot(fig)

st.divider()

def plot_season_fingerprint(season_summary_city: pd.DataFrame, city_name: str):
    season_order = ["Winter", "Spring", "Summer", "Fall"]

    dfc = season_summary_city.copy()
    dfc["season"] = pd.Categorical(dfc["season"], categories=season_order, ordered=True)
    dfc = dfc.sort_values("season")

    # Create matrix
    mat = dfc[["tavg", "prcp", "snow"]].to_numpy(dtype=float)

    # Normalize per column (for visualization only)
    mat_norm = mat.copy()
    for j in range(mat_norm.shape[1]):
        col = mat_norm[:, j]
        if np.nanmax(col) != np.nanmin(col):
            mat_norm[:, j] = (col - np.nanmin(col)) / (np.nanmax(col) - np.nanmin(col))
        else:
            mat_norm[:, j] = 0

    fig, ax = plt.subplots()
    im = ax.imshow(mat_norm, aspect="auto")

    ax.set_title(f"Season Fingerprint — {city_name}")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Avg Temp", "Total Rain", "Total Snow"])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(season_order)

    # Show real values inside cells
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if not np.isnan(val):
                text = f"{val:.1f}"
            else:
                text = "NA"
            ax.text(j, i, text, ha="center", va="center")

    plt.tight_layout()
    st.pyplot(fig)

# =========================================================
# Season  (Heatmap)
# =========================================================
st.subheader("Seasonal Matrix")

finger_df = filtered.copy()
finger_df["month"] = pd.to_datetime(finger_df["month"])  
finger_df["season"] = finger_df["month"].apply(season_from_month)

finger_summary = (finger_df.groupby(["city", "season"], as_index=False)
                  .agg(
                      tavg=("tavg", "mean"),
                      prcp=("prcp", "sum"),
                      snow=("snow", "sum")
                  ))

if finger_summary.empty:
    st.info("No data available to build Seasonal Matrix.")
else:
    season_order = ["Winter", "Spring", "Summer", "Fall"]

    # Make sure every city has all 4 seasons (fill missing with NaN)
    for c in selected_cities:
        city_df = finger_summary[finger_summary["city"] == c][["season", "tavg", "prcp", "snow"]].copy()

        # Reindex to force all seasons
        city_df["season"] = pd.Categorical(city_df["season"], categories=season_order, ordered=True)
        city_df = city_df.set_index("season").reindex(season_order).reset_index()

        if city_df.empty:
            st.info(f"No seasonal data for {c}.")
            continue

        plot_season_fingerprint(city_df, c)

st.divider()