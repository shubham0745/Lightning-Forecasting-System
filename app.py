import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime as dt
import os
import requests

# ================== PAGE CONFIG ==================
st.set_page_config(
    page_title="Lightning Forecasting System",
    page_icon="⚡",
    layout="wide",
)

# ================== LIGHT THEME + CUSTOM CSS ==================
st.markdown(
    """
    <style>
    .stApp {
        background: #f3f4f6;
        color: #111827;
        font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont;
    }
    /* more breathing room at top so header is not hidden */
    .main .block-container {
        padding-top: 2.3rem;
        padding-bottom: 2rem;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        color: #ea580c;
        letter-spacing: 0.02em;
        margin-bottom: 0.4rem;
    }
    .hero-subtitle {
        font-size: 0.98rem;
        color: #4b5563;
        max-width: 620px;
    }
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        background: #e0f2fe;
        color: #0369a1;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }
    .hero-wrapper {
        display: flex;
        gap: 2.0rem;
        align-items: stretch;
        margin-top: 1.0rem;
        margin-bottom: 1.3rem;
    }
    .hero-left {
        flex: 2;
    }
    .hero-right {
        flex: 1.3;
    }
    .glass-card {
        background: #ffffff;
        border-radius: 18px;
        padding: 16px 18px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 10px 25px rgba(148,163,184,0.22);
    }
    .storm-card {
        background: radial-gradient(circle at 10% 0%, #e5e7eb 0, #dbeafe 25%, #1d4ed8 70%, #020617 100%);
        border-radius: 18px;
        padding: 18px 18px;
        color: #e5e7eb;
        position: relative;
        overflow: hidden;
        box-shadow: 0 18px 40px rgba(15,23,42,0.55);
    }
    .storm-card h3 {
        margin: 0;
        font-size: 1.0rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #bfdbfe;
    }
    .storm-card p {
        font-size: 0.88rem;
        margin-top: 0.5rem;
        margin-bottom: 0.6rem;
    }
    .storm-icon {
        position: absolute;
        right: -8px;
        bottom: -12px;
        font-size: 4.4rem;
        opacity: 0.5;
    }
    .storm-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        background: rgba(15,23,42,0.65);
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    .storm-metric {
        font-size: 1.0rem;
        font-weight: 600;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.3rem 0 0.5rem 0;
        color: #111827;
    }
    .stat-row {
        display: flex;
        gap: 1.0rem;
        margin: 0.6rem 0 1.1rem 0;
        flex-wrap: wrap;
    }
    .stat-card {
        flex: 1;
        min-width: 170px;
        background: #ffffff;
        border-radius: 14px;
        padding: 10px 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 6px 16px rgba(148,163,184,0.18);
    }
    .stat-label {
        font-size: 0.8rem;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stat-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #111827;
    }
    .small-caption {
        font-size: 0.78rem;
        color: #6b7280;
    }
    .news-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 10px 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 10px;
        box-shadow: 0 4px 10px rgba(148,163,184,0.18);
    }
    .news-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #111827;
    }
    .news-meta {
        font-size: 0.75rem;
        color: #6b7280;
    }
    .risk-chip {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .risk-low    { background:#dcfce7; color:#15803d; }
    .risk-mod    { background:#fef9c3; color:#a16207; }
    .risk-high   { background:#fee2e2; color:#b91c1c; }
    .risk-vhigh  { background:#fecaca; color:#7f1d1d; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ================== SESSION STATE ==================
if "page" not in st.session_state:
    st.session_state.page = "Overview"
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "selected_date" not in st.session_state:
    st.session_state.selected_date = dt.date.today()
if "selected_district" not in st.session_state:
    st.session_state.selected_district = "Ranchi"
if "lat" not in st.session_state:
    st.session_state.lat = 23.36
if "lon" not in st.session_state:
    st.session_state.lon = 85.33

# ================== API KEYS FLAGS ==================
HAS_WEATHER_API = bool(os.environ.get("OPENWEATHER_API_KEY", "").strip())
HAS_NEWS_API = bool(os.environ.get("NEWS_API_KEY", "").strip())

# ================== LOAD MODEL (ENSEMBLE) ==================
@st.cache(allow_output_mutation=True)
def load_model():
    with open("lightning_ensemble_model.pkl", "rb") as f:
        ens = pickle.load(f)
    return ens


ens = load_model()
log_pipe = ens["log_pipe"]
mlp_pipe = ens["mlp_pipe"]
gb_pipe = ens["gb_pipe"]
feature_list = ens["feature_list"]

# ================== CONSTANTS ==================
DISTRICTS = [
    "Bokaro",
    "Chatra",
    "Deoghar",
    "Dhanbad",
    "Dumka",
    "Garhwa",
    "Giridih",
    "Godda",
    "Gumla",
    "Hazaribagh",
    "Jamtara",
    "Khunti",
    "Koderma",
    "Latehar",
    "Lohardaga",
    "Pakur",
    "Palamu",
    "Ramgarh",
    "Ranchi",
    "Sahibganj",
    "Simdega",
    "West Singhbhum",
]

DISTRICT_COORDS = {
    "Bokaro": (23.78, 86.20),
    "Chatra": (24.21, 84.87),
    "Deoghar": (24.48, 86.71),
    "Dhanbad": (23.80, 86.45),
    "Dumka": (24.27, 87.25),
    "Garhwa": (24.17, 83.80),
    "Giridih": (24.18, 86.30),
    "Godda": (24.83, 87.22),
    "Gumla": (23.04, 84.55),
    "Hazaribagh": (23.98, 85.36),
    "Jamtara": (23.95, 86.80),
    "Khunti": (23.08, 85.28),
    "Koderma": (24.47, 85.60),
    "Latehar": (23.75, 84.50),
    "Lohardaga": (23.43, 84.68),
    "Pakur": (24.64, 87.85),
    "Palamu": (24.07, 84.08),
    "Ramgarh": (23.64, 85.52),
    "Ranchi": (23.36, 85.33),
    "Sahibganj": (25.25, 87.65),
    "Simdega": (22.62, 84.50),
    "West Singhbhum": (22.57, 85.82),
}

# ================== API HELPERS ==================
def fetch_weather_from_openweather(lat: float, lon: float):
    api_key = os.environ.get("OPENWEATHER_API_KEY", "").strip()
    if not api_key:
        return None, "Weather API key not configured."

    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    )
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code == 401:
            return None, "Invalid or inactive OpenWeather API key (401)."
        if resp.status_code != 200:
            return None, f"OpenWeather API error: {resp.status_code}"

        data = resp.json()
        main = data.get("main", {})
        temp_c = main.get("temp", 28.0)
        rh = main.get("humidity", 70.0)
        pressure = main.get("pressure", 970.0)
        wind = data.get("wind", {}).get("speed", 0.0)
        rain_raw = 0.0
        if "rain" in data:
            rain_raw = data["rain"].get("1h", data["rain"].get("3h", 0.0))

        return {
            "temp_c": float(temp_c),
            "rh": float(rh),
            "pressure": float(pressure),
            "wind": float(wind),
            "rain_mm": float(rain_raw),
        }, None
    except Exception as e:
        return None, f"Request error: {e}"


def fetch_lightning_news(max_articles: int = 4):
    api_key = os.environ.get("NEWS_API_KEY", "").strip()
    if not api_key:
        return [], "News API key not configured."

    url = (
        "https://newsapi.org/v2/everything?"
        "q=lightning%20OR%20thunderstorm%20OR%20IMD%20weather%20alert&"
        "language=en&sortBy=publishedAt&pageSize=10&"
        f"apiKey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=6)
        if resp.status_code != 200:
            return [], f"News API error: {resp.status_code}"
        data = resp.json()
        return data.get("articles", [])[:max_articles], None
    except Exception as e:
        return [], f"News request error: {e}"

# ================== FEATURE HELPERS ==================
def compute_season(month: int, day: int):
    if month in [12, 1, 2]:
        return "Winter"
    elif month == 3:
        return "Summer"
    elif month in [4, 5]:
        return "PreMonsoon"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "PostMonsoon"


def compute_thi(temp_c: float, rh: float):
    return temp_c - ((0.55 - 0.0055 * rh) * (temp_c - 14.5))


def build_feature_row(
    date: dt.date,
    district: str,
    lat: float,
    lon: float,
    temp_c: float,
    rh: float,
    pressure: float,
    windspeed: float,
    rain_mm: float,
    rain_3day_sum: float,
    temp_3day_mean: float,
    rh_3day_mean: float,
    pres_3day_mean: float,
    pressure_drop: float,
):
    month_no = date.month
    day = date.day
    day_of_year = date.timetuple().tm_yday
    week_of_year = date.isocalendar().week
    season = compute_season(month_no, day)
    is_monsoon = 1 if season == "Monsoon" else 0
    thi = compute_thi(temp_c, rh)

    feats = {
        "Month_No": month_no,
        "Day": day,
        "Latitude": lat,
        "Longitude": lon,
        "Pressure_hPa": pressure,
        "Rh": rh,
        "Rain_mm": rain_mm,
        "Temp_C": temp_c,
        "WindSpeed": windspeed,
        "DayOfYear": day_of_year,
        "WeekOfYear": week_of_year,
        "IsMonsoon": is_monsoon,
        "THI": thi,
        "Temp_3day_mean": temp_3day_mean,
        "Rh_3day_mean": rh_3day_mean,
        "Rain_3day_sum": rain_3day_sum,
        "Pres_3day_mean": pres_3day_mean,
        "Pressure_Drop": pressure_drop,
    }

    for d in DISTRICTS:
        feats[f"Dist_{d}"] = 1 if d == district else 0

    feats["Season_Monsoon"] = 1 if season == "Monsoon" else 0
    feats["Season_PreMonsoon"] = 1 if season == "PreMonsoon" else 0
    feats["Season_PostMonsoon"] = 1 if season == "PostMonsoon" else 0
    feats["Season_Summer"] = 1 if season == "Summer" else 0
    feats["Season_Winter"] = 1 if season == "Winter" else 0

    row = pd.DataFrame([feats])
    row = row.reindex(columns=feature_list, fill_value=0)
    return row


def predict_ensemble_from_row(row: pd.DataFrame, threshold: float = 0.5):
    p_log = log_pipe.predict_proba(row)[:, 1]
    p_mlp = mlp_pipe.predict_proba(row)[:, 1]
    p_gb = gb_pipe.predict_proba(row)[:, 1]
    prob = (p_log + p_mlp + p_gb) / 3.0
    pred = int(prob[0] >= threshold)
    return pred, float(prob[0])


def risk_label(prob: float):
    if prob < 0.3:
        return "Low"
    elif prob < 0.6:
        return "Moderate"
    elif prob < 0.8:
        return "High"
    else:
        return "Very High"


def risk_chip_class(label: str):
    if label == "Low":
        return "risk-low"
    if label == "Moderate":
        return "risk-mod"
    if label == "High":
        return "risk-high"
    return "risk-vhigh"


def risk_message(label: str):
    if label == "Low":
        return "Low risk of lightning based on the current conditions."
    if label == "Moderate":
        return "Moderate risk. Stay alert and keep monitoring the sky."
    if label == "High":
        return "High risk of lightning. Outdoor activities should be minimized."
    return "Very high risk! Strong lightning safety precautions are recommended."

# ================== NAVIGATION ==================
st.sidebar.title("⚡ Navigation")
choice = st.sidebar.radio(
    "Go to:",
    ["Overview", "Prediction"],
    index=0 if st.session_state.page == "Overview" else 1,
)
st.session_state.page = choice

# ================== OVERVIEW PAGE ==================
def render_overview():
    # HERO
    st.markdown('<div class="pill">Lightning Risk · Jharkhand</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-wrapper">'
        '  <div class="hero-left">'
        '    <div class="hero-title">Lightning Forecasting System</div>'
        '    <div class="hero-subtitle">'
        '      A data-driven decision support tool built using LIS satellite lightning data '
        '      and ground-based meteorological time series (MTS) to estimate the probability '
        '      of lightning occurrence across districts of Jharkhand.'
        '    </div>'
        '  </div>'
        '  <div class="hero-right">'
        '    <div class="storm-card">'
        '      <h3>Monsoon Storm Monitor</h3>'
        '      <p>Track short-term lightning risk using machine learning that '
        'blends pressure, humidity, rainfall and seasonal patterns.</p>'
        '      <div class="storm-metric">ROC-AUC ≈ 0.97 (Ensemble)</div>'
        '      <div class="storm-badge">Logistic Regression · MLP · Gradient Boosting</div>'
        '      <div class="storm-icon">⚡</div>'
        '    </div>'
        '  </div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # STATS ROW
    st.markdown(
        """
        <div class="stat-row">
            <div class="stat-card">
                <div class="stat-label">Data Coverage</div>
                <div class="stat-value">2017 – 2022</div>
                <div class="small-caption">6 years of LIS lightning & district-wise MTS weather.</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Sample Size</div>
                <div class="stat-value">53,843 rows</div>
                <div class="small-caption">Balanced using SMOTE–Tomek to handle rare lightning days.</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Key Drivers</div>
                <div class="stat-value">Pressure drop · Rainfall</div>
                <div class="small-caption">Followed by temperature, humidity and 3-day aggregates.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">🌩️ Why Lightning Risk Matters</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Jharkhand reports **frequent pre-monsoon and monsoon thunderstorms**,  
              especially over Ranchi, Hazaribagh, Gumla and adjoining districts.  
            - Remote communities, open fields and construction sites are **highly vulnerable**.  
            - Timely risk information can help **farmers, students and workers** plan outdoor activities.  
            - The tool is designed as an early indication system – **not a replacement** for official IMD alerts.  
            """
        )

        st.markdown('<div class="section-title">🧠 What the Model Learns</div>', unsafe_allow_html=True)
        st.markdown(
            """
            - Joint behaviour of **temperature, humidity, rainfall, wind speed**.  
            - **Pressure and 3-day pressure mean / drop**, which capture developing low-pressure systems.  
            - **3-day aggregates** of temperature, humidity and rainfall (persistence of moist convection).  
            - **Calendar features** – day of year, week of year, and monsoon / pre-monsoon / winter tags.  
            - **District one-hot encoding**, so the model can learn spatial lightning hotspots.  
            """
        )

    with col2:
        st.markdown('<div class="section-title">📋 How the Model Works (Pipeline)</div>', unsafe_allow_html=True)
        st.markdown(
            """
            1. **Data Integration** – Merge LIS lightning events with IMD/MTS weather for each district & day.  
            2. **Pre-processing** – Clean missing values, remove duplicates and correct inconsistent records.  
            3. **Feature Engineering** – Create rolling 3-day statistics, THI, pressure drop and seasonal flags.  
            4. **Balancing** – Apply **SMOTE–Tomek** so that lightning and non-lightning days are comparable.  
            5. **Model Training** – Train Logistic Regression, MLP and Gradient Boosting on engineered features.  
            6. **Ensemble** – Average the three probabilities to obtain a **more stable and robust** risk estimate.  
            """
        )

        st.markdown('<div class="section-title">📰 Lightning & Weather Updates</div>', unsafe_allow_html=True)
        if not HAS_NEWS_API:
            st.caption(
                "News API key not set. (Optional) Set `NEWS_API_KEY` in environment to display real-time lightning / IMD news here."
            )
        else:
            news, err = fetch_lightning_news()
            if err:
                # make 401 nicer:
                if "401" in err:
                    st.caption("News feed unavailable: API key for NewsAPI is invalid or expired. This feature is optional.")
                else:
                    st.caption(f"Could not load news: {err}")
            elif not news:
                st.caption("No recent lightning-related news articles were found.")
            else:
                for art in news:
                    title = art.get("title", "No title")
                    src = (art.get("source") or {}).get("name", "Unknown")
                    url = art.get("url", "#")
                    published = art.get("publishedAt", "")[:10]
                    st.markdown(
                        f"""
                        <div class="news-card">
                            <div class="news-title">{title}</div>
                            <div class="news-meta">{src} · {published}</div>
                            <a href="{url}" target="_blank" style="font-size:0.8rem;color:#0284c7;">Read more ↗</a>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown("---")
    st.subheader("Try the Interactive Predictor")
    c1, c2, _ = st.columns([1.4, 2, 3])
    with c1:
        if st.button("Go to Prediction Page ⚡"):
            st.session_state.page = "Prediction"
            st.rerun()
    with c2:
        st.caption("Use today’s or forecasted weather to estimate lightning risk at district level.")

# ================== PREDICTION PAGE (unchanged logic) ==================
def render_prediction():
    st.markdown(
        '<div class="hero-title">Lightning Forecasting Tool 🔍</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Provide recent weather conditions for a district in Jharkhand. "
        "You can either enter values manually or (optionally) fetch live weather from **OpenWeatherMap**."
    )

    st.sidebar.subheader("Model Settings")
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05,
        help="If predicted probability ≥ threshold → Lightning (1), else No Lightning (0).",
    )

    # ---------- TOP ROW: DATE, DISTRICT, COORDS ----------
    col1, col2, col3 = st.columns([1.2, 1.2, 1])

    with col1:
        date = st.date_input("Date", value=st.session_state.selected_date)
        st.session_state.selected_date = date

    with col2:
        prev_dist = st.session_state.selected_district
        district = st.selectbox(
            "District",
            DISTRICTS,
            index=DISTRICTS.index(prev_dist) if prev_dist in DISTRICTS else 0,
            key="district_select",
        )
        if district != prev_dist:
            if district in DISTRICT_COORDS:
                st.session_state.lat, st.session_state.lon = DISTRICT_COORDS[district]
            st.session_state.weather_data = None
            st.session_state.selected_district = district

    with col3:
        lat = st.number_input(
            "Latitude",
            value=float(st.session_state.lat),
            format="%.4f",
            key="lat_input",
        )
        lon = st.number_input(
            "Longitude",
            value=float(st.session_state.lon),
            format="%.4f",
            key="lon_input",
        )
        st.session_state.lat = lat
        st.session_state.lon = lon

    # ---------- LIVE WEATHER ----------
    st.markdown("---")
    st.markdown("### 🌦 Live Weather (optional)")

    colW1, colW2 = st.columns([1.4, 2.0])

    with colW1:
        fetch_disabled = not HAS_WEATHER_API
        btn = st.button("Fetch from OpenWeatherMap", disabled=fetch_disabled)
        if fetch_disabled:
            st.caption("Live fetch disabled: set `OPENWEATHER_API_KEY` in environment to enable.")
        if btn and HAS_WEATHER_API:
            weather, err = fetch_weather_from_openweather(lat, lon)
            if err:
                st.warning(f"Could not fetch weather: {err}")
            else:
                st.session_state.weather_data = weather
                st.success("Weather data fetched and pre-filled in the input fields below.")
                st.rerun()

    with colW2:
        if st.session_state.weather_data:
            w = st.session_state.weather_data
            st.markdown(
                f"""
                **Live Weather Snapshot**  
                • Temperature: `{w["temp_c"]:.1f} °C`  
                • Humidity: `{w["rh"]:.0f} %`  
                • Pressure: `{w["pressure"]:.1f} hPa`  
                • Wind Speed: `{w["wind"]:.1f} m/s`  
                • Rain (last hr): `{w["rain_mm"]:.2f} mm`  
                """,
            )
        else:
            st.caption("Tip: Enter values manually, or configure the API key to enable live fetch for each district.")

    # ---------- WEATHER INPUTS ----------
    st.markdown("---")
    w = st.session_state.weather_data or {}
    default_temp = float(w.get("temp_c", 28.0))
    default_rh = float(w.get("rh", 70.0))
    default_pres = float(w.get("pressure", 970.0))
    default_wind = float(w.get("wind", 2.0))
    default_rain = float(w.get("rain_mm", 5.0))

    st.markdown("### 📥 Input Weather Conditions")

    col5, col6 = st.columns(2)
    with col5:
        temp_c = st.number_input("Temperature (°C)", value=default_temp, format="%.2f")
        rh = st.number_input(
            "Relative Humidity (%)",
            value=default_rh,
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
        )
        rain_mm = st.number_input(
            "Rainfall Today (mm)",
            value=default_rain,
            min_value=0.0,
            format="%.2f",
        )

    with col6:
        pressure = st.number_input(
            "Surface Pressure (hPa)", value=default_pres, format="%.2f"
        )
        windspeed = st.number_input(
            "Wind Speed (m/s)", value=default_wind, min_value=0.0, format="%.2f"
        )

    # ---------- LAST 3 DAYS ----------
    st.markdown("### 📊 Last 3 Days (optional, improves model)")

    col7, col8, col9 = st.columns(3)
    with col7:
        rain_3day_sum = st.number_input(
            "Rain (last 3 days, mm)", value=rain_mm, min_value=0.0, format="%.2f"
        )
    with col8:
        temp_3day_mean = st.number_input(
            "Temp mean (3 days, °C)", value=temp_c, format="%.2f"
        )
    with col9:
        rh_3day_mean = st.number_input(
            "RH mean (3 days, %)",
            value=rh,
            min_value=0.0,
            max_value=100.0,
            format="%.1f",
        )

    col10, col11 = st.columns(2)
    with col10:
        pres_3day_mean = st.number_input(
            "Pressure mean (3 days, hPa)", value=pressure, format="%.2f"
        )
    with col11:
        pressure_drop = st.number_input(
            "Pressure drop (today - 3-day mean, hPa)",
            value=float(pressure - pres_3day_mean),
            format="%.3f",
        )

    # ---------- PREDICT ----------
    st.markdown("---")
    if st.button("Predict Lightning Risk ⚡"):
        row = build_feature_row(
            date=date,
            district=district,
            lat=lat,
            lon=lon,
            temp_c=temp_c,
            rh=rh,
            pressure=pressure,
            windspeed=windspeed,
            rain_mm=rain_mm,
            rain_3day_sum=rain_3day_sum,
            temp_3day_mean=temp_3day_mean,
            rh_3day_mean=rh_3day_mean,
            pres_3day_mean=pres_3day_mean,
            pressure_drop=pressure_drop,
        )

        pred, prob = predict_ensemble_from_row(row, threshold=threshold)
        prob_pct = prob * 100
        label = risk_label(prob)
        chip_class = risk_chip_class(label)
        message = risk_message(label)

        st.subheader("Prediction Result")

        st.markdown(
            f"""
            <div class="glass-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div class="stat-label">Lightning Probability</div>
                        <div class="stat-value">{prob_pct:.2f}%</div>
                        <div class="small-caption">Decision threshold: {threshold:.2f}</div>
                    </div>
                    <div>
                        <span class="risk-chip {chip_class}">Risk: {label}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write(f"**Binary Prediction:** `{pred}`  (1 = Lightning, 0 = No Lightning)")
        st.info(message)

        with st.expander("Show Engineered Feature Vector sent to the model"):
            st.dataframe(row)

# ================== ROUTER ==================
if st.session_state.page == "Overview":
    render_overview()
else:
    render_prediction()
