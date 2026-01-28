import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Skylytics Prototype",
    page_icon="✈️",
    layout="wide"
)


# --- 1. Load Data (Fail-Safe Version) ---
@st.cache_resource
def load_data_final():
    # A. Load Random Forest Model (Required)
    try:
        model = joblib.load('flight_rf_model.pkl')
        maps = joblib.load('encoding_maps.pkl')
    except FileNotFoundError:
        st.error("⚠️ Critical Error: 'flight_rf_model.pkl' is missing. Please run your 'fix_model.py' script once.")
        st.stop()

    # B. Load Raw Data (For Live Seasonality Calculation)
    try:
        raw_data = pd.read_excel("Data_Train.xlsx")
        # Clean data immediately for the app
        raw_data.dropna(inplace=True)
        raw_data['Date_of_Journey'] = pd.to_datetime(raw_data['Date_of_Journey'], dayfirst=True)
    except FileNotFoundError:
        raw_data = pd.DataFrame()  # Return empty if missing

    return model, maps, raw_data


# Load everything into variables
model, maps, raw_data = load_data_final()

# Helper: Inverse Maps for Dropdowns
inv_airlines = {v: k for k, v in maps['airlines'].items()}
inv_dest = {v: k for k, v in maps['destinations'].items()}
source_list = ["Delhi", "Kolkata", "Bangalore", "Mumbai", "Chennai"]

# --- 2. UI Layout ---
st.title("✈️ Skylytics: AI-Driven Airfare Prediction")
st.markdown("### Dynamic Pricing & Travel Insights Prototype")

# Sidebar Inputs
st.sidebar.header("Flight Parameters")
airline_name = st.sidebar.selectbox("Airline", list(inv_airlines.values()))
source_name = st.sidebar.selectbox("Source (Origin)", source_list)
dest_name = st.sidebar.selectbox("Destination", list(inv_dest.values()))
stops = st.sidebar.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
journey_date = st.sidebar.date_input("Date of Journey", datetime.date.today())
dep_time = st.sidebar.time_input("Departure Time", datetime.time(10, 00))
arr_time = st.sidebar.time_input("Arrival Time", datetime.time(12, 30))


# --- 3. Prediction Function ---
def predict_price(airline, source, dest, stops, date, dep, arr):
    # Calculate Duration
    dep_datetime = datetime.datetime.combine(date, dep)
    arr_datetime = datetime.datetime.combine(date, arr)
    if arr_datetime < dep_datetime:
        arr_datetime += datetime.timedelta(days=1)

    duration = arr_datetime - dep_datetime
    dur_hours = duration.seconds // 3600
    dur_mins = (duration.seconds % 3600) // 60

    # Build Input Data
    input_dict = {
        'Total_Stops': maps['stops'][stops],
        'Airline': maps['airlines'][airline],
        'Destination': maps['destinations'][dest],
        'Journey_day': date.day,
        'Journey_month': date.month,
        'Dep_Time_hour': dep.hour,
        'Dep_Time_minute': dep.minute,
        'Arrival_Time_hour': arr.hour,
        'Arrival_Time_minute': arr.minute,
        'Duration_hours': dur_hours,
        'Duration_mins': dur_mins,
        'Source': source
    }

    df = pd.DataFrame([input_dict])
    valid_cols = maps['columns']
    df_final = df.reindex(columns=valid_cols, fill_value=0)

    return model.predict(df_final)[0]


# --- 4. Functionalities Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Prediction & Trends",
    "📊 Airline Comparator",
    "📈 Intraday Trends",
    "💡 AI Insights",
    "🗃️ Historical Data"
])

# TAB 1: PREDICTION + MARKET OUTLOOK

with tab1:
    # SPECIFIC FLIGHT PREDICTION
    st.subheader("✈️ Flight Price Estimator")

    col_pred_btn, col_pred_res = st.columns([1, 2])

    with col_pred_btn:
        predict_clicked = st.button("Predict Specific Fare", type="primary", use_container_width=True)

    if predict_clicked:
        price = predict_price(airline_name, source_name, dest_name, stops, journey_date, dep_time, arr_time)
        with col_pred_res:
            st.success(f"### Estimated Fare: Rs. {int(price)}")

    st.markdown("---")

    # MARKET OUTLOOK (Calculated Live)
    st.subheader("📅 Market Outlook: 30-Day Trend")
    st.caption(f"Projected price movement starting from **{journey_date}** based on historical seasonality.")

    if not raw_data.empty:
        # FAIL-SAFE CALCULATION
        # Group by Month and Day to find historical average prices
        seasonal_profile = raw_data.groupby([raw_data['Date_of_Journey'].dt.month, raw_data['Date_of_Journey'].dt.day])[
            'Price'].mean()

        # Generate next 30 days
        future_dates = [journey_date + datetime.timedelta(days=i) for i in range(30)]
        forecast_values = []

        for date in future_dates:
            try:
                # Try to find specific Month-Day average
                val = seasonal_profile.loc[(date.month, date.day)]
            except KeyError:
                # Fallback: Monthly average
                try:
                    val = seasonal_profile.loc[date.month].mean()
                except:
                    val = raw_data['Price'].mean()  # Ultimate fallback

            forecast_values.append(val)

        # Plotting the graph
        df_forecast = pd.DataFrame({'Date': future_dates, 'Trend Price': forecast_values})

        fig_forecast = px.line(
            df_forecast,
            x='Date',
            y='Trend Price',
            markers=True,
            title="Projected Seasonal Trend"
        )

        # Style: Purple line
        fig_forecast.update_traces(line_color='#8A2BE2', line_width=3)
        fig_forecast.update_layout(yaxis_title="Avg Historical Price (₹)", height=350, hovermode="x unified")

        st.plotly_chart(fig_forecast, use_container_width=True)

        # Insight
        if len(forecast_values) > 0:
            change = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100

            if change > 5:
                st.warning(f"📈 **Trend Alert:** Prices historically RISE by ~{int(change)}% in this period.")
            elif change < -5:
                st.success(f"📉 **Opportunity:** Prices historically DROP by ~{int(abs(change))}% in this period.")
            else:
                st.info("⚖️ **Stable Market:** Prices are expected to remain steady.")

    else:
        st.warning(
            "⚠️ 'Data_Train.xlsx' not found. Cannot generate market outlook. Please ensure the Excel file is in this folder.")

# TAB 2: AIRLINE COMPARATOR

with tab2:
    st.subheader("Compare Airlines")
    prices = {}
    for air in inv_airlines.values():
        p = predict_price(air, source_name, dest_name, stops, journey_date, dep_time, arr_time)
        prices[air] = p

    df_comp = pd.DataFrame(list(prices.items()), columns=['Airline', 'Price']).sort_values('Price')

    fig = px.bar(df_comp, x='Price', y='Airline', orientation='h', text_auto='.2s',
                 color='Price', color_continuous_scale='Bluyl')
    fig.update_layout(showlegend=False, height=450)
    st.plotly_chart(fig, use_container_width=True)

    cheapest = df_comp.iloc[0]['Airline']
    st.success(f"🏆 **Recommendation:** **{cheapest}** is the cheapest option.")


# TAB 3: INTRADAY TRENDS

with tab3:
    st.subheader("Best Time of Day to Fly")
    hours = list(range(0, 24, 2))
    trend_prices = []

    prog = st.progress(0)
    for i, h in enumerate(hours):
        t = datetime.time(h, 0)
        p = predict_price(airline_name, source_name, dest_name, stops, journey_date, t, arr_time)
        trend_prices.append(p)
        prog.progress((i + 1) / len(hours))
    prog.empty()

    fig_trend = px.line(x=hours, y=trend_prices, markers=True, title="Price vs. Time of Day")
    fig_trend.update_traces(line_color='#00CC96', line_width=4)
    fig_trend.update_layout(xaxis_title="Hour (24h)", yaxis_title="Price (₹)")
    st.plotly_chart(fig_trend, use_container_width=True)


# TAB 4: INSIGHTS

with tab4:
    st.subheader("Smart Insights")
    rank = maps['airlines'][airline_name]
    if rank < 3:
        st.success(f"✅ **{airline_name}** is a Budget-Friendly Carrier.")
    else:
        st.info(f"ℹ️ **{airline_name}** is a Premium/Standard Carrier.")

    if stops == "non-stop":
        st.warning("⚡ Non-stop flights command a price premium.")


# TAB 5: DATA

with tab5:
    st.subheader("Training Data Sample")
    if not raw_data.empty:
        st.dataframe(raw_data.head(50))
    else:
        st.warning("Data_Train.xlsx not found.")