import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.express as px  # NEW: For interactive charts

# Page Config
st.set_page_config(
    page_title="Skylytics Prototype",
    page_icon="✈️",
    layout="wide"
)


# --- 1. Load Data & Initialize ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('flight_rf_model.pkl')
        maps = joblib.load('encoding_maps.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model files missing! Please run 'fix_model.py' first.")
        st.stop()

    # Load ARIMA Time Series Data (New)
    try:
        ts_data = joblib.load('time_series_data.pkl')
    except FileNotFoundError:
        ts_data = None  # Handle gracefully if notebook wasn't run yet

    try:
        raw_data = pd.read_excel("Data_Train.xlsx")
    except:
        raw_data = pd.DataFrame()
    return model, maps, raw_data, ts_data


model, maps, raw_data, ts_data = load_data()

# Helper: Inverse Maps for Dropdowns
inv_airlines = {v: k for k, v in maps['airlines'].items()}
inv_dest = {v: k for k, v in maps['destinations'].items()}
source_list = ["Delhi", "Kolkata", "Bangalore", "Mumbai", "Chennai"]

# --- 2. UI Layout ---
st.title("✈️ Skylytics: AI-Driven Airfare Prediction")
st.markdown("### Dynamic Pricing & Travel Insights Prototype")
st.markdown("---")

# Sidebar Inputs
st.sidebar.header("Flight Parameters")
airline_name = st.sidebar.selectbox("Airline", list(inv_airlines.values()))
source_name = st.sidebar.selectbox("Source (Origin)", source_list)
dest_name = st.sidebar.selectbox("Destination", list(inv_dest.values()))
stops = st.sidebar.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
journey_date = st.sidebar.date_input("Date of Journey", datetime.date.today())
dep_time = st.sidebar.time_input("Departure Time", datetime.time(10, 00))
arr_time = st.sidebar.time_input("Arrival Time", datetime.time(12, 30))


# --- 3. Prediction Logic ---
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

    # Filter columns to match training data
    valid_cols = maps['columns']
    df_final = df.reindex(columns=valid_cols, fill_value=0)

    return model.predict(df_final)[0]


# --- 4. Functionalities Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Prediction",
    "📊 Airline Comparator",
    "📈 Time Trends",
    "💡 AI Insights",
    "🗃️ Historical Data"
])

# Tab 1: Prediction
# Tab 1: Prediction
with tab1:
    st.subheader("Flight Price Estimator")

    # 1. Main Prediction Button
    if st.button("Predict Fare", type="primary"):
        # A. Spot Price Prediction (Random Forest)
        price = predict_price(airline_name, source_name, dest_name, stops, journey_date, dep_time, arr_time)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Estimated Fare", f"₹ {round(price, 2)}")
        with col2:
            st.success(f"Route: {source_name} ➝ {dest_name} ({stops})")

        st.info("Spot prediction based on Random Forest Model.")

        st.markdown("---")

        # B. ARIMA Time Series Forecast
        st.subheader("📅 30-Day Market Forecast (ARIMA)")

        if ts_data is not None:
            # Smart Forecasting Logic:
            # Look for seasonal matches in the ARIMA-learned trend
            future_dates = [journey_date + datetime.timedelta(days=i) for i in range(30)]
            forecast_values = []

            for date in future_dates:
                # Find matching month/day in historical trend
                match = ts_data[
                    (ts_data['Date'].dt.month == date.month) &
                    (ts_data['Date'].dt.day == date.day)
                    ]

                if not match.empty:
                    val = match['ARIMA_Price'].values[0]
                else:
                    # Fallback to monthly average if specific day is missing
                    monthly_match = ts_data[ts_data['Date'].dt.month == date.month]
                    val = monthly_match['ARIMA_Price'].mean() if not monthly_match.empty else ts_data[
                        'ARIMA_Price'].mean()

                forecast_values.append(val)

            # Plotting
            df_forecast = pd.DataFrame({
                'Date': future_dates,
                'ARIMA Trend': forecast_values
            })

            fig_forecast = px.line(
                df_forecast,
                x='Date',
                y='ARIMA Trend',
                markers=True,
                title=f"Projected Price Trend (ARIMA Analysis)"
            )

            fig_forecast.update_traces(line_color='#8A2BE2', line_width=3)
            fig_forecast.update_layout(yaxis_title="Trend Price (₹)", hovermode="x unified", height=400)

            st.write(
                "This forecast analyzes historical seasonal patterns using an **ARIMA** model to predict market movement.")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # Trend Insight
            change = ((forecast_values[-1] - forecast_values[0]) / forecast_values[0]) * 100
            if change > 5:
                st.warning(f"📈 **Trend Alert:** Prices expected to RISE by ~{round(change)}% soon.")
            elif change < -5:
                st.success(f"📉 **Good News:** Prices expected to DROP by ~{round(abs(change))}% soon.")
            else:
                st.info("⚖️ **Stable Market:** No major price fluctuations expected.")

        else:
            st.warning(
                "⚠️ Time Series data not found. Please run 'Time_Series_Analysis.ipynb' to generate the forecast model.")

# Tab 2: Compare Airlines (Auto-Refresh Version)
with tab2:
    st.subheader("Compare Airlines")
    st.write("Real-time price comparison for your selected route.")

    # 1. Calculate prices automatically (No button needed)
    prices = {}
    for air in inv_airlines.values():
        p = predict_price(air, source_name, dest_name, stops, journey_date, dep_time, arr_time)
        prices[air] = p

    # 2. Create DataFrame for Plotly
    df_comp = pd.DataFrame(list(prices.items()), columns=['Airline', 'Price']).sort_values('Price')

    # 3. Interactive Bar Chart
    fig = px.bar(
        df_comp,
        x='Price',
        y='Airline',
        orientation='h',
        title="Price Comparison by Airline",
        text_auto='.2s',
        color='Price',
        color_continuous_scale='Bluyl'
    )

    # Customize layout
    fig.update_layout(showlegend=False, height=500)
    st.plotly_chart(fig, use_container_width=True)

    # 4. Smart Recommendation
    cheapest_airline = df_comp.iloc[0]['Airline']
    cheapest_price = df_comp.iloc[0]['Price']
    st.success(f"🏆 **Recommendation:** **{cheapest_airline}** is the cheapest option (~₹{int(cheapest_price)}).")

# Tab 3: Time Trends (The Upgraded Tab!)
with tab3:
    st.subheader("Intraday Price Trends")
    st.write("See how the price changes depending on the time of day you fly.")

    # Generate data for all 24 hours
    hours = list(range(24))
    trend_prices = []

    progress_bar = st.progress(0)
    for i, h in enumerate(hours):
        t = datetime.time(h, 0)
        p = predict_price(airline_name, source_name, dest_name, stops, journey_date, t, arr_time)
        trend_prices.append(p)
        progress_bar.progress((i + 1) / 24)
    progress_bar.empty()  # Remove bar when done

    # Create DataFrame for Plotly
    df_trend = pd.DataFrame({
        'Hour of Day': hours,
        'Estimated Price (₹)': trend_prices
    })

    # Find cheapest hour
    min_price = df_trend['Estimated Price (₹)'].min()
    best_hour = df_trend.loc[df_trend['Estimated Price (₹)'] == min_price, 'Hour of Day'].iloc[0]

    # Interactive Line Chart
    fig = px.line(
        df_trend,
        x='Hour of Day',
        y='Estimated Price (₹)',
        title=f"Price Trend for {journey_date}",
        markers=True
    )

    # Customizing the look
    fig.update_traces(line_color='#00CC96', line_width=4)
    fig.update_layout(
        xaxis_title="Hour of Day (24h)",
        yaxis_title="Price (₹)",
        hovermode="x unified"  # Shows nice tooltip line
    )

    st.plotly_chart(fig, use_container_width=True)

    st.success(f"💡 **Smart Tip:** Flying around **{best_hour}:00** seems to be the cheapest time today!")

# Tab 4: Insights
with tab4:
    st.subheader("Market Intelligence")
    rank = maps['airlines'][airline_name]

    col1, col2 = st.columns(2)
    with col1:
        if rank < 3:
            st.success(f"✅ **Budget Carrier:** {airline_name} is historically one of the most affordable.")
        else:
            st.info(f"ℹ️ **Premium/Standard:** {airline_name} typically offers more amenities but may cost more.")

    with col2:
        if stops == "non-stop":
            st.warning("⚡ **Premium for Speed:** Non-stop flights often cost 15-20% more than connecting flights.")
        else:
            st.success("💰 **Saver Choice:** You are likely saving money by choosing a flight with stops.")

# Tab 5: Data
with tab5:
    st.subheader("Historical Dataset")
    if not raw_data.empty:
        st.dataframe(raw_data.head(100))
        st.caption(f"Total Records: {len(raw_data)}")
    else:
        st.warning("Data_Train.xlsx not found.")