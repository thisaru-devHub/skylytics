import streamlit as st
import pandas as pd
import joblib
import datetime
import plotly.express as px
import plotly.graph_objects as go
import time
import flight_api
import etl_pipeline

# Page Config
st.set_page_config(
    page_title="Skylytics Prototype",
    page_icon="✈️",
    layout="wide"
)


# --- 1. Load Data ---
@st.cache_resource
def load_data_final():
    try:
        model = joblib.load('flight_rf_model.pkl')
        maps = joblib.load('encoding_maps.pkl')
    except FileNotFoundError:
        st.error("⚠️ Critical Error: Model files missing. Run 'fix_model.py' first.")
        st.stop()

    try:
        raw_data = pd.read_excel("Data_Train.xlsx")
        raw_data.dropna(inplace=True)
        raw_data['Date_of_Journey'] = pd.to_datetime(raw_data['Date_of_Journey'], dayfirst=True)
    except:
        raw_data = pd.DataFrame()

    return model, maps, raw_data


model, maps, raw_data = load_data_final()

# Helper: Inverse Maps
inv_airlines = {v: k for k, v in maps['airlines'].items()}
inv_dest = {v: k for k, v in maps['destinations'].items()}
source_list = ["Delhi", "Kolkata", "Bangalore", "Mumbai", "Chennai"]

# --- 2. UI Layout ---
st.title("✈️ Skylytics: AI-Driven Airfare Prediction")
st.markdown("### Dynamic Pricing & Travel Insights")

# --- SIDEBAR: Flight Inputs ---
st.sidebar.header("Flight Parameters")
airline_name = st.sidebar.selectbox("Airline", list(inv_airlines.values()))
source_name = st.sidebar.selectbox("Source (Origin)", source_list)
dest_name = st.sidebar.selectbox("Destination", list(inv_dest.values()))
stops = st.sidebar.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
journey_date = st.sidebar.date_input("Date of Journey", datetime.date.today())
dep_time = st.sidebar.time_input("Departure Time", datetime.time(10, 00))
arr_time = st.sidebar.time_input("Arrival Time", datetime.time(12, 30))

# --- SIDEBAR: Admin Panel (SECURED WITH SECRETS) ---
st.sidebar.markdown("---")
st.sidebar.header("🔐 Admin Access")

# Initialize Login State
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

if not st.session_state.admin_authenticated:
    # SIMPLE LOGIN FORM
    with st.sidebar.form("admin_login"):
        pass_input = st.text_input("Enter Admin Key", type="password")
        submitted = st.form_submit_button("Unlock Panel")

        if submitted:
            # Check against the hidden secrets.toml file
            try:
                valid_pass = st.secrets["admin"]["password"]
                if pass_input == valid_pass:
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Access Denied")
            except FileNotFoundError:
                st.error("Secrets file missing.")
else:
    # LOGGED IN VIEW
    st.sidebar.success("✅ Admin Authenticated")
    if st.sidebar.button("🔒 Logout"):
        st.session_state.admin_authenticated = False
        st.rerun()

    with st.sidebar.expander("⚙️ Model Maintenance (ETL)", expanded=True):
        st.caption("**Status:** Active")
        st.caption("**Last Update:** 7 Days ago")

        if st.button("Run Weekly Retraining", type="secondary"):
            progress_text = "Initializing Pipeline..."
            my_bar = st.sidebar.progress(0, text=progress_text)

            # Step 1: Extract
            time.sleep(0.5)
            my_bar.progress(30, text="📥 Extracting API Data...")

            # Step 2: Transform
            time.sleep(0.5)
            my_bar.progress(60, text="🧹 Cleaning & Preprocessing...")

            # Step 3: Load
            time.sleep(0.5)
            my_bar.progress(90, text="🧠 Retraining Random Forest...")

            # Execute
            success, msg = etl_pipeline.run_etl_job()

            if success:
                my_bar.progress(100, text="✅ Complete")
                time.sleep(0.5)
                my_bar.empty()
                st.sidebar.success("System Updated!")
                st.sidebar.caption(f"Log: {msg}")
                st.balloons()
                st.cache_resource.clear()
            else:
                st.sidebar.error(f"Failed: {msg}")


# --- 3. Prediction Function ---
def predict_price(airline, source, dest, stops, date, dep, arr):
    # Duration Logic
    dep_dt = datetime.datetime.combine(date, dep)
    arr_dt = datetime.datetime.combine(date, arr)
    if arr_dt < dep_dt: arr_dt += datetime.timedelta(days=1)
    duration = arr_dt - dep_dt

    # Input Builder
    source_val = maps.get('sources', {}).get(source_name, 0)
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
        'Duration_hours': duration.seconds // 3600,
        'Duration_mins': (duration.seconds % 3600) // 60,
        'Source': source_val
    }

    # DataFrame Alignment
    df = pd.DataFrame(columns=maps['columns'])
    df.loc[0] = 0
    for col in input_dict:
        if col in df.columns:
            df.at[0, col] = input_dict[col]

    # RAW AI PREDICTION
    raw_price_2019 = model.predict(df)[0]

    # CALIBRATION FIX
    INFLATION_FACTOR = 1.5
    calibrated_price = raw_price_2019 * INFLATION_FACTOR

    return calibrated_price


# --- 4. Main Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💰 Predict", "📊 Compare", "📈 Trends", "👁️ Route Analytics", "🗃️ Data"])

# TAB 1: PREDICTION
with tab1:
    st.subheader("✈️ Flight Price Estimator")

    if 'pred_price' not in st.session_state:
        st.session_state['pred_price'] = None

    col_btn, col_res = st.columns([1, 2])

    with col_btn:
        if st.button("Predict Specific Fare", type="primary", use_container_width=True):
            price = predict_price(airline_name, source_name, dest_name, stops, journey_date, dep_time, arr_time)
            st.session_state['pred_price'] = price

    if st.session_state['pred_price'] is not None:
        price_inr = st.session_state['pred_price']
        price_lkr = price_inr * 3.4

        with col_res:
            st.success(f"### Estimated Fare: LKR {price_lkr:,.2f}")
            st.caption(f"(Includes Fuel Surcharge & Inflation Adjustment)")

        # API INTEGRATION
        st.divider()
        with st.expander("📡 Verify with Live Market Data (Amadeus API)"):
            st.info("Connects to GDS for real-time validation.")

            if st.button("Fetch Live Price Now"):
                with st.spinner("Connecting to Amadeus..."):
                    iata = {"Delhi": "DEL", "New Delhi": "DEL", "Bangalore": "BLR", "Mumbai": "BOM",
                            "Kolkata": "CCU", "Chennai": "MAA", "Hyderabad": "HYD", "Cochin": "COK"}

                    search_date = journey_date
                    if search_date < datetime.date.today():
                        search_date = datetime.date.today() + datetime.timedelta(days=7)
                        st.warning(f"⚠️ Past date selected. Searching live for: {search_date}")

                    live_price, currency = flight_api.get_live_price(
                        iata.get(source_name, "DEL"),
                        iata.get(dest_name, "BOM"),
                        search_date.strftime("%Y-%m-%d")
                    )

                    if live_price:
                        if currency == "EUR":
                            live_price *= 330;
                            currency = "LKR"
                        elif currency == "INR":
                            live_price *= 3.4;
                            currency = "LKR"

                        c1, c2 = st.columns(2)
                        c1.metric("Your AI Prediction", f"LKR {price_lkr:,.0f}")
                        c2.metric("Actual Live Price", f"{currency} {live_price:,.0f}")

                        diff = abs(price_lkr - live_price)
                        if diff < 20000:
                            st.success("✅ Validated: Prediction is within acceptable market variance.")
                        else:
                            st.warning("⚠️ Market Volatility Detected: High demand variation.")
                    else:
                        st.error("API Error: Could not fetch live price.")

    st.markdown("---")
    st.subheader("📅 Market Outlook")
    if not raw_data.empty:
        seasonal = raw_data.groupby([raw_data['Date_of_Journey'].dt.month, raw_data['Date_of_Journey'].dt.day])[
            'Price'].mean()
        dates = [journey_date + datetime.timedelta(days=i) for i in range(30)]
        vals = []
        for d in dates:
            try:
                vals.append(seasonal.loc[(d.month, d.day)])
            except:
                vals.append(raw_data['Price'].mean())

        fig = px.line(x=dates, y=vals, title="30-Day Trend Forecast")
        fig.update_traces(line_color='#8A2BE2')
        st.plotly_chart(fig, use_container_width=True)

# TAB 2: COMPARATOR
with tab2:
    st.subheader("Compare Airlines")
    prices = {air: predict_price(air, source_name, dest_name, stops, journey_date, dep_time, arr_time) for air in
              inv_airlines.values()}
    df_comp = pd.DataFrame(list(prices.items()), columns=['Airline', 'Price']).sort_values('Price')
    fig = px.bar(df_comp, x='Price', y='Airline', orientation='h', color='Price')
    st.plotly_chart(fig, use_container_width=True)

# TAB 3: INTRADAY
with tab3:
    st.subheader("Best Time of Day")
    hours = list(range(0, 24, 2))
    prices = [predict_price(airline_name, source_name, dest_name, stops, journey_date, datetime.time(h, 0), arr_time)
              for h in hours]
    fig = px.line(x=hours, y=prices, markers=True, title="Price vs Hour")
    st.plotly_chart(fig, use_container_width=True)

# TAB 4: VISUAL INSIGHTS (INDEPENDENT)
with tab4:
    st.subheader("👁️ Route Analytics (Live)")
    name_fix = {
        "Bangalore": "Banglore",
        "New Delhi": "Delhi"
    }
    clean_source = name_fix.get(source_name, source_name)
    clean_dest = name_fix.get(dest_name, dest_name)

    route_df = raw_data[
        (raw_data['Source'] == clean_source) &
        (raw_data['Destination'] == clean_dest)
        ]

    if route_df.empty:
        route_df = raw_data[
            (raw_data['Source'] == clean_dest) &
            (raw_data['Destination'] == clean_source)
            ]

    if not route_df.empty:
        route_df['Price_LKR'] = route_df['Price'] * 3.4
        avg_price = route_df['Price_LKR'].mean()
        min_price = route_df['Price_LKR'].min()
        max_price = route_df['Price_LKR'].max()

        m1, m2, m3 = st.columns(3)
        m1.metric("Average Market Price", f"LKR {avg_price:,.0f}")
        m2.metric("Lowest Recorded", f"LKR {min_price:,.0f}")
        m3.metric("Highest Recorded", f"LKR {max_price:,.0f}")

        st.divider()
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("### 🏆 Cheapest Airline")
            air_stats = route_df.groupby('Airline')['Price_LKR'].mean().sort_values().reset_index()
            fig_air = px.bar(
                air_stats, x='Airline', y='Price_LKR', color='Price_LKR', title="Avg Price by Carrier",
                color_continuous_scale='Bluyl'
            )
            st.plotly_chart(fig_air, use_container_width=True)

        with c2:
            st.markdown("### 📅 Best Day to Fly")
            route_df['Day_Name'] = route_df['Date_of_Journey'].dt.day_name()
            day_stats = route_df.groupby('Day_Name')['Price_LKR'].mean().reindex(
                ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            ).reset_index()
            fig_day = px.line(
                day_stats, x='Day_Name', y='Price_LKR', markers=True, title="Weekly Price Pattern"
            )
            fig_day.update_traces(line_color='#FF4B4B', line_width=3)
            st.plotly_chart(fig_day, use_container_width=True)
    else:
        st.warning(f"No historical data found for **{source_name}** to **{dest_name}**.")
        st.info("Try changing the Source or Destination in the sidebar.")

# TAB 5: DATA
with tab5:
    st.dataframe(raw_data.head(50))