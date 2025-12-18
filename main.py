import streamlit as st
import pandas as pd
import joblib
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.exceptions import NotFittedError


# Page Config
st.set_page_config(page_title="Skylytics Prototype", layout="wide")


# --- 1. Load Data & Initialize ---
@st.cache_resource
def load_data():
    try:
        model = joblib.load('flight_rf_model.pkl')
        maps = joblib.load('encoding_maps.pkl')
    except FileNotFoundError:
        st.error("⚠️ Model files missing! Please run the 'Save Block' in your Jupyter Notebook.")
        st.stop()

    # Load raw data for "Historical Data" tab if available
    try:
        raw_data = pd.read_excel("Data_Train.xlsx")
    except:
        raw_data = pd.DataFrame()
    return model, maps, raw_data


model, maps, raw_data = load_data()

# Helper: Inverse Maps for Dropdowns
inv_airlines = {v: k for k, v in maps['airlines'].items()}
inv_dest = {v: k for k, v in maps['destinations'].items()}
# Note: Your notebook dropped 'Source', so we provide a static list for the UI
source_list = ["Delhi", "Kolkata", "Bangalore", "Mumbai", "Chennai"]

# --- 2. UI Layout ---
st.title("✈️ Skylytics: Airfare Prediction Prototype")
st.markdown("AI-Driven Dynamic Pricing Analysis System")

# Sidebar Inputs
st.sidebar.header("Flight Details")
airline_name = st.sidebar.selectbox("Airline", list(inv_airlines.values()))
source_name = st.sidebar.selectbox("Source (Origin)", source_list)
dest_name = st.sidebar.selectbox("Destination", list(inv_dest.values()))
stops = st.sidebar.selectbox("Total Stops", ["non-stop", "1 stop", "2 stops", "3 stops", "4 stops"])
journey_date = st.sidebar.date_input("Date of Journey", datetime.date.today())
dep_time = st.sidebar.time_input("Departure Time", datetime.time(10, 00))
arr_time = st.sidebar.time_input("Arrival Time", datetime.time(12, 30))


# --- 3. Prediction Logic (Smart Handler) ---
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
    # We create a dictionary with ALL possible data points
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
        # We include Source here just in case, but filter it out later if the model doesn't use it
        'Source': source
    }

    # Create DataFrame
    df = pd.DataFrame([input_dict])

    # CRITICAL: Filter columns to match EXACTLY what the notebook model trained on
    # This automatically drops 'Source' if your notebook dropped it
    valid_cols = maps['columns']

    # Handle missing columns gracefully
    # If the app calculated something (like Duration_mins) but the model dropped it, this fixes it.
    df_final = df.reindex(columns=valid_cols, fill_value=0)

    return model.predict(df_final)[0]


# --- 4. Functionalities Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "💰 Prediction", "📊 Airline Comparator", "📈 Time Trends", "💡 Smart Insights", "🗃️ Historical Data"
])

# Tab 1: Prediction
with tab1:
    st.subheader("Flight Price Estimator")
    if st.button("Predict Fare", type="primary"):
        price = predict_price(airline_name, source_name, dest_name, stops, journey_date, dep_time, arr_time)
        st.success(f"Estimated Price: ₹ {round(price, 2)}")
        st.info("Prediction based on Random Forest Model trained on historical data.")

# Tab 2: Compare Airlines
with tab2:
    if st.button("Compare All Airlines"):
        prices = {}
        for air in inv_airlines.values():
            p = predict_price(air, source_name, dest_name, stops, journey_date, dep_time, arr_time)
            prices[air] = p

        df_comp = pd.DataFrame(list(prices.items()), columns=['Airline', 'Price']).sort_values('Price')
        st.bar_chart(df_comp.set_index('Airline'))
        st.success(f"Cheapest Option: {df_comp.iloc[0]['Airline']}")

# Tab 3: Time Trends
with tab3:
    st.write("Price trend throughout the day:")
    hours = range(0, 24, 2)
    prices = []
    for h in hours:
        t = datetime.time(h, 0)
        prices.append(predict_price(airline_name, source_name, dest_name, stops, journey_date, t, arr_time))

    fig, ax = plt.subplots()
    ax.plot(hours, prices, marker='o')
    ax.set_xlabel("Hour")
    ax.set_ylabel("Price")
    st.pyplot(fig)

# Tab 4: Insights
with tab4:
    rank = maps['airlines'][airline_name]
    if rank < 3:
        st.success(f"✅ {airline_name} is typically a budget-friendly carrier.")
    else:
        st.info(f"ℹ️ {airline_name} is a premium/mid-range carrier.")

    if stops == "non-stop":
        st.info("⚡ Non-stop flights usually cost more due to convenience.")

# Tab 5: Data
with tab5:
    if not raw_data.empty:
        st.dataframe(raw_data.head(50))
    else:
        st.warning("Data_Train.xlsx not found.")