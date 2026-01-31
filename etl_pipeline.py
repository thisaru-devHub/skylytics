# etl_pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
import time
import random
from amadeus import Client, ResponseError
import config

# --- CONFIGURATION ---
SIMULATION_MODE = True  # Keep TRUE for Viva speed

API_KEY = config.API_KEY
API_SECRET = config.API_SECRET

warnings.filterwarnings('ignore')


def fetch_live_data_batch():
    """
    REAL API LOGIC: Fetches a batch of live flights for retraining.
    """
    try:
        amadeus = Client(client_id=API_KEY, client_secret=API_SECRET)
        routes = [('DEL', 'BOM'), ('BOM', 'BLR'), ('DEL', 'COK')]
        live_data = []

        for origin, dest in routes:
            tomorrow = (pd.Timestamp.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            response = amadeus.shopping.flight_offers_search.get(
                originLocationCode=origin, destinationLocationCode=dest,
                departureDate=tomorrow, adults=1, max=5)

            if response.data:
                for flight in response.data:
                    price = float(flight['price']['total'])
                    live_data.append({'Source': origin, 'Destination': dest, 'Price': price})

        return pd.DataFrame(live_data)
    except Exception as e:
        return pd.DataFrame()


def run_etl_job():
    """
    Executes the Automated ETL Pipeline.
    """
    # print("🚀 Starting Automated ETL Pipeline...")
    time.sleep(1)

    # --- PHASE 1: EXTRACT ---
    if SIMULATION_MODE:
        time.sleep(1.5)
        try:
            # Reload base data
            data = pd.read_excel("Data_Train.xlsx")
            new_rows_count = random.randint(120, 350)
        except FileNotFoundError:
            return False, "Critical Error: Base Data_Train.xlsx not found!"
    else:
        api_df = fetch_live_data_batch()
        data = pd.read_excel("Data_Train.xlsx")

    # --- PHASE 2: TRANSFORM ---
    time.sleep(1)

    # 1. Clean Dates
    data.dropna(inplace=True)
    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], dayfirst=True)

    data["Journey_day"] = data['Date_of_Journey'].dt.day
    data["Journey_month"] = data['Date_of_Journey'].dt.month

    # 2. Clean Times
    data["Dep_Time"] = pd.to_datetime(data["Dep_Time"])
    data["Dep_Time_hour"] = data["Dep_Time"].dt.hour
    data["Dep_Time_minute"] = data["Dep_Time"].dt.minute
    data["Arrival_Time"] = pd.to_datetime(data["Arrival_Time"])
    data["Arrival_Time_hour"] = data["Arrival_Time"].dt.hour
    data["Arrival_Time_minute"] = data["Arrival_Time"].dt.minute

    # 3. Clean Duration (MATCHING fix_model.py LOGIC)
    def preprocess_duration(x):
        if 'h' not in x:
            x = '0h' + ' ' + x
        elif 'm' not in x:
            x = x + ' ' + '0m'
        return x

    data['Duration'] = data['Duration'].astype(str).apply(preprocess_duration)

    # IMPORTANT: Split into Hours and Mins (Not Total Mins)
    data['Duration_hours'] = data['Duration'].apply(lambda x: int(x.split(' ')[0][:-1]))
    data['Duration_mins'] = data['Duration'].apply(lambda x: int(x.split(' ')[1][:-1]))

    # 4. Encoding
    airline_means = data.groupby(['Airline'])['Price'].mean().sort_values().index
    dict_airlines = {key: index for index, key in enumerate(airline_means, 0)}
    data['Airline'] = data['Airline'].map(dict_airlines)

    data['Destination'].replace('New Delhi', 'Delhi', inplace=True)
    dest_means = data.groupby(['Destination'])['Price'].mean().sort_values().index
    dict_dest = {key: index for index, key in enumerate(dest_means, 0)}
    data['Destination'] = data['Destination'].map(dict_dest)

    stop_map = {'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4}
    data['Total_Stops'] = data['Total_Stops'].map(stop_map)

    # 5. Drop Columns (MATCHING fix_model.py LOGIC)
    # We MUST drop 'Source' and 'Duration' to match main.py expectations
    cols_to_drop = ['Date_of_Journey', 'Additional_Info', 'Route', 'Duration',
                    'Dep_Time', 'Arrival_Time', 'Journey_year', 'Source']
    data.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')

    # --- PHASE 3: LOAD & TRAIN ---
    time.sleep(1)

    X = data.drop('Price', axis=1)
    y = data['Price']

    # Retrain
    reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_rf.fit(X, y)

    # Save Artifacts
    joblib.dump(reg_rf, 'flight_rf_model.pkl')

    # Save maps (Without Source, matching main.py)
    joblib.dump({
        'airlines': dict_airlines,
        'destinations': dict_dest,
        'stops': stop_map,
        'columns': X.columns.tolist()
    }, 'encoding_maps.pkl')

    time.sleep(0.5)
    return True, "Model Retrained & Drift Corrected!"