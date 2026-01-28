# etl_pipeline.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings
import time

warnings.filterwarnings('ignore')


def run_etl_job():
    """
    Simulates the Extract, Transform, Load (ETL) process.
    1. Loads data (Extract)
    2. Cleans & Encodes it (Transform)
    3. Retrains Model (Load)
    """
    print("🚀 Starting Automated ETL Pipeline...")
    time.sleep(1)  # Simulate connection time

    # --- 1. EXTRACT ---
    print("📥 Extracting data from database...")
    try:
        # In a real app, this would query a SQL DB for *new* data
        # For the demo, we reload the Excel sheet
        data = pd.read_excel("Data_Train.xlsx")
    except FileNotFoundError:
        return False, "Data file not found!"

    # --- 2. TRANSFORM ---
    print("⚙️  Preprocessing & Transforming...")
    data.dropna(inplace=True)

    # Date Handling
    data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], dayfirst=True)
    data["Journey_day"] = data['Date_of_Journey'].dt.day
    data["Journey_month"] = data['Date_of_Journey'].dt.month

    # Time Handling
    data["Dep_Time"] = pd.to_datetime(data["Dep_Time"])
    data["Dep_Time_hour"] = data["Dep_Time"].dt.hour
    data["Dep_Time_minute"] = data["Dep_Time"].dt.minute

    data["Arrival_Time"] = pd.to_datetime(data["Arrival_Time"])
    data["Arrival_Time_hour"] = data["Arrival_Time"].dt.hour
    data["Arrival_Time_minute"] = data["Arrival_Time"].dt.minute

    # Duration Handling
    def preprocess_duration(x):
        if 'h' not in x:
            x = '0h' + ' ' + x
        elif 'm' not in x:
            x = x + ' ' + '0m'
        return x

    data['Duration'] = data['Duration'].apply(preprocess_duration)
    data['Duration_total_mins'] = data['Duration'].str.replace('h', "*60").str.replace(' ', '+').str.replace('m',
                                                                                                             "*1").apply(
        eval)

    # Encoding (Target Guided Ordinal Encoding)
    # This logic ensures the new model learns from new price trends
    airline_means = data.groupby(['Airline'])['Price'].mean().sort_values().index
    dict_airlines = {key: index for index, key in enumerate(airline_means, 0)}
    data['Airline'] = data['Airline'].map(dict_airlines)

    data['Destination'].replace('New Delhi', 'Delhi', inplace=True)
    dest_means = data.groupby(['Destination'])['Price'].mean().sort_values().index
    dict_dest = {key: index for index, key in enumerate(dest_means, 0)}
    data['Destination'] = data['Destination'].map(dict_dest)

    # Source Map (Need to save this too for consistency)
    source_means = data.groupby(['Source'])['Price'].mean().sort_values().index
    dict_source = {key: index for index, key in enumerate(source_means, 0)}
    data['Source'] = data['Source'].map(dict_source)

    stop_map = {'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4}
    data['Total_Stops'] = data['Total_Stops'].map(stop_map)

    # Drop Unused
    data.drop(
        columns=['Date_of_Journey', 'Additional_Info', 'Route', 'Duration', 'Dep_Time', 'Arrival_Time', 'Journey_year'],
        axis=1, inplace=True, errors='ignore')

    # --- 3. LOAD & TRAIN ---
    print("🧠 Retraining AI Model on new batch...")
    X = data.drop('Price', axis=1)
    y = data['Price']

    reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    reg_rf.fit(X, y)

    # Save Everything
    print("💾 Saving artifacts...")
    joblib.dump(reg_rf, 'flight_rf_model.pkl')

    # Save the new maps so the App stays synced with the Model
    maps = {
        'airlines': dict_airlines,
        'destinations': dict_dest,
        'sources': dict_source,
        'stops': stop_map,
        'columns': X.columns.tolist()
    }
    joblib.dump(maps, 'encoding_maps.pkl')

    time.sleep(1)
    print("✅ ETL Job Finished Successfully.")
    return True, "Model Retrained & Saved!"