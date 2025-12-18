import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("🚀 Starting Repair Script in PyCharm...")

# 1. Load Data
try:
    data = pd.read_excel("Data_Train.xlsx")
    print("✅ Dataset loaded.")
except FileNotFoundError:
    print("❌ ERROR: 'Data_Train.xlsx' not found.")
    print("Make sure the Excel file is in the same folder as this script!")
    exit()

# 2. Preprocessing
data.dropna(inplace=True)

# --- THE FIX IS HERE ---
# Convert 'Date_of_Journey' to datetime object first!
# dayfirst=True helps pandas understand DD/MM/YYYY format correctly
data['Date_of_Journey'] = pd.to_datetime(data['Date_of_Journey'], dayfirst=True)

# Now we can extract day and month
data["Journey_day"] = data['Date_of_Journey'].dt.day
data["Journey_month"] = data['Date_of_Journey'].dt.month

# Dep_Time
data["Dep_Time"] = pd.to_datetime(data["Dep_Time"])
data["Dep_Time_hour"] = data["Dep_Time"].dt.hour
data["Dep_Time_minute"] = data["Dep_Time"].dt.minute

# Arrival_Time
data["Arrival_Time"] = pd.to_datetime(data["Arrival_Time"])
data["Arrival_Time_hour"] = data["Arrival_Time"].dt.hour
data["Arrival_Time_minute"] = data["Arrival_Time"].dt.minute

# Duration
def preprocess_duration(x):
    if 'h' not in x: x = '0h' + ' ' + x
    elif 'm' not in x: x = x + ' ' + '0m'
    return x

data['Duration'] = data['Duration'].apply(preprocess_duration)
data['Duration_hours'] = data['Duration'].apply(lambda x: int(x.split(' ')[0][:-1]))
data['Duration_mins'] = data['Duration'].apply(lambda x: int(x.split(' ')[1][:-1]))

# 3. Encoding
print("⚙️  Encoding features...")

# Airline Map
airline_means = data.groupby(['Airline'])['Price'].mean().sort_values().index
dict_airlines = {key: index for index, key in enumerate(airline_means, 0)}
data['Airline'] = data['Airline'].map(dict_airlines)

# Destination Map
data['Destination'].replace('New Delhi', 'Delhi', inplace=True)
dest_means = data.groupby(['Destination'])['Price'].mean().sort_values().index
dict_dest = {key: index for index, key in enumerate(dest_means, 0)}
data['Destination'] = data['Destination'].map(dict_dest)

# Stops Map
stop_map = {'non-stop': 0, '2 stops': 2, '1 stop': 1, '3 stops': 3, '4 stops': 4}
data['Total_Stops'] = data['Total_Stops'].map(stop_map)

# 4. Drop Columns
cols_to_drop = ['Date_of_Journey', 'Additional_Info', 'Source', 'Journey_year',
                'Route', 'Duration', 'Dep_Time', 'Arrival_Time']
data.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')

# 5. Training
print("🧠 Training the Model (Fitting)...")
X = data.drop('Price', axis=1)
y = data['Price']

reg_rf = RandomForestRegressor(n_estimators=100, random_state=42)
reg_rf.fit(X, y)

# 6. Save
print("💾 Saving fitted model...")
joblib.dump(reg_rf, 'flight_rf_model.pkl')

joblib.dump({
    'airlines': dict_airlines,
    'destinations': dict_dest,
    'stops': stop_map,
    'columns': X.columns.tolist()
}, 'encoding_maps.pkl')

print("✅ SUCCESS! The model is fixed. Now run 'streamlit run main.py' in the terminal.")