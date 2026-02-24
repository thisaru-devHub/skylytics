**Skylytics – AI-Powered Airfare Prediction System**

**Overview**

Skylytics is a machine learning-based airfare prediction system that analyzes historical flight pricing trends and integrates real-time flight data using the Amadeus REST API.
The system predicts airline ticket prices and provides booking insights based on time-series analysis.

**Features**

1. Random Forest regression model for price prediction
2. Model evaluation using R², MAE, and RMSE
3. Real-time flight data retrieval via Amadeus API
4. Data preprocessing and ETL pipeline
5. Interactive visualization using Streamlit

**Technologies Used**

Python
Pandas & NumPy
Scikit-learn
Streamlit
Amadeus REST API

**Model Evaluation Metrics**

R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Setup Instructions

**To run locally:**

Install dependencies
Create a .env file
Add:
AMADEUS_API_KEY=your_key
AMADEUS_SECRET=your_secret

Run:
streamlit run app.py
