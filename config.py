"""Settings for Skylytics.

Secrets are NOT stored in this file. They are read from (in order):
  1. Environment variables AMADEUS_API_KEY / AMADEUS_API_SECRET
  2. .streamlit/secrets.toml  ->  [amadeus] api_key / api_secret
See .streamlit/secrets.toml.example.
"""
import os


def _from_streamlit_secrets(key):
    try:
        import streamlit as st
        return st.secrets["amadeus"][key]
    except Exception:
        return None


API_KEY = os.getenv("AMADEUS_API_KEY") or _from_streamlit_secrets("api_key")
API_SECRET = os.getenv("AMADEUS_API_SECRET") or _from_streamlit_secrets("api_secret")

# --- SYSTEM SETTINGS ---
# If no keys are set, the app falls back to Simulation Mode automatically.
USE_LIVE_API = bool(API_KEY and API_SECRET)
