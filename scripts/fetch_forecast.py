import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime
import pytz

# --- CONFIGURATION ---
LAT = 30.2672
LON = -97.7431
OUTPUT_FILE = "austin_forecast_live.csv"

def main():
    print("1. Setting up Forecast API client...")
    # We use a shorter cache here (e.g., 5 mins) because forecasts change frequently!
    cache_session = requests_cache.CachedSession('.cache', expire_after=300)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    # NOTE: Different URL (api.open-meteo vs archive-api)
    url = "https://api.open-meteo.com/v1/forecast"
    
    # We request the EXACT same variables as the training script
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": [
            "temperature_2m", 
            "precipitation", 
            "visibility", 
            "wind_speed_10m", 
            "weather_code"
        ],
        "timezone": "America/Chicago",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "wind_speed_unit": "mph",
        "forecast_days": 3  # Get the next 3 days of data
    }

    print("2. Fetching Live Forecast for Austin...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    hourly = response.Hourly()
    
    # Process into a dictionary (Same structure as history script)
    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        ),
        "temperature_f": hourly.Variables(0).ValuesAsNumpy(),
        "precipitation_inch": hourly.Variables(1).ValuesAsNumpy(),
        "visibility_meters": hourly.Variables(2).ValuesAsNumpy(),
        "wind_speed_mph": hourly.Variables(3).ValuesAsNumpy(),
        "weather_code": hourly.Variables(4).ValuesAsNumpy()
    }

    df = pd.DataFrame(data=hourly_data)

    # Convert Timezone
    local_tz = pytz.timezone("America/Chicago")
    df['date'] = df['date'].dt.tz_convert(local_tz)

    # FILTER: We only care about NOW onwards
    # (The API returns data starting at midnight today, but we don't need the past hours)
    now = datetime.now(local_tz)
    df = df[df['date'] >= now]

    # Format Date
    df['formatted_date'] = df['date'].dt.strftime('%Y %b %d %I:%M:%S %p')

    # Reorder to match training data exactly
    final_df = df[[
        'formatted_date', 
        'date', 
        'temperature_f', 
        'precipitation_inch', 
        'visibility_meters', 
        'wind_speed_mph', 
        'weather_code'
    ]]

    print(f"3. Saving {len(final_df)} future hours to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\n--- UPCOMING WEATHER RISK FACTORS ---")
    print(final_df[['formatted_date', 'visibility_meters', 'precipitation_inch', 'wind_speed_mph']].head(5))

if __name__ == "__main__":
    main()