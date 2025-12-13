import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry

# --- CONFIGURATION ---
LAT = 30.2672
LON = -97.7431
START_DATE = "2017-09-26"
END_DATE = "2025-12-12" 
OUTPUT_FILE = "data/bronze/austin_weather_training_data.csv"

def main():
    print("1. Setting up API client...")
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # UPDATED: Added variables specifically for Traffic Incident Analysis
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": [
            "temperature_2m", 
            "precipitation", 
            "visibility",        # CRITICAL for crash prediction
            "wind_speed_10m",    # Important for bridges/flyovers
            "weather_code"       # 0=Clear, 61=Rain, 95=Thunderstorm
        ],
        "timezone": "America/Chicago",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "wind_speed_unit": "mph"
    }

    print(f"2. Fetching ML-ready weather data ({START_DATE} to {END_DATE})...")
    responses = openmeteo.weather_api(url, params=params)
    response = responses[0]
    
    # Process hourly data
    hourly = response.Hourly()
    
    # Create dictionary of numpy arrays
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

    print("3. Processing DataFrame...")
    df = pd.DataFrame(data=hourly_data)

    # Convert Timezone
    df['date'] = df['date'].dt.tz_convert('America/Chicago')

    # Filter Start Date (Specific Request)
    target_start = pd.Timestamp("2017-09-26 11:11:00").tz_localize("America/Chicago")
    df = df[df['date'] >= target_start]

    # Create the formatted string column (for human reading)
    df['formatted_date'] = df['date'].dt.strftime('%Y %b %d %I:%M:%S %p')

    # KEY STEP FOR HACKATHON:
    # Ensure 'date' is kept as a datetime object so you can merge it with traffic data later
    # Reorder columns
    final_df = df[[
        'formatted_date', 
        'date', 
        'temperature_f', 
        'precipitation_inch', 
        'visibility_meters', 
        'wind_speed_mph', 
        'weather_code'
    ]]

    print(f"4. Saving {len(final_df)} rows to {OUTPUT_FILE}...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nDone! Columns available for model training:")
    print(final_df.columns.tolist())

if __name__ == "__main__":
    main()
