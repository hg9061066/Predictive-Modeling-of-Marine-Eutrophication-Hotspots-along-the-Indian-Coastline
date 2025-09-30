import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
import os

def forecast_parameters(input_file):
    """
    Identifies high-risk stations and uses linear regression to forecast
    BOD and Dissolved Oxygen levels for 2025.
    
    Args:
        input_file (str): The name of the cleaned data file.
    """
    print("--- Starting Time-Series Forecasting for 2025 ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Successfully loaded '{input_file}'.")

    # --- 2. Identify High-Risk Stations ---
    # A high-risk station is any station that has been a hotspot at least once.
    hotspot_stations = df[df['hotspot'] == 1]['station_code'].unique()
    
    if len(hotspot_stations) == 0:
        print("No high-risk stations found. Exiting.")
        return

    print(f"Identified {len(hotspot_stations)} high-risk stations for analysis.")
    
    high_risk_df = df[df['station_code'].isin(hotspot_stations)].copy()

    # --- 3. Forecast for Each Station and Parameter ---
    forecast_results = []
    parameters_to_forecast = ['bod', 'dissolved_oxygen']
    forecast_year = 2025

    for station in hotspot_stations:
        for param in parameters_to_forecast:
            # Get historical data for the specific station and parameter
            station_data = high_risk_df[high_risk_df['station_code'] == station][['year', param]].dropna()
            station_data = station_data.sort_values(by='year')

            # We need at least 2 data points to fit a line
            if len(station_data) < 2:
                continue

            # Prepare data for Scikit-learn
            X = station_data[['year']]
            y = station_data[param]

            # Train the Linear Regression model
            model = LinearRegression()
            model.fit(X, y)

            # Predict the value for 2025
            predicted_value = model.predict(np.array([[forecast_year]]))[0]

            # Store the result
            forecast_results.append({
                'station_code': station,
                'location_name': df[df['station_code'] == station]['location_name'].iloc[0],
                'parameter': param,
                'forecast_2025': predicted_value,
                'trend_slope': model.coef_[0] # Positive slope = increasing, Negative = decreasing
            })

    # Convert results to a DataFrame
    forecast_df = pd.DataFrame(forecast_results)
    print(f"\nGenerated forecasts for {len(forecast_df)} station-parameter pairs.")

    # Save forecasts to CSV
    output_csv = 'parameter_forecasts_2025.csv'
    forecast_df.to_csv(output_csv, index=False)
    print(f"Forecasts saved to '{output_csv}'")
    
    print("\n--- Top 10 Forecasted Trends ---")
    print("Top 5 Worsening BOD Trends (Increasing):")
    print(forecast_df[forecast_df['parameter'] == 'bod'].sort_values(by='trend_slope', ascending=False).head())
    
    print("\nTop 5 Worsening Dissolved Oxygen Trends (Decreasing):")
    print(forecast_df[forecast_df['parameter'] == 'dissolved_oxygen'].sort_values(by='trend_slope', ascending=True).head())


    # --- 4. Visualize Trends for Top Stations ---
    # Visualize the top 2 stations with the worst BOD trends
    top_bod_stations = forecast_df[forecast_df['parameter'] == 'bod'].sort_values(by='trend_slope', ascending=False).head(2)

    if not os.path.exists('forecast_plots'):
        os.makedirs('forecast_plots')

    for index, row in top_bod_stations.iterrows():
        station_code = row['station_code']
        history = high_risk_df[high_risk_df['station_code'] == station_code]
        
        X_hist = history[['year']]
        y_hist = history['bod']
        
        model = LinearRegression()
        model.fit(X_hist, y_hist)
        
        # Create a trend line for plotting
        years_for_line = np.array([[history['year'].min()], [forecast_year]])
        trend_line = model.predict(years_for_line)

        plt.figure(figsize=(10, 6))
        plt.scatter(X_hist, y_hist, color='blue', label='Historical Data', zorder=5)
        plt.plot(years_for_line, trend_line, color='red', linestyle='--', label='Linear Trend')
        plt.scatter([forecast_year], [row['forecast_2025']], color='green', marker='*', s=200, label='2025 Forecast', zorder=5)
        
        plt.title(f'BOD Forecast for Station {int(station_code)}\n{row["location_name"][:50]}...')
        plt.xlabel('Year')
        plt.ylabel('BOD (mg/L)')
        plt.legend()
        plt.grid(True)
        
        plot_filename = f'forecast_plots/bod_forecast_station_{int(station_code)}.png'
        plt.savefig(plot_filename)
        print(f"\nSaved trend plot to '{plot_filename}'")
    
    print("\n--- Forecasting Complete ---")

if __name__ == "__main__":
    INPUT_FILENAME = r'C:\Users\white\OneDrive\Desktop\Coding\My Work\Paper 3\cleaned_water_quality_data.csv'
    forecast_parameters(INPUT_FILENAME)