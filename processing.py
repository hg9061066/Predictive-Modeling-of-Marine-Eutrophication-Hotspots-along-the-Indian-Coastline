import pandas as pd
import numpy as np
import sys

def clean_and_preprocess(input_file, output_file):
    """
    Loads the messy multi-header CSV with a specific structure, 
    cleans it, calculates averages, creates the 'hotspot' target, 
    and saves a clean, model-ready CSV.
    """
    print(f"Starting preprocessing of '{input_file}'...")

    try:
        # Load the CSV, but this time we will fix the headers manually
        df = pd.read_csv(input_file, header=None, skiprows=2)
        
        # Manually define the correct column headers based on the image
        # This is the most robust way to handle this specific file structure
        column_names = [
            'STN Code', 'Monitoring Location', 'Type Water Body', 'State Name',
            'Temp_Min', 'Temp_Max',
            'DO_Min', 'DO_Max',
            'pH_Min', 'pH_Max',
            'Conductivity_Min', 'Conductivity_Max',
            'BOD_Min', 'BOD_Max',
            'NitrateN_Min', 'NitrateN_Max',
            'FecalColiform_Min', 'FecalColiform_Max',
            'TotalColiform_Min', 'TotalColiform_Max',
            'FecalStreptococci_Min', 'FecalStreptococci_Max',
            'Col_22', 'Year' # Placeholder for blank column and Year
        ]
        
        # Assign the first 24 column names
        df.columns = column_names[:len(df.columns)]
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        sys.exit(1)

    print("Successfully loaded data and applied correct headers.")
    
    # --- 1. Select and Rename Columns ---
    # We now have simple, predictable column names
    df_clean = df.rename(columns={
        'STN Code': 'station_code',
        'Monitoring Location': 'location_name',
        'Year': 'year',
        'BOD_Min': 'bod_min',
        'BOD_Max': 'bod_max',
        'DO_Min': 'do_min',
        'DO_Max': 'do_max',
        'NitrateN_Min': 'nitrate_min',
        'NitrateN_Max': 'nitrate_max',
        'FecalColiform_Min': 'fecal_coliform_min',
        'FecalColiform_Max': 'fecal_coliform_max'
    })

    # Select only the columns we will use
    required_cols = [
        'station_code', 'location_name', 'year', 'bod_min', 'bod_max',
        'do_min', 'do_max', 'nitrate_min', 'nitrate_max',
        'fecal_coliform_min', 'fecal_coliform_max'
    ]
    df_clean = df_clean[required_cols].copy()
    print("Columns selected and renamed.")

    # --- 2. Clean Data and Calculate Averages ---
    param_cols = [
        'bod_min', 'bod_max', 'do_min', 'do_max', 'nitrate_min', 'nitrate_max',
        'fecal_coliform_min', 'fecal_coliform_max'
    ]
    
    for col in param_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    df_clean[param_cols] = df_clean[param_cols].fillna(0)

    df_clean['bod'] = (df_clean['bod_min'] + df_clean['bod_max']) / 2
    df_clean['dissolved_oxygen'] = (df_clean['do_min'] + df_clean['do_max']) / 2
    df_clean['nitrate'] = (df_clean['nitrate_min'] + df_clean['nitrate_max']) / 2
    df_clean['fecal_coliform'] = (df_clean['fecal_coliform_min'] + df_clean['fecal_coliform_max']) / 2
    print("Numeric values cleaned and parameter averages calculated.")

    # --- 3. Create 'hotspot' Target Variable ---
    cond_bod = df_clean['bod'] > 3
    cond_do = df_clean['dissolved_oxygen'] < 5
    cond_fecal = df_clean['fecal_coliform'] > 500
    cond_nitrate = df_clean['nitrate'] > 10
    
    hotspot_conditions = (cond_bod | cond_do | cond_fecal | cond_nitrate)
    df_clean['hotspot'] = hotspot_conditions.astype(int)
    print("Target variable 'hotspot' created.")
    
    # --- 4. Finalize and Save ---
    final_model_columns = [
        'station_code', 'location_name', 'year', 'bod', 
        'dissolved_oxygen', 'nitrate', 'fecal_coliform', 'hotspot'
    ]
    
    df_final = df_clean[final_model_columns]
    
    # Drop rows where year is not valid
    df_final = df_final[pd.to_numeric(df_final['year'], errors='coerce').notna()]
    df_final['year'] = df_final['year'].astype(int)

    df_final.to_csv(output_file, index=False)
    
    print("-" * 30)
    print(f"Success! Clean data saved to '{output_file}'")
    print("\nFirst 5 rows of the processed data:")
    print(df_final.head())
    print("\nHotspot Distribution (0 = Non-Hotspot, 1 = Hotspot):")
    print(df_final['hotspot'].value_counts())
    print("-" * 30)


if __name__ == "__main__":
    INPUT_FILENAME = r'C:\Users\white\OneDrive\Desktop\Coding\My Work\Paper 3\Merged file.csv'
    OUTPUT_FILENAME = r'C:\Users\white\OneDrive\Desktop\Coding\My Work\Paper 3\cleaned_water_quality_data.csv'
    
    clean_and_preprocess(INPUT_FILENAME, OUTPUT_FILENAME)