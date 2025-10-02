import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

def train_xgboost(input_file):
    """
    Loads cleaned data, trains an XGBoost model to predict hotspots,
    and evaluates its performance.
    
    Args:
        input_file (str): The name of the cleaned CSV file.
    """
    print("--- Training and Evaluating XGBoost Model ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Successfully loaded '{input_file}'.")
    df.dropna(inplace=True)

    # --- 2. Prepare Data ---
    # Encode categorical features
    df['station_code_encoded'] = pd.factorize(df['station_code'])[0]
    df['location_name_encoded'] = pd.factorize(df['location_name'])[0]
    
    features = [
        'year', 'bod', 'dissolved_oxygen', 'nitrate', 
        'fecal_coliform', 'station_code_encoded', 'location_name_encoded'
    ]
    target = 'hotspot'

    X = df[features]
    y = df[target]

    # Use the same random_state to ensure the same data split for a fair comparison
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.\n")

    # --- 3. Train the XGBoost Model ---
    # We set random_state for reproducibility and other parameters to handle updates.
    model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    print("Training the XGBoost model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 4. Evaluate the Model ---
    print("\n--- Model Performance ---")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Hotspot (0)', 'Hotspot (1)']))
    print("---------------------------------")


if __name__ == "__main__":
    INPUT_FILENAME = r'C:\Users\white\OneDrive\Desktop\Coding\My Work\Paper 3\cleaned_water_quality_data.csv'
    train_xgboost(INPUT_FILENAME)
