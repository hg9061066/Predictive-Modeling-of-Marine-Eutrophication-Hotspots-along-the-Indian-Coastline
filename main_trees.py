import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import sys

def train_decision_tree(input_file):
    """
    Loads cleaned data, trains a Decision Tree model to predict hotspots,
    and evaluates its performance.
    
    Args:
        input_file (str): The name of the cleaned CSV file.
    """
    print("--- Training and Evaluating Decision Tree Model ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    print(f"Successfully loaded '{input_file}'.")
    df.dropna(inplace=True)

    # --- 2. Prepare Data ---
    # Encode categorical features to be used by the model
    df['station_code_encoded'] = pd.factorize(df['station_code'])[0]
    df['location_name_encoded'] = pd.factorize(df['location_name'])[0]
    
    features = [
        'year', 'bod', 'dissolved_oxygen', 'nitrate', 
        'fecal_coliform', 'station_code_encoded', 'location_name_encoded'
    ]
    target = 'hotspot'

    X = df[features]
    y = df[target]

    # Split data into training and testing sets (using the same random_state for consistency)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.\n")

    # --- 3. Train the Decision Tree Model ---
    # We set random_state for reproducibility.
    model = DecisionTreeClassifier(random_state=42)
    
    print("Training the Decision Tree model...")
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
    train_decision_tree(INPUT_FILENAME)
