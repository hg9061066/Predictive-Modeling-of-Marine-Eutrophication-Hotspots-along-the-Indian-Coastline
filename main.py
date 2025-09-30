import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def train_and_evaluate_model(input_file):
    """
    Loads the cleaned water quality data, trains a Random Forest model
    to predict hotspots, evaluates its performance, and saves a
    feature importance plot.
    
    Args:
        input_file (str): The name of the cleaned CSV file.
    """
    print("--- Starting Model Training and Evaluation ---")

    # --- 1. Load Data ---
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
        print("Please make sure this script is in the same directory as your data file.")
        sys.exit(1)

    print(f"Successfully loaded '{input_file}'. Shape: {df.shape}")

    # --- 2. Feature Engineering and Preparation ---
    
    # Drop rows with missing values that might have been missed
    df.dropna(inplace=True)

    # Encode Categorical Features: We convert text columns into numbers
    # so the model can understand them.
    # 'station_code' and 'location_name' are encoded into numeric labels.
    df['station_code_encoded'] = pd.factorize(df['station_code'])[0]
    df['location_name_encoded'] = pd.factorize(df['location_name'])[0]
    
    # Define the features (predictors) and the target variable
    features = [
        'year', 
        'bod', 
        'dissolved_oxygen', 
        'nitrate', 
        'fecal_coliform',
        'station_code_encoded',
        'location_name_encoded'
    ]
    target = 'hotspot'

    X = df[features]
    y = df[target]

    print("\nFeatures prepared for modeling:")
    print(X.head())

    # --- 3. Split Data into Training and Testing Sets ---
    # We'll use 80% of the data to train the model and 20% to test its performance.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nData split into {len(X_train)} training samples and {len(X_test)} testing samples.")

    # --- 4. Train the Random Forest Model ---
    # We use a Random Forest because it's powerful and good at handling
    # different types of features. `n_estimators` is the number of "trees" in the forest.
    model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    
    print("\nTraining the Random Forest model...")
    model.fit(X_train, y_train)
    print("Model training complete.")

    # --- 5. Evaluate the Model ---
    print("\n--- Model Performance ---")
    
    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    # Print a detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Non-Hotspot (0)', 'Hotspot (1)']))
    
    # --- 6. Generate and Save Feature Importance Plot ---
    
    # Get importance scores from the trained model
    importances = model.feature_importances_
    
    # Create a DataFrame for visualization
    feature_importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Feature Importance for Hotspot Prediction', fontsize=16)
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Save the plot to a file
    plot_filename = 'feature_importance.png'
    plt.savefig(plot_filename)
    print(f"\nFeature importance plot saved as '{plot_filename}'")
    print("---------------------------------")


if __name__ == "__main__":
    INPUT_FILENAME = r'C:\Users\white\OneDrive\Desktop\Coding\My Work\Paper 3\cleaned_water_quality_data.csv'
    train_and_evaluate_model(INPUT_FILENAME)