import pandas as pd
import os
import re
import json
from datetime import datetime

LOG_DIR = "logs"  # Supposons que les logs de prédiction sont dans un sous-répertoire 'logs'
PREDICT_LOG_PATTERN = r"predict-([a-zA-Z_]+)-([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6})\.json"
API_LOG_FILE = "../api.log" # Chemin relatif depuis solution_guidance vers api.log à la racine

def analyze_prediction_logs(log_dir=LOG_DIR):
    """
    Analyzes prediction logs to extract insights.
    Assumes logs are JSON files in the format: predict-<country>-<timestamp>.json
    Each JSON file contains a list of prediction records.
    """
    if not os.path.exists(log_dir):
        print(f"Log directory '{log_dir}' not found.")
        return pd.DataFrame()

    all_predictions = []
    for filename in os.listdir(log_dir):
        match = re.match(PREDICT_LOG_PATTERN, filename)
        if match:
            country = match.group(1)
            timestamp_str = match.group(2)
            try:
                with open(os.path.join(log_dir, filename), 'r') as f:
                    log_data = json.load(f)
                    if isinstance(log_data, list): # Expecting a list of log entries
                        for entry in log_data:
                            entry['country'] = country
                            entry['log_timestamp'] = datetime.fromisoformat(timestamp_str)
                            all_predictions.append(entry)
                    else:
                        print(f"Warning: Log file {filename} does not contain a list.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from {filename}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    
    if not all_predictions:
        print("No valid prediction log entries found.")
        return pd.DataFrame()

    df = pd.DataFrame(all_predictions)
    
    # Convert relevant columns to appropriate types
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if 'y_pred' in df.columns:
        # Assuming y_pred might be a list/array string representation in some logs
        df['y_pred'] = df['y_pred'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

    return df

def analyze_api_logs(api_log_file=API_LOG_FILE):
    """
    Analyzes the main API log file for errors and endpoint usage.
    """
    if not os.path.exists(api_log_file):
        print(f"API log file '{api_log_file}' not found.")
        return None, None

    error_lines = []
    endpoint_counts = {}
    
    with open(api_log_file, 'r') as f:
        for line in f:
            if "ERROR" in line:
                error_lines.append(line.strip())
            
            # Example: 2023-10-27 10:00:00,123 - INFO - Request processed: POST /predict - Status: 200 - Duration: 0.0010s
            match = re.search(r"Request processed: (GET|POST) (/\w+).*Status: (\d+)", line)
            if match:
                method = match.group(1)
                endpoint = match.group(2)
                status = match.group(3)
                key = f"{method} {endpoint}"
                if key not in endpoint_counts:
                    endpoint_counts[key] = {'total': 0, 'success': 0, 'error': 0}
                endpoint_counts[key]['total'] += 1
                if status.startswith('2'):
                    endpoint_counts[key]['success'] += 1
                else:
                    endpoint_counts[key]['error'] += 1
                    
    return error_lines, endpoint_counts

def generate_report(predictions_df, api_errors, api_endpoint_counts):
    """
    Generates a summary report based on the analysis.
    """
    print("--- Post-Production Analysis Report ---")
    
    # Prediction Log Analysis
    print("\n--- Prediction Log Analysis ---")
    if not predictions_df.empty:
        print(f"Total predictions logged: {len(predictions_df)}")
        if 'country' in predictions_df.columns:
            print("Predictions by country:")
            print(predictions_df['country'].value_counts())
        if 'y_pred' in predictions_df.columns:
            # Ensure y_pred is numeric for describe()
            predictions_df['y_pred_numeric'] = pd.to_numeric(predictions_df['y_pred'], errors='coerce')
            print("\nSummary of y_pred (numeric predictions only):")
            print(predictions_df['y_pred_numeric'].describe())
        # Add more analysis here: e.g., prediction distribution, drift detection placeholders
        print("\nFurther analysis could include: prediction value distribution, drift over time (if applicable).")
    else:
        print("No prediction data to analyze.")

    # API Log Analysis
    print("\n--- API Log Analysis ---")
    if api_errors is not None:
        print(f"Total error lines in API log: {len(api_errors)}")
        if api_errors:
            print("Last 5 API errors:")
            for err in api_errors[-5:]:
                print(err)
    else:
        print("API log not found or not analyzed.")

    if api_endpoint_counts is not None:
        print("\nAPI Endpoint Usage:")
        for endpoint, counts in api_endpoint_counts.items():
            print(f"  {endpoint}: Total={counts['total']}, Success={counts['success']}, Error={counts['error']}")
    else:
        print("API endpoint usage data not available.")
        
    print("\n--- End of Report ---")

if __name__ == "__main__":
    # Create dummy log directory and files for testing if they don't exist
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        # Create a dummy predict log
        dummy_log_data = [
            {"timestamp": "2023-01-01T10:00:00", "y_pred": [120.5], "y_proba": None, "runtime": "000:00:01", "model_version": 0.1},
            {"timestamp": "2023-01-01T10:05:00", "y_pred": [150.0], "y_proba": None, "runtime": "000:00:01", "model_version": 0.1}
        ]
        with open(os.path.join(LOG_DIR, f"predict-all-{datetime.now().isoformat().replace(':','-')}.json"), 'w') as f:
            json.dump(dummy_log_data, f)
    
    # Create a dummy api.log for testing if it doesn't exist
    if not os.path.exists(API_LOG_FILE):
        with open(API_LOG_FILE, 'w') as f:
            f.write(f"{datetime.now().isoformat()} - INFO - Request received: POST /train\n")
            f.write(f"{datetime.now().isoformat()} - INFO - Request processed: POST /train - Status: 200 - Duration: 10.0s\n")
            f.write(f"{datetime.now().isoformat()} - ERROR - Error during prediction: Some error\n")

    predictions_df = analyze_prediction_logs()
    api_errors, api_endpoint_counts = analyze_api_logs()
    generate_report(predictions_df, api_errors, api_endpoint_counts)