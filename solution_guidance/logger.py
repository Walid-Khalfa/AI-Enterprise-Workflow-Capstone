import os
import json
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs") # Place logs à la racine du projet

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

def _log_event(log_type, country, data, runtime, model_version, model_version_note=None, test=False):
    """
    Helper function to log training or prediction events.
    """
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "log_type": log_type,
        "country": country,
        "model_version": model_version,
        "model_version_note": model_version_note,
        "runtime": runtime,
        "test_mode": test
    }
    log_entry.update(data)

    prefix = "test-" if test else ""
    log_file_name = os.path.join(LOG_DIR, f"{prefix}{log_type}-{country}-{timestamp.replace(':', '-')}.json")
    
    try:
        # Configure logger with rotation
        logger = logging.getLogger('ai_logger')
        logger.setLevel(logging.INFO)
        handler = RotatingFileHandler(
            log_file_name,
            maxBytes=10485760,  # 10MB
            backupCount=5
        )
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        # Write log entry
        with open(log_file_name, 'w') as f:
            json.dump([log_entry], f, indent=4)
        
        # Log success message
        logger.info(f"Successfully logged {log_type} event for {country}")
    except Exception as e:
        logger.error(f"Error writing to log file {log_file_name}: {e}")
        raise

def update_train_log(country, date_range, metrics, runtime, model_version, model_version_note, test=False):
    """
    Logs training event details.
    country: Tag for the data used (e.g., 'all', 'france')
    date_range: Tuple (start_date_str, end_date_str) for training data
    metrics: Dictionary of evaluation metrics (e.g., {'rmse': 100.5})
    runtime: Training duration string (e.g., "00:05:30")
    model_version: Version of the model (e.g., 0.1)
    model_version_note: Note about the model version
    test: Boolean, True if it's a test run
    """
    log_data = {
        "training_data_start_date": date_range[0],
        "training_data_end_date": date_range[1],
        "evaluation_metrics": metrics
    }
    _log_event("train", country, log_data, runtime, model_version, model_version_note, test)

def update_predict_log(country, y_pred, y_proba, query_date, runtime, model_version, test=False):
    """
    Logs prediction event details.
    country: Country for which prediction was made
    y_pred: Predicted value(s)
    y_proba: Prediction probability (if applicable)
    query_date: Date for which the prediction was made (e.g., "2023-10-01")
    runtime: Prediction duration string
    model_version: Version of the model used
    test: Boolean, True if it's a test run
    """
    log_data = {
        "query_date": query_date,
        "y_pred": y_pred.tolist() if hasattr(y_pred, 'tolist') else y_pred, # Ensure serializable
        "y_proba": y_proba.tolist() if hasattr(y_proba, 'tolist') else y_proba # Ensure serializable
    }
    _log_event("predict", country, log_data, runtime, model_version, test=test)

if __name__ == '__main__':
    # Example Usage (for testing the logger itself)
    print(f"Log directory configured at: {os.path.abspath(LOG_DIR)}")
    
    # Test train log
    update_train_log(country="all", 
                     date_range=("2022-01-01", "2022-12-31"), 
                     metrics={'rmse': 123.45, 'mae': 90.2},
                     runtime="00:10:15", 
                     model_version=0.1, 
                     model_version_note="Initial supervised learning model", 
                     test=True)
    print("Test train log created.")

    # Test predict log
    import numpy as np
    update_predict_log(country="united_kingdom", 
                       y_pred=np.array([1500.75]), 
                       y_proba=np.array([[0.2, 0.8]]),
                       query_date="2023-05-10", 
                       runtime="00:00:02", 
                       model_version=0.1, 
                       test=True)
    print("Test predict log created.")

    update_predict_log(country="france", 
                       y_pred=np.array([250.0]), 
                       y_proba=None,
                       query_date="2023-06-15", 
                       runtime="00:00:01", 
                       model_version=0.2, 
                       test=False)
    print("Non-test predict log created.")